"""Inference for FLUX 3 Image Klein with the two Halloween adapters."""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import torch
from diffusers import Flux3AutoBlocks
from huggingface_hub import hf_hub_download

BASE_MODEL = "black-forest-labs/flux-3-image-klein-base"
# Set this one value to a public HF model repo ID or an HTTPS directory URL.
# A local directory is also accepted for private, local verification.
LORA_SOURCE = os.environ.get("HALLOWEEN_LORA_SOURCE", "")
CHECKPOINTS = {
    "constant_with_warmup": (
        "klein-vqx-shift4k-warmup_000002000.safetensors",
        "e95c410acf88ba418aaf224e6f3d3b5e323ae10c5133cc5e7f82d8c58a52d3ae",
    ),
    "cosine": (
        "klein-vqx-shift4k-cosine_000002000.safetensors",
        "b3ea1920666e045e0b36663fabf9cecdbdaa4c189e4688a0c3117d81cf897efd",
    ),
}
TRIGGER = "vqx, "
CARD_REQUEST = "an old Halloween greeting card with a hand-drawn illustration of "
PROMPT_PREFIX = TRIGGER + CARD_REQUEST
# Without "flat scan", the upsampler photographs the card on a tabletop and hides the LoRA's look.
UPSAMPLE_REQUEST = "a flat, straight-on archival scan of an old Halloween greeting card filling the frame, with a hand-drawn illustration of "
# The FLUX 3 Image Klein prompt upsampler. It runs in the same ZeroGPU slot as Klein.
PUP_REPO = "black-forest-labs/flux-3-image-klein-pup"
SCARECROW = PROMPT_PREFIX + 'a scarecrow standing in a pumpkin patch at dusk under a large harvest moon, with an owl on the fence post, black crows on the scarecrow\'s arms and corn shocks behind him. The card has the clearly readable greeting "Best Wishes for Halloween" across the top.'
ASTRONAUTS = PROMPT_PREFIX + 'two astronauts sharing lunch at a table inside a lunar outpost, with Earth visible through a round window. A jack-o-lantern sits on the table beside their food and a small paper bat decorates the window. The card has the clearly readable greeting "Have a Happy Halloween" across the top.'


def resolve_prompt(subject: str, full_prompt: str = "") -> str:
    prompt = full_prompt.strip() or (PROMPT_PREFIX + subject.strip() if subject.strip() else "")
    if not prompt:
        raise ValueError("Describe what you want on your card.")
    if len(prompt) > 6000:
        raise ValueError("Keep the prompt under 6,000 characters.")
    return prompt


def build_upsampler(device="cuda"):
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    model = Qwen3_5ForConditionalGeneration.from_pretrained(PUP_REPO, dtype=torch.bfloat16, token=os.getenv("HF_TOKEN"))
    processor = AutoProcessor.from_pretrained(PUP_REPO, token=os.getenv("HF_TOKEN"))
    return model.to(device).eval(), processor


@torch.inference_mode()
def upsample(upsampler, subject: str) -> str:
    """Expand the card request into a full caption, then put the trigger in front.

    The upsampler has never seen vqx, so it gets the request without it and the
    trigger is added back to the caption it returns.
    """
    from diffusers.modular_pipelines.flux3.prompt_upsampler import upsample_prompt

    model, processor = upsampler
    return TRIGGER + upsample_prompt(model, processor, UPSAMPLE_REQUEST + subject.strip(), max_new_tokens=1024)


def validate_settings(seed, strength, checkpoint):
    if not isinstance(seed, (int, float)) or not math.isfinite(seed) or int(seed) != seed or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Choose a whole number for the seed, from 0 to 4,294,967,295.")
    if not isinstance(strength, (int, float)) or not math.isfinite(strength) or not 0 <= strength <= 1.5:
        raise ValueError("Choose a LoRA strength from 0 to 1.5.")
    if checkpoint not in CHECKPOINTS:
        raise ValueError("Choose one of the two checkpoints.")
    return int(seed), float(strength)


def verify_checkpoint(path: Path, checkpoint: str) -> Path:
    expected = CHECKPOINTS[checkpoint][1]
    with path.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    if digest != expected:
        raise ValueError(f"The {checkpoint} checkpoint does not match the inspected release file at {path}")
    return path


def checkpoint_path(checkpoint: str, source: str = LORA_SOURCE) -> Path:
    if not source:
        raise ValueError("Set HALLOWEEN_LORA_SOURCE to the weights location before making cards.")
    filename = CHECKPOINTS[checkpoint][0]
    local = Path(source).expanduser()
    if local.is_dir():
        path = local / filename
    elif urlparse(source).scheme == "https":
        cache = Path(__file__).parent / ".cache" / "checkpoints"
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / filename
        if not path.exists():
            # Exclusive creation preserves existing files. A failed partial download is
            # kept for inspection; move it aside manually before retrying.
            with urlopen(source.rstrip("/") + "/" + filename, timeout=60) as response, path.open("xb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
    elif "://" in source or source.startswith(("/", ".", "~")):
        raise ValueError("For the weights location, use an existing directory, HF repo ID, or HTTPS directory URL.")
    else:
        path = Path(hf_hub_download(repo_id=source, filename=filename, token=os.getenv("HF_TOKEN")))
    return verify_checkpoint(path, checkpoint)


def build_pipeline(*, lora_paths=None, components=None, device="cuda"):
    """Build on startup, before entering the ZeroGPU function.

    Optional components let the parity harness reuse the exact cached base
    weights without downloading a second copy. Production loads from HF.
    """
    paths = {name: Path(lora_paths[name]) if lora_paths is not None else checkpoint_path(name)
             for name in CHECKPOINTS}
    if lora_paths is not None:
        for name, path in paths.items():
            verify_checkpoint(path, name)
    blocks = Flux3AutoBlocks()
    blocks.sub_blocks.pop("prompt_upsampler")
    pipe = blocks.init_pipeline(BASE_MODEL if components is None else None)
    if components is None:
        pipe.load_components(dtype={"vae": torch.bfloat16, "default": torch.bfloat16}, token=os.getenv("HF_TOKEN"))
    else:
        pipe.update_components(**components)
    pipe.guider.guidance_scale = 3.5
    pipe.guider.use_original_formulation = False
    # Text-only cards do not need the Qwen vision tower.
    if getattr(pipe.text_encoder.model, "visual", None) is not None:
        pipe.text_encoder.model.visual = None
    pipe.tokenizer.tokenizer.padding_side = "right"
    for name in CHECKPOINTS:
        path = paths[name]
        # Diffusers recognizes diffusion_model.*.lora_A/B.weight and maps
        # original module names to its transformer layout automatically.
        pipe.load_lora_weights(str(path), adapter_name=name)
    # Match the reference's unmerged FP32 adapter computation.
    from peft.tuners.tuners_utils import BaseTunerLayer
    for module in pipe.transformer.modules():
        if isinstance(module, BaseTunerLayer):
            for adapter in module.lora_A.values():
                adapter.float()
            for adapter in module.lora_B.values():
                adapter.float()
    pipe.set_adapters("constant_with_warmup", adapter_weights=1.0)
    pipe.to(device)
    for model in (pipe.transformer, pipe.text_encoder, pipe.vae):
        model.eval().requires_grad_(False)
    return pipe


@torch.inference_mode()
def generate_image(pipe, prompt: str, seed=23, strength=1.0, checkpoint="constant_with_warmup"):
    """The same inference entry point used by the UI and parity check."""
    seed, strength = validate_settings(seed, strength, checkpoint)
    pipe.set_adapters(checkpoint, adapter_weights=strength)
    # A CPU generator matches make_card.py's initial noise source.
    return pipe(
        prompt=prompt,
        negative_prompt="",
        height=1024,
        width=1024,
        num_inference_steps=50,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        output="images",
    )[0]
