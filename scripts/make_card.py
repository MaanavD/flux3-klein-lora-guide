"""Make a Halloween card with FLUX 3 Image Klein and a LoRA checkpoint.

Run this from the AI Toolkit root, with its venv active.
  PYTHONPATH="$PWD" python /path/to/make_card.py --lora halloween-lora_000002000.safetensors \
    --subject 'a black cat on a garden fence. The card has the clearly readable greeting "Happy Halloween" across the top.'

--subject goes through the FLUX 3 Image Klein prompt upsampler first, then vqx is put in front of
the caption it returns. Use --prompt instead to pass a full caption, starting with vqx, as is.
"""
import argparse
import gc
import json

import torch
from extensions_built_in.diffusion_models.flux3.flux3 import Flux3Model
from toolkit.config_modules import GenerateImageConfig, ModelConfig, NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork

p = argparse.ArgumentParser()
p.add_argument("--lora", required=True, help="path to your .safetensors checkpoint")
g = p.add_mutually_exclusive_group(required=True)
g.add_argument("--subject", help="what's on the card; the prompt upsampler writes the full caption")
g.add_argument("--prompt", help="a full caption starting with vqx, used as is")
p.add_argument("--out", default="card.jpg")
p.add_argument("--seed", type=int, default=23)
p.add_argument("--strength", type=float, default=1.0)
p.add_argument("--steps", type=int, default=50)
p.add_argument("--guidance", type=float, default=3.5)
p.add_argument("--rank", type=int, default=32, help="use the rank you trained with")
args = p.parse_args()

# The upsampler has never seen vqx, so it gets the request without it and vqx goes in front of its caption.
# Asking for a flat scan keeps it from photographing the card on a table.
REQUEST = "a flat, straight-on archival scan of an old Halloween greeting card filling the frame, with a hand-drawn illustration of "


def upsample(subject):
    from huggingface_hub import hf_hub_download
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    pup = "black-forest-labs/flux-3-image-klein-pup"
    processor = AutoProcessor.from_pretrained(pup)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(pup, dtype=torch.bfloat16, device_map="auto").eval()
    system = json.load(open(hf_hub_download(pup, "system_prompts.json")))["t2i"]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [{"type": "text", "text": REQUEST + subject}]},
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False
    ).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    caption = processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return "vqx, " + caption


prompt = upsample(args.subject) if args.subject else args.prompt
print("prompt:", prompt)

sd = Flux3Model(device="cuda:0", dtype="bf16",
                model_config=ModelConfig(name_or_path="black-forest-labs/flux-3-image-klein-base", arch="flux3", dtype="bf16"))
sd.load_model()
for part in (sd.model, sd.text_encoder, sd.vae):
    part.requires_grad_(False)
    part.eval()

net_config = NetworkConfig(type="lora", linear=args.rank, linear_alpha=args.rank)
network = LoRASpecialNetwork(
    text_encoder=sd.text_encoder, unet=sd.get_model_to_train(), lora_dim=args.rank, alpha=args.rank,
    train_text_encoder=False, train_unet=True, is_transformer=True, transformer_only=net_config.transformer_only,
    base_model=sd, network_config=net_config, target_lin_modules=sd.target_lora_modules)
network.apply_to(sd.text_encoder, sd.get_model_to_train(), False, True)
network.force_to(torch.device("cuda:0"), torch.float32)
network._update_torch_multiplier()
network.can_merge_in = False
sd.network = network
network.load_weights(args.lora)

sd.generate_images([GenerateImageConfig(
    prompt=prompt, negative_prompt="", width=1024, height=1024, seed=args.seed,
    guidance_scale=args.guidance, num_inference_steps=args.steps, network_multiplier=args.strength,
    output_path=args.out, output_ext="jpg", add_prompt_file=False)], sampler="flowmatch")
print("saved", args.out)
