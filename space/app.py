"""Make a Halloween card. Importing on a Mac builds the UI without weights."""
import os

import spaces  # Import before any CUDA initialization for ZeroGPU.
import gradio as gr

from pipeline import (
    ASTRONAUTS, CHECKPOINTS, PROMPT_PREFIX, SCARECROW,
    build_pipeline, build_upsampler, generate_image, resolve_prompt, upsample, validate_settings,
)

UI_ONLY = os.environ.get("HALLOWEEN_UI_ONLY") == "1"
# ZeroGPU requires CUDA placement at module startup. Mac review stays CPU-only.
ON_SPACE = os.environ.get("SPACE_ID") and not UI_ONLY
PIPE = build_pipeline() if ON_SPACE else None
UPSAMPLER = build_upsampler() if ON_SPACE else None


@spaces.GPU(duration=150)
def render(prompt, subject, expand, seed, strength, checkpoint):
    if PIPE is None:
        raise gr.Error("You can try the controls here. Run on a CUDA GPU to make a card.")
    if expand:
        prompt = upsample(UPSAMPLER, subject)
    return generate_image(PIPE, prompt, seed, strength, checkpoint), prompt


def make_card(subject, expand, full_prompt, seed, strength, checkpoint):
    # An untouched textbox inside a closed accordion arrives as None.
    subject, full_prompt = subject or "", full_prompt or ""
    try:
        prompt = resolve_prompt(subject, full_prompt)
        seed, strength = validate_settings(seed, strength, checkpoint)
    except ValueError as error:
        raise gr.Error(str(error)) from error
    return render(prompt, subject, expand and not full_prompt.strip(), seed, strength, checkpoint)


with gr.Blocks(title="Make a Halloween card") as demo:
    gr.Markdown("# Make a Halloween card\nDescribe what you want on your card. FLUX 3 Image Klein makes it into a hand-drawn Halloween greeting.")
    with gr.Row():
        with gr.Column():
            subject = gr.Textbox(label="Your scene", placeholder="A black cat watching the moon from a pumpkin patch", lines=4)
            expand = gr.Checkbox(label="Expand my prompt", value=True)
            with gr.Accordion("Advanced", open=False):
                full_prompt = gr.Textbox(label="Full prompt", placeholder="Use your own prompt here, including vqx. Leave blank to use your scene with the card prompt.", lines=5)
                seed = gr.Number(label="Seed", value=23, precision=0, minimum=0, maximum=2**32 - 1)
                strength = gr.Slider(label="LoRA strength", minimum=0, maximum=1.5, value=1.0, step=0.05)
                checkpoint = gr.Dropdown(label="Checkpoint", choices=list(CHECKPOINTS), value="constant_with_warmup")
            generate = gr.Button("Make my card", variant="primary")
        with gr.Column():
            image = gr.Image(label="Your Halloween card", type="pil", format="png")
            used_prompt = gr.Textbox(label="Prompt used", lines=4, interactive=False)
    inputs = [subject, expand, full_prompt, seed, strength, checkpoint]
    gr.Examples(
        examples=[
            [SCARECROW.removeprefix(PROMPT_PREFIX), False, "", 23, 1.0, "cosine"],
            [ASTRONAUTS.removeprefix(PROMPT_PREFIX), False, "", 23, 1.0, "constant_with_warmup"],
        ],
        inputs=inputs,
        label="Try a scene",
        cache_examples=False,
    )
    generate.click(make_card, inputs=inputs, outputs=[image, used_prompt], concurrency_limit=1, concurrency_id="card")
    subject.submit(make_card, inputs=inputs, outputs=[image, used_prompt], concurrency_limit=1, concurrency_id="card")

if __name__ == "__main__":
    if not UI_ONLY and PIPE is None:
        PIPE, UPSAMPLER = build_pipeline(), build_upsampler()
    demo.queue(max_size=8).launch()
