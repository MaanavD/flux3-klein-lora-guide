"""Make a Halloween card. Importing on a Mac builds the UI without weights."""
import os
import time
from pathlib import Path

import spaces  # Import before any CUDA initialization for ZeroGPU.
import gradio as gr

from pipeline import (
    CHECKPOINTS, PUP_URL, WITCH,
    build_pipeline, build_upsampler, generate_image, resolve_prompt, upsample, upsample_endpoint, validate_settings,
)

UI_ONLY = os.environ.get("HALLOWEEN_UI_ONLY") == "1"
# ZeroGPU requires CUDA placement at module startup. Mac review stays CPU-only.
ON_SPACE = os.environ.get("SPACE_ID") and not UI_ONLY
PIPE = build_pipeline() if ON_SPACE else None
UPSAMPLER = build_upsampler() if ON_SPACE and not PUP_URL else None

GUIDE = "https://docs.bfl.ai/flux_3/flux3_image_klein_lora_guide"
LORA = "https://huggingface.co/maanavdalal/flux3-klein-halloween-lora"
REPO = "https://github.com/MaanavD/flux3-klein-lora-guide"
SCENES = [
    "A witch flying over a sleepy village",
    "A black cat on a garden fence under a crescent moon",
    "Two astronauts sharing lunch on the moon",
]

NIGHT, DUSK, LINE = "#1E1830", "#2A2342", "#433A5E"
PUMPKIN, MOON, STOCK, MUTED = "#EE7A1F", "#F2C14E", "#F0E4C9", "#B8AECB"
# The same colors in light and dark mode, so the card always sits on a night sky.
COLORS = {
    "body_background_fill": NIGHT, "body_text_color": STOCK, "body_text_color_subdued": MUTED,
    "background_fill_primary": DUSK, "background_fill_secondary": NIGHT,
    "block_background_fill": DUSK, "block_border_color": LINE, "border_color_primary": LINE,
    "block_label_background_fill": DUSK, "block_label_text_color": MUTED, "block_title_text_color": STOCK,
    "input_background_fill": NIGHT, "input_border_color": LINE, "input_border_color_focus": MOON,
    "input_placeholder_color": MUTED,
    "button_primary_background_fill": PUMPKIN, "button_primary_background_fill_hover": "#F58D3B",
    "button_primary_text_color": NIGHT, "button_primary_border_color": PUMPKIN,
    "button_secondary_background_fill": NIGHT, "button_secondary_background_fill_hover": LINE,
    "button_secondary_text_color": STOCK, "button_secondary_border_color": LINE,
    "checkbox_background_color": NIGHT, "checkbox_background_color_selected": PUMPKIN,
    "checkbox_border_color": MUTED, "checkbox_border_color_selected": PUMPKIN,
    "checkbox_label_background_fill": DUSK, "checkbox_label_text_color": STOCK,
    "color_accent": PUMPKIN, "color_accent_soft": LINE, "slider_color": PUMPKIN,
    "link_text_color": MOON, "link_text_color_hover": PUMPKIN, "link_text_color_visited": MOON,
    "accordion_text_color": STOCK, "panel_background_fill": DUSK, "border_color_accent": PUMPKIN,
}
THEME = gr.themes.Base(
    font=[gr.themes.GoogleFont("Libre Franklin"), "system-ui", "sans-serif"],
    radius_size=gr.themes.sizes.radius_sm,
)
THEME.set(**COLORS, **{name + "_dark": value for name, value in COLORS.items() if hasattr(THEME, name + "_dark")})
CSS = Path(__file__).with_name("style.css").read_text()
HEAD = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght,SOFT,WONK@9..144,600..900,0..100,0..1&display=swap">'


def gpu_seconds(prompt, subject, expand, seed, strength, checkpoint):
    # A 50-step card takes about 62 s on a ZeroGPU slice; upsampling in the slot adds about 15 s.
    return 100 if expand else 75


@spaces.GPU(duration=gpu_seconds)
def render(prompt, subject, expand, seed, strength, checkpoint):
    if PIPE is None:
        raise gr.Error("You can try the controls here. Run on a CUDA GPU to make a card.")
    start = time.time()
    if expand:
        prompt = upsample(UPSAMPLER, subject)
    upsampled = time.time()
    image = generate_image(PIPE, prompt, seed, strength, checkpoint)
    print(f"GPU time: upsample {upsampled - start:.1f}s, render {time.time() - upsampled:.1f}s")
    return image, prompt


def make_card(subject, expand, full_prompt, seed, strength, checkpoint):
    # An untouched textbox inside a closed accordion arrives as None.
    subject, full_prompt = subject or "", full_prompt or ""
    try:
        prompt = resolve_prompt(subject, full_prompt)
        seed, strength = validate_settings(seed, strength, checkpoint)
    except ValueError as error:
        raise gr.Error(str(error)) from error
    expand = expand and not full_prompt.strip()
    # The endpoint runs outside the GPU slot, as in the official Klein Space.
    if expand and PUP_URL:
        try:
            prompt = upsample_endpoint(subject)
        except Exception as error:
            print(f"Upsampling failed: {error}")
            raise gr.Error("The prompt upsampler is unavailable. Try again, or turn off Expand my prompt.") from error
        expand = False
    return render(prompt, subject, expand, seed, strength, checkpoint)


with gr.Blocks(title="Make a Halloween card") as demo:
    with gr.Column(elem_id="hero"):
        gr.HTML('<h1>Make a Halloween card</h1><p>Describe a scene, and FLUX 3 Image Klein draws it as a Halloween postcard from the early 1900s.</p>')
    with gr.Row(equal_height=False, elem_id="main"):
        with gr.Column(scale=5, min_width=300, elem_id="controls"):
            subject = gr.Textbox(label="Your scene", placeholder="A black cat watching the moon from a pumpkin patch", lines=3, max_lines=6)
            with gr.Row(elem_id="scenes"):
                scene_buttons = [gr.Button(scene, size="sm", variant="secondary", elem_classes="scene") for scene in SCENES]
            expand = gr.Checkbox(label="Expand my prompt", info="Adds detail to your scene before drawing it.", value=True)
            generate = gr.Button("Make my card", variant="primary", size="lg", elem_id="make")
            gr.HTML('<p class="note">Each card takes about a minute.</p>')
            with gr.Accordion("Advanced", open=False):
                full_prompt = gr.Textbox(label="Full prompt", placeholder="Write the whole prompt yourself, starting with vqx. Leave blank to use your scene.", lines=4)
                seed = gr.Number(label="Seed", value=23, precision=0, minimum=0, maximum=2**32 - 1)
                strength = gr.Slider(label="LoRA strength", minimum=0, maximum=1.5, value=1.0, step=0.05)
                checkpoint = gr.Dropdown(label="Checkpoint", choices=list(CHECKPOINTS), value="constant_with_warmup")
        with gr.Column(scale=7, min_width=300, elem_id="card-column"):
            image = gr.Image(value=str(Path(__file__).with_name("assets") / "witch.jpg"), type="pil", format="png",
                             show_label=False, label="Your card", elem_id="postcard")
            used_prompt = gr.Textbox(value=WITCH, label="Prompt used", lines=3, max_lines=10, interactive=False, elem_id="used")
    gr.HTML(
        f'<footer id="about"><p>The LoRA learned this look from 65 Halloween postcards printed between 1900 and 1920. '
        f'To train one on your own pictures, read the <a href="{GUIDE}" target="_blank">LoRA guide</a>. '
        f'The <a href="{LORA}" target="_blank">LoRA</a> and the <a href="{REPO}" target="_blank">dataset and training code</a> are free to download.</p></footer>'
    )
    inputs = [subject, expand, full_prompt, seed, strength, checkpoint]
    for button, scene in zip(scene_buttons, SCENES):
        button.click(lambda scene=scene: scene, outputs=subject, queue=False)
    gr.on([generate.click, subject.submit], make_card, inputs=inputs, outputs=[image, used_prompt],
          concurrency_limit=1, concurrency_id="card")

if __name__ == "__main__":
    if not UI_ONLY and PIPE is None:
        PIPE = build_pipeline()
        UPSAMPLER = None if PUP_URL else build_upsampler()
    demo.queue(max_size=16).launch(theme=THEME, css=CSS, head=HEAD)
