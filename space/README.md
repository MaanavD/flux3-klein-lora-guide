---
title: Make a Halloween card
emoji: 🎃
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 6.30.0
python_version: '3.12'
app_file: app.py
pinned: false
license: other
license_name: flux-kommunity-license
license_link: https://huggingface.co/black-forest-labs/flux-3-image-klein-base/blob/main/LICENSE.md
suggested_hardware: zero-a10g
---

# Make a Halloween card

Describe a scene and FLUX 3 Image Klein, with a LoRA trained on Halloween postcards from the early 1900s, draws it as an old Halloween greeting card. The training data, configs and checkpoints are in [MaanavD/flux3-klein-lora-guide](https://github.com/MaanavD/flux3-klein-lora-guide).

## Controls

The app puts your scene into `vqx, an old Halloween greeting card with a hand-drawn illustration of <scene>`. `vqx` is the trigger word. Under Advanced you can write your own full prompt instead (include `vqx` yourself), and change the seed (default 23), LoRA strength (0 to 1.5, default 1) and checkpoint (`constant_with_warmup` or `cosine`, both from step 2,000).

When a prompt upsampler is configured, "Expand my prompt" sends your scene to [the Klein prompt upsampler](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup) first, asking for a flat scan of an old greeting card, and adds `vqx` back to the caption it returns.

Each card is 1024 x 1024, 50 steps, guidance 3.5.

## Run locally

Making a card needs a CUDA GPU and access to [the base model](https://huggingface.co/black-forest-labs/flux-3-image-klein-base).

```bash
python -m pip install -r requirements.txt
HALLOWEEN_LORA_SOURCE=https://github.com/MaanavD/flux3-klein-lora-guide/releases/download/checkpoints-2000 python app.py
```

`HALLOWEEN_LORA_SOURCE` can also be a local folder holding both checkpoints. Set `HALLOWEEN_UI_ONLY=1` to open the controls without loading the model.

## License and attribution

FLUX 3 Image Klein is by Black Forest Labs. The base model and its derivatives, including this LoRA, are covered by the [FLUX Kommunity License v.1.0](LICENSE.md). The Qwen3-VL-4B-Instruct text encoder, Diffusers and Transformers have their own licenses. Maanav Dalal trained the LoRA with [AI Toolkit](https://github.com/ostris/ai-toolkit) at rank 32.
