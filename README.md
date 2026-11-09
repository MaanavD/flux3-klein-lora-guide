# Train a LoRA on FLUX 3 Image Klein

Datasets, AI Toolkit configs and notebooks for training a LoRA on FLUX 3 Image Klein. They train with [AI Toolkit](https://github.com/ostris/ai-toolkit) by ostris at commit `04733873cf1da8060fce01316ec45daedbd4a918`.

- **Ukiyo-e, from my video:** 246 color ukiyo-e prints with the trigger word `ukiyocolor`. Start with [ukiyocolor_lora.ipynb](ukiyocolor_lora.ipynb).
- **Halloween postcards:** 65 cards from about 1900 to 1920 with the trigger word `vqx`, plus released checkpoints and a Space. Start with [halloween_lora.ipynb](halloween_lora.ipynb).

For the full written walkthrough, from dataset and captions to picking a checkpoint, see [Train your first LoRA on FLUX 3 Image Klein](https://docs.bfl.ai/flux_3/flux3_image_klein_lora_guide).

## Files

```
ukiyoe_color/           246 color ukiyo-e prints with captions, metadata.csv
ukiyocolor_lora.ipynb   ukiyo-e training notebook
halloween/              65 .jpg images, 65 .txt captions, metadata.csv
halloween_lora.ipynb    Halloween training notebook
configs/                AI Toolkit configs
scripts/                make_card.py, which loads a checkpoint and makes an image
ukiyoe/                 156 black and white ukiyo-e prints with captions
space/                  the Hugging Face Space app
samples/                comparison cards and training previews
LICENSE                 MIT license for code and configs
LICENSE-FLUX.md         FLUX Kommunity License, for the LoRA checkpoints
NOTICE.md               attribution notice for the LoRA checkpoints
```

- `ukiyocolor.yaml`: the ukiyo-e recipe, the same settings as `halloween-lora-cosine.yaml`.
- `halloween-lora-cosine.yaml`: 4,000 steps, cosine learning rate schedule.
- `halloween-lora-warmup.yaml`: the same, with a constant schedule after 200 warmup steps.

## Ukiyo-e

Clone this repo and AI Toolkit, set `folder_path` and `training_folder` in [configs/ukiyocolor.yaml](configs/ukiyocolor.yaml), then run this from the AI Toolkit root, with its venv active.

```bash
python run.py /path/to/flux3-klein-lora-guide/configs/ukiyocolor.yaml
```

The prints come from the [Cleveland Museum of Art Open Access](https://www.clevelandart.org/open-access) collection, CC0. See [ukiyoe_color/README.md](ukiyoe_color/README.md) for captions, sources and checksums.

## Halloween postcards

### Samples

The [samples](samples/README.md) include the final card comparison at steps 2,000 and 4,000, plus the base model. There are also training previews from cosine and constant_with_warmup, my two favorite runs. The prompts and seeds are included.

### Train

Start with the [training notebook](halloween_lora.ipynb).

1. Clone this repo, then clone AI Toolkit and check out the commit above.
2. In the config, set `folder_path` to the absolute path of `halloween/` and `training_folder` to your output directory.
3. Run this from the AI Toolkit root, with its venv active.

```bash
python run.py /path/to/flux3-klein-lora-guide/configs/halloween-lora-cosine.yaml
```

### Image sources

49 images come from The Public Domain Review and 16 from Wikimedia Commons. `halloween/metadata.csv` lists the source page and license for each image.

### Make a card

My favorites were cosine and constant_with_warmup, both at step 2,000. Both checkpoints from the guide are in the [checkpoints-2000 release](https://github.com/MaanavD/flux3-klein-lora-guide/releases/tag/checkpoints-2000). They're also on Hugging Face as [maanavdalal/flux3-klein-halloween-lora](https://huggingface.co/maanavdalal/flux3-klein-halloween-lora). Run this from the AI Toolkit root, with its venv active.

```bash
PYTHONPATH="$PWD" python /path/to/flux3-klein-lora-guide/scripts/make_card.py \
  --lora klein-vqx-shift4k-cosine_000002000.safetensors \
  --subject 'a black cat on a garden fence. The card has the clearly readable greeting "Happy Halloween" across the top.' \
  --out card.jpg
```

`--subject` goes through the [FLUX 3 Image Klein prompt upsampler](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup) first, which writes the full caption, and the script puts `vqx` in front of it. To pass a full caption yourself, use `--prompt 'vqx, ...'` instead. The script makes a 1024 x 1024 card with 50 steps, guidance 3.5 and LoRA strength 1. Change `--strength`, `--seed` or `--steps` to taste.

Or make one in the browser with the [Hugging Face Space](https://huggingface.co/spaces/maanavdalal/flux3-klein-halloween-card). Its code is in `space/`.

## More data

There's also a black and white set: 156 ukiyo-e prints from the Art Institute of Chicago, all public domain. They're in `ukiyoe/`, captioned with the trigger word `ukiyobw`. See [ukiyoe/README.md](ukiyoe/README.md) for sources and checksums.

## License

Code and configs use the [MIT license](LICENSE). The images keep their own licenses, as listed in `halloween/metadata.csv`, `ukiyoe_color/metadata.csv` and the `ukiyoe/` docs. The LoRA checkpoints are a Derivative of FLUX 3 Image Klein Base and are covered by the [FLUX Kommunity License](LICENSE-FLUX.md); see [NOTICE.md](NOTICE.md).
