# FLUX 3 Image Klein Halloween postcard LoRA

I trained this LoRA on 65 Halloween postcards from about 1900 to 1920. The set uses the trigger word `vqx`. The configs train it on Klein with [AI Toolkit](https://github.com/ostris/ai-toolkit) by ostris at commit `04733873cf1da8060fce01316ec45daedbd4a918`.

## Files

```
halloween/             65 .jpg images, 65 .txt captions, metadata.csv
configs/               AI Toolkit configs
scripts/               make_card.py
halloween_lora.ipynb   training notebook
ukiyoe/                a second dataset: 156 ukiyo-e prints with captions
space/                 the Hugging Face Space app
samples/               comparison cards and training previews
LICENSE                MIT license for code and configs
```

- `halloween-lora-cosine.yaml`: 4,000 steps, cosine learning rate schedule.
- `halloween-lora-warmup.yaml`: the same, with a constant schedule after 200 warmup steps.

## Samples

The [samples](samples/README.md) include the final card comparison at steps 2,000 and 4,000, plus the base model. There are also training previews from cosine and constant_with_warmup, my two favorite runs. The prompts and seeds are included.

## Train

Start with the [training notebook](halloween_lora.ipynb).

1. Clone AI Toolkit and check out the commit above.
2. In the config, set `folder_path` to the absolute path of `halloween/` and `training_folder` to your output directory.
3. Run this from the AI Toolkit root.

```
python run.py configs/halloween-lora-cosine.yaml
```

## Image sources

49 images come from The Public Domain Review and 16 from Wikimedia Commons. `halloween/metadata.csv` lists the source page and license for each image.

Code and configs use the [MIT license](LICENSE). The images keep their own licenses, as listed in `halloween/metadata.csv` and the `ukiyoe/` docs.

## Make a card

My favorites were cosine and constant_with_warmup, both at step 2,000. Both checkpoints from the guide are in the [checkpoints-2000 release](https://github.com/MaanavD/flux3-klein-lora-guide/releases/tag/checkpoints-2000). Run this from the AI Toolkit root, with its venv active.

```bash
python /path/to/scripts/make_card.py --lora klein-vqx-shift4k-cosine_000002000.safetensors \
  --prompt 'vqx, an old Halloween greeting card with a hand-drawn illustration of a black cat on a garden fence. The card has the clearly readable greeting "Happy Halloween" across the top.' \
  --out card.jpg
```

The script makes a 1024 x 1024 card with 50 steps, guidance 3.5 and LoRA strength 1. Change `--strength`, `--seed` or `--steps` to taste.

Or make one in the browser with the [Hugging Face Space](https://huggingface.co/spaces/black-forest-labs/flux3-klein-halloween-card). Its code is in `space/`.

## Make your own

One set I'm trying next is 156 ukiyo-e prints from the Art Institute of Chicago, all public domain. They're in `ukiyoe/`, captioned with the trigger word `ukiyobw`. See [ukiyoe/README.md](ukiyoe/README.md) for sources and checksums.
