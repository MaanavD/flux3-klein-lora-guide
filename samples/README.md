# Samples

Every image here was made with FLUX 3 Image Klein plus this Halloween postcard LoRA, except for `cards/base/`, which uses the base model without the LoRA. The step 0 cosine previews were saved before training.

## Cards

`cards/` holds the guide's final comparison: 132 cards, with six prompts and seeds 23 and 907 for each arm. The five runs are `constant`, `constant_with_warmup`, `cosine`, `linear`, and `lineardecay`, each at steps 2,000 and 4,000. `base/` has the same prompts and seeds without a training step.

The `linear` and `lineardecay` folders keep the two separately named linear schedule runs. The original arm names are in [cards/manifest.json](cards/manifest.json), along with each image's schedule, step, seed, prompt index, prompt, LoRA strength, and file path. File paths are relative to `cards/`. The base records use `null` for the step and strength 0.

## Training previews

`previews/` contains the images AI Toolkit saved during training for my two favorite runs, `constant_with_warmup` and `cosine`. These use 30 sampling steps; the final cards use 50. Each saved training step has ten prompts.

- `cosine/`: 410 images, every 100 steps from 0 through 4,000.
- `constant_with_warmup/`: 200 images, every 100 steps from 2,100 through 4,000. Earlier previews were not in the staged copies.

Files are named `step-<NNNN>_prompt-<i>.jpg`. [previews/prompts.md](previews/prompts.md) lists the prompt indices, prompts, and seeds from the training configs.
