# Ukiyo-e color LoRA dataset

246 color woodblock prints, each with a caption in a `.txt` file of the same name. Every caption starts with the trigger word `ukiyocolor`. This is the dataset from my LoRA video.

The images come from the [Cleveland Museum of Art Open Access](https://www.clevelandart.org/open-access) collection, released under [CC0](https://creativecommons.org/publicdomain/zero/1.0/). `metadata.csv` lists each file's title, artist when known, date, license, artwork page and original image URL.

I resized the images to about 1.3 megapixels, a little above the 1024 px training size, to keep the repo small. The full-size originals are at each `image_url`. Duplicates were removed.

The captions describe what's in each print: people, objects, setting, and any writing. They leave out style words like "woodblock" or "ukiyo-e", because the style is what the LoRA learns. An LLM wrote them, so read a few before you train and fix any wrong details.

`SHA256SUMS` records the hashes of the images and captions. Check them from this directory.

```sh
shasum -a 256 -c SHA256SUMS
```
