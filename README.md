# 🍁 Maple Diffusion

Maple Diffusion runs Stable Diffusion models **locally** on macOS / iOS devices, in Swift, using the MPSGraph framework (not Python).

![](demonstration.jpg)

Maple Diffusion should be capable of generating a reasonable image [in a minute or two](https://twitter.com/madebyollin/status/1579213789823893504) on a recent iPhone (I get around ~2.3s / step on an iPhone 13 Pro).

To attain usable performance without tripping over iOS's 4GB memory limit, Maple Diffusion relies internally on FP16 (NHWC) tensors, operator fusion from MPSGraph, and a truly pitiable degree of swapping models to device storage.

On macOS, Maple Diffusion uses slightly more memory (~6GB), to reach <1s / step.

![](screenshot.jpg)

# Related Projects

* **Core ML Stable Diffusion** ([repo](https://github.com/apple/ml-stable-diffusion)) is Apple's recommended way of running Stable Diffusion in Swift, using CoreML instead of MPSGraph. CoreML was originally much slower than MPSGraph ([I tried it back in August](https://gist.github.com/madebyollin/86b9596ffa4ab0fa7674a16ca2aeab3d)), but Apple has improved CoreML performance a lot on recent macOS / iOS versions.
* **Native Diffusion** ([repo](https://github.com/mortenjust/native-diffusion/)) is a Swift Package-ified version of this codebase with several improvements (including image-to-image)
* **Waifu Art AI** ([announcement](https://twitter.com/dgspitzer/status/1596652212964712449), [App Store link](https://apps.apple.com/us/app/waifu-art-ai-local-generator/id6444585505)) is an iOS / macOS app for (anime-style) Stable Diffusion based on this codebase
* **Draw Things** ([announcement](https://liuliu.me/eyes/stretch-iphone-to-its-limit-a-2gib-model-that-can-draw-everything-in-your-pocket/), [App Store link](https://apps.apple.com/us/app/draw-things-ai-generation/id6444050820)) is an iOS app for Stable Diffusion (using an independent codebase with similar MPSGraph-based approach)

# Device Requirements

Maple Diffusion should run on any Apple Silicon Mac (M1, M2, etc.). Intel Macs should also work now thanks to [this PR](https://github.com/madebyollin/maple-diffusion/pull/14#issuecomment-1282166802).

Maple Diffusion should run on any iOS device with [sufficient RAM](https://blakespot.com/ios_device_specifications_grid.html) (≥6144MB RAM definitely works; 4096MB [doesn't](https://github.com/madebyollin/maple-diffusion/issues/25)). That means recent iPads should work out of the box, and recent iPhones should work if you can get the `Increase Memory Limit` capability working (to unlock 4GB of app-usable RAM). iPhone 14 variants reportedly didn't work until [iOS 16.1 stable](https://github.com/madebyollin/maple-diffusion/issues/5#issuecomment-1304410263).

Maple Diffusion currently expects **Xcode 14** and **iOS 16**; other versions may require changing build settings or just not work. iOS 16.1 (beta) was reportedly [broken](https://github.com/madebyollin/maple-diffusion/issues/8) and always generating a gray image, but I think that's fixed

# Usage

To build and run Maple Diffusion:

1. Download a Stable Diffusion PyTorch model checkpoint ([`sd-v1-4.ckpt`](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original), or some derivation thereof)

2. Download this repo

   ```bash
   git clone https://github.com/madebyollin/maple-diffusion.git && cd maple-diffusion
   ```

3. Setup & install Python with PyTorch, if you haven't already.

   ```bash
   # may need to install conda first https://github.com/conda-forge/miniforge#homebrew
   conda deactivate
   conda remove -n maple-diffusion --all
   conda create -n maple-diffusion python=3.10
   conda activate maple-diffusion
   pip install torch typing_extensions numpy Pillow requests pytorch_lightning
   ```

4. Convert the PyTorch model checkpoint into a bunch of fp16 binary blobs.

   ```bash
   ./maple-convert.py ~/Downloads/sd-v1-4.ckpt
   ```

5. Open the `maple-diffusion` Xcode project. Select the device you want to run on from the `Product > Destination` menu.

6. [Manually add](https://github.com/madebyollin/maple-diffusion/issues/5#issuecomment-1279111878) the `Increased Memory Limit` capability to the `maple-diffusion` target (this step might not be needed on iPads, but it's definitely needed on iPhones - the default limit is 3GB).

7. Build & run the project on your device with the `Product > Run` menu.
