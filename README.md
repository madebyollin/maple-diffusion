# 🍁 Maple Diffusion

Maple Diffusion runs Stable Diffusion models **locally** on macOS / iOS devices, in Swift, using the MPSGraph framework (not Python).

![](demonstration.jpg)

Maple Diffusion should be capable of generating a reasonable image [in a minute or two](https://twitter.com/madebyollin/status/1579213789823893504) on a recent iPhone (I get around ~2.3s / step on an iPhone 13 Pro).

To attain usable performance without tripping over iOS's 4GB memory limit, Maple Diffusion relies internally on FP16 (NHWC) tensors, operator fusion from MPSGraph, and a truly pitiable degree of swapping models to device storage.

On macOS, Maple Diffusion uses slightly more memory (~6GB), to reach <1s / step.

![](screenshot.jpg)

# Projects using Maple Diffusion

* https://github.com/mortenjust/maple-diffusion/ is a fork with several improvements

# Usage

Maple Diffusion should run on any recent Mac, or any iOS device with [sufficient RAM](https://blakespot.com/ios_device_specifications_grid.html) (≥6144MB RAM definitely works; 4096MB *might* but I wouldn't bet on it; anything lower than that won't work).

To build and run Maple Diffusion:

1. Download a Stable Diffusion model checkpoint ([`sd-v1-4.ckpt`](https://huggingface.co/CompVis/stable-diffusion-v1-4), or some derivation thereof)

2. Download this repo

   ```bash
   git clone https://github.com/madebyollin/maple-diffusion.git && cd maple-diffusion
   ```

3. Convert the model into a bunch of fp16 binary blobs. You might need to install PyTorch and stuff.

   ```bash
   ./maple-convert.py ~/Downloads/sd-v1-4.ckpt
   ```

4. Open the `maple-diffusion` Xcode project. Select the device you want to run on from the `Product > Destination` menu.

5. [Manually add](https://github.com/madebyollin/maple-diffusion/issues/5#issuecomment-1279111878) the `Increased Memory Limit` capability to the `maple-diffusion` target (this step might not be needed on iPads, but it's definitely needed on iPhones - the default limit is 3GB).

6. Build & run the project on your device with the `Product > Run` menu.
