# Project Invisible ERNIE Image

**Backup extension.** Restores ERNIE Image and ERNIE Image Turbo on Forge Neo builds that have removed built-in ERNIE support. The extension includes ERNIE detection, model definitions, Ministral conditioning, component loaders, tokenizer/config assets and adapter aliases. Forge supplies the shared sampling, memory management, VAE, Llama operations and normal UI/API workflow.

## Install and generate

1. Copy this folder to your Forge Neo `extensions` folder and restart Forge.
2. Put a single-file ERNIE checkpoint in `models/Stable-diffusion`, the Ministral encoder in `models/text_encoder`, and the FLUX.2 VAE in `models/VAE`.
3. Select the checkpoint and both additional modules in Forge's normal selectors.
4. Use the ordinary txt2img or img2img Generate button.

Download the matching single-file weights from [Comfy-Org/ERNIE-Image](https://huggingface.co/Comfy-Org/ERNIE-Image/tree/main).

| Component | File | Forge folder |
|---|---|---|
| Base checkpoint | `ernie-image.safetensors` | `models/Stable-diffusion` |
| Turbo checkpoint | `ernie-image-turbo.safetensors` | `models/Stable-diffusion` |
| Text encoder | `ministral-3-3b.safetensors` | `models/text_encoder` |
| VAE | `flux2-vae.safetensors` | `models/VAE` |

Existing renamed copies of these weights work. Locally tested filenames include `ERNIE-Image-BF16.safetensors`, `ERNIE-Image-Turbo-BF16.safetensors`, `Ministral-3-3B-TextEncoder-BF16--For-ERNIE-Image+Turbo.safetensors` and `FLUX.2-VAE--For-ERNIE-Image+FLUX.2-Dev+FLUX.2-Klein-4B+FLUX.2-Klein-9B+Ideogram4.safetensors`.

The old Forge ERNIE preset used Euler, Simple, 8 steps, CFG 1 and Shift 3. These settings can be entered through the normal controls. Base and Turbo quality should be evaluated separately. Prompt enhancement is outside this extension's preserved Forge engine.

## Compatibility

If Forge already provides an ERNIE engine, the extension uses that native route. Otherwise it registers its own ERNIE route. Other architectures pass through to Forge. Compatibility checks inspect required interfaces. Errors include a traceback and a clear startup message.

Reloading scripts removes hooks still owned by the extension. To uninstall, delete its folder and restart Forge. Bundled tokenizer/config assets load offline. Model weights are separate downloads.

## Validation

Checked on 24 November 2026 with Forge Neo after native ERNIE removal and an RTX 5090.

- Base and Turbo BF16 both completed normal txt2img processing at 256 x 256 with Euler, Simple, 8 steps, CFG 1 and Shift 3.
- Both completed normal img2img processing at denoising strength 0.4.
- Saved PNG pixels matched the returned images and included generation metadata.
- Regression checks covered ERNIE detection, old/new replacement signatures, wrong/missing component messages, native-engine bypass, repeated registration and hook cleanup.
- A synthetic LoRA passed alias lookup, loading and weight merging through Forge's adapter code.

Larger image quality, real LoRA/LoKr files, quantized checkpoints, browser interaction and cancellation remain unverified. These checks ran in isolated processes with other extensions disabled. Temporary test images and cache files were removed.

## Source and license

The ERNIE implementation and assets were extracted from [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/6017d53dc6c8187ca05516cc8c5676ffc0d8ead3). The restored Ministral model uses Forge's shared Llama implementation. Forge credits the ComfyUI [ERNIE transformer](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/ernie/model.py) and [prompt encoder](https://github.com/Comfy-Org/ComfyUI/blob/v0.36.0/comfy/text_encoders/ernie.py). See [LICENSE](LICENSE).
