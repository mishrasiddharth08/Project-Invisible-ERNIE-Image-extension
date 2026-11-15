# Project Invisible ERNIE Image

A Forge Neo extension extracted from the [Forge Neo ERNIE implementation](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo). It packages the ERNIE diffusion engine, transformer and Ministral prompt conditioning. Forge supplies checkpoint detection, quantized loading, tokenizer, encoder model, VAE, adapter mapping, sampling and the normal UI/API workflow.

## Install and generate

1. Copy this folder to your Forge Neo `extensions` folder and restart Forge.
2. Select an ERNIE Image or ERNIE Image Turbo checkpoint in the normal checkpoint menu.
3. For separate components, select the Ministral 3 3B text encoder and FLUX.2 VAE.
4. Use the ordinary Generate button. Forge's ERNIE preset uses Euler, Simple, 8 steps, CFG 1 and Shift 3.

The extension uses the same engine slot as built-in ERNIE and leaves other models alone. If a Forge update changes the matching native implementation, Forge continues to use its built-in engine. This release requires a Forge Neo build that already provides ERNIE support. To remove the extension, delete its folder and restart Forge.

The source was extracted from the local Forge Neo checkout at `6017d53dc6c8187ca05516cc8c5676ffc0d8ead3`, with two import paths adjusted for the extension. Forge credits the ComfyUI [ERNIE transformer](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/ernie/model.py) and [prompt encoder](https://github.com/Comfy-Org/ComfyUI/blob/v0.36.0/comfy/text_encoders/ernie.py). See [LICENSE](LICENSE).
