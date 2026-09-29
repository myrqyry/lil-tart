# Hugging Face model contract audit — 2026-09-28

## Scope

The current lil-tart model registry and packaged pipelines were cross-checked against their Hugging Face sources. The audit found **72 distinct Hugging Face model repositories directly represented by playground adapters; all 72 repository IDs resolved on the Hub**.

That is only an asset-existence result. It is not treated as inference or output verification.

For browser correctness, this audit also used exact-artifact LiteRT.js 2.5.3 sweep records from:
- https://github.com/john-rocky/edge-compat
- the relevant Hugging Face model cards under https://huggingface.co/litert-community

The important distinction is:
1. **asset exists** — the repository/file is available,
2. **contract matches** — lil-tart uses the documented tensor/signature/preprocessing contract,
3. **output verified** — a measured browser run has evidence that the backend returns matching output.

A model that merely compiles is not considered output-verified.

## Correctness fixes

- **CREPE pitch**: use the single positional `[1,1024]` input instead of inventing a named `input` binding.
- **SAM2 encoder**: use `sam2_tiny_image_encoder_v2_fp16.tflite`, whose projected outputs are compatible with the mask decoder.
- **SAM2 decoder**: use the v2 decoder artifact and its real `pred_masks` / `iou_scores` outputs. The standalone adapter remains disabled until image encoding and point-prompt encoding are exposed as one usable pipeline.
- **U-2-Net saliency**: apply per-image maximum scaling before ImageNet normalization, matching the source preprocessing.
- **Voyage 4 Nano / Nemotron-3-Embed / Harrier**: use documented pad token IDs rather than the generic zero default.
- **Runtime errors**: preserve the nested LiteRT cause in the playground error instead of hiding it behind only `Inference failed for <URL>`.

## Browser backend restrictions

`ModelAdapter` and `ModelGraph` now support a correctness-required browser backend. The playground applies that backend during load, inference, and preflight instead of assuming a successful compile means the output is trustworthy.

Exact-artifact LiteRT.js 2.5.3 browser sweeps show WebGPU output mismatches for these currently exposed graphs, so lil-tart routes them through WASM:

- DINOv2 ViT-S/14
- NAFNet SIDD width32
- MI-GAN
- Fast Neural Style: Candy, Mosaic, Rain Princess, Udnie
- SINet-V2
- PIDNet-S Cityscapes
- TwinLiteNet
- TIPSv2 B14 DPT
- GFPGAN v1.4
- Cloth Segmentation U²-Net
- M-LSD tiny
- PP-OCRv5 recognizer
- Matcha-TTS decoder and vocoder
- Mimi encoder-transformer and decoder-transformer

MoViNet-A0 Stream is also restricted to WASM in its package manifest and pipeline because the external exact-artifact LiteRT.js sweep reports WebGPU output mismatch. Its manifest verification states remain `untested` until lil-tart captures its own durable verification receipt; the external sweep is recorded as backend-selection evidence, not silently promoted into repo-owned verification.

## Runtime receipt semantics

Backend policy is now explicit in receipts:

- **selected backend** is what the user chose in the playground,
- **main graph request/resolution** is labelled as main-graph-scoped rather than presented as the whole pipeline,
- **graph backends** preserve the requested and resolved backend for every graph that actually ran.

For multi-graph adapters, the session proof renders each graph path independently instead of flattening Mimi, Matcha, PP-OCR, or other graph pipelines into one pretend-universal backend. Correctness pins are shown by graph name. Runtime fallbacks are aggregated once per graph so repeated inference calls cannot double-count a model's compile fallback. If a correctness override and a runtime fallback both happen, both are reported.

## Intentionally unavailable instead of fake-working

- **CLIPSeg** remains disabled. Its old adapter passed token IDs directly into a graph that expects `[1,77,512]` token embeddings, discarded two of the three vision features, and did not perform the required text projection. Its real host embedding/projection path must be implemented before re-enabling it.
- **SAM2 mask decoder** remains disabled as a standalone card. Asking a user to manually provide encoder feature maps and a pre-encoded sparse prompt is not a usable image-segmentation pipeline. The external sweep records a WebGPU mismatch, but the disabled adapter carries no active `requiredBackend` pin; the future live pipeline must apply and verify its backend constraint when it becomes executable.
- **YOLO26 / YOLO11 / YOLOv8 segmentation** remain disabled. Hugging Face exports now exist, but the available EdgeFirst artifacts use uint8 input and split quantized int8 outputs, which do not match the old float32 single-output placeholders.
- **Magenta RealTime 2** remains disabled. The official Hugging Face repo exists, but the product is a multi-component SpectroStream + MusicCoCa + decoder pipeline; the old single `magenta.tflite` card is not a verified model contract.
- **MusicCoCa components** remain disabled until their official Hugging Face artifacts and browser contracts are wired and qualified end-to-end.

## Regression coverage

Focused tests lock:
- CREPE positional invocation,
- SAM2 v2 artifact/output contracts and decoder disabled state without an inactive backend pin,
- CLIPSeg disabled state,
- known browser-mismatch backend restrictions,
- U-2-Net per-image-max preprocessing.

The repository's authoritative merge gate remains `pnpm verify`.

Passing unit/type/build checks is still not equivalent to output-verifying every large model. New durable model verification should include the exact artifact, runtime version, backend, input contract, and output-comparison evidence.