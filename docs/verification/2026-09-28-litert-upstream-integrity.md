# LiteRT upstream integrity hardening — 2026-09-28

## Source authority

Reviewed `google-ai-edge/litert-samples` at commit
`940336695f6f3bb119e93dc00e9eac27476095ab` and the published Hugging Face
repositories referenced by Lil Tart.

The official Qwen3-TTS reference documents three host-loop invariants already
implemented by Lil Tart:

- the MTP/code predictor executes 16 times per generated frame;
- the codec decoder consumes 64-frame windows;
- every codec window after the first carries 25 frames of left context.

These values are now named constants with focused regression coverage rather
than incidental literals.

## Artifact integrity

Published Qwen3-TTS assets now carry the exact Hugging Face LFS byte size and
SHA-256 recorded at repository revision
`528cca7d2ddf6f5c1e1127f24a7f8786f80fa6e8`.

The manifest verifier now applies integrity facts by exact asset path. This
covers `runtime-litert` requests that identify model files by their path while
also preventing dynamic assets that reuse a semantic ID from inheriting another
path's checksum. In particular, a configured voice other than
`voices/demo_speaker.npy` remains unverified unless that exact voice path has
its own manifest facts.

Both Qwen execution paths now use manifest verification: the direct playground
path and the generator/decoder worker contexts. The local experimental
`mtp_folded_int8.tflite` keeps its known byte size but deliberately has no
invented checksum.

The `browserMemoryOmni` qualification variant is mixed-source: its
`mtp_fp32.tflite` comes from
`uralstech/Qwen3-TTS-12Hz-0.6B-Base-litert-lm-omni` revision
`791880469d874546d884a0e6cf68564a61c04ca9`. Its 440,528,628-byte size and
SHA-256 are therefore recorded separately from the base repository's
same-named 440,526,692-byte MTP artifact.

The runnable example model proxy also resolves the exported
`QWEN3_TTS_UPSTREAM_REVISION` rather than mutable `main`. The pinned revision
contains the folded browser-memory MTP as well as the published base assets, so
pinning the proxy does not break the minimal browser-memory example.

## LiteRT-LM source pinning

Qwen3 0.6B, LFM2.5 Instruct/Thinking int4+int8, and Gemma 4 E2B/E4B manifests
now use immutable Hugging Face commit revisions instead of `resolve/main`.
The manifests record the published LFS byte sizes and SHA-256 values.

The LiteRT-LM pipeline intentionally streams a fetched body directly into
`Engine.create` to avoid buffering multi-gigabyte checkpoints. The pinned
revision is therefore the active runtime immutability guarantee for that path;
the SHA-256 remains authoritative provenance for resolvers that can verify
incrementally or from persisted files without destroying the streaming memory
profile.

## Verification

- `pnpm --filter @litert-playground/inference-core test`
- `pnpm --filter @litert-playground/qwen3-tts test`
- `pnpm --filter @litert-playground/text-gen test`
- package typechecks for those three packages
- repository gate: `pnpm verify`
