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

The manifest verifier now matches an asset by semantic ID or exact path. This
matters because `runtime-litert` internally asks for model paths as both the
request ID and path, while manifests can use semantic IDs such as `talker`.

Both Qwen execution paths now use manifest verification: the direct playground
path and the generator/decoder worker contexts. The local experimental
`mtp_folded_int8.tflite` keeps its known byte size but deliberately has no
invented checksum.

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
