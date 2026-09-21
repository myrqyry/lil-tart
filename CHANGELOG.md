# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Shared downstream inference compatibility now covers browser cache, text generation, encoder embeddings, ColBERT retrieval, Kokoro, Qwen3-TTS, image embeddings, and video classification from their packed public entrypoints.
- `feat(playground)`: wire adapters to Hugging Face and flag missing models (`apps/playground/src/adapters/*`, `types.ts`, `ModelList.tsx`, `ModelRunner.tsx`)
- `feat`: add model-server base URL input to `ModelRunner` (`apps/playground/src/hooks/useModelRunner.ts`, `ModelRunner.tsx`)
- `feat`: show model task-type chips in list and detail view (`ModelList.tsx`)
- `feat`: add download progress tracking for model loading
- `fix(playground)`: replace 12 empty adapters (`sam2` ×2, `vision` ×10 including `6drepnet`, `blaze-face`, `yolox`, `u2net`, `edsr`, `migan`, `style-*`) with working `prepareInputs`/`parseOutputs` (`apps/playground/src/adapters/sam2.ts:16`, `apps/playground/src/adapters/vision.ts:70`)

### Fixed
- `encoder` and `retrieval` now keep `inference-core` as a peer contract instead of leaking `workspace:*` into downstream package metadata.
- `fix`: cache LiteRT runtime in `ensureRuntime` and persist Ready badge (`packages/runtime-litert/src/context.ts:90`, `packages/runtime-litert/src/types.ts:96`)
- `fix`: make MoViNet frame commit transactional and correct WASM probes
- `fix`: terminate TTS workers on failure and probe WASM features honestly

### Docs
- Canonical Git dependency examples now use the renamed `myrqyry/lil-tart` repository while preserving the stable `@litert-playground/*` package namespace.
- `docs`: add root `AGENTS.md` and nested package guidance

## [1.0.0] - 2026-09-01

- Initial public playground release.
