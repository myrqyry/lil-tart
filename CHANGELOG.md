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
- `feat(retrieval)`: publish a `./scoring` subpath export so consumers can import late-interaction scoring without loading the ColBERT pipeline entrypoint
- `fix(playground)`: replace 12 empty adapters (`sam2` ×2, `vision` ×10 including `6drepnet`, `blaze-face`, `yolox`, `u2net`, `edsr`, `migan`, `style-*`) with working `prepareInputs`/`parseOutputs` (`apps/playground/src/adapters/sam2.ts:16`, `apps/playground/src/adapters/vision.ts:70`)

### Fixed
- Pin LiteRT-LM Hugging Face artifacts to immutable upstream revisions and record their published byte sizes and SHA-256 provenance instead of following mutable `main` paths.
- Enforce Qwen3-TTS manifest integrity on both direct and worker execution paths, including runtime requests that identify an asset by its file path rather than its semantic manifest ID.
- `encoder` and `retrieval` now keep `inference-core` as a peer contract instead of leaking `workspace:*` into downstream package metadata.
- `fix(playground)`: resolve LFM pipeline model assets from Hugging Face instead of the app origin, so packaged consumers load the same weights as the app
- `fix(text-gen)`: resolve the LiteRT-LM model to an absolute URL from an explicit model base instead of handing the repo-relative manifest path to the engine, which resolved it against the app origin and 404'd
- `fix(text-gen)`: fetch the model in the pipeline so the download is cancellable. The engine's own model fetch takes no `AbortSignal`, so switching models mid-download left the transfer running; the engine now receives the response body, which it treats identically to a URL from that point on
- `fix(playground)`: abort the in-flight LFM model download when the pipeline is disposed or the model changes
- `fix(encoder)`, `fix(retrieval)`: pass an `AbortSignal` to the model load and abort it on dispose, so encoder and ColBERT downloads are cancellable the same way text generation now is. Widens the `LiteRtRuntime.loadModel` contract to accept load options
- `fix(playground)`: the LFM panel's LiteRT WASM runtime now loads from the pinned `@litertjs/core` jsDelivr CDN; `apps/playground/public/` ships no `wasm/` directory, so origin-relative resolution could not work
- `fix(playground)`: report stored model bytes for the resolver base actually in use, and reclaim entries orphaned by a base change, so "stored locally" no longer claims weights that would be re-downloaded
- `fix(playground)`: keep weights cached under a previous model-server base visible and individually removable instead of reachable only through "clear all", which also erased the current base
- `fix(playground)`: scope model-cache removal to a base, so reclaiming a previous-base download no longer also deletes the live copy of a model that is cached under both
- `fix(text-gen)`: export `LiteRtLmTextPipelineOptions` from the package entrypoint so consumers can name the type of the constructor option they pass
- `fix(playground)`: match cache entries against the model base structurally, so a base without a trailing slash no longer claims sibling paths that share its leading characters
- `fix(playground)`: decide cached-asset membership from the base an entry was written with, so widening a nested base (`/v1/` to `/`) stops reporting unreachable bytes as stored. Entries predating the recorded base still fall back to the prefix comparison
- `fix(text-gen)`, `fix(encoder)`, `fix(retrieval)`: a dispose that lands while the engine or model is still compiling no longer lets the late result resurrect a disposed pipeline. text generation releases the orphaned engine explicitly
- `fix(text-gen)`, `fix(encoder)`, `fix(retrieval)`: a load that is cancelled because the pipeline was disposed now reports `disposed` rather than stamping `idle` over it, and a load abandoned by disposal rejects instead of resolving as if it succeeded
- `fix(text-gen)`, `fix(encoder)`, `fix(retrieval)`: an already-aborted context signal now stops a load immediately. Subscribing to a settled signal never fires, so a caller that cancelled before dialling would otherwise start a full checkpoint download
- `fix(playground)`: reject loads attempted on a disposed pipeline, and stop a tokenizer fetch that completes after disposal from repopulating pipeline state
- `test`: the LFM cancellation guard asserted an offset comparison that could not fail, since a missing needle yields -1 and -1 sorts before any real index. It now asserts both offsets exist, and is bounded to the dispose function
- `fix(retrieval)`: remove unused retrieval state that broke stricter downstream TypeScript consumers
- `fix`: cache LiteRT runtime in `ensureRuntime` and persist Ready badge (`packages/runtime-litert/src/context.ts:90`, `packages/runtime-litert/src/types.ts:96`)
- `fix`: make MoViNet frame commit transactional and correct WASM probes
- `fix`: terminate TTS workers on failure and probe WASM features honestly

### Verification
- Lock the official Qwen3-TTS host-loop contract in regression coverage: 16 MTP passes per generated frame, 64-frame codec windows, and 25-frame left context after the first window.
- Add a `pipeline-load-cancellation` runtime qualification case that streams a real cross-origin asset through the shared asset resolver in headless Chromium and observes that cancelling stops the transfer, closing the cancellation evidence gap that only stubbed `fetch` had covered. It probes `AssetResolver.resolve()`, the path production loading actually takes, since `stream()` has no production caller. Durable record in `docs/verification/2026-09-28-model-load-cancellation.md`
- `fix(playground)`: decide cached-asset membership by whether the current base resolves the entry's recorded path to its stored URL. Base-string equality was wrong in both directions: too strict for a base whose last segment is a file, which made live entries look orphaned and removable
- `fix(text-gen)`, `fix(encoder)`, `fix(retrieval)`: a load attempted after disposal now rejects instead of resolving, so a caller cannot mistake a disposed pipeline for a ready one
- `fix(text-gen)`, `fix(encoder)`, `fix(retrieval)`: a cancelled load no longer completes. Cancellation had two routes, disposal and an aborted context signal, but only disposal was checked before publishing a result, so a caller that cancelled while the engine was still compiling ended up with a live engine and a pipeline reporting itself ready

### Docs
- Canonical Git dependency examples now use the renamed `myrqyry/lil-tart` repository while preserving the stable `@litert-playground/*` package namespace.
- `docs`: add root `AGENTS.md` and nested package guidance

## [1.0.0] - 2026-09-01

- Initial public playground release.
