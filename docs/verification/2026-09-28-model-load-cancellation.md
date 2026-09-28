# Model load cancellation

This record covers the cancellation evidence for the shared model-loading path:
the pipelines resolve a repo-relative manifest path against a configured model
base, and the runtime that consumes those bytes cannot be cancelled once it
starts compiling. Cancellation therefore has to be proven at the transfer, and
until this case existed it had only been asserted against stubbed `fetch`.

## Case

`pipeline-load-cancellation` in `tests/runtime-qualification/` streams a real
local asset through the shared `createHttpAssetResolver` in headless Chromium,
aborts after the first chunk, and observes two things:

- the transfer stops well short of the full asset, and
- the resolver surfaces a cancellation (`InferenceError` with code `CANCELLED`).

The asset is 24 MiB, generated deterministically under the gitignored
`static-models/` directory rather than committed, so the case needs no
multi-megabyte fixture in the repository. It is served from a separate origin on
its own port, with CORS enabled, because a real model base is a different origin
from the app and the browser must apply the same cross-origin rules it would
against Hugging Face.

The size is load-bearing. "Fewer bytes arrived than the asset holds" is only
meaningful if the asset cannot arrive in a single chunk, so `contract.test.ts`
asserts the size floor independently of the browser observation.

## Observed

Run locally against Chromium 151.0.7922.34, requested backend WASM, wasm
runtime `@litertjs/core` 2.5.3, using the `run-qualification` runner:

```text
pipeline-load-cancellation	wasm	pass	match
```

## Falsifiability

A passing case is only worth recording if it can fail. Stopping the signal from
reaching the resolver reproduces the original defect exactly:

```text
pipeline-load-cancellation	wasm	fail	mismatch
abort did not stop the transfer: 25165824 of 25165824 bytes, cancellation code null
```

The full 24 MiB arrives when nothing aborts the transfer, which is the zombie
download this case exists to rule out. The passing run was re-confirmed after
reverting the mutation.

## What this does not prove

- It does not exercise `Engine.create` consuming a streamed body. That needs a
  real `.litertlm` checkpoint, which this case deliberately avoids, so the
  text-generation path still has no browser evidence for the engine half of the
  change.
- Tokenizer loads in the encoder and ColBERT pipelines are still not
  cancellable; `AutoTokenizer.from_pretrained` exposes no signal upstream.
- Results JSON is gitignored, so this file is the durable record. Re-run
  `pnpm qualify -- --case pipeline-load-cancellation` to regenerate it.
