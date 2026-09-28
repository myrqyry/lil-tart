# Model load cancellation

## Current scope

`pipeline-load-cancellation` now exercises `createHttpAssetResolver.resolve()`,
the buffered entrypoint used by `runtime-litert` to retrieve model bytes. Its
`onProgress` callback aborts after the first nonzero progress event. A pass requires
that the promise rejects with `CANCELLED` and that consumed bytes are greater than
zero but less than the 24 MiB fixture. Successful resolution, a different error,
zero received bytes, or consuming the entire fixture all fail the case.

The browser entry delegates to `pipeline-load-cancellation/probeTransfer.ts`.
This measures bytes consumed by the resolver, not all bytes already sent by the
server or buffered by the network stack. It does not compile a model or exercise
a pipeline's signal wiring end to end.

## Fixture and server lifecycle

The deterministic fixture lives under gitignored `static-models/`. Importing the
CLI, printing help, launching an unrelated case, or failing browser startup does
not generate it or start the probe server. Generation and the separate-origin
server start only when the cancellation case invokes its probe. The fixture path
is anchored to the generator module rather than the working directory.

The server validates the file before binding and handles read errors, including a
file disappearing after setup. Such failures become a failed case or HTTP error,
not an uncaught HTTP-listener exception. Vite, browser, context, and probe server
are closed after partial startup and after runs; one rejected close does not skip
the others.

The VCS contract now runs `git check-ignore --no-index` against the actual probe
path. Removing its ignore rule makes the test fail. The unused re-export from
`case.ts` was removed; browser-safe constants remain in `probeAsset.meta.ts`.

## Fresh evidence for this follow-up

- Real local HTTP, Node fetch, and the production resolver: cancellation passes.
- A negative control drops the fetch AbortSignal: the full 25,165,824 bytes are
  consumed, and the probe reports failure. This guards against a false green.
- The test additionally requires `resolve()` and rejects any `stream()` call.
- Missing file setup rejects normally; deletion after setup returns HTTP 500.
- Startup/cleanup tests cover failed listen, browser launch and context creation,
  unused case selection, failed fixture setup, and cleanup rejection.
- Four new CLI/lifecycle regressions failed against the original implementation
  before edits and passed after the fix.
- A browser attempt still fails because the Playwright Chromium executable is
  absent. The updated launcher exits cleanly instead of leaking its Vite listener.
  No fresh browser pass is claimed.

## Historical browser observation (superseded path)

The earlier record reported a Chromium 151.0.7922.34 pass and a negative control
that consumed the complete 24 MiB. That probe called `resolver.stream()`, which
has no current production consumer for this model-loading path. It proved only
that unused entrypoint's cancellation; it did **not** close the production
`resolve()` or pipeline evidence gap. Those results must not be relabelled as
observations of the corrected probe.

## Remaining boundaries

The current text-generation pipeline fetches with a signal and supplies a response
body to `Engine.create`; it does not use this resolver path when a model base is
configured. Consuming a real `.litertlm` stream and generating text remain separate
browser checks. Tokenizer requests in encoder/ColBERT still have no upstream abort
option. Fresh browser evidence for the corrected resolver probe is also pending.

Re-run with `pnpm qualify -- --case pipeline-load-cancellation --backend wasm`.
Results JSON is gitignored; this document distinguishes historical browser evidence
from the current Node/contract checks.
