<!-- meristem-template:v1 -->
# Active Work

Subject: aster/litert-upstream-hardening — source pinning, artifact integrity, official LiteRT contract hardening, and shared multimodal worker transport.

## Current outcome

Lil Tart now consumes upstream LiteRT model artifacts with immutable provenance,
enforces available integrity facts on real Qwen3-TTS execution paths, locks the
host-side Qwen orchestration to the current official reference contract, and
exposes a shared browser worker transport with explicit text-only LiteRT-LM and
multimodal MediaPipe GenAI engines.

## Current evidence

- google-ai-edge/litert-samples inspected at 940336695f6f3bb119e93dc00e9eac27476095ab on 2026-09-28.
- Published Hugging Face artifacts are pinned to immutable commit revisions with exact LFS sizes and SHA-256 provenance.
- Focused tests passed: inference-core 34, qwen3-tts 41, text-gen 24.
- Full repository pnpm verify passed, including typecheck, tests, boundaries, packed-consumer compatibility, runtime qualification, and build.
- Durable record: docs/verification/2026-09-28-litert-upstream-integrity.md.

### Current continuation (uncommitted)

The public `text-gen` worker transport now has an additive, explicit MediaPipe
GenAI engine for multimodal browser consumers. Existing `load(model)` and
string `generate(prompt)` callers keep the LiteRT-LM path unchanged; a consumer
can opt into `engine: 'mediapipe'` and send structured text/image/audio prompt
parts. The worker owns the MediaPipe runtime and temporary media object URLs.

This is intentionally a two-engine design. The installed `@litert-lm/core`
JavaScript contract still marks image/audio/video content as unsupported
placeholders, so Lil Tart does not pretend that adding an image-shaped object to
LiteRT-LM makes vision work. MediaPipe Tasks GenAI 0.10.29 is the browser engine
used for the modality it actually supports.

The MediaPipe path also removes a consumer-side memory trap: a remote model URL
is passed directly to MediaPipe, while Blob models are supplied as stream
readers. Consumers no longer need to download a multi-gigabyte checkpoint into
JavaScript chunks, concatenate another full buffer, and manufacture an object
URL before initialization.

Evidence established so far against this working tree:

- `pnpm --filter @litert-playground/text-gen typecheck` — pass.
- `pnpm --filter @litert-playground/text-gen test` — 32/32 tests pass.
- `pnpm test:compatibility` — all 10 packed public packages pass import,
  peer-metadata, typecheck, and external Vite build; the external fixture names
  and uses the new worker load/prompt types.
- A Live Streamer consumer was temporarily linked to this working-tree package;
  its production Vite build emitted the upgraded module worker plus a separate
  MediaPipe GenAI chunk, and a Vite/Vitest feature probe selected the shared
  multimodal path. The consumer link was then restored to its pinned Git SHA.

Full `pnpm verify` passed after this continuation: workspace typecheck,
all package/app tests, 15/15 package-boundary tests, all 10 packed-consumer
compatibility rows, runtime qualification, and production builds.

## Deliberate boundary

LiteRT-LM checkpoints remain streaming inputs to Engine.create. Do not force
multi-gigabyte checkpoints through the full-buffer hash verifier merely to hash
them; immutable revision pinning is the current runtime guarantee until
incremental or file-backed verification preserves the streaming memory profile.

LiteRT-LM's JavaScript worker remains text-only. Multimodal input is routed only
through the explicit MediaPipe engine. MediaPipe's web API does not expose
mid-generation cancellation; a worker-client abort rejects the caller and
suppresses stale/queued output, but cannot stop GPU work that MediaPipe has
already started. Do not document that as physical cancellation until upstream
provides such a primitive.

## Base continuity

This branch was created from clean fix/consumer-runtime-integration at 2200684.
That work was already committed and pushed. Its prior active-work record is
preserved below rather than discarded.

---

## Prior active-work record (preserved)

<!-- meristem-template:v1 -->
# Active Work

Subject: `fix/consumer-runtime-integration` — downstream consumer hardening and
the retrieval scoring entrypoint.

## Outcome

A downstream application can consume Lil Tart inference packages by Git SHA
through their public entrypoints, without hitting app-origin or stricter-TypeScript
breakage, and the retrieval scoring surface is a deliberate, documented choice
rather than an accidental export.

## Why it matters

This branch defines the supported downstream surface other apps are expected to
depend on. Anything published here becomes a contract: once a consumer can reach
it, removing or reshaping it is a breaking change under the project's single-revision
policy.

## Current state

**Branch context** (committed, and already pushed to
`origin/fix/consumer-runtime-integration`; not yet in `master`):

- `32e666a` *fix: harden shared consumer runtime paths* — resolve LFM pipeline
  model assets from Hugging Face instead of the app origin; remove unused
  retrieval state that broke stricter downstream TypeScript consumers.
  Touches `apps/playground/src/components/LfmPipelinePanel.tsx`,
  `packages/retrieval/src/colbert.ts`.
- `8c9cb8b` *feat: expose retrieval scoring subpath* — add
  `"./scoring": "./src/scoring.ts"` to `packages/retrieval/package.json` so
  consumers can import pure scoring without loading the ColBERT package entrypoint.

**Current working tree (dirty, uncommitted):** the branch commits above were
complete, but three defects and one missing contract surfaced in review. All are
now closed in the working tree.

Subpath coverage (from the first review round):

- `tests/fixtures/external-consumer/src/index.ts` — imports `maxSim` and
  `rankDense` from `@litert-playground/retrieval/scoring`, so the packed tarball
  must resolve the subpath for `pnpm test:compatibility` to pass.
- `tests/package-boundaries.test.ts` — new test
  *"keeps the retrieval scoring subpath a verified public entrypoint"*: pins the
  `exports` map, reads the subpath target so a rename breaks the suite, and
  asserts `scoring.ts` never imports `./colbert` or `@huggingface/transformers`
  (that independence is the subpath's entire reason to exist).
- `docs/package-revision-policy.md` — states that subpath entrypoints are
  supported on the same terms as `.`, and lists `@litert-playground/retrieval/scoring`.

The base-swap follow-ups:

- `packages/text-gen/src/litertlm-pipeline.ts` — the real bug. `load()` passed the
  manifest's repo-relative path straight to `Engine.create`, bypassing
  `context.assets` entirely, so all four `kind: 'text'` models in the LFM panel
  still resolved against the app origin and 404'd. The branch's stated purpose was
  only half-delivered. Now resolved to an absolute URL from an explicit
  `modelBase` option.
  - **First attempt was wrong and was corrected twice.** Routing through
    `context.assets.stream()` fixed the 404 but buffered the whole checkpoint
    (that resolver awaits `resolve()` first). The absolute-URL form fixed the
    memory profile but not cancellation. The final form fetches the URL *in the
    pipeline* with an `AbortSignal` and hands the engine `response.body`, which
    `modelToStream` passes through untouched. Upstream evidence for each step:
    `modelToStream` accepts a `ReadableStream` as-is; its own string fetch uses
    `credentials: 'same-origin'` and **no signal**, which is why the pipeline has
    to fetch; `EngineSettings` has no signal field, so the engine could not have
    been handed one. Memory and streaming behaviour are unchanged, because from
    `modelToStream` onward a string and a stream are handled identically.
  - Cancellation is wired from both directions: `context.signal` (which the panel
    supplies) aborts an internal controller, and `dispose()` aborts it too, so a
    consumer that wires no signal can still cancel by disposing. A cancelled load
    leaves status `idle` rather than latched to `error`, so it stays retryable.
  - Trade-off accepted: text models are fetched by the pipeline rather than the
    engine, so they do **not** participate in Cache Storage. Before this branch
    they were broken (404), so nothing working was lost. Gaining caching without
    buffering would mean making the playground resolver tee a real `fetch` body
    instead of wrapping `resolve()`, which is a larger change and is not done.
- `apps/playground/src/modelStorage.ts` — `listStoredModels(base)` counts only
  entries under that base, and `pruneSupersededModelCaches` reclaims entries
  orphaned by a base change. Cache keys embed the resolved absolute URL, so every
  pre-move entry was unreachable while still being counted.
  - `belongsToBase` compares origins and path prefixes **structurally**. A lexical
    `startsWith` was wrong for any base without a trailing slash: base
    `https://huggingface.co/litert-community` claimed
    `.../litert-community-evil/...`, and since `prune` deletes, that was a data-loss
    path reachable from free-text user input. The first version of this guard was
    only passing its test because the Hugging Face base happens to end in `/`.
  - `listOrphanedModels(base)` surfaces entries written under another base. In the
    LFM panel the base is a constant that will never change back, so pruning is
    right there. In `ModelRunner` the base is free-text user input, so pruning on
    change would irreversibly delete multi-gigabyte downloads the user may switch
    back to; those are surfaced with a per-model Remove instead, reusing
    removal scoped to the orphan set.
  - Removal is base-scoped: `removeStoredModel(modelId, base)` deletes only the
    live copy and `removeOrphanedModel(modelId, base)` only the previous-base
    bytes. A model id can hold entries under both bases at once, so a single
    unqualified delete behind a "previous base" button would destroy the live copy
    too. The orphan button also deliberately does not unload the running model,
    since the entry it removes is not the one in use.
- `packages/text-gen/src/index.ts` — `LiteRtLmTextPipelineOptions` is re-exported
  next to `LiteRtLmTextConfig`. Without it a consumer can pass `{ modelBase }` but
  cannot name the type, so the supported surface had a hole exactly where the
  packed-tarball harness should police it; the external-consumer fixture now
  imports and uses the type, so a future removal fails `test:compatibility`.
- `apps/playground/src/modelStorage.test.ts` — covers base filtering, the
  prefix-collision case that the first guard missed, the slash-less base, orphan
  listing being non-destructive, and both base-scoped removal directions.
- `tests/package-boundaries.test.ts` — added *"keeps model-cache removal
  base-scoped in storage and at the call site"*, which asserts the storage API is
  base-scoped and that the orphan row routes through the orphan variant. The unit
  tests cannot see component wiring (the playground test environment is `node`,
  with no DOM), so this is the only thing that catches the UI regressing to an
  unqualified delete.
- `apps/playground/src/components/ModelRunner.tsx` — lists orphans alongside the
  current base and renders them with per-model removal.
- `apps/playground/src/components/LfmPipelinePanel.tsx` — `MODEL_BASE` constant so
  the resolver and the text pipeline's `modelBase` cannot drift; comment recording
  that omitting `assetBase` moves the LiteRT WASM host to the pinned jsDelivr CDN
  (`apps/playground/public/` has no `wasm/` dir, so origin-relative was dead).
- `CHANGELOG.md` — entries for every change on this branch, which previously had none.
- `.meristem/` + bounded `.gitignore`/`AGENTS.md` blocks from substrate init.

Nothing is staged. No stashes.

**Base:** `master` is at `c3bd5a5` (merge of `aster/downstream-inference-surface`).
The branch is two commits ahead of it.

## Locked decisions

- The new retrieval entrypoint is a **subpath export** (`./scoring`), not a new
  package and not a deep `src` import. The revision policy explicitly forbids
  consumers importing from a package's `src` directory.
- Model assets for the LFM pipeline resolve from Hugging Face, not the app
  origin, so the app and package consumers agree on where weights come from.
- Repository is `myrqyry/lil-tart`; package namespace stays
  `@litert-playground/*`. Do not "fix" the mismatch.

## Constraints / do-not-regress

- `pnpm verify` is the pre-merge gate. Do not bypass the pre-push hook.
- Boundary tests enforce that `inference-core` stays independent, examples consume
  public entrypoints, and the playground depends on `runtime-litert`. A new
  entrypoint must not weaken these.
- Anything claimed as supported downstream surface must be packable and
  consumable by `pnpm test:compatibility`, with no `workspace:*` leakage in packed
  metadata.
- Keep the supported packed surface and `CHANGELOG.md` consistent with what
  packages actually export. The project keeps a live `[Unreleased]` section;
  shipped public API changes belong there.
- Stage by explicit path, never `git add .`.

## Current objective

Land the branch. The downstream contract for the `./scoring` entrypoint is now
decided, tested, and documented, so the remaining work is delivery:

1. Review the working-tree changes above and commit them by explicit path
   (never `git add .`).
2. Mention the `./scoring` export in the PR description so consumers know it
   exists — the one part of the review feedback not satisfiable from the repo.
3. Get the branch into `master` once CI is green.

## Verification

Established against the current tree, not inherited from earlier commits:

- `pnpm verify` — exit code 0. All six stages green: typecheck, per-package
  tests (inference-core 33, text-gen 22, encoder 11, retrieval 11, playground 141),
  boundaries 15/15, compatibility 10/10 packages, qualification 20 files, build.
- `@litert-playground/retrieval` compatibility row: `pass pass pass pass`
  (import, peers, typecheck, build) with the subpath import live.
- **Mutation check.** Three deliberate regressions were introduced and reverted;
  each was caught rather than passing silently:
  1. Renaming the export key to `./scoring-typo` — boundary test failed on the
     `exports` match, and the compatibility harness failed with
     `TS2307: Cannot find module '@litert-playground/retrieval/scoring'`.
  2. Restoring the old raw-path hand-off in text-gen — all five text-gen tests failed.
  3. Deleting the `belongsToBase` guard in `listStoredModels` — the storage tests failed.
  4. Reverting `belongsToBase` to a lexical `startsWith` — only the new
     sibling-path test failed, which is how the first guard's weakness was proven.
  5. Re-buffffering the model through `context.assets.resolve` in text-gen — five
     text-gen tests failed, including the one asserting the resolver is never called.
  6. Letting the `catch` stamp `idle` over a disposed pipeline — the dispose tests failed.
  7. Dropping the already-aborted signal check — the pre-aborted-signal tests failed.
  8. Deleting the abort line the LFM cancellation guard protected — **the old
     assertion passed it**, which is how that guard was proven unfalsifiable; the
     fixed assertion fails.
  9. Dropping the signal from the runtime resolver request in the browser probe —
     the qualification case reported the full 25,165,824 bytes arriving.
  Each mutation was reverted and the suite re-confirmed green.

Browser runtime evidence is now partially established. The `pipeline-load-cancellation`
qualification case runs in headless Chromium and streams a real cross-origin asset
through the shared resolver, observing that an abort stops the transfer well short of
24 MiB and surfaces `CANCELLED`. Dropping the signal reproduces the defect exactly
(full 25,165,824 of 25,165,824 bytes), so the case is falsifiable rather than
decorative. Record: `docs/verification/2026-09-28-model-load-cancellation.md`.

Still not established: `Engine.create` consuming a streamed body, which needs a real
`.litertlm` checkpoint, and any actual text generation in a browser. The text-gen
change is verified at the type/API boundary, by unit tests, and now for the transfer
half in a real browser; the engine half is not.

## Open uncertainty

- Whether `./scoring` duplicates logic still reachable from `.` is now
  answerable but unchecked: `src/index.ts` re-exports all five scoring
  functions, so the subpath and `.` currently overlap completely. That is
  harmless, but the subpath's independence is only meaningful if the two are
  expected to diverge. Worth a deliberate decision at some point; not blocking.
- `pruneSupersededModelCaches` is called only where the base is a constant (the LFM
  panel). It is deliberately not called from `ModelRunner`, where the base is
  free-text user input. If two panels ever share a `modelId` under different bases,
  the LFM panel's prune could reclaim the other panel's copy. Not currently the case;
  worth confirming it stays that way.
- **Text-model downloads cannot be cancelled, and this is pre-existing, not a
  regression from this branch.** A review claimed the absolute-URL fix dropped an
  abort path. Checked before accepting it: upstream `EngineSettings`
  (`@litert-lm/core@0.15.0`) is `{ model, backend?, mainExecutorSettings?,
  benchmarkEnabled? }` — no signal field, so the engine cannot be handed an
  `AbortSignal` at all. And the LFM panel never passes `signal` to
  `createLiteRtRuntime` (`LfmPipelinePanel.tsx:106`), so `context.signal` was
  `undefined` in the discarded resolver version too. Before this branch the text
  path 404'd outright, so no working cancellation was lost. The same gap applies
  to encoder/colbert, whose `context.liteRt.loadModel` also sees an undefined
  signal. Fixing it means the panel owning an `AbortController`, passing it into
  `createLiteRtRuntime`, and the pipeline consuming it — via a new engine option
  upstream, or by fetching the URL in the pipeline and handing `response.body` to
  the engine. That is a cross-cutting design change, not a review fix, and is
  **not done**. Worth deciding deliberately.
- Cache membership is decided by the base an entry was **written with** (a
  `x-lil-tart-base` header), not by a path prefix. Prefix matching is too loose once
  bases are nested: an entry cached at `https://host/v1/model.tflite` does sit under
  `https://host/`, but after widening the base the resolver asks for
  `https://host/model.tflite` and never reads it, so counting it would reintroduce
  the exact "stored locally but re-downloads" lie this work removed. Entries written
  before the header existed carry no recorded base and fall back to the structural
  prefix comparison, so no existing cache is invalidated on upgrade.
- A `dispose()` that lands while `Engine.create` or `loadModel` is still pending
  cannot cancel that work. All three pipelines carry a `loadToken`, and `dispose()`
  also sets an explicit `disposed` flag *before* aborting, so:
  - a result that arrives late is released, not published (text generation deletes
    the orphaned engine outright because WASM engine memory is not tracked by the
    runtime; encoder and colbert only skip publishing, since the runtime owns and
    disposes the compiled model);
  - a `load()` abandoned by disposal **rejects** with `CANCELLED` rather than
    resolving quietly, because the caller asked for a model and did not get one;
  - the `catch` reports `disposed` when disposal won the race, instead of stamping
    `idle` over an already disposed pipeline;
  - a tokenizer fetch completing after disposal does not repopulate state;
  - **both** cancellation routes are checked at every publication point. Disposal
    and a superseding load bump `loadToken`, but a caller aborting `context.signal`
    only flips `controller.signal` — checking the token alone let a cancelled load
    finish with a live engine and `status: 'ready'`, because a compile already under
    way cannot be aborted. An already-cancelled pipeline is now also never started:
    the model load is skipped entirely rather than started and cancelled.
  Awaiting the in-flight `load()` inside `dispose()` was considered and rejected:
  guarding the assignment sites makes correctness independent of ordering, and
  blocking `dispose()` on a multi-hundred-megabyte compile would be its own problem.
- An already-aborted `context.signal` is now handled explicitly in all three
  pipelines. Subscribing to a settled signal never fires, so `addEventListener`
  alone was an elaborate way to do nothing: a caller that cancelled before dialling
  got a full checkpoint download anyway. The playground masked this because
  `RuntimeContext.signal` is the same object the runtime holds, but a consumer
  implementing the public contract with its own signal would have hit it. This
  mirrors `startPendingLoad` in `runtime-litert/src/context.ts`, which already had
  the `if (runtimeSignal.aborted)` branch.
- **A test of mine could not fail, and it is worth recording as a lesson.** The LFM
  cancellation guard compared two `indexOf` offsets. A missing needle returns `-1`
  and `-1 < anything`, so deleting the very line the test existed to protect still
  passed. Confirmed by mutation: with the old assertion the suite reported 15/15
  green after removing the abort line; with the fix it fails. Any offset comparison
  must assert both sides are non-negative first, and the slice should be bounded to
  the function under test.
- Rejected after checking, worth recording so nobody re-raises it: the model runtime
  **is** recreated when `modelBase` changes. The `[modelBase]` effect in
  `useModelRunner.ts:322` returns a cleanup that runs on every base change and does
  `runtimePromiseRef.current = null` before disposing, so the next `ensureRuntime()`
  builds a fresh runtime with the new base. A review claimed the memoized runtime
  survives the change; it does not.
- Text models bypass Cache Storage entirely (see the trade-off note above). This
  is also why `storedInfo` is null for them in the LFM panel: the panel renders
  "Stored locally · N MB" only when `storedInfo` is set and otherwise labels the
  action "Download & load", so the readout is truthful rather than misleading.
  Restoring caching needs the playground resolver to tee a real `fetch` response
  body into the cache rather than wrapping `resolve()`, which would give every
  consumer caching without reintroducing a full buffer.
- The tokenizer load in encoder and colbert is **not** cancellable:
  `AutoTokenizer.from_pretrained(repoId)` exposes no signal, so that fetch still runs
  to completion after a cancel. Only the model load is covered. Closing it needs an
  upstream transformers.js change.
- `LiteRtRuntime` in inference-core was widened to
  `loadModel(path, options?: { signal?: AbortSignal })` (plus positional `signal` on
  `loadNpy`/`fetchBuffer`) to match what the runtime already implemented. The old
  narrow signature hid a real capability, so encoder and colbert could not pass a
  signal even though `ManagedLiteRtRuntime.loadModel` accepted one and
  `startPendingLoad` already honoured per-call signals. Additive and optional, so no
  consumer breaks.
- Worth knowing, because it changes what the panel fix alone would have achieved:
  the runtime links `LiteRtRuntimeOptions.signal` in `startPendingLoad`, so once the
  panel passed `signal: loadAbort.signal`, encoder and colbert model loads were
  *already* cancellable through the runtime-level signal. The pipeline-level
  controller added on top is what makes `dispose()` alone sufficient for a consumer
  that wires no signal, and keeps the three pipelines consistent.
- The tokenizer path in encoder/colbert calls `AutoTokenizer.from_pretrained(repoId)`,
  which resolves against the Hugging Face hub directly and therefore ignores a
  custom model base. Pre-existing, not introduced here, and out of scope.
- Whether `packages/retrieval` peer/version metadata needs any change for a
  second published entrypoint (currently `0.1.x` peer on `inference-core`).
  Judgment says no, since `./scoring` imports only types from `inference-core`.
- `docs/superpowers/` is git-ignored and holds local plans/specs; it may contain
  intent for this work that is deliberately not shared. Not inspected.
- `AGENTS.md` and `docs/package-revision-policy.md` still duplicate conventions,
  workflow rules, and the nested-agent-file list. Now that the subpath rule had
  to be written into the policy doc, the duplication cost something concrete.
  Collapsing them into one authority is a real decision still unmade.
