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
  tests (retrieval 7/7, qwen3-tts 13 files, playground 12 files, …),
  boundaries 13/13, compatibility 10/10 packages, qualification 19 files, build.
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
  Each mutation was reverted and the suite re-confirmed green.

Not established: browser runtime evidence. Nothing here was exercised in a real
browser session, so no runtime or manual UI claim is being made. In particular the
text-gen fix is verified at the type/API boundary and by unit tests, not by an
actual generation run in a browser.

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
