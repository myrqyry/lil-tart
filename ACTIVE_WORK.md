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
  - **First attempt was wrong and was corrected.** Routing through
    `context.assets.stream()` fixed the 404 but introduced a worse problem:
    `createModelLibraryAssetResolver.stream` awaits `resolve()` first, so the whole
    checkpoint was materialized into an ArrayBuffer and handed back wrapped in a
    ReadableStream. For a 1.2B-parameter model that is a full buffered read, on a
    repo whose own `qwen3-tts/AGENTS.md` exists because of browser memory limits.
    The absolute-URL form is the correct shape: the engine fetches and streams the
    checkpoint itself. A test now asserts the resolver is never called, so the
    memory profile cannot be "optimized" back into a buffer.
  - Trade-off accepted: text models are fetched by the engine directly, so they do
    **not** participate in Cache Storage. Before this branch they were broken
    (404), so nothing working was lost. Gaining caching without buffering would
    mean making the playground resolver tee a real `fetch` body instead of
    wrapping `resolve()`, which is a larger change and is not done.
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
    `removeStoredModel`.
- `apps/playground/src/modelStorage.test.ts` — covers base filtering, the
  prefix-collision case that the first guard missed, the slash-less base, and
  orphan listing being non-destructive.
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
- Text models bypass Cache Storage entirely (see the trade-off note above). If
  offline reuse of LFM/Gemma text models is wanted, the playground resolver needs to
  tee a real `fetch` response body into the cache rather than wrapping `resolve()`.
  That would restore caching for every consumer without reintroducing a full buffer.
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
