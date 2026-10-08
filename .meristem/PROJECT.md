<!-- meristem-template:v1 -->
# Project

Authority note: this file is a Meristem index, not a replacement for the
repository's own docs. Where a section below points at an established
authority, that authority wins and this file should be updated to match it.

## Purpose

Lil Tart is a local-first browser inference workspace with two faces that are
one project: the interactive app in `apps/playground` (a model launcher and
runtime qualification lab) and the reusable `@litert-playground/*` packages
that app is built from.

The distinctive premise (from `README.md`): **prove a capability once, package
it once, and consume it everywhere**, instead of rebuilding browser inference
plumbing in every app. The intended workflow is deliberately boring — browse
models without triggering a download, explicitly download and load the model
you want, observe which backend actually resolved, run preflight and real
inference, then capture enough evidence to reuse that working path elsewhere.

Coverage spans text generation, retrieval, speech, audio, vision, OCR, image
embeddings, and video classification.

## Invariants

- `pnpm verify` is the authoritative pre-merge gate. Boundary and qualification
  suites are not optional extras.
- The shared packages keep the `@litert-playground/*` namespace even though the
  repository is `myrqyry/lil-tart`. Branding did not rename package imports.
- Packages ship TypeScript source: no per-package build step,
  `exports: ".": "./src/index.ts"`, all `private: true` and `type: module`.
- A package is only part of the supported downstream surface when
  `pnpm test:compatibility` can pack, install, type-check, and build it from its
  public entrypoint. Workspace-only dependencies must not leak as `workspace:*`
  into packed metadata.
- Consumers pin one Lil Tart Git SHA per application and resolve every Lil Tart
  package from that same SHA. Do not mix revisions; peer ranges are compatibility
  information, not permission to mix.
- Product concerns (episodes, OBS, UI, orchestration, conversation semantics)
  belong in consuming apps, not in shared packages.
- A successful app session does not silently promote repository verification
  metadata. Runtime evidence and durable verification level are separate.

## Architecture and boundaries

Dependency flow: `inference-core → runtime-litert → model packages → playground`.

- `packages/inference-core` — contracts, assets, receipts, validation. No model
  dependencies; must stay independent of the runtime.
- `packages/runtime-litert` — managed LiteRT runtime, backend selection, caching,
  preflight, telemetry.
- Model packages — `text-gen`, `retrieval`, `encoder`, `image-embedding`,
  `kokoro`, `qwen3-tts`, `video-classification`, `depth-estimation`.
- `packages/browser-cache` — browser-side model/tensor cache.
- `apps/playground` — the consumer app (React + Vite + Tailwind).
- `examples/` — minimal consumer examples, deliberately small.
- `tests/` — cross-package boundary tests; `tests/runtime-qualification/` is a
  Node-environment runtime qualification suite with its own vitest config.

Where authority lives (do not duplicate these):

- Agent rules and nested routing: `AGENTS.md` plus the six nested
  `AGENTS.md` files for `runtime-litert`, `qwen3-tts`, `video-classification`,
  `inference-core`, `tests/runtime-qualification`, and `apps/playground`.
- Downstream revision policy and supported surface: `docs/package-revision-policy.md`.
- Durable verification records: `docs/verification/`.
- Change history: `CHANGELOG.md` (Keep a Changelog, SemVer, active `[Unreleased]`).
- `docs/superpowers/` is a git-ignored local working lane for plans/specs, not
  shared project knowledge.

## Known-good workflows

| Command | Meaning |
| --- | --- |
| `pnpm verify` | CI gate: typecheck + test + boundaries + compatibility + conversion assets + qualification + build |
| `pnpm test:boundaries` | Package dependency rules (examples → public entrypoints, inference-core independence, playground → runtime-litert) |
| `pnpm test:compatibility` | Pack and consume the supported external surface |
| `pnpm test:conversion-assets` | Validate pinned conversion recipes and candidate-evidence boundaries |
| `pnpm test:qualification` | Runtime qualification suite (separate vitest config, Node env) |
| `pnpm --filter <pkg> test` / `typecheck` | Single-package work |
| `pnpm dev` | Playground dev server |

Environment: Node 22+, pnpm 11+.

Workflow rules in force (from the root `AGENTS.md`): stage by explicit path, never
`git add .`; conventional commit messages with a bullet body for multi-area
changes; the pre-push hook runs automatically and real failures must be fixed
rather than bypassed.

CI mirrors this: `.github/workflows/verify.yml` on push to `master` and pull
requests; `.github/workflows/runtime-qualification.yml` is manual
`workflow_dispatch` with optional case/backend inputs.

## Verification

Distinguish evidence classes rather than collapsing them:

1. Static inspection — reading source and config.
2. Build/typecheck/test evidence — `pnpm typecheck`, `pnpm test`, boundary and
   compatibility suites.
3. Runtime evidence — actual preflight and inference in a browser, backend
   resolution observed rather than assumed.
4. Manual UI verification — only for user-visible behavior.

Only claim the level actually established. A commit that once passed verify on
an earlier branch is not evidence about the current tree.

Browser observations are a separate, recorded evidence class. They live in
`docs/verification/` as durable records, because the raw results JSON under
`tests/runtime-qualification/results/` is gitignored. Run one with
`pnpm qualify -- --case <id>`. A new browser case is only worth trusting once it
has been shown to fail for the right reason.

## Open structural uncertainty

- The supported packed surface in `docs/package-revision-policy.md` enumerates
  packages, not subpath entrypoints. `packages/retrieval` now also publishes a
  `./scoring` subpath, and it is unclear whether a subpath export counts as
  supported surface that the compatibility harness must cover, or whether
  subpaths are deliberately out of scope for the supported-surface list. Nothing
  in the policy document answers this yet. See `ACTIVE_WORK.md`.
- Duplication risk between `AGENTS.md` and `docs/package-revision-policy.md`:
  both carry conventions, workflow rules, and the nested-agent-file list. They
  currently agree, but they are two places to update. Writing the new retrieval
  `./scoring` rule into the policy document made that cost concrete. Not
  resolved here because choosing a single authority is a project decision, not
  an initialization default.

Resolved during this branch: whether subpath exports count as supported surface.
The policy document now states that subpaths are supported on the same terms as
`.` — that is, only when `pnpm test:compatibility` resolves them through the
packed tarball — and lists `@litert-playground/retrieval/scoring`.
