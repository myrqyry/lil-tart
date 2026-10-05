# Lil Tart

pnpm monorepo for LiteRT.js browser inference. Node 22+, pnpm 11+.

## Commands

| Command | What it does |
|---------|-------------|
| `pnpm install` | Install workspace deps |
| `pnpm dev` | Start playground dev server |
| `pnpm build` | Build all projects |
| `pnpm typecheck` | Type-check every package |
| `pnpm test` | Run all package tests |
| `pnpm verify` | **CI gate**: typecheck + test + boundaries + compatibility + conversion assets + qualification + build |
| `pnpm test:boundaries` | Enforce package dependency rules (examples → public entrypoints, inference-core independent, playground → runtime-litert) |
| `pnpm test:compatibility` | Pack/consume the supported external surface |
| `pnpm test:conversion-assets` | Validate pinned conversion recipes and candidate-evidence boundaries |
| `pnpm test:qualification` | Runtime qualification suite (Node env, separate vitest config) |
| `pnpm --filter <pkg> test` | Single-package tests |
| `pnpm --filter <pkg> typecheck` | Single-package typecheck |

`pnpm verify` is the authoritative pre-merge check. Do not skip boundary, conversion-asset, or qualification tests.

## Package layout

```
packages/inference-core    contracts, assets, receipts, validation (no model deps)
packages/runtime-litert    managed LiteRT runtime, backends, caching, preflight, telemetry
packages/kokoro            Kokoro TTS via kokoro-js (verified browser path)
packages/qwen3-tts         Qwen3-TTS pipeline (3 LiteRT graphs; known browser memory limitation)
packages/text-gen          LFM2.5 / Gemma 4 / Qwen3 text generation
packages/retrieval         ColBERT embeddings, late-interaction scoring
packages/encoder           Text embeddings, token classification
packages/image-embedding   Image embeddings
packages/browser-cache     Browser-side model/tensor cache
packages/video-classification  MoViNet video classification
apps/playground            React + Vite + Tailwind model lab (the consumer app)
```

Dependency flow: `inference-core → runtime-litert → model packages → playground`.
Product concepts (episodes, OBS, UI) belong in consuming apps, not shared packages.

## Conventions

- Shared packages keep the existing `@litert-playground/*` namespace even though
  the repository is now `myrqyry/lil-tart`; do not rename package imports as a
  side effect of repository branding.
- All packages: `private: true`, `type: module`, `exports: ".": "./src/index.ts"`.
  Packages ship as TypeScript source; there is no per-package build step.
- Any package intended for downstream apps must pass `pnpm test:compatibility`.
  Workspace-only dependencies must not leak as `workspace:*` in packed metadata.
- Tests co-located in `src/*.test.ts` (vitest).
- Qualification tests live in `tests/runtime-qualification/` with their own vitest config (`environment: 'node'`).
- Boundary tests in `tests/` enforce architecture invariants — keep them green.
- Vercel deploys `apps/playground/dist`; build command is `pnpm --filter playground build`.

## Workflow rules

- Stage files by explicit path; never `git add .`.
- Commit messages: conventional commits (`fix:`, `feat:`, `test:`, etc.) with a bullet body when the change spans multiple areas.
- Pre-push hook runs automatically; fix real failures, do not bypass.

## Nested agent files

Package-specific guidance lives next to the code it describes:

- `packages/runtime-litert/AGENTS.md` — backend selection, WASM probes, coordinator
- `packages/qwen3-tts/AGENTS.md` — worker lifecycle, browser memory limits
- `packages/video-classification/AGENTS.md` — transactional frame commit, SIMD probe correctness
- `packages/inference-core/AGENTS.md` — asset verification, independence constraint
- `tests/runtime-qualification/AGENTS.md` — qualification suite structure and invocation
- `apps/playground/AGENTS.md` — app entrypoint, Vercel build output

<!-- meristem:start -->
## Meristem project continuity

This repository uses Meristem project state.

Before non-trivial work:

1. Treat current repository/runtime evidence as the authority for what exists now.
2. Read `.meristem/PROJECT.md` for durable project purpose, invariants, and architecture boundaries when relevant.
3. If `ACTIVE_WORK.md` exists, read it for the current objective, locked decisions, constraints, verification state, and unresolved uncertainty.
4. Consult `.meristem/DECISIONS.md`, `LEARNINGS.md`, `ERRORS.md`, or `FEATURE_REQUESTS.md` only when they can materially change the current task.
5. Do not promote session detail into durable files merely because it is recent. Persist consequential state before dependent continuation when losing it would make later work expensive or misleading.
6. Preserve existing project conventions and working state. Do not broadly rewrite configuration or clean up unrelated code for convenience.
7. Verify user-visible outcomes before claiming completion.
8. Treat canonical `.meristem/*.md` files and `ACTIVE_WORK.md` as commit-safe repository knowledge. Keep secrets, personal continuity, raw conversations, and machine-specific state out of them; use `.meristem/local/` only for repo-local private/machine state.

Use the narrowest capable tool or specialist. Current source and runtime truth outrank stale continuation notes.
<!-- meristem:end -->
