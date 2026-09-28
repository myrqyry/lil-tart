# runtime-qualification

Qualification suite for verifying LiteRT runtime and model behavior.

## Structure

- `tests/runtime-qualification/vitest.config.ts` — separate vitest config with `environment: 'node'`.
- `shared/` — evidenceWriter and shared harness.
- `qwen-browsermemory-generator/` — Qwen browser memory qualification case.
- `pipeline-load-cancellation/` — proves a real in-flight transfer stops when cancelled.
  It probes `AssetResolver.resolve()`, the path production model loading actually takes,
  since `stream()` has no production caller, and declares `requiresProbeAsset` so the
  probe origin is only bound when the case is selected. Its probe asset is generated
  into the gitignored `static-models/` directory rather than committed, and served from
  a separate origin so the browser applies real cross-origin rules. Durable record:
  `docs/verification/2026-09-28-model-load-cancellation.md`.
- Other subdirectories (`upstream/`, `tiny-litert-baseline/`, etc.) hold per-model or per-scenario suites.

## Invocation

```bash
pnpm test:qualification     # vitest with the qualification config
pnpm qualify                # tsx runner (tests/runtime-qualification/run-qualification.ts)
```

`pnpm verify` runs `test:qualification` as one of its gates — do not skip it.

## Conventions

- Known limitations are promoted to `status: 'known-limitation'` in case results so `evidenceWriter` can match them as `pass` (not `fail`).
- `evidenceWriter` accepts observed `'fail'` or `'known-limitation'` for an expected known-limitation.
