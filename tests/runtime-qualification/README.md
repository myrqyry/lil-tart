# Runtime qualification laboratory

The Playground runtime qualification laboratory records model, backend,
browser, worker, and LiteRT.js runtime behavior. Deterministic contract tests
run in normal CI. Real-browser qualification is opt-in because it downloads
assets and requires installed browser binaries.

## Commands

Run the deterministic lane with:

```bash
pnpm test:qualification
```

Run the real browser lane with the verified baseline:

```bash
pnpm qualify -- --case tiny-litert-wasm-baseline --backend wasm
```

Use repeated `--case` and `--backend` flags to select multiple browser-backed
entries. Contract-only cases are available to deterministic tests but are
blocked from browser qualification until a real reproduction exists. Use
`--browser chromium|firefox|webkit` and `--headed` when browser inspection is
needed.

## Evidence

Generated results use `schemaVersion: 1` and record the case,
`evidenceKind`, full Playground revision, LiteRT.js package and version,
browser capabilities, requested and resolved backend, model revision, expected
status, observed status, structured error, and `matchesExpectation`. Only
`browser-observation` results can feed manifest qualification. Files are
written under `results/` and are ignored by Git except for `.gitkeep`.

When numeric comparison evidence is present, `passed: false` always makes
`matchesExpectation` false and prevents promotion to a qualified manifest,
even if inference returned `status: 'pass'`. The original observation and
comparison remain in the evidence for diagnosis. Cases without numeric
comparison keep their existing status/error matching rules.

A reproduced known limitation is an observed failure that matches the case
expectation. If an upstream fix makes a known-limitation case pass, the result
does not match and the qualification command exits nonzero. This prevents
upstream behavior changes from being silently accepted.

## Asset policy

Large model files stay outside Git. Browser cases use immutable HTTPS asset
descriptors with an exact byte count and lowercase SHA-256. The deterministic
lane validates descriptors but never downloads them. The tiny baseline is the
only checked-in model descriptor intended for immediate execution.

## Runtime lanes

Production qualification uses LiteRT.js 2.5.3. Upstream or prerelease runtime
testing is isolated and cannot change the production dependency pin. The
optional standalone model-tester lane is also isolated from production package
metadata.

## Boundaries

Playground owns generic runtime, model, browser, backend, and error evidence.
Live Streamer and PodQast own their product workflows and may add fields to
the JSON result without replacing the generic fields. Product repositories
must use one Playground Git SHA for all Playground packages.
