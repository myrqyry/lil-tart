# PR #15 merged Verify failure follow-up — 2026-10-05

## Diagnosis

PR [#15](https://github.com/myrqyry/lil-tart/pull/15) is merged. Current master
is `1e72863a44fe1a250e55aac7cfcb9f48b2101564`, with the same tree as PR head
`a7d14b49129b950ae1f96f67b7d84c9581b0991e`. Remote master was checked again
after validation and still pointed to that merge commit.

Gmail's Verify notifications identify three comparable failed runs:

| Tree | Run | Job | First failing boundary |
| --- | --- | --- | --- |
| Pre-PR master `4bdb3ec` | [37348794065](https://github.com/myrqyry/lil-tart/actions/runs/37348794065) | `111894098739` | Example test config import |
| PR head `a7d14b4`, synthetic merge `ec5b0f9` | [37350922270](https://github.com/myrqyry/lil-tart/actions/runs/37350922270) | `111901245910` | Same error |
| Merged master `1e72863` | [37351543963](https://github.com/myrqyry/lil-tart/actions/runs/37351543963) | `111903329394` | Same error |

All three logs show Node 22.16.0 and pnpm 11.17.0. Dependency installation
and workspace typechecking succeed. `pnpm test` stops at the minimal Kokoro
example with `ERR_UNKNOWN_FILE_EXTENSION` for
`packages/qwen3-tts/src/provenance.ts`, imported by `examples/vite.config.ts`.
The remaining gate stages are therefore not reached in those runs.

This is a real, pre-existing CI configuration/runtime incompatibility, not a
regression introduced by PR #15. There is no evidence of environment drift
between these three runs. Vite's default bundle config loader externalizes the
public workspace package import; Node 22.16.0 then receives TypeScript directly,
without native type stripping enabled by default. The same error reproduces
locally with the exact CI Node version. The public import was introduced before
PR #15, in `235068d`.

The Vercel bot reports Ready at 2026-10-05 17:46:22 UTC, but Vercel's playground
build does not exercise this example test config or the full Verify gate.
Kilo's later comment explicitly says its review did not run because the
provider content filter blocked it; it supplies no new correctness evidence.

## Fixes

- Example dev, build, and test commands use `--configLoader runner`, which
  transforms the public workspace TypeScript import. The shared example config
  uses `import.meta.dirname`, compatible with the runner's ESM evaluation.
  Public package boundaries and pinned runtime/CI versions are preserved.
- Both qualification expectation matchers reject `numericComparison.passed:
  false`, regardless of the reported inference status. Manifest promotion also
  returns `limited` for failed parity. Raw observations and numeric evidence
  are preserved for diagnosis. Cases without numeric evidence retain their
  existing matching behavior; known limitations without failed parity still
  use the existing status/error rules.
- The conversion validator compares complete expected parsed structures,
  including every option and the exact two-rule order. Missing/extra properties,
  changed values, appended/removed/reordered rules all fail. Object key order
  and JSON formatting remain immaterial. The expected structures were checked
  against both upstream recipe files at the documented
  `john-rocky/LiteRT-Models@460d52221c22388b6a9a8e8a44b61dde72976b4e` pin.
- Conversion CLI mutation tests are included in `test:conversion-assets`, and
  therefore in the authoritative `pnpm verify` gate. Fixtures are temporary;
  the real recipes and catalog are never mutated by the tests.

## Fresh evidence

| Check | Result |
| --- | --- |
| Original example test under Node 22.16.0 | Reproduces the hosted `.ts` import startup error |
| New parity tests against unchanged merged implementation | Four failures: both matchers, result construction, and manifest mapping |
| Recipe CLI mutation tests against merged validator in an isolated fixture | Eleven rejected-test failures: old validator incorrectly accepts semantic drift; four tests pass |
| `pnpm install --frozen-lockfile` using pnpm 11.17.0 | Exit 0; lockfile unchanged |
| `pnpm verify` using Node 22.16.0 / pnpm 11.17.0 | Exit 0 across all seven gates |
| Package/example tests | 426 passing executions; the shared example config discovers both extraction files in each example command |
| Boundary tests | 15/15 pass |
| Packed consumer compatibility | All 10 package rows pass import, peer, typecheck, and build checks |
| Conversion gate | 15/15 tests pass, then 10 candidates and both recipes validate |
| Qualification contracts | 73/73 pass |
| Production builds | Playground and both example commands pass; existing large-chunk advisories remain |
| Example dev server under Node 22.16.0 | Starts; Kokoro HTML returns HTTP 200; all three worker shells return HTTP 200 with exact source bytes |
| `git diff --check` | Exit 0 |

The recipe mutations cover asymmetric weights, compute precision,
dequantization, disabled checks, minimum weight size, regex, embedding algorithm,
dtype, missing/extra options, extra/missing rules, and rule order. Both pinned
recipes are exercised independently. Checked-in recipe contents match the full
upstream parsed structures.

## Remaining limits and delivery state

The source fixes are in the local master checkout, uncommitted and unpushed.
Hosted Verify has not run on this patch; the historical runs remain failed.
The two PR #15 threads remain unresolved remotely. No review comments were
posted and no deployment was initiated.

This work establishes deterministic qualification contracts and build/config
behavior. It does not establish fresh browser model inference, WebGPU numerical
parity, conversion quality, or qualification of any candidate model. Candidate
browser qualification remains unverified. Existing historical browser evidence
is preserved and is not presented as a fresh result.
