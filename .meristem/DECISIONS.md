<!-- meristem-template:v1 -->
# Decisions

Record decisions only when the choice or its rationale is likely to matter later.

Existing authority: settled downstream-consumption decisions (single Git SHA per
application, no mixed revisions, no `src` imports, supported packed surface) are
already durable in `docs/package-revision-policy.md` and are not duplicated here.
Add entries below only for choices that are *not* already recorded there.

Use entries like:

## YYYY-MM-DD — Decision

**Decision:**  
**Why:**  
**Alternatives considered:**  
**Constraints / consequences:**  
**Evidence / references:**  
**Supersedes:** none

## 2026-09-30 — Keep conversion evidence separate from browser qualification

**Decision:** Android LiteRT/CompiledModel conversion results may enter Lil Tart as pinned conversion evidence and candidate-model metadata, but they do not promote browser verification. Browser qualification still requires Lil Tart's own recorded browser observation.

**Why:** Conversion success, delegate compilation, and even successful inference can diverge from numerical correctness across runtimes and devices.

**Alternatives considered:** Treat upstream Android success as generic LiteRT qualification; rejected because it would collapse distinct runtime evidence classes.

**Constraints / consequences:** Candidate models remain browser-unverified until qualified here. Applicable runtime cases should attach numeric parity evidence when a trusted reference output exists.

**Evidence / references:** `conversion/`, `tests/runtime-qualification/shared/numericParity.ts`, source snapshot `john-rocky/LiteRT-Models@460d52221c22388b6a9a8e8a44b61dde72976b4e`.

**Supersedes:** none
