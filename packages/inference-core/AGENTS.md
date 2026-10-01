# inference-core

Contracts, asset resolvers, receipts, and validation. No model-specific logic.

## Key constraints

- **Must stay independent**: no references to kokoro, qwen3-tts, @litertjs/core, or any model package.
- **Asset verification**: `verifyAssetIntegrity(asset, buffer)` checks declared byte length and SHA-256. `createManifestVerifyingAssetResolver()` may receive `maxSha256Bytes` for browser paths that cannot safely digest very large contiguous buffers. Known-length streamed assets above the ceiling remain streaming with byte-length verification; unknown-length streamed assets remain streaming/unhashed because there is no safe pre-buffer decision point. Buffered `resolve()` calls may still hash an unknown-length asset after its actual size is known to remain within the ceiling.
- GOTCHA: manifest integrity is **exact-path authoritative**. A semantic ID may be reused by dynamic assets (for example TTS voices), so an ID match must never make one path inherit another path's hash. If no manifest path matches, the caller-supplied asset facts are used; with no supplied facts that path is intentionally unverified.

## Verification

```bash
pnpm --filter @litert-playground/inference-core typecheck
pnpm --filter @litert-playground/inference-core test
```
