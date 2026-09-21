# Package revision policy

Consuming applications must use one **Lil Tart** Git revision for all shared
inference packages. This keeps package contracts, runtime behavior, model
manifests, and qualification metadata aligned.

The repository was renamed from `myrqyry/litert-playground` to
`myrqyry/lil-tart`. The package namespace remains
`@litert-playground/*` intentionally so the repository rename does not force a
package-API rename.

## Rules

- Select one Lil Tart Git SHA for each consuming application.
- Resolve every Lil Tart package used by that application from the same SHA.
- Use the canonical repository name `myrqyry/lil-tart` in new dependency specs.
- Do not mix packages from different Lil Tart revisions.
- Use SHA pinning as the supported revision mechanism today.
- Treat peer dependency ranges as compatibility information, not as permission
  to mix revisions.
- Do not import files from a package's `src` directory; consume the public
  `.` entrypoint.
- Future release tags may improve ergonomics, but they do not change the
  single-revision rule.

A Git-pinned consumer entry looks like:

```json
{
  "@litert-playground/inference-core": "github:myrqyry/lil-tart#<sha>&path:packages/inference-core",
  "@litert-playground/runtime-litert": "github:myrqyry/lil-tart#<sha>&path:packages/runtime-litert",
  "@litert-playground/browser-cache": "github:myrqyry/lil-tart#<sha>&path:packages/browser-cache"
}
```

Every entry in one application should use the same `<sha>`.

## Supported packed compatibility surface

`pnpm test:compatibility` packs and consumes the packages below as an external
Vite/TypeScript application. A package is not part of the supported downstream
surface merely because it exists in the workspace; it belongs here only when the
compatibility harness can install, type-check, and build it from its public
entrypoint.

- `@litert-playground/inference-core`
- `@litert-playground/runtime-litert`
- `@litert-playground/browser-cache`
- `@litert-playground/text-gen`
- `@litert-playground/encoder`
- `@litert-playground/retrieval`
- `@litert-playground/kokoro`
- `@litert-playground/qwen3-tts`
- `@litert-playground/image-embedding`
- `@litert-playground/video-classification`

For the local conversational-inference stack specifically, the reusable boundary
is:

```text
inference-core
├─ runtime-litert
├─ browser-cache
├─ text-gen
├─ encoder
├─ retrieval
├─ kokoro
└─ qwen3-tts
```

Product applications still own their own worker orchestration, UI state,
conversation semantics, audio playback policy, and app-specific persistence.
Those concerns should not leak back into Lil Tart's shared packages.
