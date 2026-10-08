# @litert-playground/react-litert

A small React 19 adapter for Lil Tart's **existing managed LiteRT.js runtime**.
Inspired by the ergonomics of Mutesa-Cedric/react-litert, but does **not**
instantiate a second runtime, load a legacy LiteRT.js build, or own global
WebGPU/WASM caches.

```tsx
import { useManagedLiteRtModel } from '@litert-playground/react-litert'
import type { ManagedLiteRtRuntime } from '@litert-playground/runtime-litert'

function ModelPane({ runtime, modelPath }: {
  runtime: ManagedLiteRtRuntime
  modelPath: string
}) {
  const model = useManagedLiteRtModel({ runtime, modelPath, accelerator: 'auto' })
  // No implicit download or compilation: the user explicitly clicks Load.
  return <button onClick={() => void model.load()} disabled={model.status === 'loading'}>
    {model.status === 'ready' ? 'Ready' : 'Load model'}
  </button>
}
```

The caller constructs/owns `ManagedLiteRtRuntime` with an asset resolver and
controls its final `dispose()`. When a component unmounts or switches models,
the hook aborts **its own** in-flight subscriber without disposing shared cached
models. `run()` invokes real `runtime.predict()` / `predictWithSignature()`
with the same backend selection; callers own input tensors and successful
result tensors. If a cancelled run eventually returns, orphaned outputs are
disposed by the adapter.

**Evidence boundary:** the package tests with fake runtimes prove adapter
lifecycles and cancellation invariants; they do not qualify GPU/WebNN/WASM
inference. For execution proof, use Lil Tart's managed-runtime qualification
suite and browser/device-specific receipts.

**Validation (2026-10-08):** `pnpm verify` passed on the current checkout, including TypeScript, all package tests, package boundaries, packed external-consumer typecheck/build, conversion assets, runtime qualification, and builds. 8 focused adapter tests passed using a fake `ManagedLiteRtRuntime` with real React server rendering for the SSR case. No browser GPU/WebNN/WASM inference was executed through this adapter. Existing playground `ModelRunner` remains the established consumer and is not replaced by this package.
