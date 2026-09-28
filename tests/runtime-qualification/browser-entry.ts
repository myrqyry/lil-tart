import { loadAndCompile, loadLiteRt, setWebGpuDevice, Tensor } from '@litertjs/core'
import { createLiteRtRuntime } from '../../packages/runtime-litert/src/context'
import { createHttpAssetResolver } from '../../packages/inference-core/src/assets/http-resolver'
import { ABORT_PROBE_BYTES } from './pipeline-load-cancellation/probeAsset.meta'
import { GeneratorPhase } from '../../packages/qwen3-tts/src/phases/generator'
import {
  createQualificationTypedArray,
  createQualificationZeroTypedArray,
  serializeQualificationInput,
} from './shared/tensorBridge'
import { verifyQualificationAsset } from './shared/assetVerification'
import type {
  ModelAssetDescriptor,
  QwenGeneratorRequest,
  QwenGeneratorRunResult,
} from './schema/types'

const models = new Map<number, Awaited<ReturnType<typeof loadAndCompile>>>()
let nextModelId = 1

function describeAndStoreModel(
  model: Awaited<ReturnType<typeof loadAndCompile>>,
  includeDetails = true,
) {
  const details = (runner: {
    getInputDetails(): readonly { name: string; shape: Int32Array; dtype: 'float32' | 'int32' | 'uint8' }[]
    getOutputDetails(): readonly { name: string; shape: Int32Array; dtype: 'float32' | 'int32' | 'uint8' }[]
  }) => ({
    inputs: runner.getInputDetails().map((detail) => ({
      name: detail.name,
      shape: Array.from(detail.shape.slice(0, 16)),
      dtype: detail.dtype,
    })),
    outputs: runner.getOutputDetails().map((detail) => ({
      name: detail.name,
      shape: Array.from(detail.shape.slice(0, 16)),
      dtype: detail.dtype,
    })),
  })
  const id = nextModelId++
  models.set(id, model)
  return includeDetails ? { id, ...details(model) } : { id, inputs: [], outputs: [] }
}

Object.assign(window, {
  litertQualification: {
    async initialize(path: string) {
      await loadLiteRt(path)
      const gpu = (navigator as Navigator & {
        gpu?: { requestAdapter(): Promise<GPUAdapter | null> }
      }).gpu
      if (gpu) {
        const adapter = await gpu.requestAdapter()
        if (adapter) setWebGpuDevice(await adapter.requestDevice())
      }
    },
    async fetchAsset(descriptor: {
      id: string
      url: string
      bytes: number
      sha256: string
    }) {
      const response = await fetch(descriptor.url)
      if (!response.ok) throw new Error(`Asset request failed: ${response.status}`)
      const buffer = await response.arrayBuffer()
      const verified = await verifyQualificationAsset(buffer, descriptor)
      return Array.from(new Uint8Array(verified))
    },
    // Streams a real local asset through the shared resolver, aborts after the first
    // chunk, and reports what actually happened. The engine cannot be cancelled once
    // it starts compiling, so the transfer is the only place cancellation can be proven.
    async probeAbortStopsTransfer(origin: string) {
      const url = new URL('/asset', origin)
      const resolver = createHttpAssetResolver(url.origin)
      const controller = new AbortController()
      const asset = { id: 'abort-probe.bin', path: url.pathname.replace(/^\//, '') }
      if (!resolver.stream) {
        return { status: 'fail' as const, stage: 'probe', error: { message: 'resolver cannot stream' } }
      }
      const stream = await resolver.stream(asset, { signal: controller.signal })
      const reader = stream.getReader()
      const first = await reader.read()
      if (first.done || !first.value) {
        return {
          status: 'fail' as const,
          stage: 'probe',
          error: { message: 'asset completed before any abort could be observed' },
        }
      }
      const bytesAtAbort = first.value.byteLength
      controller.abort()

      let bytesAfterAbort = bytesAtAbort
      let cancellationCode: string | null = null
      try {
        for (;;) {
          const next = await reader.read()
          if (next.done) break
          bytesAfterAbort += next.value?.byteLength ?? 0
        }
      } catch (error) {
        const code = (error as { code?: string }).code
        cancellationCode = typeof code === 'string' ? code : 'uncoded'
      }

      const transferStopped = bytesAfterAbort < ABORT_PROBE_BYTES
      if (!transferStopped || cancellationCode !== 'CANCELLED') {
        return {
          status: 'fail' as const,
          stage: 'probe',
          error: {
            message:
              `abort did not stop the transfer: ${bytesAfterAbort} of ${ABORT_PROBE_BYTES} bytes, ` +
              `cancellation code ${String(cancellationCode)}`,
          },
        }
      }
      return { status: 'pass' as const }
    },
    runModuleWorkerLoader() {
      const worker = new Worker(
        new URL('./module-worker-loader/worker.ts', import.meta.url),
        { type: 'module' },
      )
      return new Promise<{
        status: 'pass' | 'fail'
        stage?: string
        error?: { message: string }
      }>((resolve) => {
        worker.onmessage = (event) => {
          worker.terminate()
          resolve(event.data)
        }
        worker.onerror = (event) => {
          worker.terminate()
          resolve({
            status: 'fail',
            stage: 'worker-load',
            error: { message: event.message || 'Module worker failed to load' },
          })
        }
      })
    },
    async loadAndCompile(bytes: number[], accelerator: 'wasm' | 'webgpu') {
      return describeAndStoreModel(await loadAndCompile(new Uint8Array(bytes), { accelerator }))
    },
    getSignatureDetails(id: number, signature: string) {
      const model = models.get(id)
      if (!model) throw new Error(`Unknown qualification model: ${id}`)
      const runner = model.signatures[signature]
      if (!runner) throw new Error(`Unknown LiteRT signature: ${signature}`)
      const result = {
        inputs: runner.getInputDetails().map((detail) => ({
          name: detail.name,
          shape: Array.from(detail.shape.slice(0, 16)),
          dtype: detail.dtype,
        })),
        outputs: runner.getOutputDetails().map((detail) => ({
          name: detail.name,
          shape: Array.from(detail.shape.slice(0, 16)),
          dtype: detail.dtype,
        })),
      }
      return result
    },
    getModelDetails(id: number) {
      const model = models.get(id)
      if (!model) throw new Error(`Unknown qualification model: ${id}`)
      return {
        inputs: model.getInputDetails().map((detail) => ({
          name: detail.name,
          shape: Array.from(detail.shape.slice(0, 16)),
          dtype: detail.dtype,
        })),
        outputs: model.getOutputDetails().map((detail) => ({
          name: detail.name,
          shape: Array.from(detail.shape.slice(0, 16)),
          dtype: detail.dtype,
        })),
      }
    },
    async runSignatureWithZeros(id: number, signature: string) {
      const model = models.get(id)
      if (!model) throw new Error(`Unknown qualification model: ${id}`)
      const runner = model.signatures[signature]
      if (!runner) throw new Error(`Unknown LiteRT signature: ${signature}`)
      const tensors = runner.getInputDetails().map((detail) => {
        const length = detail.shape.reduce((size, value) => size * value, 1)
        return Tensor.fromTypedArray(
          createQualificationZeroTypedArray(detail.dtype, length) as any,
          detail.shape,
        )
      })
      try {
        const outputs = await runner.run(tensors)
        const outputTensors: Tensor[] = Array.isArray(outputs)
          ? outputs
          : Object.values(outputs) as Tensor[]
        outputTensors.forEach((output) => output.delete())
      } finally {
        tensors.forEach((tensor) => tensor.delete())
      }
    },
    async runWithZeros(id: number) {
      const model = models.get(id)
      if (!model) throw new Error(`Unknown qualification model: ${id}`)
      const tensors = model.getInputDetails().map((detail) => {
        const length = detail.shape.reduce((size, value) => size * value, 1)
        return Tensor.fromTypedArray(
          createQualificationZeroTypedArray(detail.dtype, length) as any,
          detail.shape,
        )
      })
      try {
        const outputs = await model.run(tensors)
        const outputTensors: Tensor[] = Array.isArray(outputs)
          ? outputs
          : Object.values(outputs) as Tensor[]
        outputTensors.forEach((output) => output.delete())
      } finally {
        tensors.forEach((tensor) => tensor.delete())
      }
    },
    async loadAndCompileAsset(
      descriptor: ModelAssetDescriptor,
      accelerator: 'wasm' | 'webgpu',
    ) {
      const response = await fetch(descriptor.url)
      if (!response.ok) throw new Error(`Asset request failed: ${response.status}`)
      const buffer = await response.arrayBuffer()
      const verified = await verifyQualificationAsset(buffer, descriptor)
      const compiled = await loadAndCompile(new Uint8Array(verified), { accelerator })
      const result = describeAndStoreModel(
        compiled,
        false,
      )
      return result.id
    },
    async runQwenGenerator(request: QwenGeneratorRequest): Promise<QwenGeneratorRunResult> {
      const receipts = [] as QwenGeneratorRunResult['receipts']
      let stage: string | undefined
      const resolver = createQwenAssetResolver(request.assets)
      const runtime = await createLiteRtRuntime({
        assets: resolver,
        backend: request.backend,
        packageName: '@litert-playground/qwen3-tts',
        assetBase: '/node_modules/@litertjs/core/',
      })
      const phase = new GeneratorPhase(request.variant, {
        onTrace: (event) => {
          stage = event.stage
          receipts.push(event)
        },
      })
      try {
        stage = 'talker-compile'
        await phase.load(runtime)
        await phase.generate(request.text ? { text: request.text } : { text: 'Testing one two three.' }, {
          ...request.config,
          maxFrames: 1,
        })
        return {
          observation: {
            status: 'pass',
            resolvedBackend: runtime.backend === 'webgpu' ? 'webgpu' : 'wasm',
          },
          receipts,
        }
      } catch (error) {
        const cause = error as { code?: string; stage?: string; message?: string }
        return {
          observation: {
            status: 'fail',
            stage: stage ?? cause.stage,
            error: {
              code: cause.code,
              stage: stage ?? cause.stage,
              message: cause.message ?? String(error),
            },
          },
          receipts,
        }
      } finally {
        phase.dispose()
        runtime.liteRt.dispose()
      }
    },
    async run(
      id: number,
      request: ReturnType<typeof serializeQualificationInput>,
    ) {
      const model = models.get(id)
      if (!model) throw new Error(`Unknown qualification model: ${id}`)
      const createTensor = (serialized: {
        data: number[]
        shape: number[]
        dtype: 'float32' | 'int32' | 'uint8'
      }) => Tensor.fromTypedArray(
        createQualificationTypedArray(serialized.dtype, serialized.data) as any,
        serialized.shape,
      )
      const tensors = request.input.kind === 'positional'
        ? request.input.tensors.map(createTensor)
        : Object.fromEntries(
          Object.entries(request.input.tensors).map(([name, tensor]) => [name, createTensor(tensor)]),
        )
      const runner = request.signature ? model.signatures[request.signature] : model
      if (!runner) throw new Error(`Unknown LiteRT signature: ${request.signature}`)
      try {
        const outputs = await runner.run(tensors)
        const outputTensors = Array.isArray(outputs) ? outputs : Object.values(outputs)
        try {
          const serialize = (output: Tensor) => ({
            data: Array.from(output.toTypedArray()),
            shape: Array.from(output.type.layout.dimensions),
            dtype: output.type.dtype,
          })
          return Array.isArray(outputs)
            ? {
              kind: 'positional' as const,
              tensors: outputs.map(serialize),
            }
            : {
              kind: 'named' as const,
              tensors: Object.fromEntries(
                Object.entries(outputs).map(([name, output]) => [name, serialize(output)]),
              ),
            }
        } finally {
          outputTensors.forEach((output) => output.delete())
        }
      } finally {
        if (Array.isArray(tensors)) tensors.forEach((tensor) => tensor.delete())
        else Object.values(tensors).forEach((tensor) => tensor.delete())
      }
    },
    delete(id: number) {
      models.get(id)?.delete()
      models.delete(id)
    },
  },
})

function createQwenAssetResolver(assets: QwenGeneratorRequest['assets']) {
  const byPath = new Map(assets.map((asset) => [asset.path, asset]))
  const cache = new Map<string, Promise<ArrayBuffer>>()
  return {
    resolve(asset: { path: string }): Promise<ArrayBuffer> {
      const descriptor = byPath.get(asset.path)
      if (!descriptor) throw new Error(`Unknown Qwen asset: ${asset.path}`)
      const existing = cache.get(asset.path)
      if (existing) return existing
      const pending = fetch(descriptor.url).then(async (response) => {
        if (!response.ok) throw new Error(`Asset request failed: ${response.status}`)
        const buffer = await response.arrayBuffer()
        return verifyQualificationAsset(buffer, descriptor)
      })
      cache.set(asset.path, pending)
      return pending
    },
  }
}
