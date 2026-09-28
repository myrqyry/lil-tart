import type {
  QualificationBackend,
  QualificationCase,
  QualificationContext,
  QualificationEnvironment,
  QualificationResult,
  QualificationRunOptions,
  QualificationSelection,
  QualificationTensor,
  QualificationTensorInput,
  QwenGeneratorRequest,
  QwenGeneratorRunResult,
} from '../schema/types'
import type { ModelAssetDescriptor } from '../schema/types'
import { captureBrowserEnvironment } from './environment'
import { runQualificationMatrix } from './matrix'
import { createServer, type ViteDevServer } from 'vite'
import { createReadStream, statSync } from 'node:fs'
import { createServer as createNodeServer } from 'node:http'
import { join } from 'node:path'
import { createQualificationTypedArray, serializeQualificationInput } from './tensorBridge'
import { ABORT_PROBE_PATH } from '../pipeline-load-cancellation/probeAsset.meta'

interface BrowserRuntimeApi {
  initialize(path: string): Promise<void>
  fetchAsset(descriptor: Pick<ModelAssetDescriptor, 'id' | 'url' | 'bytes' | 'sha256'>): Promise<number[]>
  runModuleWorkerLoader(): Promise<{
    status: 'pass' | 'fail'
    stage?: string
    error?: { message: string }
  }>
  loadAndCompile(model: number[], accelerator: QualificationBackend): Promise<{
    id: number
    inputs: Array<{ name: string; shape: number[]; dtype: 'float32' | 'int32' | 'uint8' }>
    outputs: Array<{ name: string; shape: number[]; dtype: 'float32' | 'int32' | 'uint8' }>
  }>
  getSignatureDetails(id: number, signature: string): Promise<{
    inputs: Array<{ name: string; shape: number[]; dtype: 'float32' | 'int32' | 'uint8' }>
    outputs: Array<{ name: string; shape: number[]; dtype: 'float32' | 'int32' | 'uint8' }>
  }>
  getModelDetails(id: number): Promise<{
    inputs: Array<{ name: string; shape: number[]; dtype: 'float32' | 'int32' | 'uint8' }>
    outputs: Array<{ name: string; shape: number[]; dtype: 'float32' | 'int32' | 'uint8' }>
  }>
  runSignatureWithZeros(id: number, signature: string): Promise<void>
  runWithZeros(id: number): Promise<void>
  loadAndCompileAsset(asset: ModelAssetDescriptor, accelerator: QualificationBackend): Promise<number>
  run(id: number, request: ReturnType<typeof serializeQualificationInput>): Promise<BrowserSerializedOutput>
  delete(id: number): void
  runQwenGenerator(request: QwenGeneratorRequest): Promise<QwenGeneratorRunResult>
  probeAbortStopsTransfer(origin: string): Promise<{
    status: 'pass' | 'fail'
    stage?: string
    error?: { message: string }
  }>
}

interface BrowserSerializedTensor {
  data: number[]
  shape: number[]
  dtype: 'float32' | 'int32' | 'uint8'
}

type BrowserSerializedOutput =
  | { kind: 'positional'; tensors: BrowserSerializedTensor[] }
  | { kind: 'named'; tensors: Record<string, BrowserSerializedTensor> }

declare global {
  interface Window {
    litertQualification: BrowserRuntimeApi
  }
}

export interface BrowserLaunchOptions {
  browserName: 'chromium' | 'firefox' | 'webkit'
  headless: boolean
}

export interface BrowserQualificationOptions extends QualificationRunOptions {
  launch: BrowserLaunchOptions
  selection: QualificationSelection
  launcher?: BrowserLauncher
}

export interface BrowserCapabilities {
  browser: string
  browserVersion: string
  operatingSystem?: string
  device?: string
  gpu?: string
  webgpuAvailable: boolean
}

export type BrowserLauncher = <T>(
  options: BrowserLaunchOptions,
  run: (
    capabilities: BrowserCapabilities,
    runtime: QualificationContext['runtime'],
    fetchAsset: QualificationContext['fetchAsset'],
  ) => Promise<T>,
) => Promise<T>

export async function runBrowserQualification(
  cases: QualificationCase[],
  options: BrowserQualificationOptions,
): Promise<QualificationResult[]> {
  const selectedCases = options.selection.caseIds
    ? cases.filter((item) => options.selection.caseIds?.includes(item.id))
    : cases
  const contractOnly = selectedCases.filter(
    (qualificationCase) => qualificationCase.evidenceKind !== 'browser-observation',
  )
  if (contractOnly.length > 0) {
    throw new Error(
      `Browser qualification requires browser-observation cases: ${contractOnly.map((item) => item.id).join(', ')}`,
    )
  }
  const launcher = options.launcher ?? createPlaywrightLauncher()
  return launcher(options.launch, async (capabilities, runtime, fetchAsset) => {
    const contexts: QualificationContext[] = cases
      .flatMap((qualificationCase) => qualificationCase.environments)
      .filter((environment, index, environments) =>
        environments.findIndex((candidate) =>
          candidate.requestedBackend === environment.requestedBackend,
        ) === index,
      )
      .map((environment) => createContext(
        environment.requestedBackend,
        capabilities,
        runtime,
        fetchAsset,
      ))

    return runQualificationMatrix(cases, options.selection, contexts, options)
  })
}

function createContext(
  requestedBackend: QualificationBackend,
  capabilities: BrowserCapabilities,
  browserRuntime: QualificationContext['runtime'],
  browserFetch: QualificationContext['fetchAsset'],
): QualificationContext {
  const environment: QualificationEnvironment = captureBrowserEnvironment(
    requestedBackend,
    capabilities,
  )
  return {
    requestedBackend,
    environment,
    fetchAsset: browserFetch,
    runtime: browserRuntime,
  }
}

function createPlaywrightLauncher(): BrowserLauncher {
  return async (options, run) => {
    const playwright = await import('playwright')
    const server = await createQualificationServer()
  const probeAsset = join(process.cwd(), ABORT_PROBE_PATH)
  const probeServer = await createProbeAssetServer(probeAsset)
    const browserType = playwright[options.browserName]
    const browser = await browserType.launch({ headless: options.headless })
    const context = await browser.newContext()
    try {
      const page = await context.newPage()
      const baseUrl = server.resolvedUrls?.local[0]
      if (!baseUrl) throw new Error('Qualification server did not expose a local URL')
      await page.goto(`${baseUrl}tests/runtime-qualification/browser-entry.html`)
      const capabilities: BrowserCapabilities = {
        browser: options.browserName,
        browserVersion: browser.version(),
        webgpuAvailable: await page.evaluate(async () => {
          const gpu = (navigator as Navigator & {
            gpu?: { requestAdapter(): Promise<unknown> }
          }).gpu
          if (!gpu) return false
          try {
            return Boolean(await gpu.requestAdapter())
          } catch {
            return false
          }
        }),
      }
      const createCompiledModel = (result: Awaited<ReturnType<BrowserRuntimeApi['loadAndCompile']>>) => ({
        getInputDetails: () => result.inputs,
        getOutputDetails: () => result.outputs,
        getSignatureDetails: async (signature: string) => {
          const details = await page.evaluate(
            ({ id, signature }) => window.litertQualification.getSignatureDetails(id, signature),
            { id: result.id, signature },
          )
          return details
        },
        getModelDetails: () => page.evaluate(
          (id) => window.litertQualification.getModelDetails(id),
          result.id,
        ),
        runSignatureWithZeros: (signature: string) => page.evaluate(
          ({ id, signature }) => window.litertQualification.runSignatureWithZeros(id, signature),
          { id: result.id, signature },
        ),
        runWithZeros: () => page.evaluate(
          (id) => window.litertQualification.runWithZeros(id),
          result.id,
        ),
        run: async (
          input: QualificationTensorInput,
          signature?: string,
        ) => {
          const request = serializeQualificationInput({ input, signature })
          const output = await page.evaluate(
            ({ id, request }) => window.litertQualification.run(id, request),
            { id: result.id, request },
          )
          const deserialize = (tensor: {
            data: number[]
            shape: number[]
            dtype: 'float32' | 'int32' | 'uint8'
          }): QualificationTensor => ({
            data: createQualificationTypedArray(tensor.dtype, tensor.data),
            shape: tensor.shape,
            dtype: tensor.dtype,
          })
          return output.kind === 'positional'
            ? output.tensors.map(deserialize)
            : Object.fromEntries(
              Object.entries(output.tensors).map(([name, tensor]) => [name, deserialize(tensor)]),
            )
        },
        delete: async () => {
          await page.evaluate((id) => window.litertQualification.delete(id), result.id)
        },
      })
      const runtime = {
        initialize: () => page.evaluate((path) => window.litertQualification.initialize(path), '/node_modules/@litertjs/core/wasm'),
        runModuleWorkerLoader: () => page.evaluate(() => window.litertQualification.runModuleWorkerLoader()),
        probeAbortStopsTransfer: () => page.evaluate((origin) => window.litertQualification.probeAbortStopsTransfer(origin), probeServer.origin),
        loadAndCompile: async (model: Uint8Array, compileOptions: { accelerator: QualificationBackend }) => {
          const result = await page.evaluate(async ({ model, accelerator }) => window.litertQualification.loadAndCompile(model, accelerator), {
              model: Array.from(model),
            accelerator: compileOptions.accelerator,
          })
          return createCompiledModel(result)
        },
        loadAndCompileAsset: async (asset: ModelAssetDescriptor, compileOptions: { accelerator: QualificationBackend }) => {
          const result = await page.evaluate(
            ({ asset, accelerator }) => window.litertQualification.loadAndCompileAsset(asset, accelerator),
            { asset, accelerator: compileOptions.accelerator },
          )
          return createCompiledModel({ id: result, inputs: [], outputs: [] })
        },
        runQwenGenerator: (request: QwenGeneratorRequest) => page.evaluate(
          (value) => window.litertQualification.runQwenGenerator(value),
          request,
        ),
      }
      const fetchAsset = async (asset: ModelAssetDescriptor): Promise<ArrayBuffer> => {
        const bytes = await page.evaluate(
          (descriptor) => window.litertQualification.fetchAsset(descriptor),
          asset,
        )
        return new Uint8Array(bytes).buffer
      }
      return await run(capabilities, runtime, fetchAsset)
    } finally {
      await context.close()
      await browser.close()
      await server.close()
      await probeServer.close()
    }
  }
}

async function createQualificationServer(): Promise<ViteDevServer> {
  const server = await createServer({ root: process.cwd(), server: { port: 0 } })
  await server.listen()
  return server
}

// A separate origin for the abort probe, on purpose: the real model base is a different
// origin from the app (Hugging Face), so cancellation has to hold cross-origin. Served
// from its own port rather than the Vite dev server, which does not expose arbitrary
// files from the project root.
function createProbeAssetServer(file: string): Promise<{ origin: string; close(): Promise<void> }> {
  const server = createNodeServer((req, res) => {
    if (!req.url?.startsWith('/asset')) {
      res.statusCode = 404
      res.end()
      return
    }
    // Cross-origin on purpose, matching a real model base, so the browser applies CORS
    // exactly as it would against Hugging Face.
    res.setHeader('Access-Control-Allow-Origin', '*')
    res.setHeader('Content-Type', 'application/octet-stream')
    res.setHeader('Content-Length', String(statSync(file).size))
    createReadStream(file).pipe(res)
  })
  return new Promise((resolvePromise) => {
    server.listen(0, '127.0.0.1', () => {
      const address = server.address()
      const port = typeof address === 'object' && address ? address.port : 0
      resolvePromise({
        origin: `http://127.0.0.1:${port}`,
        close: () => new Promise<void>((done) => server.close(() => done())),
      })
    })
  })
}
