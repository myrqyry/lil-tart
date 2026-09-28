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
import { createQualificationTypedArray, serializeQualificationInput } from './tensorBridge'
import { ensureAbortProbeAsset } from '../pipeline-load-cancellation/probeAsset'
import { createProbeAssetServer } from './probeAssetServer'
import type { Browser, BrowserContext } from 'playwright'

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
  // Set when a selected case declares it needs the generated probe asset origin, so an
  // unrelated run never binds that port or opens that file.
  requiresProbeAsset?: boolean
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
  const launch: BrowserLaunchOptions = {
    ...options.launch,
    requiresProbeAsset: selectedCases.some((item) => item.requiresProbeAsset),
  }
  return launcher(launch, async (capabilities, runtime, fetchAsset) => {
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
    let server: ViteDevServer | undefined
    let probeServer: Awaited<ReturnType<typeof createProbeAssetServer>> | undefined
    let browser: Browser | undefined
    let context: BrowserContext | undefined
    let failed = false
    try {
      server = await createServer({ root: process.cwd(), server: { port: 0 } })
      await server.listen()
      browser = await playwright[options.browserName].launch({ headless: options.headless })
      context = await browser.newContext()
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
        probeAbortStopsTransfer: async () => {
          if (!options.requiresProbeAsset) throw new Error('No selected case declared a probe asset')
          // No fixture or second listener until the selected case actually probes.
          probeServer ??= await createProbeAssetServer(ensureAbortProbeAsset())
          return page.evaluate((origin) => window.litertQualification.probeAbortStopsTransfer(origin), probeServer.origin)
        },
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
    } catch (error) {
      failed = true
      throw error
    } finally {
      // Attempt every close, including resources acquired before a startup failure.
      // Preserve the original error if the run itself failed.
      const errors: unknown[] = []
      for (const resource of [context, browser, probeServer, server]) {
        try { await resource?.close() } catch (error) { errors.push(error) }
      }
      if (!failed && errors.length > 0) throw errors[0]
    }
  }
}
