import { readFile } from 'node:fs/promises'
import { join } from 'node:path'
import { efficientDetDynamicOutputCase } from './efficientdet-dynamic-output/case'
import { moduleWorkerLoaderCase } from './module-worker-loader/case'
import { qwenOmniMtpStandaloneCase } from './qwen-omni-mtp-standalone/case'
import { qwenBrowserMemoryGeneratorCase } from './qwen-browsermemory-generator/case'
import { runBrowserQualification } from './shared/browserHarness'
import type { QualificationBackend, QualificationResult, QualificationSelection } from './schema/types'
import { tinyLitertBaselineCase } from './tiny-litert-baseline/case'
import { pipelineLoadCancellationCase } from './pipeline-load-cancellation/case'
import { ensureAbortProbeAsset } from './pipeline-load-cancellation/probeAsset'

export interface QualificationCliOptions {
  caseIds?: string[]
  backends?: QualificationBackend[]
  browserName: 'chromium' | 'firefox' | 'webkit'
  headed: boolean
}

ensureAbortProbeAsset()

const cases = [
  tinyLitertBaselineCase,
  pipelineLoadCancellationCase,
  efficientDetDynamicOutputCase,
  qwenOmniMtpStandaloneCase,
  qwenBrowserMemoryGeneratorCase,
  moduleWorkerLoaderCase,
]

export function parseQualificationArgs(args: string[]): QualificationCliOptions {
  if (args[0] === '--') args = args.slice(1)
  if (args[0] === '--help') {
    console.log('Usage: pnpm qualify -- [--case ID] [--backend wasm|webgpu] [--browser chromium|firefox|webkit] [--headed]')
    return { browserName: 'chromium', headed: false }
  }
  const options: QualificationCliOptions = {
    browserName: 'chromium',
    headed: false,
  }
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index]
    if (argument === '--case') options.caseIds = [...(options.caseIds ?? []), requireValue(args, ++index, argument)]
    else if (argument === '--backend') {
      const backend = requireValue(args, ++index, argument) as QualificationBackend
      if (backend !== 'wasm' && backend !== 'webgpu') throw new Error(`Invalid backend: ${backend}`)
      options.backends = [...(options.backends ?? []), backend]
    } else if (argument === '--browser') {
      const browser = requireValue(args, ++index, argument) as QualificationCliOptions['browserName']
      if (!['chromium', 'firefox', 'webkit'].includes(browser)) throw new Error(`Invalid browser: ${browser}`)
      options.browserName = browser
    } else if (argument === '--headed') options.headed = true
    else throw new Error(`Unknown argument: ${argument}`)
  }
  return options
}

export function formatQualificationMatrix(results: QualificationResult[]): string {
  return results.map((result) => [
    result.caseId,
    result.environment.requestedBackend,
    result.observed.status,
    result.matchesExpectation ? 'match' : 'mismatch',
  ].join('\t')).join('\n')
}

async function main(): Promise<void> {
  const args = process.argv.slice(2)
  if (args.includes('--help')) return
  const options = parseQualificationArgs(args)
  const selection: QualificationSelection = {
    caseIds: options.caseIds,
    backends: options.backends,
  }
  const packageJson = JSON.parse(await readFile(join(process.cwd(), 'package.json'), 'utf8')) as {
    dependencies?: Record<string, string>
  }
  const browserCases = options.caseIds
    ? cases
    : cases.filter((qualificationCase) => qualificationCase.evidenceKind === 'browser-observation')
  const results = await runBrowserQualification(browserCases, {
    launch: { browserName: options.browserName, headless: !options.headed },
    selection,
    playgroundRevision: process.env.GITHUB_SHA ?? 'working-tree',
    runtimePackage: '@litertjs/core',
    runtimeVersion: packageJson.dependencies?.['@litertjs/core'] ?? '2.5.3',
    resultsDirectory: join(process.cwd(), 'tests/runtime-qualification/results'),
  })
  console.log(formatQualificationMatrix(results))
  if (results.some((result) => !result.matchesExpectation)) process.exitCode = 1
}

function requireValue(args: string[], index: number, argument: string): string {
  const value = args[index]
  if (!value || value.startsWith('--')) throw new Error(`Missing value for ${argument}`)
  return value
}

if (import.meta.url === `file://${process.argv[1]}`) await main()
