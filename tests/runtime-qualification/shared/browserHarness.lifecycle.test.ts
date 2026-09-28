import { beforeEach, describe, expect, it, vi } from 'vitest'
import { runBrowserQualification } from './browserHarness'
import { pipelineLoadCancellationCase } from '../pipeline-load-cancellation/case'

const doubles = vi.hoisted(() => ({
  createServer: vi.fn(), launch: vi.fn(), ensureAsset: vi.fn(), probeServer: vi.fn(),
}))
vi.mock('vite', () => ({ createServer: doubles.createServer }))
vi.mock('./probeAssetServer', () => ({ createProbeAssetServer: doubles.probeServer }))
vi.mock('playwright', () => ({ chromium: { launch: doubles.launch } }))
vi.mock('../pipeline-load-cancellation/probeAsset', () => ({ ensureAbortProbeAsset: doubles.ensureAsset }))

const options = {
  launch: { browserName: 'chromium' as const, headless: true },
  selection: {}, playgroundRevision: 'test', runtimePackage: '@litertjs/core', runtimeVersion: '2.5.3',
}

beforeEach(() => vi.resetAllMocks())

describe('browser startup cleanup', () => {
  it('closes Vite after browser launch fails without preparing an unused probe', async () => {
    const close = vi.fn().mockResolvedValue(undefined)
    doubles.createServer.mockResolvedValue({ listen: vi.fn(), close })
    doubles.launch.mockRejectedValue(new Error('launch failed'))
    await expect(runBrowserQualification([pipelineLoadCancellationCase], options)).rejects.toThrow('launch failed')
    expect(close).toHaveBeenCalledTimes(1)
    expect(doubles.ensureAsset).not.toHaveBeenCalled()
  })

  it('closes Vite even when listen fails', async () => {
    const close = vi.fn().mockResolvedValue(undefined)
    doubles.createServer.mockResolvedValue({ listen: vi.fn().mockRejectedValue(new Error('listen failed')), close })
    await expect(runBrowserQualification([pipelineLoadCancellationCase], options)).rejects.toThrow('listen failed')
    expect(close).toHaveBeenCalledTimes(1)
    expect(doubles.launch).not.toHaveBeenCalled()
  })

  it('closes the browser and server when context creation fails', async () => {
    const closeServer = vi.fn().mockResolvedValue(undefined)
    const closeBrowser = vi.fn().mockResolvedValue(undefined)
    doubles.createServer.mockResolvedValue({ listen: vi.fn(), close: closeServer })
    doubles.launch.mockResolvedValue({ close: closeBrowser, newContext: vi.fn().mockRejectedValue(new Error('context failed')) })
    await expect(runBrowserQualification([pipelineLoadCancellationCase], options)).rejects.toThrow('context failed')
    expect(closeBrowser).toHaveBeenCalledTimes(1)
    expect(closeServer).toHaveBeenCalledTimes(1)
  })
})

it('importing the CLI and parsing help never generates a fixture', async () => {
  const log = vi.spyOn(console, 'log').mockImplementation(() => {})
  try {
    const { parseQualificationArgs } = await import('../run-qualification')
    parseQualificationArgs(['--help'])
    expect(doubles.ensureAsset).not.toHaveBeenCalled()
    expect(log).toHaveBeenCalledWith(expect.stringContaining('Usage:'))
  } finally {
    log.mockRestore()
  }
})


function runningBrowser() {
  const page = { goto: vi.fn(), evaluate: vi.fn().mockResolvedValue(false) }
  const context = { newPage: vi.fn().mockResolvedValue(page), close: vi.fn().mockResolvedValue(undefined) }
  const browser = { newContext: vi.fn().mockResolvedValue(context), version: () => 'test', close: vi.fn().mockResolvedValue(undefined) }
  const server = { listen: vi.fn(), close: vi.fn().mockResolvedValue(undefined), resolvedUrls: { local: ['http://test/'] } }
  doubles.createServer.mockResolvedValue(server)
  doubles.launch.mockResolvedValue(browser)
  return { page, context, browser, server }
}

it('does not prepare the cancellation fixture when another case is selected', async () => {
  runningBrowser()
  const otherCase = { ...pipelineLoadCancellationCase, id: 'other', run: async () => ({ status: 'pass' as const }) }
  const results = await runBrowserQualification([pipelineLoadCancellationCase, otherCase], {
    ...options, selection: { caseIds: ['other'] },
  })
  expect(results).toHaveLength(1)
  expect(results[0].matchesExpectation).toBe(true)
  expect(doubles.ensureAsset).not.toHaveBeenCalled()
  expect(doubles.probeServer).not.toHaveBeenCalled()
})

it('reports missing fixture setup as a failed case and closes the browser resources', async () => {
  const { context, browser, server } = runningBrowser()
  doubles.ensureAsset.mockReturnValue('/missing')
  doubles.probeServer.mockRejectedValue(new Error('missing fixture'))
  const results = await runBrowserQualification([pipelineLoadCancellationCase], options)
  expect(results[0].observed).toMatchObject({ status: 'fail', error: { message: 'missing fixture' } })
  expect(context.close).toHaveBeenCalledTimes(1)
  expect(browser.close).toHaveBeenCalledTimes(1)
  expect(server.close).toHaveBeenCalledTimes(1)
})

it('closes every acquired resource even if context cleanup rejects', async () => {
  const { context, browser, server, page } = runningBrowser()
  const closeProbe = vi.fn().mockResolvedValue(undefined)
  doubles.ensureAsset.mockReturnValue('/fixture')
  doubles.probeServer.mockResolvedValue({ origin: 'http://probe/', close: closeProbe })
  page.evaluate.mockResolvedValueOnce(false).mockResolvedValueOnce({ status: 'pass' })
  context.close.mockRejectedValue(new Error('context close failed'))
  await expect(runBrowserQualification([pipelineLoadCancellationCase], options)).rejects.toThrow('context close failed')
  expect(browser.close).toHaveBeenCalledTimes(1)
  expect(closeProbe).toHaveBeenCalledTimes(1)
  expect(server.close).toHaveBeenCalledTimes(1)
})
