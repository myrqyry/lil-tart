import { mkdtemp, rm, truncate, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createProbeAssetServer } from '../shared/probeAssetServer'
import { probeAbortStopsTransfer } from './probeTransfer'
import * as httpResolver from '../../../packages/inference-core/src/assets/http-resolver'
import { ABORT_PROBE_BYTES } from './probeAsset.meta'

let directory: string | undefined
let server: Awaited<ReturnType<typeof createProbeAssetServer>> | undefined

afterEach(async () => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  await server?.close()
  server = undefined
  if (directory) await rm(directory, { recursive: true, force: true })
  directory = undefined
})

async function fixture() {
  directory = await mkdtemp(join(tmpdir(), 'lil-tart-abort-'))
  const file = join(directory, 'asset.bin')
  await writeFile(file, new Uint8Array(1))
  await truncate(file, ABORT_PROBE_BYTES)
  server = await createProbeAssetServer(file)
  return { file, server }
}

describe('buffered resolver cancellation over HTTP', () => {
  it('aborts resolve() after receiving real bytes', async () => {
    const { server } = await fixture()
    const makeResolver = httpResolver.createHttpAssetResolver
    const resolver = makeResolver(server.origin)
    const resolve = vi.fn(resolver.resolve)
    const stream = vi.fn().mockRejectedValue(new Error('stream() is not the production path'))
    vi.spyOn(httpResolver, 'createHttpAssetResolver').mockReturnValue({ resolve, stream })
    expect(await probeAbortStopsTransfer(server.origin)).toEqual({ status: 'pass' })
    expect(resolve).toHaveBeenCalledTimes(1)
    expect(stream).not.toHaveBeenCalled()
  })

  it('fails if the fetch boundary loses the abort signal', async () => {
    const { server } = await fixture()
    const nativeFetch = globalThis.fetch
    vi.stubGlobal('fetch', (input: RequestInfo | URL) => nativeFetch(input))
    expect(await probeAbortStopsTransfer(server.origin)).toMatchObject({
      status: 'fail', error: { message: expect.stringContaining(`${ABORT_PROBE_BYTES} of ${ABORT_PROBE_BYTES} bytes`) },
    })
  })

  it('rejects a missing fixture before starting a listener', async () => {
    directory = await mkdtemp(join(tmpdir(), 'lil-tart-missing-'))
    await expect(createProbeAssetServer(join(directory, 'missing.bin'))).rejects.toMatchObject({ code: 'ENOENT' })
  })

  it('returns HTTP 500 if the file disappears after server creation', async () => {
    const { file, server } = await fixture()
    await rm(file)
    const response = await fetch(`${server.origin}/asset`)
    expect(response.status).toBe(500)
    expect(response.headers.get('access-control-allow-origin')).toBe('*')
    expect(await response.text()).toBe('Probe asset unavailable')
  })
})
