import { describe, expect, it, vi } from 'vitest'
import {
  EncoderPipeline,
  encoder230mManifest,
  encoderPolicyLinterManifest,
  encoderSpellcheckerManifest,
  meanPool,
  selectEncoderManifest,
} from './index'

vi.mock('@huggingface/transformers', () => ({
  AutoTokenizer: {
    from_pretrained: vi.fn(async () => ({
      encode: vi.fn(async () => ({
        input_ids: { data: new Int32Array([1, 1, 1, 1]), dims: [1, 4] },
      })),
    })),
  },
}))

const context = {
  backend: 'wasm' as const,
  assets: { resolve: vi.fn() },
  signal: undefined,
  liteRt: {
    loadModel: vi.fn(async () => fakeModel),
    loadNpy: vi.fn(),
    fetchBuffer: vi.fn(),
  },
}

const fakeModel = {
  run: vi.fn(async () => [{ data: async () => new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]) }]),
}

describe('selectEncoderManifest', () => {
  it('maps capability to the right manifest', () => {
    expect(selectEncoderManifest('text-embedding')).toBe(encoder230mManifest)
    expect(selectEncoderManifest('token-classification')).toBe(encoderSpellcheckerManifest)
    expect(selectEncoderManifest('policy-classification')).toBe(encoderPolicyLinterManifest)
  })
})

describe('EncoderPipeline', () => {
  it('declares the embedding capability', () => {
    expect(encoder230mManifest.capabilities).toContain('text-embedding')
  })

  it('loads model and tokenizer from manifest paths', async () => {
    const pipeline = new EncoderPipeline({ manifest: encoder230mManifest })
    await pipeline.load(context)
    expect(context.liteRt.loadModel).toHaveBeenCalledWith(
      'litert-community/LFM2.5-Encoder-230M/resolve/main/LFM2.5-Encoder-230M_fp16.tflite',
      { signal: expect.any(AbortSignal) },
    )
    expect(pipeline.status).toBe('ready')
  })

  it('mean-pools embeddings over tokens', async () => {
    const pipeline = new EncoderPipeline({ manifest: encoder230mManifest })
    await pipeline.load(context)
    const result = await pipeline.run({ text: 'hello' }, { maxTokens: 4 })
    expect(result).toEqual({ kind: 'embedding', values: meanPool(new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]), 4, 2), dimensions: 2 })
  })

  it('returns raw token scores for token-classification manifests', async () => {
    const pipeline = new EncoderPipeline({ manifest: encoderSpellcheckerManifest })
    await pipeline.load(context)
    const result = await pipeline.run({ text: 'helo' }, { maxTokens: 4 })
    expect(result.kind).toBe('token-classification')
    if (result.kind === 'token-classification') {
      expect(result.tokens).toBe(4)
      expect(result.dimensions).toBe(2)
    }
  })
})

describe('meanPool', () => {
  it('averages rows', () => {
    const pooled = meanPool(new Float32Array([2, 4, 6, 8]), 2, 2)
    expect(Array.from(pooled)).toEqual([4, 6])
  })

  it('returns zeros for empty input', () => {
    expect(Array.from(meanPool(new Float32Array(0), 2, 3))).toEqual([0, 0, 0])
  })
})

async function waitFor(predicate: () => boolean, label: string): Promise<void> {
  for (let i = 0; i < 100; i++) {
    if (predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  throw new Error(`timed out waiting for ${label}`)
}

describe('EncoderPipeline cancellation', () => {
  function hangingContext() {
    const seen = { signal: null as AbortSignal | null }
    return {
      seen,
      context: {
        ...context,
        liteRt: {
          ...context.liteRt,
          loadModel: vi.fn((_path: string, options?: { signal?: AbortSignal }) => {
            seen.signal = options?.signal ?? null
            return new Promise((_resolve, reject) => {
              const fail = () => reject(new Error('CANCELLED'))
              if (options?.signal?.aborted) fail()
              else options?.signal?.addEventListener('abort', fail)
            })
          }),
        },
      },
    }
  }

  it('aborts an in-flight model load when disposed', async () => {
    const { seen, context: ctx } = hangingContext()
    const pipeline = new EncoderPipeline({ manifest: encoder230mManifest })
    const loading = pipeline.load(ctx as never)
    // Time the dispose so it lands while the model load is genuinely in flight.
    await waitFor(() => seen.signal !== null, 'loadModel to start')
    await pipeline.dispose()

    await expect(loading).rejects.toThrow()
    expect(seen.signal?.aborted).toBe(true)
    // A cancelled load must not stamp 'idle' over an already disposed pipeline.
    expect(pipeline.status).toBe('disposed')
  })

  it("follows the caller's signal and stays retryable", async () => {
    const { seen, context: ctx } = hangingContext()
    const controller = new AbortController()
    const pipeline = new EncoderPipeline({ manifest: encoder230mManifest })
    const loading = pipeline.load({ ...ctx, signal: controller.signal } as never)
    controller.abort()

    await expect(loading).rejects.toThrow()
    expect(seen.signal?.aborted).toBe(true)
    // Cancelled is not latched to 'error', so a retry is possible.
    expect(pipeline.status).toBe('idle')
  })

  it('refuses to load through an already-aborted signal', async () => {
    const { seen, context: ctx } = hangingContext()
    const controller = new AbortController()
    controller.abort()
    const pipeline = new EncoderPipeline({ manifest: encoder230mManifest })

    await expect(
      pipeline.load({ ...ctx, signal: controller.signal } as never),
    ).rejects.toThrow()
    // addEventListener on a settled signal never fires, so this only passes if the
    // already-aborted case is handled explicitly.
    expect(seen.signal?.aborted).toBe(true)
  })

  it('does not resurrect a disposed pipeline when the model resolves late', async () => {
    const slow = fakeModel
    let release = () => {}
    const pending = new Promise<typeof slow>((resolve) => {
      release = () => resolve(slow)
    })
    let seenLoad = false
    const ctx = {
      ...context,
      liteRt: {
        ...context.liteRt,
        loadModel: vi.fn(() => {
          seenLoad = true;
          return pending;
        }),
      },
    }
    const pipeline = new EncoderPipeline({ manifest: encoder230mManifest })
    const loading = pipeline.load(ctx as never)
    await waitFor(() => seenLoad, 'loadModel to start')
    await pipeline.dispose()
    release()
    await expect(loading).rejects.toThrow(/disposed/i)

    expect(pipeline.status).toBe('disposed')
  })
})
