import { describe, expect, it, vi } from 'vitest'
import type { MultiVectorEmbeddingResult } from '@litert-playground/inference-core'
import { cosineSimilarity, dot, maxSim, rankColBert, rankDense } from './scoring'
import { ColBertPipeline } from './colbert'
import { colbertManifest } from './manifest'

vi.mock('@huggingface/transformers', () => ({
  AutoTokenizer: {
    from_pretrained: vi.fn(async () => ({
      encode: vi.fn(async (_text: string, opts: Record<string, unknown>) => ({
        input_ids: { data: new Int32Array(opts.max_length as number).fill(1), dims: [1, opts.max_length] },
      })),
    })),
  },
}))

function mv(rows: number[][]): MultiVectorEmbeddingResult {
  return {
    kind: 'multi-vector-embedding',
    values: new Float32Array(rows.flat()),
    tokens: rows.length,
    dimensions: rows[0]?.length ?? 0,
  }
}

describe('scoring', () => {
  it('dot aligns rows by token offset', () => {
    const a = new Float32Array([1, 2, 3, 4])
    expect(dot(a, a, 0, 0, 2)).toBe(5)
    expect(dot(a, a, 2, 2, 2)).toBe(25)
  })

  it('cosineSimilarity normalizes', () => {
    const a = new Float32Array([1, 0])
    const b = new Float32Array([0, 1])
    expect(cosineSimilarity(a, b)).toBeCloseTo(0)
    expect(cosineSimilarity(a, a)).toBeCloseTo(1)
  })

  it('maxSim sums per-query-token best doc match', () => {
    const q = mv([[1, 0]])
    const d = mv([[0.9, 0], [0, 1]])
    expect(maxSim(q, d)).toBeCloseTo(0.9)
  })

  it('rankColBert sorts descending with optional topK', () => {
    const q = mv([[1, 0]])
    const docs = [
      { id: 'a', embedding: mv([[0.3, 0]]) },
      { id: 'b', embedding: mv([[0.9, 0]]) },
    ]
    const all = rankColBert(q, docs)
    expect(all.map((r) => r.id)).toEqual(['b', 'a'])
    expect(rankColBert(q, docs, 1)).toHaveLength(1)
  })

  it('rankDense sorts by cosine', () => {
    const q = { kind: 'embedding' as const, values: new Float32Array([1, 0]), dimensions: 2 }
    const docs = [
      { id: 'far', embedding: { kind: 'embedding' as const, values: new Float32Array([0, 1]), dimensions: 2 } },
      { id: 'near', embedding: { kind: 'embedding' as const, values: new Float32Array([0.8, 0.2]), dimensions: 2 } },
    ]
    expect(rankDense(q, docs).map((r) => r.id)).toEqual(['near', 'far'])
  })
})

describe('ColBertPipeline', () => {
  const fakeModel = {
    run: vi.fn(async () => [
      { data: async () => new Float32Array([1, 2, 3, 4, 5, 6]) },
    ]),
  }

  const context = {
    backend: 'wasm' as const,
    assets: { resolve: vi.fn() },
    signal: undefined,
    liteRt: { loadModel: vi.fn(async () => fakeModel) },
  }

  it('reports manifest capabilities and loads model from manifest asset path', async () => {
    expect(colbertManifest.capabilities).toContain('multi-vector-embedding')
    expect(colbertManifest.capabilities).toContain('reranking')

    const pipeline = new ColBertPipeline()
    await pipeline.load(context as never)
    expect(context.liteRt.loadModel).toHaveBeenCalledWith(
      colbertManifest.assets[0].path,
      { signal: expect.any(AbortSignal) },
    )
  })

  it('runs tokenization through the graph into a multi-vector embedding', async () => {
    const pipeline = new ColBertPipeline()
    await pipeline.load(context as never)
    const result = await pipeline.run({ text: 'hello' }, { maxTokens: 2 })
    expect(result.kind).toBe('multi-vector-embedding')
    expect(result.tokens).toBe(2)
    expect(result.dimensions).toBe(3)
    expect(Array.from(result.values)).toEqual([1, 2, 3, 4, 5, 6])
  })
})

async function waitFor(predicate: () => boolean, label: string): Promise<void> {
  for (let i = 0; i < 100; i++) {
    if (predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  throw new Error(`timed out waiting for ${label}`)
}

describe('ColBertPipeline cancellation', () => {
  function hangingContext() {
    const seen = { signal: null as AbortSignal | null }
    return {
      seen,
      context: {
        backend: 'wasm' as const,
        assets: { resolve: vi.fn() },
        signal: undefined,
        liteRt: {
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
    const pipeline = new ColBertPipeline()
    const loading = pipeline.load(ctx as never)
    // Time the dispose so it lands while the model load is genuinely in flight.
    await waitFor(() => seen.signal !== null, 'loadModel to start')
    await pipeline.dispose()

    await expect(loading).rejects.toThrow()
    expect(seen.signal?.aborted).toBe(true)
    expect(pipeline.status).toBe('disposed')
  })

  it("follows the caller's signal and stays retryable", async () => {
    const { seen, context: ctx } = hangingContext()
    const controller = new AbortController()
    const pipeline = new ColBertPipeline()
    const loading = pipeline.load({ ...ctx, signal: controller.signal } as never)
    // Abort once the model load is genuinely in flight, not before it starts.
    await waitFor(() => seen.signal !== null, 'loadModel to start')
    controller.abort()

    await expect(loading).rejects.toThrow()
    expect(seen.signal?.aborted).toBe(true)
    expect(pipeline.status).toBe('idle')
  })

  it('refuses to load through an already-aborted signal', async () => {
    const { seen, context: ctx } = hangingContext()
    const controller = new AbortController()
    controller.abort()
    const pipeline = new ColBertPipeline()

    await expect(pipeline.load({ ...ctx, signal: controller.signal } as never)).rejects.toThrow()
    // addEventListener on a settled signal never fires, so this only passes if the
    // already-aborted case is handled explicitly. The model load is then never started.
    expect(seen.signal).toBeNull()
  })

  it('does not resurrect a disposed pipeline when the model resolves late', async () => {
    const model = { run: vi.fn() }
    let release = () => {}
    const pending = new Promise<typeof model>((resolve) => {
      release = () => resolve(model)
    })
    const { context: base } = hangingContext()
    let seenLoad = false
    const pipeline = new ColBertPipeline()
    const loading = pipeline.load({
      ...base,
      liteRt: {
        ...base.liteRt,
        loadModel: vi.fn(() => {
          seenLoad = true;
          return pending;
        }),
      },
    } as never)
    await waitFor(() => seenLoad, 'loadModel to start')
    await pipeline.dispose()
    release()
    await expect(loading).rejects.toThrow(/cancelled/i)

    expect(pipeline.status).toBe('disposed')
  })
})
