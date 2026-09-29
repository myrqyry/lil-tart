import { CompiledModel, Tensor } from '@litertjs/core'
import { sample, SampleOpts } from './sampler'
import { traceArray, traceTensor, type GeneratorTraceEvent } from './generator-trace'

const MTP_CACHE_SLOTS = 17
export const QWEN3_TTS_MTP_PASSES_PER_FRAME = 16
const MTP_CODEBOOKS = QWEN3_TTS_MTP_PASSES_PER_FRAME - 1
const HIDDEN = 1024
const CODEC_VOCAB = 3072
const NEG_INF = -1e9

export interface MTPConfig {
  mtpEmbeddings: Float32Array
  codecEmbeddings: Float32Array
  numCacheSlots?: number
  numCodebooks?: number
  cacheShape?: number[]
  accelerator?: 'wasm' | 'webgpu'
  onTrace?: (event: GeneratorTraceEvent) => void
}

async function toInputTensor(data: Float32Array | Int32Array, shape: number[], accelerator: 'wasm' | 'webgpu'): Promise<Tensor> {
  return new Tensor(data, shape).moveTo(accelerator)
}

function readFloat32(tensor: Tensor): Float32Array {
  const arr = tensor.toTypedArray()
  return arr instanceof Float32Array ? arr : Float32Array.from(arr)
}

export class MTP {
  private mtpEmb: Float32Array
  private codecEmb: Float32Array
  private numCacheSlots: number
  private numCodebooks: number
  private cacheShape: number[]
  private accelerator: 'wasm' | 'webgpu'
  private onTrace?: (event: GeneratorTraceEvent) => void

  constructor(
    private model: CompiledModel,
    config: MTPConfig,
  ) {
    this.mtpEmb = config.mtpEmbeddings
    this.codecEmb = config.codecEmbeddings
    this.numCacheSlots = config.numCacheSlots ?? MTP_CACHE_SLOTS
    this.numCodebooks = config.numCodebooks ?? MTP_CODEBOOKS
    this.cacheShape = config.cacheShape ?? [1, this.numCacheSlots, HIDDEN]
    this.accelerator = config.accelerator ?? 'wasm'
    this.onTrace = config.onTrace
  }

  async predict(
    hiddenState: Float32Array,
    cb0: number,
    opts?: Partial<SampleOpts>,
  ): Promise<number[]> {
    const sampleOpts: SampleOpts = {
      temperature: opts?.temperature ?? 0,
      topK: opts?.topK ?? 0,
      repetitionPenalty: 1,
      prevTokens: [],
    }

    const cacheSize = this.cacheShape.reduce((a, b) => a * b, 1)
    const kAll = new Float32Array(cacheSize)
    const vAll = new Float32Array(cacheSize)

    const codes: number[] = []

    for (let t = 0; t < this.numCodebooks + 1; t++) {
      let embed: Float32Array
      if (t === 0) {
        embed = hiddenState
      } else if (t === 1) {
        embed = this.codecEmb.slice(cb0 * HIDDEN, (cb0 + 1) * HIDDEN)
      } else {
        const step = t - 2
        const prevCode = codes[codes.length - 1]
        const offset = (step * CODEC_VOCAB + prevCode) * HIDDEN
        embed = this.mtpEmb.slice(offset, offset + HIDDEN)
      }

      const mask = new Float32Array(this.numCacheSlots).fill(NEG_INF)
      for (let i = 0; i <= t; i++) mask[i] = 0

      this.onTrace?.({
        stage: 'mtp-input-build',
        frame: t,
        phase: 'start',
        tensors: [
          traceArray('args_0', 'float32', [1, 1, HIDDEN]),
          traceArray('args_1', 'int32', [1]),
          traceArray('args_2', 'float32', [1, 1, 1, this.numCacheSlots]),
          traceArray('args_3', 'float32', this.cacheShape),
          traceArray('args_4', 'float32', this.cacheShape),
        ],
      })

      const inputs: Record<string, Promise<Tensor>> = {
        'args_0': toInputTensor(embed, [1, 1, HIDDEN], this.accelerator),
        'args_1': toInputTensor(new Int32Array([t]), [1], this.accelerator),
        'args_2': toInputTensor(mask, [1, 1, 1, this.numCacheSlots], this.accelerator),
        'args_3': toInputTensor(kAll, this.cacheShape, this.accelerator),
        'args_4': toInputTensor(vAll, this.cacheShape, this.accelerator),
      }

      const resolved: Record<string, Tensor> = {}
      for (const [key, promise] of Object.entries(inputs)) resolved[key] = await promise
      this.onTrace?.({
        stage: 'mtp-input-build',
        frame: t,
        phase: 'end',
        tensors: Object.entries(resolved).map(([name, tensor]) => traceTensor(name, tensor)),
      })
      const startedAt = performance.now()
      this.onTrace?.({ stage: 'mtp-run', frame: t, phase: 'start' })
      const result = await this.model.run(resolved)
      this.onTrace?.({ stage: 'mtp-run', frame: t, phase: 'end', durationMs: performance.now() - startedAt })

      const kUpd = readFloat32(result['output_1'] as Tensor)
      const vUpd = readFloat32(result['output_2'] as Tensor)
      this.onTrace?.({
        stage: 'mtp-output-read',
        frame: t,
        tensors: [
          traceTensor('output_0', result['output_0'] as Tensor),
          traceTensor('output_1', result['output_1'] as Tensor),
          traceTensor('output_2', result['output_2'] as Tensor),
        ],
      })
      kAll.set(kUpd)
      vAll.set(vUpd)

      if (t >= 1) {
        const logits = readFloat32(result['output_0'] as Tensor)
        const headLogits = logits.slice((t - 1) * CODEC_VOCAB, t * CODEC_VOCAB)
        codes.push(sample(headLogits, sampleOpts))
      }
    }

    return codes
  }
}
