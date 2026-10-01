import { CompiledModel, Tensor } from '@litertjs/core'

const NUM_CODE_GROUPS = 16
const UPSAMPLE = 1920
export const QWEN3_TTS_CODEC_WINDOW_FRAMES = 64
export const QWEN3_TTS_CODEC_LEFT_CONTEXT_FRAMES = 25

export interface CodecConfig {
  chunkSize?: number
  overlapLeft?: number
  accelerator?: 'wasm' | 'webgpu'
}

async function toInputTensor(data: Float32Array | Int32Array, shape: number[], accelerator: 'wasm' | 'webgpu'): Promise<Tensor> {
  return new Tensor(data, shape).moveTo(accelerator)
}

function readFloat32(tensor: Tensor): Float32Array {
  const arr = tensor.toTypedArray()
  return arr instanceof Float32Array ? arr : Float32Array.from(arr)
}

export class CodecDecoder {
  private chunkSize: number
  private overlapLeft: number
  private accelerator: 'wasm' | 'webgpu'

  constructor(
    private model: CompiledModel,
    config?: CodecConfig,
  ) {
    this.chunkSize = config?.chunkSize ?? QWEN3_TTS_CODEC_WINDOW_FRAMES
    this.overlapLeft = config?.overlapLeft ?? QWEN3_TTS_CODEC_LEFT_CONTEXT_FRAMES
    this.accelerator = config?.accelerator ?? 'wasm'
  }

  async decode(frameCodes: number[][]): Promise<Float32Array> {
    const pieces: Float32Array[] = []
    const codes = frameCodes
    const chunk = this.chunkSize
    const ctx = this.overlapLeft

    let i = 0
    while (i < codes.length) {
      const c = Math.min(ctx, i)
      const j = Math.min(i + chunk - c, codes.length)
      const windowLen = j - (i - c)

      const buf = new Int32Array(NUM_CODE_GROUPS * chunk).fill(0)
      for (let g = 0; g < NUM_CODE_GROUPS; g++) {
        for (let f = 0; f < windowLen; f++) {
          buf[g * chunk + f] = codes[i - c + f][g]
        }
      }

      const inputs: Record<string, Promise<Tensor>> = {
        'args_0': toInputTensor(buf, [1, NUM_CODE_GROUPS, chunk], this.accelerator),
      }

      const resolved: Record<string, Tensor> = {}
      for (const [key, promise] of Object.entries(inputs)) resolved[key] = await promise
      const result = await this.model.run(resolved)
      const outKey = Object.keys(result)[0]
      const wav = readFloat32(result[outKey] as Tensor)

      const validFrames = windowLen - c
      if (validFrames > 0) {
        const ctxSamples = c * UPSAMPLE
        const validSamples = validFrames * UPSAMPLE
        pieces.push(wav.slice(ctxSamples, ctxSamples + validSamples))
      }

      i = j
    }

    if (pieces.length === 0) return new Float32Array(0)
    const totalLen = pieces.reduce((s, p) => s + p.length, 0)
    const out = new Float32Array(totalLen)
    let off = 0
    for (const p of pieces) { out.set(p, off); off += p.length }
    return out
  }
}
