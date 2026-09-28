import { describe, expect, it, vi } from 'vitest'
import type { InferenceContext } from './types'
import { crepeAdapter } from './audio'
import { clipsegAdapter, normalizeU2NetSaliencyPixels } from './batch2'
import { dinov2Adapter } from './dinov2'
import { mimiAdapter } from './mimi'
import { nafnetAdapter } from './nafnet'
import { ppOcrAdapter } from './ocr'
import { sam2DecoderAdapter, sam2EncoderAdapter } from './sam2'
import { matchaTtsAdapter } from './tts'
import { visionAdapters } from './vision'

describe('Hugging Face model contracts', () => {
  it('invokes CREPE through its single positional TFLite input', async () => {
    const activation = new Float32Array(360)
    activation[100] = 1

    const predict = vi.fn(async (_graph, inputs) => ({
      output: { data: async () => activation },
    }))
    const ctx = {
      predict,
      createTensor: (data: Float32Array | Int32Array, shape: number[]) => ({ data, shape }),
    } as unknown as InferenceContext

    await crepeAdapter.run?.({ audio: new Float32Array(1024) }, ctx)

    expect(predict).toHaveBeenCalledTimes(1)
    const inputs = predict.mock.calls[0][1]
    expect(Array.isArray(inputs)).toBe(true)
    expect(inputs).toHaveLength(1)
  })

  it('uses the decoder-ready SAM2 v2 artifacts and real decoder output names', () => {
    expect(sam2EncoderAdapter.metadata.modelPath).toMatch(/sam2_tiny_image_encoder_v2_fp16\.tflite$/)
    expect(sam2DecoderAdapter.metadata.modelPath).toMatch(/sam2_tiny_mask_decoder_v2_fp16\.tflite$/)
    expect(sam2DecoderAdapter.outputSpecs.map((spec) => spec.name)).toEqual(['pred_masks', 'iou_scores'])
    expect(sam2DecoderAdapter.disabled).toBe(true)
    expect(sam2DecoderAdapter.requiredBackend).toBe('wasm')
  })

  it('keeps incomplete CLIPSeg unavailable instead of exposing the old fake pipeline', () => {
    expect(clipsegAdapter.disabled).toBe(true)
    expect(clipsegAdapter.metadata.description).toMatch(/host token-embedding lookup/i)
    expect(clipsegAdapter.graphs?.find((graph) => graph.name === 'vision')?.requiredBackend).toBe('wasm')
  })

  it('pins browser graphs with measured WebGPU output mismatches to WASM', () => {
    expect(ppOcrAdapter.graphs?.find((graph) => graph.name === 'rec')?.requiredBackend).toBe('wasm')
    expect(matchaTtsAdapter.graphs?.find((graph) => graph.name === 'decoder')?.requiredBackend).toBe('wasm')
    expect(matchaTtsAdapter.graphs?.find((graph) => graph.name === 'vocoder')?.requiredBackend).toBe('wasm')
    expect(mimiAdapter.graphs?.find((graph) => graph.name === 'enc_tx')?.requiredBackend).toBe('wasm')
    expect(mimiAdapter.graphs?.find((graph) => graph.name === 'dec_tx')?.requiredBackend).toBe('wasm')
    expect(dinov2Adapter.requiredBackend).toBe('wasm')
    expect(nafnetAdapter.requiredBackend).toBe('wasm')

    const requiredWasm = new Set([
      'migan',
      'style-candy',
      'style-mosaic',
      'style-rain-princess',
      'style-udnie',
      'sinet-v2',
      'pidnet-s-cityscapes',
      'twinlitenet',
      'tipsv2-b14-dpt',
      'gfpgan-v1.4',
      'cloth-segmentation',
      'm-lsd-tiny',
    ])

    for (const adapter of visionAdapters.filter((candidate) => requiredWasm.has(candidate.modelId))) {
      expect(adapter.requiredBackend, adapter.modelId).toBe('wasm')
    }
    expect(visionAdapters.filter((candidate) => requiredWasm.has(candidate.modelId))).toHaveLength(requiredWasm.size)
  })

  it('matches U-2-Net per-image-max preprocessing rather than plain /255 normalization', () => {
    const image = {
      width: 1,
      height: 1,
      data: new Uint8ClampedArray([128, 64, 0, 255]),
    } as ImageData

    const pixels = normalizeU2NetSaliencyPixels(image)

    expect(Array.from(pixels)).toHaveLength(3)
    expect(pixels[0]).toBeCloseTo((1 - 0.485) / 0.229, 5)
    expect(pixels[1]).toBeCloseTo((0.5 - 0.456) / 0.224, 5)
    expect(pixels[2]).toBeCloseTo((0 - 0.406) / 0.225, 5)
  })
})
