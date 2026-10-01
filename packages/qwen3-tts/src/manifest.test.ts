import { describe, expect, it } from 'vitest'
import { qwen3TtsManifest, createQwen3TtsManifest, qwen3TtsVariants } from './manifest'

if (false) {
  void qwen3TtsVariants.fp32
  // @ts-expect-error Unknown variant keys must not be typed as present.
  void qwen3TtsVariants['fp32 ']
}

describe('Qwen variants', () => {
  it('keeps artifact filenames in variant metadata', () => {
    const manifest = createQwen3TtsManifest(qwen3TtsVariants.int4)
    expect(manifest.assets.find((asset) => asset.id === 'talker')?.path).toBe('talker_int4.tflite')
    expect(manifest.name).toContain('int4')
  })

  it('uses the qualified Omni MTP integrity facts even though its filename matches base fp32', () => {
    const manifest = createQwen3TtsManifest(qwen3TtsVariants.browserMemoryOmni)
    const mtp = manifest.assets.find((asset) => asset.id === 'mtp')

    expect(mtp).toMatchObject({
      path: 'mtp_fp32.tflite',
      bytes: 440_528_628,
      sha256: '7e808fb554fdf443e70e5ccdd3fdccd3cd74cdec606d3375fa4c5877d4f46e0b',
    })
  })

  it('matches the official Qwen3-TTS LiteRT repository', () => {
    expect(qwen3TtsManifest.modelId).toBe('qwen3-tts-12hz-0.6b-base')
    expect(qwen3TtsManifest.assets.find(asset => asset.id === 'tokenizer')?.bytes)
      .toBe(11_424_262)
    expect(qwen3TtsManifest.assets.find(asset => asset.id === 'talker')?.path)
      .toBe('talker_fp32.tflite')
    expect(qwen3TtsManifest.memory.downloadBytes).toBe(3_415_668_132)
  })
})
