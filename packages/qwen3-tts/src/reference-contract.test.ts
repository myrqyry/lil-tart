import { describe, expect, it } from 'vitest'
import {
  QWEN3_TTS_CODEC_LEFT_CONTEXT_FRAMES,
  QWEN3_TTS_CODEC_WINDOW_FRAMES,
} from './codec'
import { QWEN3_TTS_MTP_PASSES_PER_FRAME } from './mtp'
import {
  QWEN3_TTS_UPSTREAM_REVISION,
  createQwen3TtsManifest,
  qwen3TtsVariants,
} from './manifest'

describe('official Qwen3-TTS LiteRT host contract', () => {
  it('keeps the published host-loop invariants explicit', () => {
    expect(QWEN3_TTS_MTP_PASSES_PER_FRAME).toBe(16)
    expect(QWEN3_TTS_CODEC_WINDOW_FRAMES).toBe(64)
    expect(QWEN3_TTS_CODEC_LEFT_CONTEXT_FRAMES).toBe(25)
  })

  it('pins the upstream revision used to source integrity metadata', () => {
    expect(QWEN3_TTS_UPSTREAM_REVISION).toBe('528cca7d2ddf6f5c1e1127f24a7f8786f80fa6e8')
  })

  it('carries SHA-256 for every published asset in the int4 path', () => {
    const manifest = createQwen3TtsManifest(qwen3TtsVariants.int4)
    const required = manifest.assets.filter(asset => !asset.optional)
    expect(required).toHaveLength(8)
    expect(required.every(asset => /^[0-9a-f]{64}$/.test(asset.sha256 ?? ''))).toBe(true)
    expect(manifest.memory.downloadBytes).toBe(1_887_776_836)
  })

  it('does not invent a checksum for the local experimental folded MTP', () => {
    const manifest = createQwen3TtsManifest(qwen3TtsVariants.browserMemory)
    const mtp = manifest.assets.find(asset => asset.id === 'mtp')
    expect(mtp?.path).toBe('mtp_folded_int8.tflite')
    expect(mtp?.bytes).toBe(229_608_368)
    expect(mtp?.sha256).toBeUndefined()
  })
})
