import { type ModelAsset, type ModelManifest } from '@litert-playground/inference-core'
import {
  QWEN3_TTS_UPSTREAM_REPOSITORY,
  QWEN3_TTS_UPSTREAM_REVISION,
} from './provenance'
export {
  QWEN3_TTS_UPSTREAM_REPOSITORY,
  QWEN3_TTS_UPSTREAM_REVISION,
  QWEN3_TTS_MAX_IN_MEMORY_SHA256_BYTES,
} from './provenance'

interface PublishedAsset {
  bytes: number
  sha256: string
}

const QWEN3_TTS_OMNI_MTP: PublishedAsset = {
  bytes: 440_528_628,
  sha256: '7e808fb554fdf443e70e5ccdd3fdccd3cd74cdec606d3375fa4c5877d4f46e0b',
}

const PUBLISHED_ASSETS: Readonly<Record<string, PublishedAsset>> = {
  'tokenizer.json': {
    bytes: 11_424_262,
    sha256: 'a7d41145c408f3062e824f965be1c29854cd809e111cdf8170aa8b0bcd5d1fab',
  },
  'talker_fp32.tflite': {
    bytes: 1_783_890_064,
    sha256: '0b137d39421d8fcd4653a06c245511a426710e48aff5169de7d111e1989eae60',
  },
  'talker_int4.tflite': {
    bytes: 255_998_768,
    sha256: 'e03df54e73ed1f88b2ae6d47bbf82dd64ea90a3620d753a0f3c8d6a8d60848db',
  },
  'mtp_fp32.tflite': {
    bytes: 440_526_692,
    sha256: '7cc01b402637c850432ee9171c9cc9103813c0406b70c455ea8752833adf1f03',
  },
  'codec_decoder_fp32.tflite': {
    bytes: 456_820_324,
    sha256: '491e10c263c498d9f305ae655dbcce18455d54e44421480afe2d7da2b07da39d',
  },
  'tables/codec_embedding_fp32.npy': {
    bytes: 12_583_040,
    sha256: '47fa9e30f98b1528fc9b332d314f22a32fa33e187509a4d3537f8b2c31199e39',
  },
  'tables/mtp_embeddings_fp16.npy': {
    bytes: 62_914_688,
    sha256: 'fea581b6a04f1cbec20b49511c36a00011411ccfba31f89b7571f82fe6b36706',
  },
  'tables/text_embedding_fp16.npy': {
    bytes: 622_329_984,
    sha256: '6fab9de0a8bc144aa3efefdacb5e8292b8499a9ae1d60fcee240bac528b7441e',
  },
  'tables/text_projection_fp32.npz': {
    bytes: 25_179_078,
    sha256: 'ebb0f6a7aaacdbc903e825e33480b2da4d4b71c43c90acb0e89988049c77c100',
  },
  'voices/demo_speaker.npy': {
    bytes: 4_224,
    sha256: 'b1527f54f68f44ca98bfddcaa9dc0018deb2db62590c9d2699efabbd0dfc1c3c',
  },
}

export interface Qwen3TtsVariant {
  id: string
  talker: string
  mtp: string
  codec: string
  quantization: string
  backendSupport: Partial<Record<'webgpu' | 'wasm' | 'webnn', boolean | 'experimental'>>
}

const QWEN3_TTS_VARIANT_DEFINITIONS = {
  fp32: {
    id: 'fp32', talker: 'talker_fp32.tflite', mtp: 'mtp_fp32.tflite', codec: 'codec_decoder_fp32.tflite',
    quantization: 'fp32', backendSupport: { webgpu: 'experimental', wasm: true },
  },
  int4: {
    id: 'int4', talker: 'talker_int4.tflite', mtp: 'mtp_fp32.tflite', codec: 'codec_decoder_fp32.tflite',
    quantization: 'int4 talker / fp32 auxiliary graphs', backendSupport: { webgpu: 'experimental', wasm: true },
  },
  browserMemory: {
    id: 'browserMemory', talker: 'talker_int4.tflite', mtp: 'mtp_folded_int8.tflite', codec: 'codec_decoder_fp32.tflite',
    quantization: 'int4 talker / int8 folded mtp / fp32 codec', backendSupport: { webgpu: 'experimental', wasm: true },
  },
  browserMemoryOmni: {
    id: 'browserMemoryOmni', talker: 'talker_int4.tflite', mtp: 'mtp_fp32.tflite', codec: 'codec_decoder_fp32.tflite',
    quantization: 'int4 talker / Omni fp32 mtp / fp32 codec', backendSupport: { webgpu: 'experimental', wasm: true },
  },
} as const satisfies Record<string, Qwen3TtsVariant>

export type Qwen3TtsVariantId = keyof typeof QWEN3_TTS_VARIANT_DEFINITIONS

// Keep the historical broad lookup surface for downstream callers while the
// exact id union above lets first-party UIs fail at compile time if definitions
// and labels drift apart.
export const qwen3TtsVariants: Record<string, Qwen3TtsVariant> =
  QWEN3_TTS_VARIANT_DEFINITIONS

function publishedAsset(
  id: string,
  path: string,
  fallbackBytes?: number,
  optional?: boolean,
  publishedOverride?: PublishedAsset,
): ModelAsset {
  const published = publishedOverride ?? PUBLISHED_ASSETS[path]
  return {
    id,
    path,
    ...(published ? published : fallbackBytes !== undefined ? { bytes: fallbackBytes } : {}),
    ...(optional ? { optional: true } : {}),
  }
}

function assetsFor(variant: Qwen3TtsVariant): ModelAsset[] {
  return [
    publishedAsset('tokenizer', 'tokenizer.json'),
    publishedAsset('talker', variant.talker),
    publishedAsset(
      'mtp',
      variant.mtp,
      variant.id === 'browserMemory' ? 229_608_368 : undefined,
      false,
      variant.id === 'browserMemoryOmni' ? QWEN3_TTS_OMNI_MTP : undefined,
    ),
    publishedAsset('codec-decoder', variant.codec),
    publishedAsset('codec-embedding', 'tables/codec_embedding_fp32.npy'),
    publishedAsset('mtp-embeddings', 'tables/mtp_embeddings_fp16.npy'),
    publishedAsset('text-embedding', 'tables/text_embedding_fp16.npy'),
    publishedAsset('text-projection', 'tables/text_projection_fp32.npz'),
    publishedAsset('voice', 'voices/demo_speaker.npy', undefined, true),
  ]
}

export function createQwen3TtsManifest(variant: Qwen3TtsVariant = qwen3TtsVariants.fp32): ModelManifest {
  const assets = assetsFor(variant)
  const requiredDownloadBytes = assets
    .filter(asset => !asset.optional)
    .reduce((total, asset) => total + (asset.bytes ?? 0), 0)
  return {
    modelId: 'qwen3-tts-12hz-0.6b-base',
    name: 'Qwen3-TTS 0.6B (' + variant.quantization + ')',
    version: '0.4.0',
    capabilities: ['text-to-speech'],
    backends: variant.backendSupport,
    memory: { downloadBytes: requiredDownloadBytes, residentBytes: 2_500_000_000 },
    assets,
    verification: {
      assets: 'untested',
      compile: 'untested',
      inference: 'untested',
      output: 'untested',
      qualification: 'unverified',
      upstreamRevision: QWEN3_TTS_UPSTREAM_REVISION,
    },
  }
}

export const qwen3TtsManifest = createQwen3TtsManifest()
