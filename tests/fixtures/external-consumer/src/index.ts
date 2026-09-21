import { InferenceError, createHttpAssetResolver } from '@litert-playground/inference-core'
import { createLiteRtRuntime } from '@litert-playground/runtime-litert'
import { createBrowserCacheAssetResolver } from '@litert-playground/browser-cache'
import { LiteRtLmTextPipeline } from '@litert-playground/text-gen'
import { EncoderPipeline } from '@litert-playground/encoder'
import { ColBertPipeline } from '@litert-playground/retrieval'
import { KokoroPipeline } from '@litert-playground/kokoro'
import { Qwen3TtsPipeline } from '@litert-playground/qwen3-tts'
import { ClipImageEmbeddingPipeline } from '@litert-playground/image-embedding'
import { MoViNetPipeline } from '@litert-playground/video-classification'

export const importedPackages = [
  InferenceError,
  createHttpAssetResolver,
  createLiteRtRuntime,
  createBrowserCacheAssetResolver,
  LiteRtLmTextPipeline,
  EncoderPipeline,
  ColBertPipeline,
  KokoroPipeline,
  Qwen3TtsPipeline,
  ClipImageEmbeddingPipeline,
  MoViNetPipeline,
]
