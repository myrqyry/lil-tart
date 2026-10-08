import { InferenceError, createHttpAssetResolver } from '@litert-playground/inference-core'
import { createLiteRtRuntime } from '@litert-playground/runtime-litert'
import { ManagedLiteRtModelController, useManagedLiteRtModel } from '@litert-playground/react-litert'
import { createBrowserCacheAssetResolver } from '@litert-playground/browser-cache'
import {
  LiteRtLmTextPipeline,
  LiteRtLmWorkerClient,
  type LiteRtLmTextPipelineOptions,
  type LiteRtLmWorkerLoadOptions,
  type LiteRtLmWorkerPrompt,
} from '@litert-playground/text-gen'
import { EncoderPipeline } from '@litert-playground/encoder'
import { ColBertPipeline } from '@litert-playground/retrieval'
import { maxSim, rankDense } from '@litert-playground/retrieval/scoring'
import { KokoroPipeline } from '@litert-playground/kokoro'
import { Qwen3TtsPipeline } from '@litert-playground/qwen3-tts'
import { ClipImageEmbeddingPipeline } from '@litert-playground/image-embedding'
import { MoViNetPipeline } from '@litert-playground/video-classification'
import {
  DepthEstimationPipeline,
  da3SmallPortraitManifest,
  type DepthFrame,
} from '@litert-playground/depth-estimation'

// A constructor option whose type is unreachable from the entrypoint is a hole in the
// supported surface, so the packed consumer has to be able to name it.
export const textPipelineOptions: LiteRtLmTextPipelineOptions = {
  modelBase: 'https://huggingface.co/',
}

// The lightweight downstream loader is part of the supported public surface:
// consumers with their own model cache only need to provide the execution backend.
export async function loadCachedTextModel(model: Blob): Promise<void> {
  const pipeline = new LiteRtLmTextPipeline('gemma-4-e2b-it', { model })
  await pipeline.loadForBackend('webgpu')
  await pipeline.dispose()
}

export const multimodalWorkerLoadOptions: LiteRtLmWorkerLoadOptions = {
  engine: 'mediapipe',
  mediaPipe: { maxNumImages: 1, supportAudio: true },
}

export const depthFrameContract: Pick<
  DepthFrame,
  'kind' | 'width' | 'height' | 'sourceToDepthUv' | 'nearFarConvention'
> = {
  kind: 'depth',
  width: 1,
  height: 1,
  sourceToDepthUv: { scaleX: 1, scaleY: 1, offsetX: 0, offsetY: 0 },
  nearFarConvention: 'unknown',
}

export const depthPipeline = new DepthEstimationPipeline()
export const depthManifest = da3SmallPortraitManifest

export const multimodalWorkerPrompt: LiteRtLmWorkerPrompt = [
  { type: 'text', text: 'Describe this frame.' },
  { type: 'image', data: new Blob(['frame']) },
]

export const importedPackages = [
  InferenceError,
  createHttpAssetResolver,
  createLiteRtRuntime,
  ManagedLiteRtModelController,
  useManagedLiteRtModel,
  createBrowserCacheAssetResolver,
  LiteRtLmTextPipeline,
  LiteRtLmWorkerClient,
  EncoderPipeline,
  ColBertPipeline,
  maxSim,
  rankDense,
  KokoroPipeline,
  Qwen3TtsPipeline,
  ClipImageEmbeddingPipeline,
  MoViNetPipeline,
  DepthEstimationPipeline,
  da3SmallPortraitManifest,
]
