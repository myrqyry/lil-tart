import type { ModelManifest } from '@litert-playground/inference-core';

const UPSTREAM_REVISION = '5cd25d936e3fde2edef68f53b4123401454ac9c8';
const MODEL_URL =
  'https://huggingface.co/litert-community/Depth-Anything-3-Small/resolve/' +
  UPSTREAM_REVISION +
  '/da3_small_gpu_fp16.tflite';

export const da3SmallPortraitManifest: ModelManifest = {
  modelId: 'depth-anything-3-small-portrait',
  name: 'Depth Anything 3 Small — portrait LiteRT',
  version: '0.1.0',
  capabilities: ['image-depth'],
  // Candidate browser paths only. Durable browser verification remains unpromoted.
  backends: { webgpu: 'experimental', wasm: 'experimental' },
  memory: { downloadBytes: 55_000_000, residentBytes: 500_000_000 },
  assets: [
    {
      id: 'model',
      path: MODEL_URL,
      sha256: 'e170369a72ba1bba7486a4d2de555639fccd0595a9bb5b5349f7733ed4aebd1f',
      role: 'monocular depth model',
    },
  ],
  verification: {
    assets: 'untested',
    compile: 'untested',
    inference: 'untested',
    output: 'untested',
    qualification: 'limited',
    upstreamRevision: UPSTREAM_REVISION,
    expectedOutput: {
      preprocessing: [
        'fixed 504x896 portrait input (W x H)',
        'RGB float32 NCHW tensor [1,3,896,504]',
        'divide RGB by 255',
        'ImageNet mean [0.485,0.456,0.406]',
        'ImageNet standard deviation [0.229,0.224,0.225]',
      ],
      outputShape: [1, 1, 896, 504],
      behavior: [
        'package normalizes finite model output per frame to [0,1]',
        'source/depth UV mapping is carried explicitly on DepthFrame',
        'browser backend and output quality remain unverified until qualification runs',
      ],
    },
  },
};

export const da3SmallPortraitSpec = {
  manifest: da3SmallPortraitManifest,
  inputShape: [1, 3, 896, 504] as const,
  outputShape: [1, 1, 896, 504] as const,
  mean: [0.485, 0.456, 0.406] as const,
  std: [0.229, 0.224, 0.225] as const,
  nearFarConvention: 'unknown' as const,
  maxAspectRatioError: 0.03,
};
