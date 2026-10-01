import { type ModelAsset, type ModelManifest } from '@litert-playground/inference-core';

interface PinnedLiteRtLmArtifact {
  repository: string;
  revision: string;
  file: string;
  bytes: number;
  sha256: string;
}

const QWEN3_06B: PinnedLiteRtLmArtifact = {
  repository: 'litert-community/Qwen3-0.6B',
  revision: 'a3c5d805ae362dff7f580bc25f2dfb9a5a7eaa76',
  file: 'Qwen3-0.6B.litertlm',
  bytes: 614_236_160,
  sha256: '555579ff2f4fd13379abe69c1c3ab5200f7338bc92471557f1d6614a6e5ab0b4',
};

const LFM25_INSTRUCT_INT4: PinnedLiteRtLmArtifact = {
  repository: 'litert-community/LFM2.5-1.2B-Instruct',
  revision: '4ef1641b562b70ffca66ce4fee7bef4446b374b6',
  file: 'LFM2.5-1.2B-Instruct_int4.litertlm',
  bytes: 736_015_744,
  sha256: '96041e6c72b1d2e9b73a122870606bbf18a3819bf14c43c3f2b74685ed62c24d',
};

const LFM25_INSTRUCT_INT8: PinnedLiteRtLmArtifact = {
  repository: 'litert-community/LFM2.5-1.2B-Instruct',
  revision: '4ef1641b562b70ffca66ce4fee7bef4446b374b6',
  file: 'LFM2.5-1.2B-Instruct_int8.litertlm',
  bytes: 1_247_091_440,
  sha256: '41c192feac3a028cfd35f4a0c5db2a70961dc388ecb4ad2cac9a094be0a078ea',
};

const LFM25_THINKING_INT4: PinnedLiteRtLmArtifact = {
  repository: 'litert-community/LFM2.5-1.2B-Thinking',
  revision: '1b1e49ad9dccdededc9d03bc0fe3071d83595d75',
  file: 'LFM2.5-1.2B-Thinking_int4.litertlm',
  bytes: 736_015_744,
  sha256: '0dfe157d03f1b16c78595e4570e84a1c0bf224324cebd95d2245e1ede35e4766',
};

const LFM25_THINKING_INT8: PinnedLiteRtLmArtifact = {
  repository: 'litert-community/LFM2.5-1.2B-Thinking',
  revision: '1b1e49ad9dccdededc9d03bc0fe3071d83595d75',
  file: 'LFM2.5-1.2B-Thinking_int8.litertlm',
  bytes: 1_244_594_224,
  sha256: 'bf610c452264fc09865e6ffff7945377c450b29297b957800ff34b18326b17cd',
};

const GEMMA4_E2B: PinnedLiteRtLmArtifact = {
  repository: 'litert-community/gemma-4-E2B-it-litert-lm',
  revision: 'b3ca0d2f076785a8f4b2219ddbd2bdb99954eae1',
  file: 'gemma-4-E2B-it.litertlm',
  bytes: 2_588_147_712,
  sha256: '181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c',
};

const GEMMA4_E4B: PinnedLiteRtLmArtifact = {
  repository: 'litert-community/gemma-4-E4B-it-litert-lm',
  revision: '2eee7ac325f20eb8c9ac1d0e972f7c84663062da',
  file: 'gemma-4-E4B-it.litertlm',
  bytes: 3_659_530_240,
  sha256: '0b2a8980ce155fd97673d8e820b4d29d9c7d99b8fa6806f425d969b145bd52e0',
};

function pinnedAsset(artifact: PinnedLiteRtLmArtifact): ModelAsset {
  return {
    id: 'model',
    path: artifact.repository + '/resolve/' + artifact.revision + '/' + artifact.file,
    bytes: artifact.bytes,
    sha256: artifact.sha256,
  };
}

function manifest(
  modelId: string,
  name: string,
  version: string,
  capabilities: ModelManifest['capabilities'],
  artifact: PinnedLiteRtLmArtifact,
): ModelManifest {
  return {
    modelId,
    name,
    version,
    capabilities,
    backends: { webgpu: true, wasm: true },
    memory: { downloadBytes: artifact.bytes, residentBytes: artifact.bytes },
    assets: [pinnedAsset(artifact)],
    verification: {
      assets: 'untested',
      compile: 'untested',
      inference: 'untested',
      output: 'untested',
      qualification: 'unverified',
      upstreamRevision: artifact.revision,
    },
  };
}

export const litertLmManifest = manifest(
  'qwen3-0.6b', 'Qwen 3 0.6B (LiteRT-LM)', '0.6.0', ['text-generation'], QWEN3_06B,
);

export const lfm2_5InstructManifest = manifest(
  'lfm2.5-1.2b-instruct',
  'LFM2.5 1.2B Instruct (LiteRT-LM, int4)',
  '1.2.0',
  ['text-generation'],
  LFM25_INSTRUCT_INT4,
);

export const lfm2_5InstructInt8Manifest = manifest(
  'lfm2.5-1.2b-instruct-int8',
  'LFM2.5 1.2B Instruct (LiteRT-LM, int8)',
  '1.2.0',
  ['text-generation'],
  LFM25_INSTRUCT_INT8,
);

export const lfm2_5ThinkingManifest = manifest(
  'lfm2.5-1.2b-thinking',
  'LFM2.5 1.2B Thinking (LiteRT-LM, int4)',
  '1.2.0',
  ['text-generation', 'reasoning'],
  LFM25_THINKING_INT4,
);

export const lfm2_5ThinkingInt8Manifest = manifest(
  'lfm2.5-1.2b-thinking-int8',
  'LFM2.5 1.2B Thinking (LiteRT-LM, int8)',
  '1.2.0',
  ['text-generation', 'reasoning'],
  LFM25_THINKING_INT8,
);

export const gemma4E2bManifest = manifest(
  'gemma-4-e2b-it', 'Gemma 4 E2B Instruct (LiteRT-LM)', '4.0.0', ['text-generation'], GEMMA4_E2B,
);

export const gemma4E4bManifest = manifest(
  'gemma-4-e4b-it', 'Gemma 4 E4B Instruct (LiteRT-LM)', '4.0.0', ['text-generation'], GEMMA4_E4B,
);

export type TextGenCapability = 'text-generation' | 'reasoning';
export type TextGenPreference = 'low_latency' | 'deep';

export function selectTextGenerationManifest(
  capability: TextGenCapability,
  preference: TextGenPreference = 'low_latency',
): ModelManifest {
  const thinking = capability === 'reasoning';
  return preference === 'deep'
    ? thinking
      ? lfm2_5ThinkingInt8Manifest
      : lfm2_5InstructInt8Manifest
    : thinking
      ? lfm2_5ThinkingManifest
      : lfm2_5InstructManifest;
}

export const transformersTextManifest: ModelManifest = {
  modelId: 'qwen3-0.6b-litertlm',
  name: 'Qwen 3 0.6B (Transformers.js)',
  version: '0.6.0',
  capabilities: ['text-generation'],
  backends: { webgpu: true },
  memory: { downloadBytes: 629_145_600, residentBytes: 629_145_600 },
  assets: [
    {
      id: 'model',
      path: 'onnx-community/Qwen3-0.6B-ONNX',
      bytes: 629_145_600,
    },
  ],
};
