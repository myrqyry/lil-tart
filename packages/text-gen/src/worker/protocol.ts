export interface LiteRtLmWorkerMessage {
  role: 'user' | 'assistant' | 'model';
  content: string;
}

export interface LiteRtLmWorkerGenerationConfig {
  temperature?: number;
  maxOutputTokens?: number;
  topK?: number;
  topP?: number;
  seed?: number;
  maxContextTokens?: number;
  history?: LiteRtLmWorkerMessage[];
  systemPrompt?: string;
}

export type LiteRtLmWorkerEngine = 'litert-lm' | 'mediapipe';

export interface LiteRtLmWorkerMediaPipeLoadOptions {
  wasmBaseUrl?: string;
  maxTokens?: number;
  topK?: number;
  temperature?: number;
  randomSeed?: number;
  maxNumImages?: number;
  supportAudio?: boolean;
  disableRewinding?: boolean;
}

export interface LiteRtLmWorkerLoadOptions {
  engine?: LiteRtLmWorkerEngine;
  mediaPipe?: LiteRtLmWorkerMediaPipeLoadOptions;
  /**
   * Optional caller-owned identity for Blob-backed models. Supply this when
   * independently-created Blob objects represent the same cached model and
   * should deduplicate an in-flight load.
   */
  loadKey?: string;
}

export type LiteRtLmWorkerPromptPart =
  | { type: 'text'; text: string }
  | { type: 'image'; data: Blob }
  | {
      type: 'audio';
      /**
       * PCM samples remain caller-owned. The current client structured-clones
       * this array when posting to the worker, so large clips incur one buffer
       * copy rather than detaching the caller's data.
       */
      audioSamples: Float32Array;
      audioSampleRateHz: number;
    };

export type LiteRtLmWorkerPrompt = string | LiteRtLmWorkerPromptPart[];

export type LiteRtLmWorkerRequest =
  | { type: 'load'; model: string | Blob; options?: LiteRtLmWorkerLoadOptions }
  | {
      type: 'generate';
      id: string;
      prompt: LiteRtLmWorkerPrompt;
      config?: LiteRtLmWorkerGenerationConfig;
    }
  | { type: 'cancel'; id: string }
  | { type: 'dispose' };

export type LiteRtLmWorkerResponse =
  | { type: 'ready' }
  | { type: 'token'; id: string; text: string }
  | { type: 'reasoning'; id: string; text: string }
  | { type: 'complete'; id: string }
  | { type: 'cancelled'; id: string }
  | { type: 'error'; id?: string; message: string }
  | { type: 'disposed' };
