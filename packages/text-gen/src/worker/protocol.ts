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
}

export type LiteRtLmWorkerPromptPart =
  | { type: 'text'; text: string }
  | { type: 'image'; data: Blob }
  | { type: 'audio'; data: Blob };

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
  | { type: 'error'; id?: string; message: string }
  | { type: 'disposed' };
