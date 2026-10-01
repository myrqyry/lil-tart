/// <reference lib="webworker" />
import { createLiteRtRuntime } from '@litert-playground/runtime-litert';
import {
  createCachingAssetResolver,
  createHttpAssetResolver,
  createManifestVerifyingAssetResolver,
  type ModelManifest,
} from '@litert-playground/inference-core';
import { GeneratorPhase } from '../phases/generator';
import type { GeneratorWorkerRequest, GeneratorWorkerResponse } from './protocol';
import { serializeError } from './protocol';

let phase: GeneratorPhase | undefined;

async function buildContext(modelBase: string, manifest: ModelManifest) {
  const inner = createCachingAssetResolver(createHttpAssetResolver(modelBase));
  // Dedicated workers keep strict SHA-256 verification. The 64 MiB
  // whole-buffer ceiling is a main-thread playground safeguard only.
  const assets = createManifestVerifyingAssetResolver(manifest, inner);
  return createLiteRtRuntime({ assets });
}

self.onmessage = async (event: MessageEvent<GeneratorWorkerRequest>) => {
  const req = event.data;
  try {
    if (req.type === 'initialize') {
      phase = new GeneratorPhase(req.variant, {
        onProgress: (progress) => {
          self.postMessage({ type: 'progress', progress } satisfies GeneratorWorkerResponse);
        },
      });
      const context = await buildContext(req.modelBase, phase.manifest);
      await phase.load(context);
      self.postMessage({ type: 'ready' } satisfies GeneratorWorkerResponse);
      return;
    }
    if (req.type === 'generate') {
      if (!phase) throw new Error('generator not initialized');
      const frames = await phase.generate(req.input, req.config);
      const phaseReceipt = {
        name: phase.name,
        backend: 'wasm' as const,
        loadMs: phase.loadMs,
        compileMs: phase.compileMs,
        inferenceMs: phase.inferenceMs,
      };
      self.postMessage(
        { type: 'frames', requestId: req.requestId, frames, phaseReceipt } satisfies GeneratorWorkerResponse,
        [frames.frames.buffer],
      );
      return;
    }
    if (req.type === 'cancel') {
      // Host terminates the worker for cancellation; nothing to do here.
      return;
    }
  } catch (cause) {
    const requestId = 'requestId' in req ? req.requestId : undefined;
    self.postMessage({ type: 'error', requestId, error: serializeError(cause) } satisfies GeneratorWorkerResponse);
  }
};

// Signal the host that this worker's onmessage handler is installed and ready
// to receive the initialize message (the host must not post initialize before
// this arrives, or the message is dropped in the pre-handler event loop).
self.postMessage({ type: 'booted' } satisfies GeneratorWorkerResponse);

export {};
