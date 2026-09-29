/// <reference lib="webworker" />
import { createLiteRtRuntime } from '@litert-playground/runtime-litert';
import {
  createCachingAssetResolver,
  createHttpAssetResolver,
  createManifestVerifyingAssetResolver,
  type ModelManifest,
} from '@litert-playground/inference-core';
import { DecoderPhase } from '../phases/decoder';
import type { DecoderWorkerRequest, DecoderWorkerResponse } from './protocol';
import { serializeError } from './protocol';

let phase: DecoderPhase | undefined;

async function buildContext(modelBase: string, manifest: ModelManifest) {
  const inner = createCachingAssetResolver(createHttpAssetResolver(modelBase));
  const assets = createManifestVerifyingAssetResolver(manifest, inner);
  return createLiteRtRuntime({ assets });
}

self.onmessage = async (event: MessageEvent<DecoderWorkerRequest>) => {
  const req = event.data;
  try {
    if (req.type === 'initialize') {
      phase = new DecoderPhase(req.variant);
      const context = await buildContext(req.modelBase, phase.manifest);
      await phase.load(context);
      self.postMessage({ type: 'ready' } satisfies DecoderWorkerResponse);
      return;
    }
    if (req.type === 'decode') {
      if (!phase) throw new Error('decoder not initialized');
      const audio = await phase.decode(req.frames);
      const phaseReceipt = {
        name: phase.name,
        backend: 'wasm' as const,
        loadMs: phase.loadMs,
        compileMs: phase.compileMs,
        inferenceMs: phase.inferenceMs,
      };
      self.postMessage(
        { type: 'audio', requestId: req.requestId, audio, phaseReceipt } satisfies DecoderWorkerResponse,
        [audio.buffer],
      );
      return;
    }
  } catch (cause) {
    const requestId = 'requestId' in req ? req.requestId : undefined;
    self.postMessage({ type: 'error', requestId, error: serializeError(cause) } satisfies DecoderWorkerResponse);
  }
};

// Signal the host that this worker's onmessage handler is installed and ready
// to receive the initialize message (the host must not post initialize before
// this arrives, or the message is dropped in the pre-handler event loop).
self.postMessage({ type: 'booted' } satisfies DecoderWorkerResponse);

export {};
