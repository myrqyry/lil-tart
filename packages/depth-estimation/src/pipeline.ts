import { InferenceError } from '@litert-playground/inference-core';
import type {
  Backend,
  DepthFrame,
  DepthNearFarConvention,
  ModelManifest,
  Pipeline,
  PipelineProgress,
  PipelineStatus,
  RuntimeContext,
} from '@litert-playground/inference-core';
import type {
  LiteRtModelOptions,
  ManagedLiteRtRuntime,
} from '@litert-playground/runtime-litert';
import {
  assertSourceAspectCompatible,
  createDepthFrame,
  inspectDepthTensorContract,
} from './contract';
import type { DepthModelSpec, DepthTensorContract } from './contract';
import { da3SmallPortraitSpec } from './manifest';

export interface DepthInput {
  canvas: HTMLCanvasElement | OffscreenCanvas;
  frameId?: string | number;
  timestampMs?: number;
}

export interface DepthConfig {
  allowAspectMismatch?: boolean;
  invert?: boolean;
  nearFarConvention?: DepthNearFarConvention;
}

export class DepthEstimationPipeline
  implements Pipeline<DepthInput, DepthFrame, DepthConfig>
{
  readonly manifest: ModelManifest;
  status: PipelineStatus = 'idle';
  onProgress?: (progress: PipelineProgress) => void;

  private runtime: ManagedLiteRtRuntime | null = null;
  private contract: DepthTensorContract | null = null;
  private fallbackBackend: Backend | null = null;
  private readonly modelOptions: LiteRtModelOptions;

  constructor(private readonly spec: DepthModelSpec = da3SmallPortraitSpec) {
    this.manifest = spec.manifest;
    this.modelOptions = { supportedBackends: this.manifest.backends };
  }

  async load(context: RuntimeContext): Promise<void> {
    if (this.status === 'ready') return;
    this.status = 'loading';
    this.report({ phase: 'loading', step: 0, total: 1 });

    try {
      const runtime = context.liteRt as unknown as ManagedLiteRtRuntime;
      this.fallbackBackend = context.backend;
      const model = await runtime.loadModel(this.modelUrl, this.modelOptions);
      this.contract = inspectDepthTensorContract(
        model.getInputDetails(),
        model.getOutputDetails(),
        this.spec,
      );
      this.runtime = runtime;
      this.status = 'ready';
      this.report({ phase: 'loading', step: 1, total: 1 });
    } catch (cause) {
      this.status = 'error';
      throw cause instanceof InferenceError
        ? cause
        : new InferenceError('MODEL_COMPILE_FAILED', String(cause), {
            stage: 'compile',
            cause,
          });
    }
  }

  async run(
    input: DepthInput,
    config: DepthConfig = {},
    signal?: AbortSignal,
  ): Promise<DepthFrame> {
    if (this.status !== 'ready' || !this.runtime || !this.contract) {
      throw new InferenceError('INFERENCE_FAILED', 'Depth pipeline not ready');
    }
    if (signal?.aborted) {
      throw new InferenceError('CANCELLED', 'Depth inference cancelled', {
        stage: 'preprocess',
      });
    }

    const sourceWidth = input.canvas.width;
    const sourceHeight = input.canvas.height;
    if (!config.allowAspectMismatch) {
      assertSourceAspectCompatible(
        sourceWidth,
        sourceHeight,
        this.contract,
        this.spec.maxAspectRatioError,
      );
    }

    this.status = 'running';
    const startedAt = performance.now();
    const tensor = this.canvasToTensor(input.canvas, this.contract);

    try {
      const rawOutput = await this.runtime.predict(this.modelUrl, tensor, {
        ...this.modelOptions,
        signal,
        label: 'depth-estimation',
      });
      const outputs = Array.isArray(rawOutput)
        ? rawOutput
        : Object.values(rawOutput);

      try {
        if (outputs.length !== 1) {
          throw new InferenceError(
            'OUTPUT_INVALID',
            'Depth model returned ' + outputs.length + ' outputs; expected 1',
            { stage: 'postprocess' },
          );
        }

        const rawDepth = new Float32Array(
          this.runtime.readTensor<Float32Array>(outputs[0]),
        );
        const info = this.runtime.getModelInfo(this.modelUrl, this.modelOptions);

        const frame = createDepthFrame(
          rawDepth,
          this.contract.outputWidth,
          this.contract.outputHeight,
          {
            sourceWidth,
            sourceHeight,
            ...(input.frameId !== undefined ? { frameId: input.frameId } : {}),
            ...(input.timestampMs !== undefined
              ? { timestampMs: input.timestampMs }
              : {}),
            modelId: this.manifest.modelId,
            backend: info?.resolvedBackend ?? this.fallbackBackend ?? 'wasm',
            inferenceMs: performance.now() - startedAt,
            nearFarConvention:
              config.nearFarConvention ?? this.spec.nearFarConvention,
            invert: config.invert,
          },
        );
        this.status = 'ready';
        return frame;
      } finally {
        outputs.forEach((output) => output.delete());
      }
    } catch (cause) {
      this.status = 'ready';
      if (cause instanceof InferenceError) throw cause;
      if (signal?.aborted) {
        throw new InferenceError('CANCELLED', 'Depth inference cancelled', {
          stage: 'inference',
          cause,
        });
      }
      throw new InferenceError('INFERENCE_FAILED', 'Depth inference failed', {
        stage: 'inference',
        cause,
      });
    } finally {
      tensor.delete();
    }
  }

  async dispose(): Promise<void> {
    this.runtime?.disposeModel(this.modelUrl);
    this.runtime = null;
    this.contract = null;
    this.fallbackBackend = null;
    this.status = 'disposed';
  }

  private get modelUrl(): string {
    const asset = this.manifest.assets.find((candidate) => candidate.id === 'model');
    if (!asset) {
      throw new InferenceError('ASSET_FETCH_FAILED', 'Depth model asset is missing');
    }
    return asset.path;
  }

  private canvasToTensor(
    canvas: HTMLCanvasElement | OffscreenCanvas,
    contract: DepthTensorContract,
  ) {
    const target = this.createCanvas(contract.inputWidth, contract.inputHeight);
    const context = target.getContext('2d');
    if (!context) {
      throw new InferenceError(
        'INFERENCE_FAILED',
        'Unable to create depth preprocessing context',
        { stage: 'preprocess' },
      );
    }

    context.drawImage(
      canvas,
      0,
      0,
      contract.inputWidth,
      contract.inputHeight,
    );

    const rgba = context.getImageData(
      0,
      0,
      contract.inputWidth,
      contract.inputHeight,
    ).data;

    const pixels = contract.inputWidth * contract.inputHeight;
    const values = new Float32Array(3 * pixels);
    for (let pixel = 0; pixel < pixels; pixel += 1) {
      const rgbaOffset = pixel * 4;
      for (let channel = 0; channel < 3; channel += 1) {
        values[channel * pixels + pixel] =
          (rgba[rgbaOffset + channel] / 255 - this.spec.mean[channel]) /
          this.spec.std[channel];
      }
    }

    return this.runtime!.createTensor(values, [...contract.inputShape]);
  }

  private createCanvas(
    width: number,
    height: number,
  ): OffscreenCanvas | HTMLCanvasElement {
    if (typeof OffscreenCanvas !== 'undefined') {
      return new OffscreenCanvas(width, height);
    }
    if (typeof document !== 'undefined') {
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      return canvas;
    }
    throw new InferenceError(
      'BACKEND_UNAVAILABLE',
      'Canvas preprocessing is unavailable in this environment',
      { stage: 'preprocess' },
    );
  }

  private report(progress: PipelineProgress): void {
    this.onProgress?.(progress);
  }
}
