import { Tensor } from '@litertjs/core';
import { InferenceError } from '@litert-playground/inference-core';
import type {
  ModelManifest,
  Pipeline,
  PipelineProgress,
  PipelineStatus,
  RuntimeContext,
} from '@litert-playground/inference-core';
import type { ManagedLiteRtRuntime } from '@litert-playground/runtime-litert';
import { moViNetManifest } from './manifest';
import kinetics600Labels from './kinetics600_labels.json';

export interface MoViNetPrediction {
  topClasses: { label: string; index: number; score: number }[];
  logits: Float32Array;
  frameNum: number;
  latencyMs: number;
}

export interface MoViNetInput {
  canvas: HTMLCanvasElement | OffscreenCanvas;
}

export interface MoViNetConfig {
  topK?: number;
}

const INPUT_SIZE = 172;
const CONV_DEPTHS = [2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 4];
const CONV_LAYERS = 11;
const POOL_LAYERS = 16;

class MoViNetState {
  private convBuffers: Float32Array[][] = [];
  private poolSums: Float32Array[] = [];
  private frameNum = 0;
  private constant: Float32Array = new Float32Array([1]);
  private initialized = false;
  private convOffsets: number[] = [];

  private inputShapes: number[][] = [];
  private convShapes: number[][] = [];

  init(inputDetails: { shape: number[]; index: number }[]): void {
    if (this.initialized) return;

    this.convOffsets = [];
    let o = 0;
    for (const d of CONV_DEPTHS) {
      this.convOffsets.push(o);
      o += d;
    }

    this.inputShapes = inputDetails.sort((a, b) => a.index - b.index).map(d => d.shape);
    this.convShapes = [];
    for (let c = 0; c < CONV_LAYERS; c++) {
      const bufferShapes: number[][] = [];
      for (let i = 0; i < CONV_DEPTHS[c]; i++) {
        bufferShapes.push(this.inputShapes[1 + this.convOffsets[c] + i]);
      }
      this.convShapes.push(this.inputShapes[1 + this.convOffsets[c]]);
    }

    this.allocateBuffers();
    this.initialized = true;
  }

  reset(): void {
    this.frameNum = 0;
    if (this.inputShapes.length > 0) {
      this.allocateBuffers();
      this.initialized = true;
      return;
    }

    this.convBuffers = [];
    this.poolSums = [];
    this.initialized = false;
  }

  // ponytail: frameNum commits only after the whole frame succeeds — no
  // rollback path needed.
  commitFrame(nextFrame: number): void {
    this.frameNum = nextFrame;
  }

  async buildInputTensors(frameTensor: Tensor): Promise<{ inputs: Tensor[]; nextFrame: number }> {
    const nextFrame = this.frameNum + 1;
    const n = nextFrame;
    const invCount = new Float32Array([1 / n]);
    const inputs: Tensor[] = [frameTensor];

    for (let c = 0; c < CONV_LAYERS; c++) {
      for (let i = 0; i < CONV_DEPTHS[c]; i++) {
        const shape = this.inputShapes[1 + this.convOffsets[c] + i];
        inputs.push(Tensor.fromTypedArray(this.convBuffers[c][i], shape));
      }
    }
    for (let i = 0; i < POOL_LAYERS; i++) {
      const shape = this.inputShapes[29 + i];
      inputs.push(Tensor.fromTypedArray(this.poolSums[i], shape));
    }
    inputs.push(Tensor.fromTypedArray(invCount, [1, 1, 1, 1]));
    inputs.push(Tensor.fromTypedArray(this.constant, [1, 1, 1, 1]));

    return { inputs, nextFrame };
  }

  private allocateBuffers(): void {
    this.convBuffers = this.convShapes.map((_shape, c) =>
      Array.from({ length: CONV_DEPTHS[c] }, (_, i) =>
        new Float32Array(this.product(this.inputShapes[1 + this.convOffsets[c] + i])),
      ),
    );
    this.poolSums = Array.from({ length: POOL_LAYERS }, (_, i) => {
      const shape = this.inputShapes[29 + i];
      return new Float32Array(this.product(shape));
    });
  }

  updateState(outputTensors: Tensor[]): void {
    for (let c = 0; c < CONV_LAYERS; c++) {
      const currentFrame = outputTensors[1 + c];
      const data = currentFrame.toTypedArray() as Float32Array;
      this.convBuffers[c] = [
        ...this.convBuffers[c].slice(1),
        new Float32Array(data),
      ];
    }
    for (let i = 0; i < POOL_LAYERS; i++) {
      const mean = outputTensors[12 + i];
      const data = mean.toTypedArray() as Float32Array;
      for (let j = 0; j < data.length; j++) {
        this.poolSums[i][j] += data[j];
      }
    }
  }

  private product(shape: number[]): number {
    return shape.reduce((a, b) => a * b, 1);
  }
}

export class MoViNetPipeline implements Pipeline<MoViNetInput, MoViNetPrediction, MoViNetConfig> {
  readonly manifest: ModelManifest = moViNetManifest;
  status: PipelineStatus = 'idle';
  onProgress?: (progress: PipelineProgress) => void;

  private context: RuntimeContext | null = null;
  private runtime: ManagedLiteRtRuntime | null = null;
  private state = new MoViNetState();
  private isLoaded = false;

  async load(context: RuntimeContext): Promise<void> {
    if (this.status === 'ready') return;
    this.context = context;
    this.status = 'loading';
    this.report({ phase: 'loading', step: 0, total: 1 });
    try {
      const runtime = context.liteRt as unknown as ManagedLiteRtRuntime;
      const model = await runtime.loadModel(this.modelUrl, { accelerator: 'wasm' });
      const details = model.getInputDetails();
      this.state.init(details.map(d => ({
        shape: Array.from(d.shape),
        index: d.index,
      })));
      this.runtime = runtime;
      this.isLoaded = true;
      this.status = 'ready';
      this.report({ phase: 'loading', step: 1, total: 1 });
    } catch (e) {
      this.status = 'error';
      throw e instanceof InferenceError ? e : new InferenceError('MODEL_COMPILE_FAILED', String(e), { cause: e });
    }
  }

  async run(input: MoViNetInput, config?: MoViNetConfig, signal?: AbortSignal): Promise<MoViNetPrediction> {
    if (this.status !== 'ready' || !this.runtime) {
      throw new InferenceError('INFERENCE_FAILED', 'Pipeline not ready');
    }
    this.status = 'running';
    const cfg = { topK: config?.topK ?? 5 };
    const inferenceStart = performance.now();
    try {
      if (signal?.aborted) throw new Error('CANCELLED');
      const frameTensor = this.canvasToTensor(input.canvas);
      const { inputs, nextFrame } = await this.state.buildInputTensors(frameTensor);
      try {
        const rawOutput = await this.runtime.predict(this.modelUrl, inputs, { signal, accelerator: 'wasm' });
        const outputs = Array.isArray(rawOutput) ? rawOutput : Object.values(rawOutput);
        try {
          const logitsArr = new Float32Array(this.runtime.readTensor<Float32Array>(outputs[0]));
          this.state.updateState(outputs);
          this.state.commitFrame(nextFrame);
          const probs = this.softmax(logitsArr);
          const result: MoViNetPrediction = {
            topClasses: this.topK(probs, cfg.topK),
            logits: logitsArr,
            frameNum: this.state['frameNum'],
            latencyMs: Math.round(performance.now() - inferenceStart),
          };
          this.status = 'ready';
          return result;
        } finally {
          outputs.forEach((output) => output.delete());
        }
      } finally {
        inputs.forEach((input) => input.delete());
      }
    } catch (e) {
      this.status = 'ready';
      throw e;
    }
  }

  async dispose(): Promise<void> {
    if (this.runtime && this.isLoaded) {
      this.runtime.disposeModel(this.modelUrl);
    }
    this.state.reset();
    this.runtime = null;
    this.context = null;
    this.isLoaded = false;
    this.status = 'disposed';
  }

  private get modelUrl(): string {
    return this.manifest.assets.find(a => a.id === 'model')?.path ?? '';
  }

  private canvasToTensor(canvas: HTMLCanvasElement | OffscreenCanvas): Tensor {
    const offscreen = new OffscreenCanvas(INPUT_SIZE, INPUT_SIZE);
    const ctx = offscreen.getContext('2d')!;
    ctx.drawImage(canvas, 0, 0, INPUT_SIZE, INPUT_SIZE);
    const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
    const { data, width, height } = imageData;

    const floatData = new Float32Array(3 * height * width);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const p = (y * width + x) * 4;
        const c = y * width + x;
        floatData[0 * height * width + c] = data[p] / 255;
        floatData[1 * height * width + c] = data[p + 1] / 255;
        floatData[2 * height * width + c] = data[p + 2] / 255;
      }
    }

    return Tensor.fromTypedArray(floatData, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  }

  private softmax(logits: Float32Array): Float32Array {
    let max = -Infinity;
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] > max) max = logits[i];
    }
    const exp = new Float32Array(logits.length);
    let sum = 0;
    for (let i = 0; i < logits.length; i++) {
      exp[i] = Math.exp(logits[i] - max);
      sum += exp[i];
    }
    for (let i = 0; i < exp.length; i++) {
      exp[i] /= sum;
    }
    return exp;
  }

  private topK(probs: Float32Array, k: number): { label: string; index: number; score: number }[] {
    const indexed: { score: number; index: number }[] = [];
    for (let i = 0; i < probs.length; i++) {
      indexed.push({ score: probs[i], index: i });
    }
    indexed.sort((a, b) => b.score - a.score);
    return indexed.slice(0, k).map(({ score, index }) => ({
      label: (kinetics600Labels as string[])[index] ?? `class_${index}`,
      index,
      score,
    }));
  }

  private report(progress: PipelineProgress): void {
    this.onProgress?.(progress);
  }
}
