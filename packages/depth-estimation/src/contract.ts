import { InferenceError } from '@litert-playground/inference-core';
import type {
  Backend,
  DepthFrame,
  DepthNearFarConvention,
  ModelManifest,
} from '@litert-playground/inference-core';

export interface DepthModelSpec {
  manifest: ModelManifest;
  inputShape: readonly [1, 3, number, number];
  outputShape: readonly [1, 1, number, number];
  mean: readonly [number, number, number];
  std: readonly [number, number, number];
  nearFarConvention: DepthNearFarConvention;
  maxAspectRatioError?: number;
}

export interface DepthTensorContract {
  inputShape: [number, number, number, number];
  outputShape: [number, number, number, number];
  inputWidth: number;
  inputHeight: number;
  outputWidth: number;
  outputHeight: number;
}

export interface DepthFrameMetadata {
  sourceWidth: number;
  sourceHeight: number;
  frameId?: string | number;
  timestampMs?: number;
  modelId: string;
  backend: Backend;
  inferenceMs?: number;
  nearFarConvention?: DepthNearFarConvention;
  invert?: boolean;
}

interface TensorDetailLike {
  shape: ArrayLike<number>;
}

function tuple4(value: ArrayLike<number>): [number, number, number, number] {
  const shape = Array.from(value);
  if (shape.length !== 4 || shape.some((dim) => !Number.isInteger(dim) || dim <= 0)) {
    throw new InferenceError(
      'OUTPUT_INVALID',
      'Expected a rank-4 positive tensor shape, got [' + shape.join(',') + ']',
      { stage: 'model-contract' },
    );
  }
  return shape as [number, number, number, number];
}

function sameShape(actual: readonly number[], expected: readonly number[]): boolean {
  return actual.length === expected.length
    && actual.every((value, index) => value === expected[index]);
}

export function inspectDepthTensorContract(
  inputDetails: readonly TensorDetailLike[],
  outputDetails: readonly TensorDetailLike[],
  spec: DepthModelSpec,
): DepthTensorContract {
  if (inputDetails.length !== 1 || outputDetails.length !== 1) {
    throw new InferenceError(
      'OUTPUT_INVALID',
      'Depth model must expose exactly one input and one output; got ' +
        inputDetails.length + '/' + outputDetails.length,
      { stage: 'model-contract' },
    );
  }

  const inputShape = tuple4(inputDetails[0].shape);
  const outputShape = tuple4(outputDetails[0].shape);

  if (!sameShape(inputShape, spec.inputShape)) {
    throw new InferenceError(
      'OUTPUT_INVALID',
      'Depth model input shape [' + inputShape.join(',') +
        '] does not match pinned contract [' + spec.inputShape.join(',') + ']',
      { stage: 'model-contract' },
    );
  }
  if (!sameShape(outputShape, spec.outputShape)) {
    throw new InferenceError(
      'OUTPUT_INVALID',
      'Depth model output shape [' + outputShape.join(',') +
        '] does not match pinned contract [' + spec.outputShape.join(',') + ']',
      { stage: 'model-contract' },
    );
  }

  return {
    inputShape,
    outputShape,
    inputWidth: inputShape[3],
    inputHeight: inputShape[2],
    outputWidth: outputShape[3],
    outputHeight: outputShape[2],
  };
}

export function assertSourceAspectCompatible(
  sourceWidth: number,
  sourceHeight: number,
  contract: DepthTensorContract,
  maxAspectRatioError = 0.03,
): void {
  if (sourceWidth <= 0 || sourceHeight <= 0) {
    throw new InferenceError('INVALID_INPUT', 'Depth source dimensions must be positive');
  }

  const sourceAspect = sourceWidth / sourceHeight;
  const modelAspect = contract.inputWidth / contract.inputHeight;
  const error = Math.abs(sourceAspect - modelAspect) / modelAspect;

  if (error > maxAspectRatioError) {
    throw new InferenceError(
      'INVALID_INPUT',
      'Source aspect ' + sourceWidth + 'x' + sourceHeight +
        ' differs from the model fixed ' + contract.inputWidth + 'x' +
        contract.inputHeight + ' aspect by ' + (error * 100).toFixed(1) +
        '%. Use a model converted for the source aspect or explicitly allow distortion.',
      { stage: 'preprocess' },
    );
  }
}

export function normalizeDepthValues(
  values: Float32Array,
  invert = false,
): { data: Float32Array; min: number; max: number } {
  if (values.length === 0) {
    throw new InferenceError('OUTPUT_INVALID', 'Depth model returned an empty tensor', {
      stage: 'postprocess',
    });
  }

  let min = Infinity;
  let max = -Infinity;
  for (const value of values) {
    if (!Number.isFinite(value)) {
      throw new InferenceError('OUTPUT_INVALID', 'Depth model returned a non-finite value', {
        stage: 'postprocess',
      });
    }
    if (value < min) min = value;
    if (value > max) max = value;
  }

  const range = max - min;
  const data = new Float32Array(values.length);
  if (range <= 1e-12) return { data, min, max };

  for (let index = 0; index < values.length; index += 1) {
    const normalized = (values[index] - min) / range;
    data[index] = invert ? 1 - normalized : normalized;
  }
  return { data, min, max };
}

function invertConvention(
  convention: DepthNearFarConvention,
  invert: boolean,
): DepthNearFarConvention {
  if (!invert || convention === 'unknown') return convention;
  return convention === 'higher-near' ? 'higher-far' : 'higher-near';
}

export function createDepthFrame(
  rawValues: Float32Array,
  width: number,
  height: number,
  metadata: DepthFrameMetadata,
): DepthFrame {
  if (rawValues.length !== width * height) {
    throw new InferenceError(
      'OUTPUT_INVALID',
      'Depth tensor length ' + rawValues.length + ' does not match ' +
        width + 'x' + height,
      { stage: 'postprocess' },
    );
  }

  const invert = metadata.invert ?? false;
  const normalized = normalizeDepthValues(rawValues, invert);
  return {
    kind: 'depth',
    data: normalized.data,
    width,
    height,
    sourceWidth: metadata.sourceWidth,
    sourceHeight: metadata.sourceHeight,
    sourceToDepthUv: { scaleX: 1, scaleY: 1, offsetX: 0, offsetY: 0 },
    ...(metadata.frameId !== undefined ? { frameId: metadata.frameId } : {}),
    timestampMs: metadata.timestampMs ?? Date.now(),
    nearFarConvention: invertConvention(
      metadata.nearFarConvention ?? 'unknown',
      invert,
    ),
    normalization: 'frame-minmax',
    rawRange: { min: normalized.min, max: normalized.max },
    modelId: metadata.modelId,
    backend: metadata.backend,
    ...(metadata.inferenceMs !== undefined ? { inferenceMs: metadata.inferenceMs } : {}),
  };
}
