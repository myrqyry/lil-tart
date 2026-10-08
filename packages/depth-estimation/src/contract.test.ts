import { describe, expect, it } from 'vitest';
import { InferenceError } from '@litert-playground/inference-core';
import {
  assertSourceAspectCompatible,
  createDepthFrame,
  inspectDepthTensorContract,
  normalizeDepthValues,
} from './contract';
import { da3SmallPortraitSpec } from './manifest';

describe('Depth tensor contract', () => {
  it('accepts the pinned DA3 portrait tensor shapes', () => {
    const contract = inspectDepthTensorContract(
      [{ shape: [1, 3, 896, 504] }],
      [{ shape: [1, 1, 896, 504] }],
      da3SmallPortraitSpec,
    );
    expect(contract).toMatchObject({
      inputWidth: 504,
      inputHeight: 896,
      outputWidth: 504,
      outputHeight: 896,
    });
  });

  it('rejects stale square tensor assumptions', () => {
    expect(() =>
      inspectDepthTensorContract(
        [{ shape: [1, 3, 518, 518] }],
        [{ shape: [1, 1, 518, 518] }],
        da3SmallPortraitSpec,
      ),
    ).toThrow(/does not match pinned contract/);
  });

  it('rejects large source-aspect distortion by default', () => {
    const contract = inspectDepthTensorContract(
      [{ shape: [1, 3, 896, 504] }],
      [{ shape: [1, 1, 896, 504] }],
      da3SmallPortraitSpec,
    );
    expect(() => assertSourceAspectCompatible(256, 224, contract)).toThrow(
      InferenceError,
    );
    expect(() => assertSourceAspectCompatible(504, 896, contract)).not.toThrow();
  });
});

describe('Depth normalization', () => {
  it('normalizes finite depth to 0..1 while preserving order', () => {
    const result = normalizeDepthValues(new Float32Array([2, 4, 6]));
    expect(Array.from(result.data)).toEqual([0, 0.5, 1]);
    expect(result.min).toBe(2);
    expect(result.max).toBe(6);
  });

  it('inverts normalized depth only when requested', () => {
    const result = normalizeDepthValues(new Float32Array([2, 4, 6]), true);
    expect(Array.from(result.data)).toEqual([1, 0.5, 0]);
  });

  it('rejects non-finite depth values', () => {
    expect(() =>
      normalizeDepthValues(new Float32Array([0, Number.NaN])),
    ).toThrow(/non-finite/);
  });
});

describe('DepthFrame', () => {
  it('carries frame identity, source dimensions, and explicit UV mapping', () => {
    const frame = createDepthFrame(
      new Float32Array([10, 20, 30, 40]),
      2,
      2,
      {
        sourceWidth: 640,
        sourceHeight: 360,
        frameId: 17,
        timestampMs: 1234,
        modelId: 'test-depth',
        backend: 'webgpu',
        nearFarConvention: 'higher-near',
      },
    );

    expect(frame).toMatchObject({
      kind: 'depth',
      width: 2,
      height: 2,
      sourceWidth: 640,
      sourceHeight: 360,
      frameId: 17,
      timestampMs: 1234,
      sourceToDepthUv: {
        scaleX: 1,
        scaleY: 1,
        offsetX: 0,
        offsetY: 0,
      },
      nearFarConvention: 'higher-near',
      normalization: 'frame-minmax',
      modelId: 'test-depth',
      backend: 'webgpu',
    });
  });

  it('flips a known near/far convention when data is inverted', () => {
    const frame = createDepthFrame(
      new Float32Array([0, 1]),
      2,
      1,
      {
        sourceWidth: 2,
        sourceHeight: 1,
        modelId: 'test-depth',
        backend: 'wasm',
        nearFarConvention: 'higher-near',
        invert: true,
      },
    );
    expect(frame.nearFarConvention).toBe('higher-far');
    expect(Array.from(frame.data)).toEqual([1, 0]);
  });
});
