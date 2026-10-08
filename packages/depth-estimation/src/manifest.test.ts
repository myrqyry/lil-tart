import { describe, expect, it } from 'vitest';
import { da3SmallPortraitManifest, da3SmallPortraitSpec } from './manifest';

describe('DA3 portrait candidate manifest', () => {
  it('pins the artifact and keeps browser verification unpromoted', () => {
    const model = da3SmallPortraitManifest.assets.find(
      (asset) => asset.id === 'model',
    );
    expect(model?.path).toContain(
      '/resolve/5cd25d936e3fde2edef68f53b4123401454ac9c8/da3_small_gpu_fp16.tflite',
    );
    expect(model?.sha256).toBe(
      'e170369a72ba1bba7486a4d2de555639fccd0595a9bb5b5349f7733ed4aebd1f',
    );
    expect(da3SmallPortraitManifest.verification).toMatchObject({
      compile: 'untested',
      inference: 'untested',
      output: 'untested',
      qualification: 'limited',
    });
  });

  it('records the measured fixed portrait tensor contract', () => {
    expect(da3SmallPortraitSpec.inputShape).toEqual([1, 3, 896, 504]);
    expect(da3SmallPortraitSpec.outputShape).toEqual([1, 1, 896, 504]);
  });
});
