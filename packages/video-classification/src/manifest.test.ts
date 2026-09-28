import { describe, expect, it } from 'vitest';
import { moViNetManifest } from './manifest';

describe('MoViNet manifest', () => {
  it('records the pinned model and qualification contract', () => {
    expect(moViNetManifest.assets).toEqual([
      expect.objectContaining({ id: 'model', role: 'model' }),
    ]);
    expect(moViNetManifest.backends).toEqual({ webgpu: false, wasm: true });
    expect(moViNetManifest.verification).toMatchObject({
      qualification: 'limited',
      upstreamRevision: 'c2ceda0efa7344ba5a95c3eeaa9925cb0940e453',
      environments: [
        expect.objectContaining({ backend: 'wasm' }),
      ],
      expectedOutput: {
        outputShape: [1, 600],
        labels: {
          assetId: 'labels',
          count: 600,
          mapping: 'zero-based output index to Kinetics-600 label',
        },
      },
    });
  });

  it('describes preprocessing and score behavior as JSON data', () => {
    const verification = moViNetManifest.verification;
    expect(verification?.expectedOutput?.preprocessing).toEqual([
      'resize frames to 172x172',
      'RGB planar channels',
      'normalize channel values by dividing by 255',
    ]);
    expect(verification?.expectedOutput?.behavior).toEqual([
      'softmax scores sum to 1',
      'top classes use Kinetics-600 labels',
    ]);
    expect(JSON.parse(JSON.stringify(moViNetManifest))).toEqual(moViNetManifest);
  });
});
