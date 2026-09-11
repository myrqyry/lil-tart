import { describe, it, expect } from 'vitest'
import { visionAdapters } from './vision'
import { adapters13 } from './batch2'
import { sam2Adapters } from './sam2'
import { dinov2Adapter } from './dinov2'
import { nafnetAdapter } from './nafnet'
import { realesrganAdapter } from './realesrgan'

// ImageInput emits the raw ImageData under `image`; every single-image adapter must read it.
const imageAdapters = [
  ...visionAdapters,
  ...adapters13.filter(a => !a.isPipeline && a.inputSpecs.some(s => s.shape.length === 4)),
  sam2Adapters[0],
  dinov2Adapter,
  nafnetAdapter,
  realesrganAdapter,
]

describe('image input contract', () => {
  for (const adapter of imageAdapters) {
    it(`${adapter.modelId} reads values.image`, () => {
      expect(() => adapter.prepareInputs({})).toThrow(/image data not provided/i)
    })
  }
})
