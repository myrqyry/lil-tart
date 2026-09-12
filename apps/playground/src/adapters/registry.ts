import type { ModelAdapter } from './types'
import { realesrganAdapter } from './realesrgan'
import { nafnetAdapter } from './nafnet'
import { dinov2Adapter } from './dinov2'
import { magentaAdapters } from './magenta'
import { musicCocaAdapters } from './musiccoca'
import { visionAdapters } from './vision'
import { sam2Adapters } from './sam2'
import { adapters13 } from './batch2'
import { audioAdapters } from './audio'

export const registeredAdapters: ModelAdapter[] = [
  ...audioAdapters,
  ...magentaAdapters,
  ...musicCocaAdapters,
  realesrganAdapter,
  nafnetAdapter,
  dinov2Adapter,
  ...visionAdapters,
  ...sam2Adapters,
  ...adapters13,
]
