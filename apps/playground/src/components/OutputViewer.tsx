import ImageOutput, { isImageShape } from './ImageOutput'
import type { TensorSpec } from '../adapters/types'
import type { RawTensor } from '../hooks/useModelRunner'

interface OutputViewerProps {
  outputs: Record<string, any> | null
  outputTensors?: Record<string, RawTensor> | null
  outputSpecs?: TensorSpec[]
}

export default function OutputViewer({ outputs, outputTensors, outputSpecs }: OutputViewerProps) {
  if (!outputs) {
    return (
      <div>
        <h2 className="mb-2 text-sm font-semibold text-on-surface-variant uppercase tracking-wide">Outputs</h2>
        <p className="text-sm text-on-surface-variant">Run inference to see results</p>
      </div>
    )
  }
  return (
    <div>
      <h2 className="mb-3 text-sm font-semibold text-on-surface-variant uppercase tracking-wide">Outputs</h2>
      {Object.entries(outputs).map(([key, value]) => {
        const raw = outputTensors?.[key]
        const spec = outputSpecs?.find(s => s.name === key)
        const shape = raw?.shape ?? spec?.shape ?? []

        return (
          <div key={key} className="mb-4">
            <div className="mb-1 text-sm font-semibold text-on-surface">{key}</div>
            {typeof ImageData !== 'undefined' && value instanceof ImageData ? (
              <ImageOutput data={value.data} shape={[1, value.height, value.width]} label={key} />
            ) : raw && isImageShape(shape) ? (
              <ImageOutput data={raw.data} shape={shape} label={key} />
            ) : (
              <pre className="max-h-48 overflow-auto rounded-lg bg-surface-container px-4 py-3 font-mono text-xs text-on-surface [scrollbar-width:thin]">
                {typeof value === 'string' ? value : JSON.stringify(value, null, 2)}
              </pre>
            )}
          </div>
        )
      })}
    </div>
  )
}
