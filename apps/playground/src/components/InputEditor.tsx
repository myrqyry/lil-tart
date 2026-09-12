import { useState, useEffect } from 'react'
import type { TensorSpec } from '../adapters/types'

interface InputEditorProps {
  specs: TensorSpec[]
  onChange: (values: Record<string, any>) => void
}

function defaultValues(specs: TensorSpec[]): Record<string, any> {
  const v: Record<string, any> = {}
  for (const spec of specs) {
    if (spec.dtype === 'string') {
      v[spec.name] = ''
    } else if (spec.constraints?.enum) {
      v[spec.name] = spec.constraints.enum[0]
    } else {
      v[spec.name] = 0
    }
  }
  return v
}

export default function InputEditor({ specs, onChange }: InputEditorProps) {
  const [values, setValues] = useState<Record<string, any>>(() => defaultValues(specs))

  useEffect(() => { onChange(values) }, [values, onChange])

  const set = (name: string, raw: string) => {
    const spec = specs.find(s => s.name === name)
    if (spec?.dtype === 'string') {
      setValues(prev => ({ ...prev, [name]: raw }))
      return
    }
    let val: any = raw
    const isArray = spec && (spec.shape.length > 1 || spec.shape[0] > 1)
    if (isArray) {
      try {
        val = JSON.parse(raw)
      } catch {
        // keep raw string while user is typing invalid JSON
      }
    } else if (spec?.dtype === 'float32') {
      val = parseFloat(raw)
      if (!isNaN(val) && spec.constraints) {
        if (spec.constraints.min !== undefined) val = Math.max(spec.constraints.min, val)
        if (spec.constraints.max !== undefined) val = Math.min(spec.constraints.max, val)
      }
    }
    setValues(prev => ({ ...prev, [name]: val }))
  }

  return (
    <div>
      <h2 className="mb-3 text-sm font-semibold text-on-surface-variant uppercase tracking-wide">Inputs</h2>
      {specs.map(spec => (
        <div key={spec.name} className="mb-4">
          <label className="mb-1 block text-sm font-semibold text-on-surface">
            {spec.name}
            <span className="ml-2 text-xs font-normal text-on-surface-variant">
              {spec.dtype} {JSON.stringify(spec.shape)}
            </span>
          </label>
          <p className="mb-1 text-xs text-on-surface-variant">{spec.description}</p>
          {spec.dtype === 'string' ? (
            <textarea
              value={String(values[spec.name] ?? '')}
              onChange={e => set(spec.name, e.target.value)}
              rows={2}
              placeholder={spec.description}
              className="w-full rounded-lg border border-outline bg-surface-container px-3 py-2 text-sm text-on-surface transition-colors focus:border-primary focus:ring-2 focus:ring-primary/30 focus:outline-none"
            />
          ) : spec.constraints?.enum ? (
            <select
              value={String(values[spec.name] ?? '')}
              onChange={e => set(spec.name, e.target.value)}
              className="w-full rounded-lg border border-outline bg-surface-container px-3 py-2 text-sm text-on-surface transition-colors focus:border-primary focus:ring-2 focus:ring-primary/30 focus:outline-none"
            >
              {spec.constraints.enum.map((opt, i) => (
                <option key={i} value={opt} className="bg-surface-dim">
                  {spec.constraints?.items?.[i] ?? opt}
                </option>
              ))}
            </select>
          ) : spec.shape.length > 1 || spec.shape[0] > 1 ? (
            <textarea
              defaultValue={JSON.stringify(values[spec.name] ?? defaultValues([spec])[spec.name])}
              onChange={e => set(spec.name, e.target.value)}
              rows={4}
              className="w-full rounded-lg border border-outline bg-surface-container px-3 py-2 font-mono text-sm text-on-surface transition-colors focus:border-primary focus:ring-2 focus:ring-primary/30 focus:outline-none"
            />
          ) : (
            <input
              type="number"
              step={spec.dtype === 'float32' ? 'any' : '1'}
              value={Number(values[spec.name] ?? 0)}
              onChange={e => set(spec.name, e.target.value)}
              className="w-full rounded-lg border border-outline bg-surface-container px-3 py-2 text-sm text-on-surface transition-colors focus:border-primary focus:ring-2 focus:ring-primary/30 focus:outline-none"
            />
          )}
        </div>
      ))}
    </div>
  )
}
