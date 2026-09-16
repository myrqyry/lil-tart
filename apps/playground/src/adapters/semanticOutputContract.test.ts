import { describe, expect, it } from 'vitest'
import { audioAdapters } from './audio'
import { ppOcrAdapter } from './ocr'
import { textAdapters } from './text'

const adapters = [...audioAdapters, ...textAdapters, ppOcrAdapter]

const semanticStringOutputs = [
  ['moonshine-tiny', 'text'],
  ['whisper-tiny', 'text'],
  ['whisper-base', 'text'],
  ['whisper-medium', 'text'],
  ['whisper-large-v3-turbo', 'text'],
  ['crepe-pitch', 'pitch'],
  ['wav2vec2-960h', 'text'],
  ['wav2vec2-kws', 'keyword'],
  ['panns-cnn14', 'tags'],
  ['basic-pitch', 'notes'],
  ['granite-speech', 'text'],
  ['parakeet-ja', 'text'],
  ['ettin-reranker', 'scores'],
  ['lfm2.5-pii', 'entities'],
  ['lfm2.5-policy-linter', 'violations'],
  ['lfm2.5-prompt-router', 'routes'],
  ['lfm2.5-spellchecker', 'corrected'],
  ['pp-ocrv5', 'text'],
] as const

describe('semantic output metadata', () => {
  for (const [modelId, outputName] of semanticStringOutputs) {
    it(`${modelId}.${outputName} is typed as a host-decoded string`, () => {
      const adapter = adapters.find(candidate => candidate.modelId === modelId)
      expect(adapter, `missing adapter ${modelId}`).toBeDefined()

      const output = adapter?.outputSpecs.find(spec => spec.name === outputName)
      expect(output, `missing output ${modelId}.${outputName}`).toBeDefined()
      expect(output?.dtype).toBe('string')
    })
  }
})
