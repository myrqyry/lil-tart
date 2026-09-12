import { describe, expect, it } from 'vitest'
import { HfTokenizer, type HfTokenizerJson } from './hfTokenizer'

const json: HfTokenizerJson = {
  normalizer: { type: 'NFC' },
  pre_tokenizer: { type: 'ByteLevel', add_prefix_space: false, use_regex: true },
  model: { type: 'BPE', vocab: { a: 10, b: 11, ab: 12, Ġ: 13 }, merges: [['a', 'b']] },
  post_processor: {
    type: 'TemplateProcessing',
    single: [{ SpecialToken: { id: '[CLS]' } }, { Sequence: { id: 'A' } }, { SpecialToken: { id: '[SEP]' } }],
    special_tokens: { '[CLS]': { ids: [1] }, '[SEP]': { ids: [2] } },
  },
}

const tokenizer = new HfTokenizer(json)

describe('HfTokenizer', () => {
  it('applies ByteLevel byte mapping and BPE merges inside the template', () => {
    expect(tokenizer.encode('ab')).toEqual([1, 12, 2])
  })

  it('keeps a leading space as its own byte-level token', () => {
    expect(tokenizer.encode(' ab')).toEqual([1, 13, 12, 2])
  })

  it('truncates the body to the max length including template specials', () => {
    expect(tokenizer.encode('ab', { maxLength: 2 })).toEqual([1, 2])
  })

  it('pads to an explicit length', () => {
    expect(tokenizer.encode('ab', { length: 5, padId: 9 })).toEqual([1, 12, 2, 9, 9])
  })

  it('normalizes combining characters before tokenizing', () => {
    expect(tokenizer.encode('\u0065\u0301')).toEqual(tokenizer.encode('\u00e9'))
  })

  it('throws on an unsupported node type instead of guessing', () => {
    const broken = new HfTokenizer({ ...json, normalizer: { type: 'BertNormalizer' } })
    expect(() => broken.encode('ab')).toThrow(/unsupported normalizer/)
  })
})
