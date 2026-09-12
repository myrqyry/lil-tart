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

describe('HfTokenizer (granite-style: literal Replace + Split)', () => {
  const granite = new HfTokenizer({
    normalizer: { type: 'Replace', pattern: { String: ' ' }, content: '▁' },
    pre_tokenizer: { type: 'Split', pattern: { String: ' ' }, behavior: 'MergedWithPrevious', invert: false },
    model: { type: 'BPE', vocab: { '▁': 20, a: 10, b: 11, ab: 12, '▁ab': 21 }, merges: [['a', 'b'], ['▁', 'ab']] },
    post_processor: {
      type: 'TemplateProcessing',
      single: [{ SpecialToken: { id: '<bos>' } }, { Sequence: { id: 'A' } }],
      special_tokens: { '<bos>': { ids: [2] } },
    },
  })

  it('does not byte-map when the pre-tokenizer has no ByteLevel node', () => {
    expect(granite.encode('ab')).toEqual([2, 12])
  })

  it('replaces spaces with the sentinel before BPE', () => {
    expect(granite.encode('a b')).toEqual([2, 10, 20, 11])
  })
})

describe('HfTokenizer (LFM-style: regex Split + post_processor Sequence)', () => {
  const lfm = new HfTokenizer({
    normalizer: null,
    pre_tokenizer: {
      type: 'Sequence',
      pretokenizers: [
        { type: 'Split', pattern: { Regex: '\\s+|\\p{L}+' }, behavior: 'Isolated', invert: false },
        { type: 'ByteLevel', add_prefix_space: false, use_regex: false },
      ],
    },
    model: { type: 'BPE', vocab: { Ġ: 13, a: 10, b: 11, ab: 12 }, merges: [['a', 'b']] },
    post_processor: {
      type: 'Sequence',
      processors: [
        { type: 'ByteLevel', add_prefix_space: true, use_regex: true },
        {
          type: 'TemplateProcessing',
          single: [{ SpecialToken: { id: '<s>' } }, { Sequence: { id: 'A' } }],
          special_tokens: { '<s>': { ids: [1] } },
        },
      ],
    },
  })

  it('splits on the regex, byte-maps, and honors the post_processor prefix space', () => {
    expect(lfm.encode('ab')).toEqual([1, 13, 12])
  })

  it('accepts an inline (?i:) group that JavaScript cannot express', () => {
    const ci = new HfTokenizer({
      model: { type: 'BPE', vocab: { "'": 7, S: 8, s: 9, "'S": 6, "'s": 5 }, merges: [["'", 'S'], ["'", 's']] },
      pre_tokenizer: { type: 'Split', pattern: { Regex: "(?i:'s|\\p{L}+)" }, behavior: 'Isolated', invert: false },
    })
    expect(ci.encode("'S")).toEqual([6])
    expect(ci.encode("'s")).toEqual([5])
  })
})
