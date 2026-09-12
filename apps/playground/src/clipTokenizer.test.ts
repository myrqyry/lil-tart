import { describe, expect, it } from 'vitest'
import { ClipTokenizer, type ClipTokenizerJson } from './clipTokenizer'

const json: ClipTokenizerJson = {
  model: {
    vocab: { '<|startoftext|>': 49406, '<|endoftext|>': 49407, a: 1, b: 2, ab: 3, ac: 4, c: 5 },
    merges: ['a b', 'a c'],
  },
  added_tokens: [
    { content: '<|startoftext|>', id: 49406 },
    { content: '<|endoftext|>', id: 49407 },
  ],
}

describe('ClipTokenizer', () => {
  const tokenizer = new ClipTokenizer(json)

  it('applies byte-pair merges', () => {
    expect(tokenizer.encode('ab', 5)).toEqual([49406, 3, 49407, 49407, 49407])
  })

  it('splits on whitespace and drops it', () => {
    expect(tokenizer.encode('a b', 5)).toEqual([49406, 1, 2, 49407, 49407])
  })

  it('lowercases and collapses whitespace runs', () => {
    expect(tokenizer.encode('  AB ', 5)).toEqual([49406, 3, 49407, 49407, 49407])
  })

  it('falls back to characters when no merge applies', () => {
    expect(tokenizer.encode('ba', 5)).toEqual([49406, 2, 1, 49407, 49407])
  })

  it('passes special tokens through without byte-encoding', () => {
    expect(tokenizer.encode('<|startoftext|>', 4)).toEqual([49406, 49406, 49407, 49407])
  })

  it('pads and truncates to the requested length', () => {
    expect(tokenizer.encode('a', 8)).toHaveLength(8)
    expect(tokenizer.encode('ab', 2)).toEqual([49406, 49407])
  })
})
