import { describe, expect, it } from 'vitest'
import { decodeSentencePiece, makeCausalMask, resampleToMono16k, windowsOf } from './audioUtils'

describe('resampleToMono16k', () => {
  it('downmixes stereo to mono at the same rate', () => {
    const left = new Float32Array([1, 0, 1, 0])
    const right = new Float32Array([0, 1, 0, 1])
    expect(Array.from(resampleToMono16k([left, right], 16000))).toEqual([0.5, 0.5, 0.5, 0.5])
  })

  it('halves the length when downsampling 32 kHz to 16 kHz', () => {
    const mono = new Float32Array(320)
    mono.fill(1)
    expect(resampleToMono16k([mono], 32000).length).toBe(160)
  })

  it('returns empty for empty input', () => {
    expect(resampleToMono16k([new Float32Array(0)], 16000).length).toBe(0)
  })
})

describe('makeCausalMask', () => {
  it('masks future positions and keeps the diagonal', () => {
    const mask = makeCausalMask(3)
    expect(Array.from(mask)).toEqual([0, -1e9, -1e9, 0, 0, -1e9, 0, 0, 0])
  })
})

describe('decodeSentencePiece', () => {
  const vocab = ['<unk>', '<s>', '</s>', '<0x68>', '<0x69>', '▁hello', '<0xC3>', '<0xA9>']

  it('strips the leading space and joins word pieces', () => {
    expect(decodeSentencePiece([5, 3, 4], vocab)).toBe('hellohi')
  })

  it('reassembles multi-byte UTF-8 from byte-fallback tokens', () => {
    expect(decodeSentencePiece([6, 7], vocab)).toBe('é')
  })

  it('skips unknown ids', () => {
    expect(decodeSentencePiece([5, 999], vocab)).toBe('hello')
  })
})

describe('windowsOf', () => {
  it('chunks into fixed windows with a shorter tail', () => {
    const w = windowsOf(new Float32Array(10), 4)
    expect(w.map(x => x.length)).toEqual([4, 4, 2])
  })

  it('returns a single empty window for empty audio', () => {
    expect(windowsOf(new Float32Array(0), 4)).toHaveLength(1)
  })
})
