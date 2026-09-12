import { describe, expect, it } from 'vitest'
import { computeDeltas, decodeSentencePiece, decodeUnigram, encodeWav, fft, logMelSpectrogram, makeCausalMask, melFilterbank, melFilterbankSlaney, melSpectrogram, resampleToMono, windowsOf } from './audioUtils'

describe('resampleToMono', () => {
  it('downmixes stereo to mono at the same rate', () => {
    const left = new Float32Array([1, 0, 1, 0])
    const right = new Float32Array([0, 1, 0, 1])
    expect(Array.from(resampleToMono([left, right], 16000))).toEqual([0.5, 0.5, 0.5, 0.5])
  })

  it('halves the length when downsampling 32 kHz to 16 kHz', () => {
    const mono = new Float32Array(320)
    mono.fill(1)
    expect(resampleToMono([mono], 32000).length).toBe(160)
    expect(resampleToMono([mono], 32000, 32000).length).toBe(320)
  })

  it('returns empty for empty input', () => {
    expect(resampleToMono([new Float32Array(0)], 16000).length).toBe(0)
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

describe('decodeUnigram', () => {
  const pieces = ['<unk>', '<s>', '▁', 'あ', 'い', 'う', '。']

  it('joins pieces and turns metaspace into spaces', () => {
    expect(decodeUnigram([2, 3, 4, 6], pieces)).toBe('あい。')
  })

  it('drops special tokens and trims', () => {
    expect(decodeUnigram([1, 2, 5], pieces)).toBe('う')
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

describe('fft', () => {
  it('transforms an impulse into a flat unit spectrum', () => {
    const re = new Float32Array([1, 0, 0, 0])
    const im = new Float32Array(4)
    fft(re, im)
    expect(Array.from(re).map(v => Math.round(v * 1e6) / 1e6)).toEqual([1, 1, 1, 1])
    expect(Array.from(im).map(v => Math.round(v * 1e6) / 1e6)).toEqual([0, 0, 0, 0])
  })
})

describe('logMelSpectrogram', () => {
  it('returns frames*nMels finite log values (-100 dB floor for silence)', () => {
    const basis = new Float32Array(4 * 513).fill(1)
    const logmel = logMelSpectrogram(new Float32Array(320000), basis, 4, 1024, 320, 32000)
    expect(logmel.length).toBe(1001 * 4)
    expect(logmel.every(v => v === -100)).toBe(true)
  })
})

describe('melSpectrogram', () => {
  it('honours a shorter centred window and returns raw power', () => {
    const basis = new Float32Array(4 * 257).fill(1)
    const power = melSpectrogram(new Float32Array(80000), basis, 4, 512, 160, 16000, 400, 80000)
    expect(power.length).toBe((1 + 80000 / 160) * 4)
    expect(power.every(v => v === 0)).toBe(true)
  })
})

describe('melFilterbank', () => {
  it('builds non-negative HTK triangles with one peak per mel', () => {
    const basis = melFilterbank(16000, 512, 80, 0, 8000)
    expect(basis.length).toBe(80 * 257)
    expect(basis.every(v => v >= 0 && v <= 1)).toBe(true)
    for (let m = 0; m < 80; m++) {
      const row = basis.subarray(m * 257, (m + 1) * 257)
      expect(Math.max(...row)).toBeGreaterThan(0)
    }
  })
})

describe('melFilterbankSlaney', () => {
  it('builds finite slaney-normalized triangles for the whisper front-end', () => {
    const basis = melFilterbankSlaney(16000, 400, 80, 0, 8000)
    expect(basis.length).toBe(80 * 201)
    expect(basis.every(v => Number.isFinite(v) && v >= 0)).toBe(true)
    for (let m = 0; m < 80; m++) {
      const row = basis.subarray(m * 201, (m + 1) * 201)
      expect(Math.max(...row)).toBeGreaterThan(0)
    }
  })
})

describe('computeDeltas', () => {
  it('matches torchaudio on a linear ramp (replicate edges)', () => {
    const deltas = computeDeltas(new Float32Array([0, 1, 2, 3, 4]), 1, 5, 3)
    expect(Array.from(deltas)).toEqual([0.5, 1, 1, 1, 0.5])
  })
})

describe('encodeWav', () => {
  it('writes a 16-bit mono RIFF header and clamps samples', () => {
    const wav = encodeWav(new Float32Array([0, 1, -2, 0.5]), 22050)
    const v = new DataView(wav)
    const tag = (off: number) => String.fromCharCode(v.getUint8(off), v.getUint8(off + 1), v.getUint8(off + 2), v.getUint8(off + 3))
    expect(tag(0)).toBe('RIFF')
    expect(tag(8)).toBe('WAVE')
    expect(v.getUint16(22, true)).toBe(1)
    expect(v.getUint32(24, true)).toBe(22050)
    expect(v.getUint32(40, true)).toBe(8)
    expect(v.getInt16(44, true)).toBe(0)
    expect(v.getInt16(46, true)).toBe(0x7fff)
    expect(v.getInt16(48, true)).toBe(-0x8000)
  })
})
