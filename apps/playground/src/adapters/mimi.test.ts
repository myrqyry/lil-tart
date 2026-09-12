import { describe, expect, it } from 'vitest'
import { parseRvq, rvqDecode, rvqEncode, transposeFeature, type RvqTables } from './mimi'

describe('parseRvq', () => {
  it('splits the flat buffer in the documented order', () => {
    const data = new Float32Array([
      1, 0, 0, 1, // semWin
      2, 0, 0, 1, // acoWin
      3, 0, 0, 1, // semWout
      4, 0, 0, 1, // acoWout
      5, 0, 0, 1, // semCb
      6, 0, 0, 1, // acoCb[0]
    ])
    const tables = parseRvq(data, 1, 2, 2, 2)
    expect(tables.semWin[0]).toBe(1)
    expect(tables.acoWin[0]).toBe(2)
    expect(tables.semWout[0]).toBe(3)
    expect(tables.acoWout[0]).toBe(4)
    expect(tables.semCb[0]).toBe(5)
    expect(tables.acoCb[0][0]).toBe(6)
  })
})

describe('split RVQ', () => {
  const identity = new Float32Array([1, 0, 0, 1])
  const tables: RvqTables = {
    dim: 2,
    size: 2,
    hidden: 2,
    semWin: identity,
    acoWin: identity,
    semWout: identity,
    acoWout: identity,
    semCb: identity,
    acoCb: [identity],
  }

  it('encodes a residual to its nearest codebook rows', () => {
    expect(Array.from(rvqEncode(new Float32Array([1, 0]), 2, 1, tables))).toEqual([0, 0])
  })

  it('decodes codes back through the quantizer matrices', () => {
    expect(Array.from(rvqDecode(new Int32Array([0, 0]), 1, tables))).toEqual([2, 0])
  })
})

describe('transposeFeature', () => {
  it('moves channels to the last axis', () => {
    expect(Array.from(transposeFeature(new Float32Array([1, 2, 3, 4]), 2, 2))).toEqual([1, 3, 2, 4])
  })
})
