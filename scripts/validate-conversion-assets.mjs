import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { deepStrictEqual } from 'node:assert'

const root = resolve(import.meta.dirname, '..')
const sourceRevision = '460d52221c22388b6a9a8e8a44b61dde72976b4e'

async function readJson(path) {
  return JSON.parse(await readFile(resolve(root, path), 'utf8'))
}

function assert(condition, message) {
  if (!condition) throw new Error(message)
}

function validateRecipe(recipe, expectedGranularity, expectedAlgorithm) {
  // Full structures from the pinned revision, independent of the checked-in JSON.
  // Object key order/formatting may vary; every option and rule must remain exact.
  const rule = (operation, algorithm, numBits, granularity) => ({
    regex: '.*',
    operation,
    algorithm_key: algorithm,
    op_config: {
      weight_tensor_config: {
        num_bits: numBits,
        symmetric: true,
        granularity,
        dtype: 'INT',
      },
      compute_precision: 'INTEGER',
      explicit_dequantize: false,
      skip_checks: false,
      min_weight_elements: 0,
    },
  })
  deepStrictEqual(recipe, [
    rule('*', expectedAlgorithm, 4, expectedGranularity),
    rule('EMBEDDING_LOOKUP', 'min_max_uniform_quantize', 8, 'CHANNELWISE'),
  ], `pinned recipe contents drifted from ${sourceRevision}`)
}

const block32 = await readJson('conversion/litert-lm/recipes/int4_block32_octav.json')
const block128 = await readJson('conversion/litert-lm/recipes/int4_block128.json')
validateRecipe(block32, 'BLOCKWISE_32', 'OCTAV')
validateRecipe(block128, 'BLOCKWISE_128', 'min_max_uniform_quantize')

const catalog = await readJson('conversion/model-candidates.json')
assert(catalog.schemaVersion === 1, 'candidate catalog schemaVersion must be 1')
assert(catalog.source?.repository === 'john-rocky/LiteRT-Models', 'candidate source repository drifted')
assert(catalog.source?.revision === sourceRevision, 'candidate source revision drifted')
assert(Array.isArray(catalog.candidates) && catalog.candidates.length > 0, 'candidate catalog is empty')

const ids = new Set()
for (const candidate of catalog.candidates) {
  assert(typeof candidate.id === 'string' && candidate.id.length > 0, 'candidate id is required')
  assert(!ids.has(candidate.id), `duplicate candidate id: ${candidate.id}`)
  ids.add(candidate.id)
  assert(candidate.browserQualification === 'unverified', `${candidate.id} must not inherit Android evidence as browser qualification`)
  assert(typeof candidate.upstreamPath === 'string' && candidate.upstreamPath.length > 0, `${candidate.id} missing upstreamPath`)
  assert(typeof candidate.upstreamEvidence === 'string' && candidate.upstreamEvidence.length > 0, `${candidate.id} missing upstreamEvidence`)
}

console.log(
  `Validated ${catalog.candidates.length} conversion candidates and 2 pinned LiteRT-LM recipes from ${sourceRevision.slice(0, 12)}`,
)
