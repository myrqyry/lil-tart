import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'

const root = resolve(import.meta.dirname, '..')
const sourceRevision = '460d52221c22388b6a9a8e8a44b61dde72976b4e'

async function readJson(path) {
  return JSON.parse(await readFile(resolve(root, path), 'utf8'))
}

function assert(condition, message) {
  if (!condition) throw new Error(message)
}

function validateRecipe(recipe, expectedGranularity, expectedAlgorithm) {
  assert(Array.isArray(recipe) && recipe.length >= 2, 'recipe must contain weight and embedding rules')
  const [weights, embedding] = recipe
  const weightConfig = weights?.op_config?.weight_tensor_config
  const embeddingConfig = embedding?.op_config?.weight_tensor_config

  assert(weights?.operation === '*', 'first recipe rule must target all operations')
  assert(weights?.algorithm_key === expectedAlgorithm, `unexpected recipe algorithm: ${weights?.algorithm_key}`)
  assert(weightConfig?.num_bits === 4, 'general weights must remain int4')
  assert(weightConfig?.granularity === expectedGranularity, `expected ${expectedGranularity}`)
  assert(embedding?.operation === 'EMBEDDING_LOOKUP', 'embedding rule must target EMBEDDING_LOOKUP')
  assert(embeddingConfig?.num_bits === 8, 'embedding weights must remain int8')
  assert(embeddingConfig?.granularity === 'CHANNELWISE', 'embedding weights must remain channelwise')
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
