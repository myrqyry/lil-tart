import { spawnSync } from 'node:child_process'
import { cp, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { strict as assert } from 'node:assert'
import { test } from 'node:test'

const root = resolve(import.meta.dirname, '..')

async function validateMutation(t, mutate) {
  const fixture = await mkdtemp(join(tmpdir(), 'lil-tart-conversion-'))
  t.after(() => rm(fixture, { recursive: true, force: true }))
  await mkdir(join(fixture, 'scripts'))
  await cp(join(root, 'scripts/validate-conversion-assets.mjs'), join(fixture, 'scripts/validate-conversion-assets.mjs'))
  await cp(join(root, 'conversion'), join(fixture, 'conversion'), { recursive: true })
  for (const name of ['int4_block32_octav.json', 'int4_block128.json']) {
    const path = join(fixture, 'conversion/litert-lm/recipes', name)
    const recipe = JSON.parse(await readFile(path, 'utf8'))
    mutate(recipe)
    await writeFile(path, JSON.stringify(recipe))
    const result = spawnSync(process.execPath, [join(fixture, 'scripts/validate-conversion-assets.mjs')], { encoding: 'utf8' })
    assert.equal(result.status, 1, `${name} must reject recipe drift: ${result.stdout}`)
    assert.match(result.stderr, /Error/)
    // Keep the other recipe pristine so each rejection identifies this file.
    await cp(join(root, 'conversion/litert-lm/recipes', name), path)
  }
}

test('the checked-in recipes and candidate boundaries pass', () => {
  const result = spawnSync(process.execPath, [join(root, 'scripts/validate-conversion-assets.mjs')], { encoding: 'utf8' })
  assert.equal(result.status, 0, result.stderr)
  assert.match(result.stdout, /10 conversion candidates and 2 pinned/)
})

for (const [name, mutate] of [
  ['asymmetric weights', recipe => { recipe[0].op_config.weight_tensor_config.symmetric = false }],
  ['float compute precision', recipe => { recipe[0].op_config.compute_precision = 'FLOAT' }],
  ['explicit dequantization', recipe => { recipe[1].op_config.explicit_dequantize = true }],
  ['disabled checks', recipe => { recipe[0].op_config.skip_checks = true }],
  ['changed minimum weight size', recipe => { recipe[1].op_config.min_weight_elements = 1 }],
  ['changed regex', recipe => { recipe[0].regex = 'decoder.*' }],
  ['changed embedding algorithm', recipe => { recipe[1].algorithm_key = 'OCTAV' }],
  ['changed dtype', recipe => { recipe[1].op_config.weight_tensor_config.dtype = 'FLOAT' }],
  ['missing property', recipe => { delete recipe[0].op_config.compute_precision }],
  ['extra property', recipe => { recipe[0].op_config.extra = true }],
  ['third rule', recipe => { recipe.push(structuredClone(recipe[0])) }],
  ['missing rule', recipe => { recipe.pop() }],
  ['reordered rules', recipe => { recipe.reverse() }],
]) {
  test(`rejects ${name} in either pinned recipe`, t => validateMutation(t, mutate))
}

test('JSON formatting and object key order do not change recipe semantics', async t => {
  const fixture = await mkdtemp(join(tmpdir(), 'lil-tart-conversion-'))
  t.after(() => rm(fixture, { recursive: true, force: true }))
  await mkdir(join(fixture, 'scripts'))
  await cp(join(root, 'scripts/validate-conversion-assets.mjs'), join(fixture, 'scripts/validate-conversion-assets.mjs'))
  await cp(join(root, 'conversion'), join(fixture, 'conversion'), { recursive: true })
  const path = join(fixture, 'conversion/litert-lm/recipes/int4_block32_octav.json')
  const recipe = JSON.parse(await readFile(path, 'utf8'))
  recipe[0] = Object.fromEntries(Object.entries(recipe[0]).reverse())
  await writeFile(path, JSON.stringify(recipe))
  const result = spawnSync(process.execPath, [join(fixture, 'scripts/validate-conversion-assets.mjs')], { encoding: 'utf8' })
  assert.equal(result.status, 0, result.stderr)
})
