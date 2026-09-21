import { readFile } from 'node:fs/promises'
import { describe, expect, it } from 'vitest'

async function text(path: string): Promise<string> {
  return readFile(new URL(`../${path}`, import.meta.url), 'utf8')
}

describe('workspace package boundaries', () => {
  it('keeps examples on public package entrypoints', async () => {
    const qwen = await text('examples/minimal-qwen3-tts/main.tsx')
    const kokoro = await text('examples/minimal-kokoro/main.ts')
    expect(`${qwen}\n${kokoro}`).not.toMatch(/\.\.\/\.\/src|apps\/playground|podqast/)
  })

  it('keeps package manifests directed from generic to specific code', async () => {
    const core = await text('packages/inference-core/package.json')
    const runtime = await text('packages/runtime-litert/package.json')
    const kokoro = await text('packages/kokoro/package.json')
    const qwen = await text('packages/qwen3-tts/package.json')
    expect(core).not.toMatch(/kokoro|qwen3-tts|@litertjs\/core/)
    expect(runtime).toContain('@litert-playground/inference-core')
    expect(kokoro).toContain('@litert-playground/inference-core')
    expect(qwen).toContain('@litert-playground/inference-core')
  })

  it('keeps the playground on the shared managed runtime instead of rebuilding LiteRT plumbing', async () => {
    const runner = await text('apps/playground/src/hooks/useModelRunner.ts')
    expect(runner).toContain("from '@litert-playground/runtime-litert'")
    expect(runner).toContain('createLiteRtRuntime')
    expect(runner).not.toMatch(/\bloadLiteRt\b|\bloadAndCompile\b/)
  })

  it('keeps managed runtime policy generic and product-independent', async () => {
    const runtime = [
      await text('packages/runtime-litert/src/context.ts'),
      await text('packages/runtime-litert/src/coordinator.ts'),
      await text('packages/runtime-litert/src/types.ts'),
    ].join('\n')
    expect(runtime).not.toMatch(/podqast|episode|streamer|earthbound|\bobs\b/i)
    expect(runtime).toContain('preflight')
    expect(runtime).toContain('telemetry')
  })

  it('keeps Kokoro externally consumable through the shared core contract', async () => {
    const manifest = JSON.parse(await text('packages/kokoro/package.json')) as {
      dependencies?: Record<string, string>
      peerDependencies?: Record<string, string>
      devDependencies?: Record<string, string>
    }
    const source = await text('packages/kokoro/src/pipeline.ts')
    const entrypoint = await text('packages/kokoro/src/index.ts')

    expect(manifest.dependencies).toMatchObject({ 'kokoro-js': '^1.2.1' })
    expect(manifest.dependencies?.['@litert-playground/inference-core']).toBeUndefined()
    expect(manifest.peerDependencies).toMatchObject({ '@litert-playground/inference-core': '0.1.x' })
    expect(manifest.devDependencies).toMatchObject({ '@litert-playground/inference-core': 'workspace:*' })
    expect(source).toMatch(
      /from ['"]@litert-playground\/inference-core['"]/,
    )
    expect(entrypoint).toContain('KokoroPipeline')
    expect(entrypoint).toContain('kokoroManifest')
    expect(entrypoint).toContain('KokoroInput')
    expect(entrypoint).toContain('KokoroConfig')
  })

  it('keeps text-gen externally consumable through the shared core contract', async () => {
    const manifest = JSON.parse(await text('packages/text-gen/package.json')) as {
      dependencies?: Record<string, string>
      peerDependencies?: Record<string, string>
      devDependencies?: Record<string, string>
    }
    const pipeline = await text('packages/text-gen/src/litertlm-pipeline.ts')
    const entrypoint = await text('packages/text-gen/src/index.ts')

    expect(manifest.dependencies).toMatchObject({ '@litert-lm/core': '^0.15.0' })
    expect(manifest.dependencies?.['@litert-playground/inference-core']).toBeUndefined()
    expect(manifest.peerDependencies).toMatchObject({ '@litert-playground/inference-core': '0.1.x' })
    expect(manifest.devDependencies).toMatchObject({ '@litert-playground/inference-core': 'workspace:*' })
    expect(pipeline).toMatch(/from ['"]@litert-playground\/inference-core['"]/)
    expect(entrypoint).toContain('LiteRtLmTextPipeline')
    expect(entrypoint).toContain('litertLmManifest')
    expect(entrypoint).toContain('selectTextGenerationManifest')
    expect(entrypoint).toContain('LiteRtLmWorkerClient')
    expect(entrypoint).toContain('LiteRtLmTextConfig')
  })


  it('keeps encoder and retrieval externally consumable through the shared core contract', async () => {
    for (const packageName of ['encoder', 'retrieval']) {
      const manifest = JSON.parse(await text(`packages/${packageName}/package.json`)) as {
        dependencies?: Record<string, string>
        peerDependencies?: Record<string, string>
        devDependencies?: Record<string, string>
      }

      expect(manifest.dependencies).toMatchObject({
        '@huggingface/transformers': '^3.8.1',
      })
      expect(manifest.dependencies?.['@litert-playground/inference-core']).toBeUndefined()
      expect(manifest.peerDependencies).toMatchObject({
        '@litert-playground/inference-core': '0.1.x',
      })
      expect(manifest.devDependencies).toMatchObject({
        '@litert-playground/inference-core': 'workspace:*',
      })
    }

    const encoderEntrypoint = await text('packages/encoder/src/index.ts')
    const retrievalEntrypoint = await text('packages/retrieval/src/index.ts')
    expect(encoderEntrypoint).toContain('EncoderPipeline')
    expect(encoderEntrypoint).toContain('selectEncoderManifest')
    expect(retrievalEntrypoint).toContain('ColBertPipeline')
    expect(retrievalEntrypoint).toContain('rankColBert')
  })

  it('keeps runtime and Qwen packages externally consumable through peer contracts', async () => {
    const runtime = JSON.parse(await text('packages/runtime-litert/package.json')) as {
      dependencies?: Record<string, string>
      peerDependencies?: Record<string, string>
      devDependencies?: Record<string, string>
    }
    const qwen = JSON.parse(await text('packages/qwen3-tts/package.json')) as {
      dependencies?: Record<string, string>
      peerDependencies?: Record<string, string>
      devDependencies?: Record<string, string>
    }

    expect(runtime.dependencies?.['@litert-playground/inference-core']).toBeUndefined()
    expect(runtime.peerDependencies).toMatchObject({ '@litert-playground/inference-core': '0.1.x' })
    expect(runtime.devDependencies).toMatchObject({ '@litert-playground/inference-core': 'workspace:*' })

    expect(qwen.dependencies?.['@litert-playground/inference-core']).toBeUndefined()
    expect(qwen.dependencies?.['@litert-playground/runtime-litert']).toBeUndefined()
    expect(qwen.peerDependencies).toMatchObject({
      '@litert-playground/inference-core': '0.1.x',
      '@litert-playground/runtime-litert': '0.1.x',
    })
    expect(qwen.devDependencies).toMatchObject({
      '@litert-playground/inference-core': 'workspace:*',
      '@litert-playground/runtime-litert': 'workspace:*',
    })
    expect(qwen.dependencies?.['@litertjs/core']).toBe('^2.5.3')
  })

  it('keeps video-classification externally consumable through the shared core contract', async () => {
    const manifest = JSON.parse(await text('packages/video-classification/package.json')) as {
      dependencies?: Record<string, string>
      peerDependencies?: Record<string, string>
      devDependencies?: Record<string, string>
    }
    const pipeline = await text('packages/video-classification/src/pipeline.ts')
    const entrypoint = await text('packages/video-classification/src/index.ts')

    expect(manifest.dependencies?.['@litertjs/core']).toBe('^2.5.3')
    expect(manifest.dependencies?.['@litert-playground/inference-core']).toBeUndefined()
    expect(manifest.dependencies?.['@litert-playground/runtime-litert']).toBeUndefined()
    expect(manifest.peerDependencies).toMatchObject({
      '@litert-playground/inference-core': '0.1.x',
      '@litert-playground/runtime-litert': '0.1.x',
    })
    expect(manifest.devDependencies).toMatchObject({
      '@litert-playground/inference-core': 'workspace:*',
      '@litert-playground/runtime-litert': 'workspace:*',
    })
    expect(pipeline).toMatch(/from ['"]@litert-playground\/inference-core['"]/)
    expect(entrypoint).toContain('MoViNetPipeline')
    expect(entrypoint).toContain('moViNetManifest')
    expect(entrypoint).toContain('MoViNetInput')
    expect(entrypoint).toContain('MoViNetConfig')
  })

  it('keeps image-embedding externally consumable through the shared core contract', async () => {
    const manifest = JSON.parse(await text('packages/image-embedding/package.json')) as {
      dependencies?: Record<string, string>
      peerDependencies?: Record<string, string>
      devDependencies?: Record<string, string>
    }
    const pipeline = await text('packages/image-embedding/src/pipeline.ts')
    const entrypoint = await text('packages/image-embedding/src/index.ts')

    expect(manifest.dependencies?.['@litertjs/core']).toBe('^2.5.3')
    expect(manifest.dependencies?.['@litert-playground/inference-core']).toBeUndefined()
    expect(manifest.dependencies?.['@litert-playground/runtime-litert']).toBeUndefined()
    expect(manifest.peerDependencies).toMatchObject({
      '@litert-playground/inference-core': '0.1.x',
      '@litert-playground/runtime-litert': '0.1.x',
    })
    expect(manifest.devDependencies).toMatchObject({
      '@litert-playground/inference-core': 'workspace:*',
      '@litert-playground/runtime-litert': 'workspace:*',
    })
    expect(pipeline).toMatch(/from ['"]@litert-playground\/inference-core['"]/)
    expect(entrypoint).toContain('ClipImageEmbeddingPipeline')
    expect(entrypoint).toContain('clipImageEmbeddingManifest')
    expect(entrypoint).toContain('ClipImageInput')
    expect(entrypoint).toContain('ClipImageConfig')
  })

  it('keeps the packed consumer fixture on the full public inference surface', async () => {
    const consumer = await text('tests/fixtures/external-consumer/src/index.ts')

    for (const packageName of [
      'inference-core',
      'runtime-litert',
      'browser-cache',
      'text-gen',
      'encoder',
      'retrieval',
      'kokoro',
      'qwen3-tts',
      'image-embedding',
      'video-classification',
    ]) {
      expect(consumer).toContain(`from '@litert-playground/${packageName}'`)
    }

    expect(consumer).not.toMatch(/packages\/.*\/src/)
    expect(consumer).not.toMatch(/apps\/playground/)
  })

  it('keeps browser caching separate from core and runtime', async () => {
    const browserCacheManifest = JSON.parse(
      await text('packages/browser-cache/package.json'),
    ) as {
      dependencies?: Record<string, string>
      peerDependencies?: Record<string, string>
    }
    const browserCache = await text('packages/browser-cache/src/cache.ts')
    const core = [
      await text('packages/inference-core/src/index.ts'),
      await text('packages/inference-core/src/assets/manifest-resolver.ts'),
    ].join('\n')
    const runtime = [
      await text('packages/runtime-litert/src/context.ts'),
      await text('packages/runtime-litert/src/coordinator.ts'),
      await text('packages/runtime-litert/src/types.ts'),
    ].join('\n')

    expect(browserCacheManifest.dependencies).toBeUndefined()
    expect(browserCacheManifest.peerDependencies).toMatchObject({
      '@litert-playground/inference-core': '0.1.x',
    })
    expect(browserCache).toMatch(/from ['"]@litert-playground\/inference-core['"]/)
    expect(browserCache).toContain('createOpfsAssetStore')
    expect(core).not.toMatch(/browser-cache|navigator\.storage|FileSystemDirectoryHandle/)
    expect(runtime).not.toMatch(/browser-cache|navigator\.storage|FileSystemDirectoryHandle/)
  })
})
