import { cp, mkdtemp, readFile, readdir, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'

const run = promisify(execFile)
const root = resolve(new URL('..', import.meta.url).pathname)
const packages = [
  ['inference-core', 'inference-core.tgz'],
  ['runtime-litert', 'runtime-litert.tgz'],
  ['browser-cache', 'browser-cache.tgz'],
  ['text-gen', 'text-gen.tgz'],
  ['encoder', 'encoder.tgz'],
  ['retrieval', 'retrieval.tgz'],
  ['kokoro', 'kokoro.tgz'],
  ['qwen3-tts', 'qwen3-tts.tgz'],
  ['image-embedding', 'image-embedding.tgz'],
  ['video-classification', 'video-classification.tgz'],
]

const tempRoot = await mkdtemp(join(tmpdir(), 'lil-tart-compat-'))

try {
  const packRoot = join(tempRoot, 'packs')
  const fixtureRoot = join(tempRoot, 'consumer')
  const archiveRoot = join(fixtureRoot, 'archives')
  await cp(join(root, 'tests/fixtures/external-consumer'), fixtureRoot, {
    recursive: true,
  })

  const rows = []
  for (const [packageDirectory, archiveName] of packages) {
    const sourceRoot = join(root, 'packages', packageDirectory)
    const destination = join(packRoot, packageDirectory)
    await run('pnpm', ['pack', '--pack-destination', destination], {
      cwd: sourceRoot,
    })
    const [packedName] = await readdir(destination)
    const archivePath = join(destination, packedName)
    const targetPath = join(archiveRoot, archiveName)
    const packageManifest = JSON.parse(
      await run('tar', ['-xOf', archivePath, 'package/package.json']).then(
        ({ stdout }) => stdout,
      ),
    )
    const publishableDependencies = {
      ...packageManifest.dependencies,
      ...packageManifest.peerDependencies,
    }
    if (Object.values(publishableDependencies).includes('workspace:*')) {
      throw new Error(`${packageManifest.name} contains workspace:* in packed metadata`)
    }
    await cp(archivePath, targetPath)
    rows.push({
      package: packageManifest.name,
      archive: archiveName,
      import: 'pending',
      peers: 'pending',
      typecheck: 'pending',
      build: 'pending',
    })
  }

  await run(
    'pnpm',
    ['install', '--ignore-workspace', '--no-frozen-lockfile', '--ignore-scripts'],
    {
      cwd: fixtureRoot,
    },
  )
  await run('pnpm', ['run', 'typecheck'], { cwd: fixtureRoot })
  await run('pnpm', ['run', 'build'], { cwd: fixtureRoot })

  for (const row of rows) {
    row.import = 'pass'
    row.peers = 'pass'
    row.typecheck = 'pass'
    row.build = 'pass'
    console.log(
      `${row.package}\t${row.archive}\t${row.import}\t${row.peers}\t${row.typecheck}\t${row.build}`,
    )
  }
} finally {
  await rm(tempRoot, { recursive: true, force: true })
}
