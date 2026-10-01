import {
  InferenceError,
  type AssetRequestOptions,
  type AssetResolver,
  type ModelAsset,
  type ModelManifest,
} from '../types'

function assetFromManifest(manifest: ModelManifest, asset: ModelAsset): ModelAsset {
  // Integrity facts describe bytes at a concrete path. Semantic IDs can be
  // reused for dynamic assets (for example different TTS voices), so an ID
  // match must never make one path inherit another path's size/hash.
  return manifest.assets.find((candidate) => candidate.path === asset.path) ?? asset
}

function bytesFromStream(stream: ReadableStream<Uint8Array>): Promise<ArrayBuffer> {
  return (async () => {
    const reader = stream.getReader()
    const chunks: Uint8Array[] = []
    let total = 0
    while (true) {
      const next = await reader.read()
      if (next.done) break
      chunks.push(next.value)
      total += next.value.byteLength
    }
    const result = new Uint8Array(total)
    let offset = 0
    for (const chunk of chunks) {
      result.set(chunk, offset)
      offset += chunk.byteLength
    }
    return result.buffer
  })()
}

async function sha256(buffer: ArrayBuffer): Promise<string> {
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', buffer))
  return Array.from(digest, (value) => value.toString(16).padStart(2, '0')).join('')
}

export async function verifyAssetIntegrity(asset: ModelAsset, buffer: ArrayBuffer): Promise<ArrayBuffer> {
  if (asset.bytes !== undefined && buffer.byteLength !== asset.bytes) {
    throw new InferenceError(
      'ASSET_INTEGRITY_FAILED',
      `${asset.id}: expected ${asset.bytes} bytes, received ${buffer.byteLength}`,
      { asset: asset.id, stage: 'assets' },
    )
  }

  if (asset.sha256 !== undefined) {
    const actual = await sha256(buffer)
    if (actual.toLowerCase() !== asset.sha256.toLowerCase()) {
      throw new InferenceError(
        'ASSET_INTEGRITY_FAILED',
        `${asset.id}: expected SHA-256 ${asset.sha256}, received ${actual}`,
        { asset: asset.id, stage: 'assets' },
      )
    }
  }

  return buffer
}

export function createManifestVerifyingAssetResolver(
  manifest: ModelManifest,
  inner: AssetResolver,
): AssetResolver {
  return {
    async resolve(asset: ModelAsset, options?: AssetRequestOptions): Promise<ArrayBuffer> {
      return verifyAssetIntegrity(assetFromManifest(manifest, asset), await inner.resolve(asset, options))
    },

    stream: inner.stream
      ? async (asset: ModelAsset, options?: AssetRequestOptions): Promise<ReadableStream<Uint8Array>> => {
          const verified = await verifyAssetIntegrity(
            assetFromManifest(manifest, asset),
            await bytesFromStream(await inner.stream!(asset, options)),
          )
          return new ReadableStream({
            start(controller) {
              controller.enqueue(new Uint8Array(verified))
              controller.close()
            },
          })
        }
      : undefined,
  }
}
