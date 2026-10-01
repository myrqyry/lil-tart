import {
  InferenceError,
  type AssetRequestOptions,
  type AssetResolver,
  type ModelAsset,
  type ModelManifest,
} from '../types'

export interface AssetIntegrityVerificationOptions {
  /**
   * Largest artifact that may be fully buffered for SHA-256 verification.
   * Undefined keeps strict hashing for every declared SHA-256.
   */
  maxSha256Bytes?: number
}

function assetFromManifest(manifest: ModelManifest, asset: ModelAsset): ModelAsset {
  // Integrity facts describe bytes at a concrete path. Semantic IDs can be
  // reused for dynamic assets (for example different TTS voices), so an ID
  // match must never make one path inherit another path's size/hash.
  return manifest.assets.find((candidate) => candidate.path === asset.path) ?? asset
}

function shouldVerifySha256(
  asset: ModelAsset,
  actualBytes: number | undefined,
  options: AssetIntegrityVerificationOptions,
): boolean {
  if (asset.sha256 === undefined) return false
  const max = options.maxSha256Bytes
  if (max === undefined) return true
  const bytes = asset.bytes ?? actualBytes
  return bytes === undefined || bytes <= max
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

function lengthVerifyingStream(
  asset: ModelAsset,
  stream: ReadableStream<Uint8Array>,
): ReadableStream<Uint8Array> {
  const expectedBytes = asset.bytes
  if (expectedBytes === undefined) return stream

  const reader = stream.getReader()
  let total = 0
  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      try {
        const next = await reader.read()
        if (next.done) {
          if (total !== expectedBytes) {
            controller.error(new InferenceError(
              'ASSET_INTEGRITY_FAILED',
              `${asset.id}: expected ${expectedBytes} bytes, received ${total}`,
              { asset: asset.id, stage: 'assets' },
            ))
            return
          }
          controller.close()
          return
        }
        total += next.value.byteLength
        if (total > expectedBytes) {
          await reader.cancel('asset exceeds declared byte length')
          controller.error(new InferenceError(
            'ASSET_INTEGRITY_FAILED',
            `${asset.id}: expected ${expectedBytes} bytes, received more than ${expectedBytes}`,
            { asset: asset.id, stage: 'assets' },
          ))
          return
        }
        controller.enqueue(next.value)
      } catch (error) {
        controller.error(error)
      }
    },
    cancel(reason) {
      return reader.cancel(reason)
    },
  })
}

async function sha256(buffer: ArrayBuffer): Promise<string> {
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', buffer))
  return Array.from(digest, (value) => value.toString(16).padStart(2, '0')).join('')
}

export async function verifyAssetIntegrity(
  asset: ModelAsset,
  buffer: ArrayBuffer,
  options: AssetIntegrityVerificationOptions = {},
): Promise<ArrayBuffer> {
  if (asset.bytes !== undefined && buffer.byteLength !== asset.bytes) {
    throw new InferenceError(
      'ASSET_INTEGRITY_FAILED',
      `${asset.id}: expected ${asset.bytes} bytes, received ${buffer.byteLength}`,
      { asset: asset.id, stage: 'assets' },
    )
  }

  if (shouldVerifySha256(asset, buffer.byteLength, options)) {
    const actual = await sha256(buffer)
    if (actual.toLowerCase() !== asset.sha256!.toLowerCase()) {
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
  verificationOptions: AssetIntegrityVerificationOptions = {},
): AssetResolver {
  return {
    async resolve(asset: ModelAsset, options?: AssetRequestOptions): Promise<ArrayBuffer> {
      return verifyAssetIntegrity(
        assetFromManifest(manifest, asset),
        await inner.resolve(asset, options),
        verificationOptions,
      )
    },

    stream: inner.stream
      ? async (asset: ModelAsset, options?: AssetRequestOptions): Promise<ReadableStream<Uint8Array>> => {
          const manifestAsset = assetFromManifest(manifest, asset)
          const stream = await inner.stream!(asset, options)
          if (!shouldVerifySha256(manifestAsset, undefined, verificationOptions)) {
            // Large artifacts can keep streaming without a second full buffer;
            // declared byte length is still enforced as chunks pass through.
            return lengthVerifyingStream(manifestAsset, stream)
          }

          const verified = await verifyAssetIntegrity(
            manifestAsset,
            await bytesFromStream(stream),
            verificationOptions,
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
