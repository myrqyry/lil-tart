import { createReadStream } from 'node:fs'
import { stat } from 'node:fs/promises'
import { createServer } from 'node:http'

// Separate origin so the browser applies real CORS rules. Validate before binding:
// a missing fixture rejects the case setup rather than throwing in an HTTP callback.
export async function createProbeAssetServer(file: string): Promise<{ origin: string; close(): Promise<void> }> {
  const metadata = await stat(file)
  if (!metadata.isFile()) throw new Error(`Probe asset is not a file: ${file}`)
  const server = createServer((req, res) => {
    res.setHeader('Access-Control-Allow-Origin', '*')
    if (req.url !== '/asset') {
      res.writeHead(404).end()
      return
    }
    const source = createReadStream(file)
    // The file can disappear after stat(), so handle open/read failures too.
    source.on('error', (error) => {
      if (res.headersSent) res.destroy(error)
      else {
        res.removeHeader('Content-Length')
        res.writeHead(500).end('Probe asset unavailable')
      }
    })
    res.on('close', () => source.destroy())
    res.setHeader('Content-Type', 'application/octet-stream')
    res.setHeader('Content-Length', String(metadata.size))
    source.pipe(res)
  })
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject)
    server.listen(0, '127.0.0.1', resolve)
  })
  const address = server.address()
  if (!address || typeof address === 'string') {
    server.close()
    throw new Error('Probe server did not bind a TCP address')
  }
  return {
    origin: `http://127.0.0.1:${address.port}`,
    close: () => new Promise<void>((resolve, reject) => {
      server.close(error => error ? reject(error) : resolve())
      server.closeAllConnections()
    }),
  }
}
