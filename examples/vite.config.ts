import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'node:fs'
import path from 'node:path'
import { Readable } from 'node:stream'
import type { Connect, Plugin } from 'vite'
import {
  QWEN3_TTS_UPSTREAM_REPOSITORY,
  QWEN3_TTS_UPSTREAM_REVISION,
} from '../packages/qwen3-tts/src/manifest'

const root = path.resolve(__dirname, '..')
const modelPrefix = '/models/qwen3-tts/'
const litertWasmPrefix = '/litert-wasm/'
const litertWasmUpstream = 'https://cdn.jsdelivr.net/npm/@litertjs/core@2.5.3/wasm/'
const residencyWorkerFile = path.resolve(__dirname, 'minimal-qwen3-tts/residency-worker.js')
const generatorWorkerFile = path.resolve(__dirname, 'minimal-qwen3-tts/generator-worker.js')
const decoderWorkerFile = path.resolve(__dirname, 'minimal-qwen3-tts/decoder-worker.js')
const workerShells: Record<string, string> = {
  'residency-worker.js': residencyWorkerFile,
  'generator-worker.js': generatorWorkerFile,
  'decoder-worker.js': decoderWorkerFile,
}

function qwenModelProxy(): Plugin {
  const middleware: Connect.NextHandleFunction = async (req, res, next) => {
    const requestPath = req.url?.split('?')[0] ?? ''
    if (!requestPath.startsWith(modelPrefix)) {
      next()
      return
    }

    const encodedPath = requestPath.slice(modelPrefix.length)
    let modelPath: string
    try {
      modelPath = decodeURIComponent(encodedPath)
    } catch {
      res.statusCode = 400
      res.end('Invalid model path')
      return
    }
    if (!modelPath || modelPath.includes('..')) {
      res.statusCode = 400
      res.end('Invalid model path')
      return
    }

    const upstream = new URL(
      `https://huggingface.co/${QWEN3_TTS_UPSTREAM_REPOSITORY}/resolve/${QWEN3_TTS_UPSTREAM_REVISION}/${modelPath}`,
    )
    const headers = new Headers()
    for (const name of ['range', 'if-range', 'if-none-match', 'if-modified-since']) {
      const value = req.headers[name]
      if (value) headers.set(name, Array.isArray(value) ? value[0] : value)
    }

    try {
      const response = await fetch(upstream, {
        method: req.method === 'HEAD' ? 'HEAD' : 'GET',
        headers,
      })
      res.statusCode = response.status
      response.headers.forEach((value, name) => res.setHeader(name, value))
      if (req.method === 'HEAD' || !response.body) {
        res.end()
        return
      }
      Readable.fromWeb(response.body).pipe(res)
    } catch (cause) {
      res.statusCode = 502
      res.end(`Model proxy failed: ${String(cause)}`)
    }
  }

  return {
    name: 'qwen-model-proxy',
    configureServer(server) {
      server.middlewares.use(middleware)
    },
    configurePreviewServer(server) {
      server.middlewares.use(middleware)
    },
  }
}

function litertWasmProxy(): Plugin {
  const middleware: Connect.NextHandleFunction = async (req, res, next) => {
    const requestPath = req.url?.split('?')[0] ?? ''
    if (!requestPath.startsWith(litertWasmPrefix)) {
      next()
      return
    }

    const rest = decodeURIComponent(requestPath.slice(litertWasmPrefix.length))

    if (workerShells[rest]) {
      const file = workerShells[rest]
      res.setHeader('content-type', 'application/javascript')
      res.setHeader('cache-control', 'no-store')
      res.end(fs.readFileSync(file))
      return
    }

    if (!rest || rest.includes('..')) {
      res.statusCode = 400
      res.end('Invalid wasm path')
      return
    }

    const upstream = new URL(litertWasmUpstream + rest)
    try {
      const response = await fetch(upstream, {
        method: req.method === 'HEAD' ? 'HEAD' : 'GET',
      })
      res.statusCode = response.status
      response.headers.forEach((value, name) => {
        if (name === 'content-encoding' || name === 'content-length') return
        res.setHeader(name, value)
      })
      if (req.method === 'HEAD' || !response.body) {
        res.end()
        return
      }
      Readable.fromWeb(response.body).pipe(res)
    } catch (cause) {
      res.statusCode = 502
      res.end(`Wasm proxy failed: ${String(cause)}`)
    }
  }

  return {
    name: 'litert-wasm-proxy',
    configureServer(server) {
      server.middlewares.use(middleware)
    },
    configurePreviewServer(server) {
      server.middlewares.use(middleware)
    },
  }
}

export default defineConfig({
  root,
  plugins: [react(), qwenModelProxy(), litertWasmProxy()],
  build: {
    assetsInlineLimit: 0,
    rollupOptions: {
      input: {
        qwen3Tts: path.resolve(root, 'examples/minimal-qwen3-tts/index.html'),
        kokoro: path.resolve(root, 'examples/minimal-kokoro/index.html'),
      },
    },
  },
})
