import { describe, expect, it, vi } from 'vitest'
import { createElement } from 'react'
import { renderToString } from 'react-dom/server'
import type { ManagedLiteRtRuntime } from '@litert-playground/runtime-litert'
import {
  ManagedLiteRtModelController, ModelRequestCancelledError, useManagedLiteRtModel,
} from './index'

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (error: Error) => void
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no })
  return { promise, resolve, reject }
}

const modelInfo = {
  modelPath: '/test.tflite', requestedBackend: 'auto' as const,
  resolvedBackend: 'wasm' as const, fallbackCount: 0, compileDurationMs: 12,
}
function fakeRuntime() {
  const load = vi.fn().mockResolvedValue({ fake: true })
  const predict = vi.fn().mockResolvedValue([])
  const predictWithSignature = vi.fn().mockResolvedValue([])
  const getInfo = vi.fn().mockReturnValue(modelInfo)
  const dispose = vi.fn()
  const runtime = {
    loadModel: load, predict, predictWithSignature,
    getModelInfo: getInfo, dispose,
  } as unknown as ManagedLiteRtRuntime
  return { runtime, load, predict, predictWithSignature, getInfo, dispose }
}

describe('managed React LiteRT adapter', () => {
  it('starts idle during server render without loading anything', () => {
    const fake = fakeRuntime()
    function Test() {
      const result = useManagedLiteRtModel({ runtime: fake.runtime, modelPath: '/test.tflite' })
      return createElement('span', {}, result.status)
    }
    expect(renderToString(createElement(Test))).toContain('idle')
    expect(fake.load).not.toHaveBeenCalled()
  })

  it('requires explicit load and surfaces the resolved backend from the runtime', async () => {
    const fake = fakeRuntime()
    const controller = new ManagedLiteRtModelController({
      runtime: fake.runtime, modelPath: '/test.tflite', accelerator: 'auto',
    })
    const states: string[] = []
    controller.subscribe(() => { states.push(controller.getSnapshot().status) })
    expect(controller.getSnapshot().status).toBe('idle')
    expect(await controller.load()).toEqual(modelInfo)
    expect(states).toEqual(['loading', 'ready'])
    expect(fake.getInfo).toHaveBeenCalledWith('/test.tflite', { accelerator: 'auto' })
    expect(fake.load.mock.calls[0][1].signal).toBeInstanceOf(AbortSignal)
    controller.dispose()
    expect(fake.dispose).not.toHaveBeenCalled()
  })

  it('shares a single in-flight model load for duplicate clicks', async () => {
    const pending = deferred<unknown>()
    const fake = fakeRuntime()
    fake.load.mockReturnValue(pending.promise)
    const model = new ManagedLiteRtModelController({ runtime: fake.runtime, modelPath: '/test.tflite' })
    const first = model.load()
    const second = model.load()
    expect(fake.load).toHaveBeenCalledTimes(1)
    pending.resolve({})
    await Promise.all([first, second])
    model.dispose()
  })

  it('aborting an abandoned load cannot revive its stale state', async () => {
    const pending = deferred<unknown>()
    const fake = fakeRuntime()
    fake.load.mockReturnValue(pending.promise)
    const model = new ManagedLiteRtModelController({ runtime: fake.runtime, modelPath: '/test.tflite' })
    const wait = model.load()
    const forwardedSignal = fake.load.mock.calls[0][1].signal as AbortSignal
    model.cancel()
    expect(forwardedSignal.aborted).toBe(true)
    pending.resolve({})
    await expect(wait).rejects.toBeInstanceOf(ModelRequestCancelledError)
    expect(model.getSnapshot().status).toBe('idle')
    model.dispose()
  })

  it('survives React Strict Mode setup-cleanup-setup without reviving stale loads', async () => {
    const fake = fakeRuntime()
    const firstLoad = deferred<unknown>()
    fake.load.mockReturnValueOnce(firstLoad.promise).mockResolvedValueOnce({ fake: true })
    const model = new ManagedLiteRtModelController({ runtime: fake.runtime, modelPath: '/test.tflite' })
    model.retain()
    const stale = model.load().catch((e: unknown) => e)
    model.release()
    expect((fake.load.mock.calls[0][1].signal as AbortSignal).aborted).toBe(true)
    model.retain()
    await Promise.resolve()
    firstLoad.resolve({})
    expect(await stale).toBeInstanceOf(ModelRequestCancelledError)
    expect(await model.load()).toEqual(modelInfo)
    model.release()
    await Promise.resolve()
    await expect(model.load()).rejects.toBeInstanceOf(ModelRequestCancelledError)
    expect(fake.dispose).not.toHaveBeenCalled()
  })

  it('does not deliver inference results after cancellation and cleans orphan tensors', async () => {
    const pending = deferred<unknown>()
    const fake = fakeRuntime()
    const model = new ManagedLiteRtModelController({ runtime: fake.runtime, modelPath: '/test.tflite' })
    await model.load()
    fake.predict.mockReturnValue(pending.promise)
    const tensor = { delete: vi.fn() }
    const wait = model.run([]).catch((e: unknown) => e)
    model.cancel()
    expect((fake.predict.mock.calls[0][2].signal as AbortSignal).aborted).toBe(true)
    pending.resolve([tensor])
    expect(await wait).toBeInstanceOf(ModelRequestCancelledError)
    expect(tensor.delete).toHaveBeenCalledOnce()
    model.dispose()
    expect(fake.dispose).not.toHaveBeenCalled()
  })

  it('preserves real model inference failure for display, without pretending success', async () => {
    const fake = fakeRuntime()
    fake.load.mockRejectedValue(new Error('WASM unavailable'))
    const model = new ManagedLiteRtModelController({ runtime: fake.runtime, modelPath: '/test.tflite' })
    await expect(model.load()).rejects.toThrow('WASM unavailable')
    expect(model.getSnapshot().status).toBe('error')
    expect(model.getSnapshot().error?.message).toBe('WASM unavailable')
    model.dispose()
  })

  it('routes signature calls through the managed runtime with a cancellation signal', async () => {
    const fake = fakeRuntime()
    const model = new ManagedLiteRtModelController({ runtime: fake.runtime, modelPath: '/test.tflite' })
    await model.load()
    await model.run([], { signature: 'embed', label: 'test' })
    expect(fake.predictWithSignature.mock.calls[0].slice(0, 3)).toEqual(['/test.tflite', 'embed', []])
    expect(fake.predictWithSignature.mock.calls[0][3].label).toBe('test')
    model.dispose()
  })
})
