import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { RuntimeContext } from '@litert-playground/inference-core';
import { TransformersTextPipeline } from "./transformers-pipeline";
import { LiteRtLmTextPipeline } from "./litertlm-pipeline";
import {
  gemma4E2bManifest,
  gemma4E4bManifest,
  lfm2_5InstructManifest,
  lfm2_5InstructInt8Manifest,
  lfm2_5ThinkingManifest,
  lfm2_5ThinkingInt8Manifest,
  litertLmManifest,
  selectTextGenerationManifest,
} from "./manifest";

const { mockPipeline, mockEngineCreate, mockCreateConversation, mockSendMessageStreaming } = vi.hoisted(() => ({
  mockPipeline: vi.fn(),
  mockEngineCreate: vi.fn(),
  mockCreateConversation: vi.fn(),
  mockSendMessageStreaming: vi.fn(),
}));

vi.mock("@huggingface/transformers", () => ({
  pipeline: mockPipeline,
}));

vi.mock("@litert-lm/core", () => ({
  Engine: {
    create: mockEngineCreate,
  },
}));

function fakeContext(): RuntimeContext {
  return {
    backend: "wasm",
    assets: {} as RuntimeContext["assets"],
    liteRt: {} as RuntimeContext["liteRt"],
  };
}

// A model fetch stub that hands back a real response body, which is what the
// pipeline must pass to the engine in place of a bare URL.
function stubFetch() {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new Uint8Array([1]));
      controller.close();
    },
  });
  // Behaves like native fetch: an already-aborted signal rejects immediately rather
  // than resolving, which is the whole point of threading the signal through.
  const mock = vi.fn((_url: unknown, init?: { signal?: AbortSignal }) => {
    if (init?.signal?.aborted) {
      return Promise.reject(new DOMException("aborted", "AbortError"));
    }
    return Promise.resolve({ ok: true, status: 200, body });
  });
  const original = globalThis.fetch;
  globalThis.fetch = mock as unknown as typeof fetch;
  afterEach(() => {
    globalThis.fetch = original;
  });
  return mock;
}

// Waits until `predicate` holds, so a dispose can be timed to land while the call
// under test is genuinely in flight rather than before it starts.
async function waitFor(predicate: () => boolean, label: string): Promise<void> {
  for (let i = 0; i < 100; i++) {
    if (predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  throw new Error(`timed out waiting for ${label}`);
}

function streamOf(chunks: string[]): ReadableStream<{ text: string }> {  return new ReadableStream({
    start(controller) {
      for (const c of chunks) controller.enqueue({ text: c });
      controller.close();
    },
  });
}

beforeEach(() => {
  mockPipeline.mockReset();
  mockEngineCreate.mockReset();
  mockCreateConversation.mockReset();
  mockSendMessageStreaming.mockReset();

  mockPipeline.mockResolvedValue(async () => ({
    output: [{ generated_text: "Hello from transformers" }],
  }));

  mockCreateConversation.mockResolvedValue({
    sendMessage: async () => ({ text: "" }),
    sendMessageStreaming: mockSendMessageStreaming,
    cancel: () => {},
    delete: async () => {},
  });

  mockEngineCreate.mockResolvedValue({
    createConversation: mockCreateConversation,
    delete: async () => {},
  });

  mockSendMessageStreaming.mockReturnValue(
    streamOf(["Hel", "lo", " from ", "litert-lm"])
  );
});

describe("TransformersTextPipeline", () => {
  it("throws on run before load", async () => {
    const p = new TransformersTextPipeline();
    await expect(
      p.run({ messages: [{ role: "user", content: "hi" }] })
    ).rejects.toThrow("Pipeline not ready");
  });

  it("has correct manifest", () => {
    const p = new TransformersTextPipeline();
    expect(p.manifest.capabilities).toContain("text-generation");
  });

  it("generates text via pipeline", async () => {
    const p = new TransformersTextPipeline();
    await p.load(fakeContext());

    const result = await p.run(
      { systemPrompt: "Be brief.", messages: [{ role: "user", content: "hi" }] },
      { modelId: "onnx-community/Qwen3-0.6B-ONNX", device: "wasm" }
    );

    expect(result.kind).toBe("text");
    expect(result.text).toBe("Hello from transformers");
    expect(mockPipeline).toHaveBeenCalledWith(
      "text-generation",
      "onnx-community/Qwen3-0.6B-ONNX",
      expect.objectContaining({ device: "wasm" })
    );
  });

  it("honors an aborted signal", async () => {
    const p = new TransformersTextPipeline();
    await p.load(fakeContext());

    const controller = new AbortController();
    controller.abort();

    await expect(
      p.run(
        { messages: [{ role: "user", content: "x" }] },
        { modelId: "onnx-community/Qwen3-0.6B-ONNX", device: "wasm" },
        controller.signal
      )
    ).rejects.toThrow("CANCELLED");
  });
});

describe("LiteRtLmTextPipeline", () => {
  it("throws on run before load", async () => {
    const p = new LiteRtLmTextPipeline();
    await expect(
      p.run({ messages: [{ role: "user", content: "hi" }] })
    ).rejects.toThrow("Pipeline not ready");
  });

  it("has correct manifest", () => {
    const p = new LiteRtLmTextPipeline();
    expect(p.manifest.capabilities).toContain("text-generation");
  });

  it("replays history then streams the prompt", async () => {
    const p = new LiteRtLmTextPipeline();
    await p.load(fakeContext());

    const result = await p.run(
      {
        systemPrompt: "Be brief.",
        messages: [
          { role: "user", content: "hi" },
          { role: "assistant", content: "yo" },
        ],
      },
      { model: "litert-community/Qwen3-0.6B/resolve/main/Qwen3-0.6B.litertlm" }
    );

    expect(result.kind).toBe("text");
    expect(result.text).toBe("Hello from litert-lm");
    expect(mockEngineCreate).toHaveBeenCalledWith(
      expect.objectContaining({
        mainExecutorSettings: { maxNumTokens: 4096 },
      })
    );
    expect(mockSendMessageStreaming).toHaveBeenCalled();
  });

  it("passes tools through the preface and forwards streamed tool calls", async () => {
    const p = new LiteRtLmTextPipeline();
    await p.load(fakeContext());
    const onToolCall = vi.fn();
    mockSendMessageStreaming.mockReturnValue(
      new ReadableStream({
        start(controller) {
          controller.enqueue({
            tool_calls: [
              {
                type: "function",
                function: { name: "calculate", arguments: { expression: "2+2" } },
              },
            ],
          });
          controller.close();
        },
      }),
    );

    await p.run(
      { systemPrompt: "Use tools when needed.", messages: [{ role: "user", content: "2+2?" }] },
      {
        tools: [
          {
            name: "calculate",
            description: "Evaluate a mathematical expression",
            parameters: {
              type: "object",
              properties: { expression: { type: "string" } },
              required: ["expression"],
            },
          },
        ],
        onToolCall,
      },
    );

    expect(mockCreateConversation).toHaveBeenCalledWith(
      expect.objectContaining({
        preface: expect.objectContaining({
          messages: [{ role: "system", content: "Use tools when needed." }],
          tools: [expect.objectContaining({ name: "calculate" })],
        }),
      }),
    );
    expect(onToolCall).toHaveBeenCalledWith([
      { name: "calculate", arguments: { expression: "2+2" } },
    ]);
  });

  it("batches complete tool calls until stream completion and normalizes JSON-string arguments", async () => {
    const onToolCall = vi.fn();
    let streamController!: ReadableStreamDefaultController<{ tool_calls?: unknown[] }>;
    mockSendMessageStreaming.mockReturnValue(
      new ReadableStream<{ tool_calls?: unknown[] }>({
        start(controller) {
          streamController = controller;
        },
      }),
    );

    const p = new LiteRtLmTextPipeline();
    await p.load(fakeContext());
    const running = p.run(
      { messages: [{ role: "user", content: "2+2?" }] },
      {
        tools: [
          {
            name: "calculate",
            description: "Evaluate a mathematical expression",
            parameters: {
              type: "object",
              properties: { expression: { type: "string" } },
              required: ["expression"],
            },
          },
        ],
        onToolCall,
      },
    );

    streamController.enqueue({
      tool_calls: [
        {
          type: "function",
          function: {
            name: "calculate",
            arguments: '{"expression":"2+2"}',
          },
        },
      ],
    });
    await Promise.resolve();
    expect(onToolCall).not.toHaveBeenCalled();

    streamController.enqueue({
      tool_calls: [
        {
          type: "function",
          function: {
            name: "lookup",
            arguments: { topic: "math" },
          },
        },
      ],
    });
    streamController.close();
    await running;

    expect(mockCreateConversation).toHaveBeenCalledWith(
      expect.objectContaining({
        preface: {
          tools: [expect.objectContaining({ name: "calculate" })],
        },
      }),
    );
    expect(onToolCall).toHaveBeenCalledTimes(1);
    expect(onToolCall).toHaveBeenCalledWith([
      { name: "calculate", arguments: { expression: "2+2" } },
      { name: "lookup", arguments: { topic: "math" } },
    ]);
  });

  it("rejects malformed streamed tool calls instead of dropping them silently", async () => {
    mockSendMessageStreaming.mockReturnValue(
      new ReadableStream({
        start(controller) {
          controller.enqueue({
            tool_calls: [
              {
                type: "function",
                function: { name: "", arguments: {} },
              },
            ],
          });
          controller.close();
        },
      }),
    );

    const p = new LiteRtLmTextPipeline();
    await p.load(fakeContext());

    await expect(
      p.run(
        { messages: [{ role: "user", content: "use the tool" }] },
        { onToolCall: vi.fn() },
      ),
    ).rejects.toMatchObject({ code: "OUTPUT_INVALID" });
  });

  it("rejects a non-array tool_calls payload at the runtime boundary", async () => {
    mockSendMessageStreaming.mockReturnValue(
      new ReadableStream({
        start(controller) {
          controller.enqueue({ tool_calls: "not-an-array" });
          controller.close();
        },
      }),
    );

    const p = new LiteRtLmTextPipeline();
    await p.load(fakeContext());

    await expect(
      p.run(
        { messages: [{ role: "user", content: "use the tool" }] },
        { onToolCall: vi.fn() },
      ),
    ).rejects.toMatchObject({ code: "OUTPUT_INVALID" });
  });

  it("does not emit buffered tool calls from an aborted turn", async () => {
    let streamController!: ReadableStreamDefaultController<{
      tool_calls?: unknown;
      text?: string;
    }>;
    let waitingForNextChunk = false;
    mockSendMessageStreaming.mockReturnValue(
      new ReadableStream({
        start(controller) {
          streamController = controller;
          controller.enqueue({
            tool_calls: [
              {
                type: "function",
                function: { name: "calculate", arguments: { expression: "2+2" } },
              },
            ],
          });
        },
        pull() {
          waitingForNextChunk = true;
        },
      }),
    );

    const controller = new AbortController();
    const onToolCall = vi.fn();
    const p = new LiteRtLmTextPipeline();
    await p.load(fakeContext());
    const running = p.run(
      { messages: [{ role: "user", content: "2+2?" }] },
      { onToolCall },
      controller.signal,
    );

    await waitFor(() => waitingForNextChunk, "stream to wait for its next chunk");
    controller.abort();
    streamController.enqueue({ text: "" });
    streamController.close();

    await expect(running).rejects.toThrow("CANCELLED");
    expect(onToolCall).not.toHaveBeenCalled();
  });

  it("releases each completed conversation before the next run", async () => {
    const deletes: Array<ReturnType<typeof vi.fn>> = [];
    const createConversation = vi.fn(async () => {
      const deleteConversation = vi.fn(async () => undefined);
      deletes.push(deleteConversation);
      return {
        sendMessage: async () => ({ text: "" }),
        sendMessageStreaming: mockSendMessageStreaming,
        cancel: vi.fn(),
        delete: deleteConversation,
      };
    });
    mockEngineCreate.mockResolvedValue({
      createConversation,
      delete: vi.fn(async () => undefined),
    });

    const p = new LiteRtLmTextPipeline();
    await p.load(fakeContext());
    await p.run({ messages: [{ role: "user", content: "first" }] });
    await p.run({ messages: [{ role: "user", content: "second" }] });

    expect(createConversation).toHaveBeenCalledTimes(2);
    expect(deletes).toHaveLength(2);
    expect(deletes[0]).toHaveBeenCalledTimes(1);
    expect(deletes[1]).toHaveBeenCalledTimes(1);
    expect(p).toHaveProperty("conversation", null);
  });

  it("loads the manifest's model path instead of a hardcoded default", async () => {
    const p = new LiteRtLmTextPipeline(lfm2_5ThinkingManifest);
    await p.load(fakeContext());

    expect(mockEngineCreate).toHaveBeenCalledWith(
      expect.objectContaining({
        model:
          "litert-community/LFM2.5-1.2B-Thinking/resolve/1b1e49ad9dccdededc9d03bc0fe3071d83595d75/LFM2.5-1.2B-Thinking_int4.litertlm",
        backend: "wasm",
      })
    );
  });

  it("accepts a caller-owned model input without fetching it again", async () => {
    const original = globalThis.fetch;
    const fetchMock = vi.fn();
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const cachedModel = new Blob([new Uint8Array([1, 2, 3])]);

    try {
      const p = new LiteRtLmTextPipeline(lfm2_5ThinkingManifest, {
        model: cachedModel,
        maxContextTokens: 8192,
      });
      await p.loadForBackend("webgpu");

      expect(fetchMock).not.toHaveBeenCalled();
      expect(mockEngineCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          model: cachedModel,
          backend: "webgpu",
          mainExecutorSettings: { maxNumTokens: 8192 },
        })
      );
    } finally {
      globalThis.fetch = original;
    }
  });

  it("loads from a backend without requiring unrelated runtime services", async () => {
    const p = new LiteRtLmTextPipeline(lfm2_5ThinkingManifest);
    await p.loadForBackend("wasm");

    expect(p.status).toBe("ready");
    expect(mockEngineCreate).toHaveBeenCalledWith(
      expect.objectContaining({ backend: "wasm" })
    );
  });

  it("rejects a concurrent load instead of racing two native engines", async () => {
    let releaseEngine!: () => void;
    const engineReady = new Promise<void>((resolve) => {
      releaseEngine = resolve;
    });
    mockEngineCreate.mockImplementationOnce(async () => {
      await engineReady;
      return {
        createConversation: mockCreateConversation,
        delete: vi.fn(async () => undefined),
      };
    });

    const p = new LiteRtLmTextPipeline(lfm2_5ThinkingManifest);
    const firstLoad = p.loadForBackend("wasm");

    await vi.waitFor(() => expect(p.status).toBe("loading"));
    await vi.waitFor(() => expect(mockEngineCreate).toHaveBeenCalledTimes(1));
    await expect(p.loadForBackend("wasm")).rejects.toMatchObject({
      code: "INVALID_INPUT",
    });
    expect(mockEngineCreate).toHaveBeenCalledTimes(1);

    releaseEngine();
    await expect(firstLoad).resolves.toBeUndefined();
    expect(p.status).toBe("ready");
    expect(mockEngineCreate).toHaveBeenCalledTimes(1);
  });

  it("lets RuntimeContext WebNN degrade to the engine's supported backend selection", async () => {
    const p = new LiteRtLmTextPipeline(lfm2_5ThinkingManifest);
    await p.load({ ...fakeContext(), backend: "webnn" });

    expect(mockEngineCreate).toHaveBeenCalledWith(
      expect.objectContaining({ backend: undefined }),
    );
  });

  it("rejects explicit WebNN requests through the backend-only entrypoint", async () => {
    const p = new LiteRtLmTextPipeline(lfm2_5ThinkingManifest);
    await expect(
      p.loadForBackend("webnn" as never),
    ).rejects.toMatchObject({ code: "BACKEND_UNAVAILABLE" });
    expect(mockEngineCreate).not.toHaveBeenCalled();
  });

  it("requests a fresh caller-owned stream from the factory on retry", async () => {
    const sources: ReadableStream<Uint8Array>[] = [];
    const modelFactory = vi.fn(() => {
      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(new Uint8Array([sources.length + 1]));
          controller.close();
        },
      });
      sources.push(stream);
      return stream;
    });
    mockEngineCreate
      .mockRejectedValueOnce(new Error("compile failed"))
      .mockResolvedValueOnce({
        createConversation: mockCreateConversation,
        delete: vi.fn(async () => undefined),
      });

    const p = new LiteRtLmTextPipeline(lfm2_5ThinkingManifest, {
      model: modelFactory,
    });

    await expect(p.loadForBackend("wasm")).rejects.toThrow("compile failed");
    await expect(p.loadForBackend("wasm")).resolves.toBeUndefined();

    expect(modelFactory).toHaveBeenCalledTimes(2);
    expect(sources).toHaveLength(2);
    expect(mockEngineCreate.mock.calls[0][0].model).toBe(sources[0]);
    expect(mockEngineCreate.mock.calls[1][0].model).toBe(sources[1]);
  });

  // The manifest path is repo-relative. Handing it to the engine raw makes the
  // browser resolve it against the app origin, which 404s, so it has to be made
  // absolute. The pipeline hands the response body to the engine: routing through
  // context.assets would materialize the whole model into an ArrayBuffer first,
  // which is the wrong memory trade for a 1.2B-parameter model in a browser.
  it("fetches the absolute URL and hands the response stream to the engine", async () => {
    const fetchMock = stubFetch();
    const p = new LiteRtLmTextPipeline(lfm2_5ThinkingManifest, {
      modelBase: "https://huggingface.co/",
    });
    await p.load(fakeContext());

    expect(fetchMock).toHaveBeenCalledWith(
      "https://huggingface.co/litert-community/LFM2.5-1.2B-Thinking/resolve/1b1e49ad9dccdededc9d03bc0fe3071d83595d75/LFM2.5-1.2B-Thinking_int4.litertlm",
      expect.objectContaining({ credentials: "same-origin" })
    );

    // The engine must receive the response body, not the URL, so the transfer is ours.
    const passed = mockEngineCreate.mock.calls[0][0].model;
    expect(passed).toBeInstanceOf(ReadableStream);
    expect(typeof passed).not.toBe("string");
  });

  it("never buffers the checkpoint through the asset resolver", async () => {
    const fetchMock = stubFetch();
    const resolve = vi.fn();
    const stream = vi.fn();
    const context: RuntimeContext = {
      ...fakeContext(),
      assets: { resolve, stream } as unknown as RuntimeContext["assets"],
    };

    const p = new LiteRtLmTextPipeline(lfm2_5ThinkingManifest, {
      modelBase: "https://huggingface.co/",
    });
    await p.load(context);

    // Buffering here is the regression this test exists to prevent.
    expect(resolve).not.toHaveBeenCalled();
    expect(stream).not.toHaveBeenCalled();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("cancels the model download when the caller's signal aborts", async () => {
    const controller = new AbortController();
    const seenSignal = { current: null as AbortSignal | null };
    const original = globalThis.fetch;
    globalThis.fetch = vi.fn((_url: unknown, init?: { signal?: AbortSignal }) => {
      seenSignal.current = init?.signal ?? null;
      return Promise.reject(new DOMException("aborted", "AbortError"));
    }) as unknown as typeof fetch;
    const context: RuntimeContext = { ...fakeContext(), signal: controller.signal };

    try {
      const p = new LiteRtLmTextPipeline(lfm2_5ThinkingManifest, {
        modelBase: "https://huggingface.co/",
      });
      const loading = p.load(context);
      controller.abort();
      await expect(loading).rejects.toThrow(/cancelled/i);

      // The signal handed to fetch is the one we control.
      expect(seenSignal.current?.aborted).toBe(true);
      // Cancelled is retryable, not latched to 'error'.
      expect(p.status).toBe("idle");
    } finally {
      globalThis.fetch = original;
    }
  });

  it("cancels an in-flight download when the pipeline is disposed", async () => {
    const seenSignal = { current: null as AbortSignal | null };
    const original = globalThis.fetch;
    globalThis.fetch = vi.fn((_url: unknown, init?: { signal?: AbortSignal }) => {
      seenSignal.current = init?.signal ?? null;
      return new Promise((_resolve, reject) => {
        const fail = () => reject(new DOMException("aborted", "AbortError"));
        // Native fetch rejects immediately for an already-aborted signal.
        if (init?.signal?.aborted) fail();
        else init?.signal?.addEventListener("abort", fail);
      });
    }) as unknown as typeof fetch;

    try {
      const p = new LiteRtLmTextPipeline(lfm2_5ThinkingManifest, {
        modelBase: "https://huggingface.co/",
      });
      const loading = p.load(fakeContext());
      await p.dispose();
      await expect(loading).rejects.toThrow(/cancelled/i);
      expect(seenSignal.current?.aborted).toBe(true);
    } finally {
      globalThis.fetch = original;
    }
  });

  it("reports a failed model fetch as a fetch error, not a cancellation", async () => {
    const original = globalThis.fetch;
    globalThis.fetch = vi.fn().mockResolvedValue({ ok: false, status: 404, body: null });
    try {
      const p = new LiteRtLmTextPipeline(lfm2_5ThinkingManifest, {
        modelBase: "https://huggingface.co/",
      });
      await expect(p.load(fakeContext())).rejects.toThrow(/HTTP 404/);
      expect(p.status).toBe("error");
    } finally {
      globalThis.fetch = original;
    }
  });

  it("leaves the engine's own fetch alone when no model base is set", async () => {
    const original = globalThis.fetch;
    globalThis.fetch = vi.fn();
    try {
      const p = new LiteRtLmTextPipeline(lfm2_5ThinkingManifest);
      await p.load(fakeContext());
      expect(globalThis.fetch).not.toHaveBeenCalled();
      expect(mockEngineCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          model:
            "litert-community/LFM2.5-1.2B-Thinking/resolve/1b1e49ad9dccdededc9d03bc0fe3071d83595d75/LFM2.5-1.2B-Thinking_int4.litertlm",
        })
      );
    } finally {
      globalThis.fetch = original;
    }
  });

  it("does not resurrect a disposed pipeline when the engine finishes late", async () => {
    stubFetch();
    // The response body is already consumed; only the engine compile is pending,
    // and an abort cannot cancel that.
    let releaseEngine = () => {};
    const compiling = new Promise<unknown>((resolve) => {
      releaseEngine = () =>
        resolve({ createConversation: vi.fn(), delete: vi.fn(async () => undefined) });
    });
    mockEngineCreate.mockReturnValue(compiling);

    const p = new LiteRtLmTextPipeline(lfm2_5ThinkingManifest, {
      modelBase: "https://huggingface.co/",
    });
    const loading = p.load(fakeContext());
    await waitFor(() => mockEngineCreate.mock.calls.length > 0, "engine compile to start");
    await p.dispose();
    releaseEngine();

    // Rejects rather than resolving: the caller asked for a model and did not get one.
    await expect(loading).rejects.toThrow(/cancelled/i);
    // Disposed stays disposed, and the late engine is released rather than kept.
    expect(p.status).toBe("disposed");
  });

  // The response body is consumed and the compile is under way when the caller cancels.
  // Aborting cannot stop a compile, so the guard has to notice controller.signal
  // rather than only the dispose token, or the cancelled load still ends up 'ready'.
  it("discards an engine that completes after the caller aborts", async () => {
    stubFetch();
    const lateEngine = {
      createConversation: vi.fn(),
      delete: vi.fn(async () => undefined),
    };
    let releaseEngine = () => {};
    mockEngineCreate.mockReturnValue(
      new Promise<unknown>((resolve) => {
        releaseEngine = () => resolve(lateEngine);
      })
    );

    const controller = new AbortController();
    const p = new LiteRtLmTextPipeline(lfm2_5ThinkingManifest, {
      modelBase: "https://huggingface.co/",
    });
    const loading = p.load({ ...fakeContext(), signal: controller.signal });
    await waitFor(() => mockEngineCreate.mock.calls.length > 0, "engine compile to start");
    controller.abort();
    releaseEngine();

    await expect(loading).rejects.toThrow(/cancelled/i);
    expect(lateEngine.delete).toHaveBeenCalledTimes(1);
    expect(p.status).not.toBe("ready");
  });

  it("deletes an engine that resolves after disposal", async () => {
    stubFetch();
    const lateEngine = {
      createConversation: vi.fn(),
      delete: vi.fn(async () => undefined),
    };
    let releaseEngine = () => {};
    mockEngineCreate.mockReturnValue(
      new Promise<unknown>((resolve) => {
        releaseEngine = () => resolve(lateEngine);
      })
    );

    const p = new LiteRtLmTextPipeline(lfm2_5ThinkingManifest, {
      modelBase: "https://huggingface.co/",
    });
    const loading = p.load(fakeContext());
    await waitFor(() => mockEngineCreate.mock.calls.length > 0, "engine compile to start");
    await p.dispose();
    releaseEngine();
    await expect(loading).rejects.toThrow(/cancelled/i);

    expect(lateEngine.delete).toHaveBeenCalledTimes(1);
  });

  it("refuses to download through an already-aborted signal", async () => {
    const fetchMock = stubFetch();
    const controller = new AbortController();
    controller.abort();
    const p = new LiteRtLmTextPipeline(lfm2_5ThinkingManifest, {
      modelBase: "https://huggingface.co/",
    });

    await expect(p.load({ ...fakeContext(), signal: controller.signal })).rejects.toThrow(
      /cancelled/i
    );
    // addEventListener on a settled signal never fires, so without the explicit
    // pre-abort check the fetch would start anyway.
    const passedSignal = fetchMock.mock.calls[0][1]?.signal as AbortSignal | undefined;
    expect(passedSignal?.aborted).toBe(true);
    expect(p.status).toBe("idle");
  });

  it("resolves against a model base that has no trailing slash", async () => {
    const fetchMock = stubFetch();
    const p = new LiteRtLmTextPipeline(lfm2_5ThinkingManifest, {
      modelBase: "https://huggingface.co/litert-community",
    });
    await p.load(fakeContext());

    expect(fetchMock).toHaveBeenCalledWith(
      "https://huggingface.co/litert-community/LFM2.5-1.2B-Thinking/resolve/1b1e49ad9dccdededc9d03bc0fe3071d83595d75/LFM2.5-1.2B-Thinking_int4.litertlm",
      expect.anything()
    );
  });
});

describe("pinned LiteRT-LM model artifacts", () => {
  it("uses immutable Hugging Face revisions with exact integrity metadata", () => {
    for (const manifest of [
      litertLmManifest,
      lfm2_5InstructManifest,
      lfm2_5InstructInt8Manifest,
      lfm2_5ThinkingManifest,
      lfm2_5ThinkingInt8Manifest,
      gemma4E2bManifest,
      gemma4E4bManifest,
    ]) {
      const asset = manifest.assets[0];
      expect(asset.path).toMatch(/\/resolve\/[0-9a-f]{40}\//);
      expect(asset.sha256).toMatch(/^[0-9a-f]{64}$/);
      expect(asset.bytes).toBe(manifest.memory.downloadBytes);
    }
  });
});

describe("selectTextGenerationManifest", () => {
  it("selects the reasoning model for the reasoning capability", () => {
    expect(selectTextGenerationManifest("reasoning")).toBe(lfm2_5ThinkingManifest);
    expect(selectTextGenerationManifest("reasoning", "deep")).toBe(lfm2_5ThinkingInt8Manifest);
  });

  it("selects the instruct model for text generation", () => {
    expect(selectTextGenerationManifest("text-generation")).toBe(lfm2_5InstructManifest);
    expect(selectTextGenerationManifest("text-generation", "deep")).toBe(
      lfm2_5InstructInt8Manifest
    );
  });

  it("advertises the reasoning capability only on thinking manifests", () => {
    expect(lfm2_5ThinkingManifest.capabilities).toContain("reasoning");
    expect(lfm2_5InstructManifest.capabilities).not.toContain("reasoning");
  });
});


describe('LiteRtLmTextPipeline disposal boundaries', () => {
  it('rejects loading a disposed instance', async () => {
    const p = new LiteRtLmTextPipeline()
    await p.dispose()
    await expect(p.load(fakeContext())).rejects.toMatchObject({ code: 'CANCELLED' })
    expect(mockEngineCreate).not.toHaveBeenCalled()
  })

  it('preserves cancellation when deleting a late engine fails', async () => {
    const cleanupError = new Error('engine delete failed')
    const engine = { delete: vi.fn().mockRejectedValue(cleanupError) }
    let release!: () => void
    mockEngineCreate.mockReturnValue(new Promise(resolve => { release = () => resolve(engine) }))
    const p = new LiteRtLmTextPipeline()
    const loading = p.load(fakeContext())
    await waitFor(() => mockEngineCreate.mock.calls.length > 0, 'engine compile to start')
    await p.dispose()
    release()
    await expect(loading).rejects.toMatchObject({ code: 'CANCELLED', cause: cleanupError })
    expect(engine.delete).toHaveBeenCalledTimes(1)
    expect(p.status).toBe('disposed')
  })

  it('preserves completed inference when conversation cleanup fails and dispose retries it', async () => {
    const cleanupError = new Error('conversation delete failed')
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    const conversation = {
      sendMessage: vi.fn(), sendMessageStreaming: mockSendMessageStreaming, cancel: vi.fn(),
      delete: vi.fn()
        .mockRejectedValueOnce(cleanupError)
        .mockResolvedValueOnce(undefined),
    }
    const engine = {
      createConversation: vi.fn().mockResolvedValue(conversation),
      delete: vi.fn().mockResolvedValue(undefined),
    }
    mockEngineCreate.mockResolvedValue(engine)
    const p = new LiteRtLmTextPipeline()
    await p.load(fakeContext())

    await expect(p.run({ messages: [{ role: 'user', content: 'hi' }] })).resolves.toMatchObject({
      kind: 'text',
      text: 'Hello from litert-lm',
    })
    expect(conversation.delete).toHaveBeenCalledTimes(1)
    expect(p).toHaveProperty('conversation', conversation)
    expect(p.status).toBe('error')
    expect(warn).toHaveBeenCalledWith(
      '[text-gen] LiteRT-LM conversation cleanup failed',
      cleanupError,
    )

    await expect(p.load(fakeContext())).rejects.toMatchObject({ code: 'INFERENCE_FAILED' })
    expect(mockEngineCreate).toHaveBeenCalledTimes(1)

    await expect(p.dispose()).resolves.toBeUndefined()
    expect(conversation.delete).toHaveBeenCalledTimes(2)
    expect(engine.delete).toHaveBeenCalledTimes(1)
    expect(p.status).toBe('disposed')
    warn.mockRestore()
  })

  it('preserves the primary inference failure when conversation cleanup also fails', async () => {
    const primaryError = new Error('generation failed')
    const cleanupError = new Error('conversation delete failed')
    const conversation = {
      sendMessage: vi.fn(),
      sendMessageStreaming: vi.fn(() => new ReadableStream({
        start(controller) { controller.error(primaryError) },
      })),
      cancel: vi.fn(),
      delete: vi.fn()
        .mockRejectedValueOnce(cleanupError)
        .mockResolvedValueOnce(undefined),
    }
    const engine = {
      createConversation: vi.fn().mockResolvedValue(conversation),
      delete: vi.fn().mockResolvedValue(undefined),
    }
    mockEngineCreate.mockResolvedValue(engine)
    const p = new LiteRtLmTextPipeline()
    await p.load(fakeContext())

    let caught: Error & { cause?: unknown } | undefined
    try {
      await p.run({ messages: [{ role: 'user', content: 'hi' }] })
    } catch (error) {
      caught = error as Error & { cause?: unknown }
    }

    expect(caught).toBe(primaryError)
    expect(caught?.cause).toBe(cleanupError)
    expect(p.status).toBe('error')
    await expect(p.dispose()).resolves.toBeUndefined()
  })

  it('does not publish ready or delete the engine before conversation cleanup settles', async () => {
    let releaseCleanup!: () => void
    const cleanup = new Promise<void>((resolve) => { releaseCleanup = resolve })
    const conversation = {
      sendMessage: vi.fn(),
      sendMessageStreaming: mockSendMessageStreaming,
      cancel: vi.fn(),
      delete: vi.fn(() => cleanup),
    }
    const engine = {
      createConversation: vi.fn().mockResolvedValue(conversation),
      delete: vi.fn().mockResolvedValue(undefined),
    }
    mockEngineCreate.mockResolvedValue(engine)
    const p = new LiteRtLmTextPipeline()
    await p.load(fakeContext())

    const running = p.run({ messages: [{ role: 'user', content: 'hi' }] })
    await waitFor(() => conversation.delete.mock.calls.length === 1, 'conversation cleanup to start')

    expect(p.status).toBe('running')
    await expect(p.run({ messages: [{ role: 'user', content: 'second' }] })).rejects.toThrow(
      'Pipeline not ready',
    )

    const disposing = p.dispose()
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(engine.delete).not.toHaveBeenCalled()

    releaseCleanup()
    await running
    await disposing
    expect(engine.delete).toHaveBeenCalledTimes(1)
    expect(p.status).toBe('disposed')
  })

  it('finishes disposal even when engine cleanup rejects', async () => {
    const cleanupError = new Error('engine delete failed')
    const conversation = {
      sendMessage: vi.fn(), sendMessageStreaming: mockSendMessageStreaming, cancel: vi.fn(),
      delete: vi.fn().mockResolvedValue(undefined),
    }
    const engine = {
      createConversation: vi.fn().mockResolvedValue(conversation),
      delete: vi.fn().mockRejectedValue(cleanupError),
    }
    mockEngineCreate.mockResolvedValue(engine)
    const p = new LiteRtLmTextPipeline()
    await p.load(fakeContext())
    await p.run({ messages: [{ role: 'user', content: 'hi' }] })

    await expect(p.dispose()).rejects.toBe(cleanupError)
    expect(conversation.delete).toHaveBeenCalledTimes(1)
    expect(engine.delete).toHaveBeenCalledTimes(1)
    expect(p.status).toBe('disposed')
    expect(p).toHaveProperty('conversation', null)
    expect(p).toHaveProperty('engine', null)
    await expect(p.load(fakeContext())).rejects.toMatchObject({ code: 'CANCELLED' })
  })
})


it('keeps text generation disposed when an in-flight stream finishes', async () => {
  let finish!: () => void
  mockSendMessageStreaming.mockReturnValue(new ReadableStream({
    start(controller) { finish = () => controller.close() },
  }))
  const p = new LiteRtLmTextPipeline()
  await p.load(fakeContext())
  const running = p.run({ messages: [{ role: 'user', content: 'hi' }] })
  await waitFor(() => mockSendMessageStreaming.mock.calls.length > 0, 'generation stream to start')
  await p.dispose()
  finish()
  await running
  expect(p.status).toBe('disposed')
})
