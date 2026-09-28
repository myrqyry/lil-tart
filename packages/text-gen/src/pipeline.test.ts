import { beforeEach, describe, expect, it, vi } from "vitest";
import type { RuntimeContext } from '@litert-playground/inference-core';
import { TransformersTextPipeline } from "./transformers-pipeline";
import { LiteRtLmTextPipeline } from "./litertlm-pipeline";
import {
  lfm2_5InstructManifest,
  lfm2_5InstructInt8Manifest,
  lfm2_5ThinkingManifest,
  lfm2_5ThinkingInt8Manifest,
  selectTextGenerationManifest,
} from "./manifest";

const { mockPipeline, mockEngineCreate, mockSendMessageStreaming } = vi.hoisted(() => ({
  mockPipeline: vi.fn(),
  mockEngineCreate: vi.fn(),
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

function streamOf(chunks: string[]): ReadableStream<{ text: string }> {
  return new ReadableStream({
    start(controller) {
      for (const c of chunks) controller.enqueue({ text: c });
      controller.close();
    },
  });
}

beforeEach(() => {
  mockPipeline.mockReset();
  mockEngineCreate.mockReset();
  mockSendMessageStreaming.mockReset();

  mockPipeline.mockResolvedValue(async () => ({
    output: [{ generated_text: "Hello from transformers" }],
  }));

  mockEngineCreate.mockResolvedValue({
    createConversation: async () => ({
      sendMessage: async () => ({ text: "" }),
      sendMessageStreaming: mockSendMessageStreaming,
      cancel: () => {},
      delete: async () => {},
    }),
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

  it("loads the manifest's model path instead of a hardcoded default", async () => {
    const p = new LiteRtLmTextPipeline(lfm2_5ThinkingManifest);
    await p.load(fakeContext());

    expect(mockEngineCreate).toHaveBeenCalledWith(
      expect.objectContaining({
        model:
          "litert-community/LFM2.5-1.2B-Thinking/resolve/main/LFM2.5-1.2B-Thinking_int4.litertlm",
        backend: "wasm",
      })
    );
  });

  // The manifest path is repo-relative. Handing it straight to the engine makes the
  // browser resolve it against the app origin, which 404s; it has to go through the
  // runtime's asset resolver so the configured model base applies.
  it("resolves the model through the asset resolver instead of passing the raw path", async () => {
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new Uint8Array([1]));
        controller.close();
      },
    });
    const assets = {
      resolve: vi.fn(),
      stream: vi.fn().mockResolvedValue(stream),
    };
    const context: RuntimeContext = {
      ...fakeContext(),
      assets: assets as unknown as RuntimeContext["assets"],
    };

    const p = new LiteRtLmTextPipeline(lfm2_5ThinkingManifest);
    await p.load(context);

    expect(assets.stream).toHaveBeenCalledWith(
      {
        id: "litert-community/LFM2.5-1.2B-Thinking/resolve/main/LFM2.5-1.2B-Thinking_int4.litertlm",
        path: "litert-community/LFM2.5-1.2B-Thinking/resolve/main/LFM2.5-1.2B-Thinking_int4.litertlm",
      },
      { signal: undefined }
    );
    expect(assets.resolve).not.toHaveBeenCalled();
    expect(mockEngineCreate).toHaveBeenCalledWith(
      expect.objectContaining({ model: stream })
    );
  });

  it("falls back to a Blob when the resolver cannot stream", async () => {
    const bytes = new ArrayBuffer(4);
    const assets = { resolve: vi.fn().mockResolvedValue(bytes) };
    const context: RuntimeContext = {
      ...fakeContext(),
      assets: assets as unknown as RuntimeContext["assets"],
    };

    const p = new LiteRtLmTextPipeline(lfm2_5ThinkingManifest);
    await p.load(context);

    expect(assets.resolve).toHaveBeenCalledTimes(1);
    const passed = mockEngineCreate.mock.calls[0][0].model;
    expect(passed).toBeInstanceOf(Blob);
    expect(await (passed as Blob).arrayBuffer()).toEqual(bytes);
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
