# Conversion evidence lane

This directory holds reproducible conversion inputs and candidate-model evidence.
It does **not** make a model part of Lil Tart's supported runtime surface.

Source material imported on 2026-09-30 is pinned to:

- repository: `john-rocky/LiteRT-Models`
- revision: `460d52221c22388b6a9a8e8a44b61dde72976b4e`
- upstream license: MIT for the repository scripts/recipes; converted model licenses vary

The two LiteRT-LM recipes under `litert-lm/recipes/` are copied verbatim from that
revision. They remain conversion inputs, not claims about any Lil Tart model.
`pnpm test:conversion-assets` compares their complete parsed structures with
that pin, including every option and the exact two-rule order, and tests
that semantic mutations are rejected. JSON formatting and object key order
do not affect this check.

## Qualification boundary

Upstream Android `CompiledModel` results are useful evidence about conversion and
delegate behavior, but they are **not** browser WebGPU qualification. Candidate
entries therefore remain `browserQualification: "unverified"` until Lil Tart's
own browser harness records evidence.

The qualification sequence for a promoted model should be:

1. pin immutable source/artifact revisions and integrity facts;
2. verify conversion output and tensor contracts;
3. compile on the requested Lil Tart backend;
4. run real inference;
5. compare numeric output against a trusted reference when applicable;
6. run behavior/quality checks appropriate to the task;
7. only then promote manifest verification metadata.

## LiteRT-LM quantization recipes

`int4_block32_octav.json` uses blockwise-32 int4 with OCTAV for general weights
and keeps `EMBEDDING_LOOKUP` at channelwise int8.

`int4_block128.json` uses blockwise-128 int4 for general weights and keeps
`EMBEDDING_LOOKUP` at channelwise int8.

The upstream source presents block32+OCTAV as the stronger data-free quality
choice and block128 as a speed/quality tradeoff with fewer scales. Lil Tart
preserves those recipes as references rather than changing existing manifests
or assuming they are superior for every model.

Do not replace a working published artifact merely because a recipe exists here.
A new export must re-enter the qualification sequence above.

## GPU conversion lessons worth carrying forward

The upstream conversion guide documents several failure classes that matter to
future Lil Tart work: graph conversion success is not equivalent to delegate
compatibility, delegate compatibility is not equivalent to numerical correctness,
and some failures only appear on-device.

Examples include non-finite or silently wrong reductions, fp16 normalization
overflow, unsupported higher-rank intermediates, constant-only computation
quirks, and conversion paths that preserve CNNs but corrupt attention layouts.

Lil Tart does not import the Android-specific patch toolkit wholesale because its
runtime target is browser LiteRT.js/WebGPU. Instead, the durable mechanism adopted
here is stronger evidence: numeric parity can now be recorded in runtime
qualification results, and Android-only evidence stays clearly scoped.
