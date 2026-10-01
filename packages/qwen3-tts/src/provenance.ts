export const QWEN3_TTS_UPSTREAM_REPOSITORY = 'litert-community/Qwen3-TTS-12Hz-0.6B-Base'
export const QWEN3_TTS_UPSTREAM_REVISION = '528cca7d2ddf6f5c1e1127f24a7f8786f80fa6e8'

// Web Crypto SHA-256 is whole-buffer only. Keep strict hashes for the small
// tokenizer/tables while avoiding another giant contiguous digest allocation
// for browser-scale model graphs; declared byte length remains verified.
export const QWEN3_TTS_MAX_IN_MEMORY_SHA256_BYTES = 64 * 1024 * 1024
