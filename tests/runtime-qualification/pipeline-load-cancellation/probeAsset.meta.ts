// Browser-safe constants, imported by both the Node-side generator and the in-browser
// probe. Deliberately free of node: imports: pulling those into the browser entry
// fails module resolution and takes the whole harness page down with it.
export const ABORT_PROBE_BYTES = 24 * 1024 * 1024
export const ABORT_PROBE_PATH = 'static-models/qualification-abort-probe.bin'
