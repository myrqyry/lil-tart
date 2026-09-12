export interface TensorSpec {
  name: string
  dtype: 'float32' | 'int32' | 'int8' | 'uint8'
  shape: number[]
  description: string
  constraints?: {
    min?: number
    max?: number
    enum?: string[]
    items?: string[]
  }
}

export interface ModelMetadata {
  name: string
  description: string
  modelPath: string
  tags: string[]
}

/** ponytail: extra graphs loaded alongside `metadata.modelPath` for split models (DETR A/B). */
export interface ModelGraph {
  name: string
  modelPath: string
}

/** Passed to `ModelAdapter.run`; routes each graph through the same loaded runtime. */
export interface InferenceContext {
  predict(
    graph: string,
    inputs: Record<string, import('@litertjs/core').Tensor> | import('@litertjs/core').Tensor[],
    signature?: string,
  ): Promise<Record<string, import('@litertjs/core').Tensor>>
  createTensor(data: Float32Array | Int32Array, shape: number[]): import('@litertjs/core').Tensor
}

export interface ModelAdapter {
  modelId: string
  metadata: ModelMetadata
  inputSpecs: TensorSpec[]
  outputSpecs: TensorSpec[]
  prepareInputs(values: Record<string, any>): Record<string, import('@litertjs/core').Tensor>
  parseOutputs(outputs: Record<string, import('@litertjs/core').Tensor>): Promise<Record<string, any>>
  isPipeline?: true
  /** ponytail: no browser-fetchable .tflite yet (verify via HEAD 200 CORS *). UI disables the entry. */
  disabled?: boolean
  /** ponytail: multi-graph models; every graph is loaded and driven from `run`. */
  graphs?: ModelGraph[]
  /** ponytail: when present, replaces the single-graph predict path in `runInference`. */
  run?: (values: Record<string, any>, ctx: InferenceContext) => Promise<Record<string, any>>
}
