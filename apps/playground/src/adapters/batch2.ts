import type { ModelAdapter } from './types'
import { Tensor } from '@litertjs/core'
import { resizeImageData, normalizeAndFormatImageData, tensorToImageData } from '../imageUtils'
import { loadClipTokenizer } from '../clipTokenizer'

const s = (n: string, d: 'float32' | 'int32', sh: number[], desc: string) =>
  ({ name: n, dtype: d, shape: sh, description: desc })

export const depth3Adapter: ModelAdapter = {
  modelId: 'depth-anything-3',
  metadata: { name: 'Depth Anything 3', description: 'Monocular depth estimation (896×504)', modelPath: 'https://huggingface.co/litert-community/Depth-Anything-3-Small/resolve/main/da3_small_gpu_fp16.tflite', tags: ['vision', 'depth'] },
  inputSpecs: [s('input', 'float32', [1, 3, 896, 504], 'RGB ImageNet-normalized NCHW')],
  outputSpecs: [s('depth', 'float32', [1, 1, 896, 504], 'Depth map')],
  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const imageData = values['image'] as ImageData;
    if (!imageData) {
      throw new Error('Image data not provided for depth3Adapter');
    }
    const [_, C, H, W] = this.inputSpecs[0].shape; // Get target shape from inputSpecs
    const resized = resizeImageData(imageData, W, H);
    const nchwTensor = normalizeAndFormatImageData(resized, [1, C, H, W], {
      dataFormat: 'NCHW',
      colorOrder: 'RGB',
      normalization: 'imagenet',
    });
    return { 'input': nchwTensor };
  },
  parseOutputs(outputs: Record<string, Tensor>): Promise<Record<string, any>> {
    const depthTensor = outputs['depth']
    if (!depthTensor) throw new Error('Depth tensor not found in outputs')
    const [_, _c, H, W] = this.outputSpecs[0].shape
    return tensorToImageData(depthTensor, W, H).then(d => ({ depth: d }))
  },
}

export const rtmposeFaceAdapter: ModelAdapter = {
  modelId: 'rtmpose-face',
  metadata: { name: 'RTMPose Face', description: '98-point face alignment WFLW (256×256)', modelPath: 'https://huggingface.co/litert-community/RTMPose-Face-WFLW-LiteRT/resolve/main/rtm_face_fp16.tflite', tags: ['vision', 'pose'] },
  inputSpecs: [s('input', 'float32', [1, 3, 256, 256], 'RGB mmpose mean/std NCHW')],
  outputSpecs: [
    s('simcc_x', 'float32', [1, 98, 512], 'SimCC X bins (argmax/2 → pixel)'),
    s('simcc_y', 'float32', [1, 98, 512], 'SimCC Y bins (argmax/2 → pixel)'),
  ],
  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const imageData = values['image'] as ImageData
    if (!imageData) throw new Error('Image data not provided for rtmposeFaceAdapter')
    const [_, C, H, W] = this.inputSpecs[0].shape
    const resized = resizeImageData(imageData, W, H)
    const t = normalizeAndFormatImageData(resized, [1, C, H, W], { dataFormat: 'NCHW', colorOrder: 'RGB', normalization: 'imagenet' })
    return { input: t }
  },
  parseOutputs(outputs: Record<string, Tensor>): Promise<Record<string, any>> {
    return Promise.resolve({ simcc_x: outputs.simcc_x, simcc_y: outputs.simcc_y })
  },
}
  export const rtmposeHandAdapter: ModelAdapter = {
  modelId: 'rtmpose-hand',
  metadata: { name: 'RTMPose Hand', description: '21-keypoint hand pose (256×256)', modelPath: 'https://huggingface.co/litert-community/RTMPose-Hand-LiteRT/resolve/main/rtmhand_fp16.tflite', tags: ['vision', 'pose'] },
  inputSpecs: [s('input', 'float32', [1, 3, 256, 256], 'RGB ImageNet norm NCHW')],
  outputSpecs: [
    s('simcc_x', 'float32', [1, 21, 512], 'SimCC X bins'),
    s('simcc_y', 'float32', [1, 21, 512], 'SimCC Y bins'),
  ],
  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const img = values['image'] as ImageData
    if (!img) throw new Error('Image data not provided')
    const [_, C, H, W] = this.inputSpecs[0].shape
    const resized = resizeImageData(img, W, H)
    const t = normalizeAndFormatImageData(resized, [1, C, H, W], { dataFormat: 'NCHW', colorOrder: 'RGB', normalization: 'imagenet' })
    return { input: t }
  },
  parseOutputs(o: Record<string, Tensor>) { return Promise.resolve({ simcc_x: o.simcc_x, simcc_y: o.simcc_y }) },
}

export const rtmposeSAdapter: ModelAdapter = {
  modelId: 'rtmpose-s',
  metadata: { name: 'RTMPose-s', description: '17-keypoint body pose (256×192)', modelPath: 'https://huggingface.co/litert-community/RTMPose-s-LiteRT/resolve/main/rtmpose_s_fp16.tflite', tags: ['vision', 'pose'] },
  inputSpecs: [s('input', 'float32', [1, 3, 256, 192], 'RGB mmpose mean/std NCHW')],
  outputSpecs: [
    s('simcc_x', 'float32', [1, 17, 384], 'SimCC X bins'),
    s('simcc_y', 'float32', [1, 17, 512], 'SimCC Y bins'),
  ],
  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const img = values['image'] as ImageData
    if (!img) throw new Error('Image data not provided')
    const [_, C, H, W] = this.inputSpecs[0].shape
    const resized = resizeImageData(img, W, H)
    const t = normalizeAndFormatImageData(resized, [1, C, H, W], { dataFormat: 'NCHW', colorOrder: 'RGB', normalization: 'imagenet' })
    return { input: t }
  },
  parseOutputs(o: Record<string, Tensor>) { return Promise.resolve({ simcc_x: o.simcc_x, simcc_y: o.simcc_y }) },
}

export const rtmwAdapter: ModelAdapter = {
  modelId: 'rtmw-wholebody',
  metadata: { name: 'RTMW-m WholeBody', description: '133-keypoint whole-body pose (256×192)', modelPath: 'https://huggingface.co/litert-community/RTMW-m-WholeBody-LiteRT/resolve/main/rtmw_fp16.tflite', tags: ['vision', 'pose'] },
  inputSpecs: [s('input', 'float32', [1, 3, 256, 192], 'RGB mmpose mean/std NCHW')],
  outputSpecs: [
    s('simcc_x', 'float32', [1, 133, 384], 'SimCC X bins'),
    s('simcc_y', 'float32', [1, 133, 512], 'SimCC Y bins'),
  ],
  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const img = values['image'] as ImageData
    if (!img) throw new Error('Image data not provided')
    const [_, C, H, W] = this.inputSpecs[0].shape
    const resized = resizeImageData(img, W, H)
    const t = normalizeAndFormatImageData(resized, [1, C, H, W], { dataFormat: 'NCHW', colorOrder: 'RGB', normalization: 'imagenet' })
    return { input: t }
  },
  parseOutputs(o: Record<string, Tensor>) { return Promise.resolve({ simcc_x: o.simcc_x, simcc_y: o.simcc_y }) },
}

export const yolactAdapter: ModelAdapter = {
  modelId: 'yolact',
  metadata: { name: 'YOLACT — Instance Seg', description: 'Real-time instance segmentation ResNet50 (550×550)', modelPath: 'https://huggingface.co/litert-community/YOLACT-ResNet50-LiteRT/resolve/main/yolact.tflite', tags: ['vision', 'segmentation'] },
  inputSpecs: [s('input', 'float32', [1, 3, 550, 550], 'BGR normalized NCHW')],
  outputSpecs: [
    s('loc', 'float32', [1, 19248, 4], 'Box regressors'),
    s('conf', 'float32', [1, 19248, 81], 'Class scores (softmax)'),
    s('mask', 'float32', [1, 19248, 32], 'Mask coefficients'),
    s('proto', 'float32', [1, 138, 138, 32], 'Prototype masks'),
  ],
  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const img = values['image'] as ImageData
    if (!img) throw new Error('Image data not provided')
    const [_, C, H, W] = this.inputSpecs[0].shape
    const resized = resizeImageData(img, W, H)
    const t = normalizeAndFormatImageData(resized, [1, C, H, W], {
      dataFormat: 'NCHW',
      colorOrder: 'BGR',
      normalization: 'imagenet',
      mean: [103.94 / 255, 116.78 / 255, 123.68 / 255],
      std: [57.38 / 255, 57.12 / 255, 58.40 / 255],
    })
    return { input: t }
  },
  parseOutputs(o: Record<string, Tensor>) { return Promise.resolve({ loc: o.loc, conf: o.conf, mask: o.mask, proto: o.proto }) },
}

export const ormbgAdapter: ModelAdapter = {
  modelId: 'ormbg',
  metadata: { name: 'ORMBG — BG Removal', description: 'Background removal (1024×1024)', modelPath: 'https://huggingface.co/litert-community/ormbg-LiteRT/resolve/main/ormbg.tflite', tags: ['vision', 'matting'] },
  inputSpecs: [s('input', 'float32', [1, 3, 1024, 1024], 'RGB x/255 NCHW')],
  outputSpecs: [s('alpha', 'float32', [1, 1, 1024, 1024], 'Alpha matte')],
  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const img = values['image'] as ImageData
    if (!img) throw new Error('Image data not provided')
    const [_, C, H, W] = this.inputSpecs[0].shape
    const resized = resizeImageData(img, W, H)
    const t = normalizeAndFormatImageData(resized, [1, C, H, W], { dataFormat: 'NCHW', colorOrder: 'RGB', normalization: '0-1' })
    return { input: t }
  },
  parseOutputs(o: Record<string, Tensor>) {
    const [_, _c, _h, W] = this.outputSpecs[0].shape
    return tensorToImageData(o.alpha, W, _h).then(d => ({ alpha: d }))
  },
}

export const u2netSalientAdapter: ModelAdapter = {
  modelId: 'u2net-salient',
  metadata: { name: 'U-2-Net — Saliency', description: 'Salient object detection (320×320)', modelPath: 'https://huggingface.co/litert-community/U-2-Net/resolve/main/u2net_fp16.tflite', tags: ['vision', 'segmentation'] },
  inputSpecs: [s('input', 'float32', [1, 3, 320, 320], 'RGB ImageNet-norm NCHW')],
  outputSpecs: [s('saliency', 'float32', [1, 1, 320, 320], 'Saliency mask [0,1]')],
  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const img = values['image'] as ImageData
    if (!img) throw new Error('Image data not provided')
    const [_, C, H, W] = this.inputSpecs[0].shape
    const resized = resizeImageData(img, W, H)
    const t = normalizeAndFormatImageData(resized, [1, C, H, W], { dataFormat: 'NCHW', colorOrder: 'RGB', normalization: 'imagenet' })
    return { input: t }
  },
  parseOutputs(o: Record<string, Tensor>) {
    const [_, _c, H, W] = this.outputSpecs[0].shape
    return tensorToImageData(o.saliency, W, H).then(d => ({ saliency: d }))
  },
}

export const modnetAdapter: ModelAdapter = {
  modelId: 'modnet',
  metadata: { name: 'MODNet — Matting', description: 'Trimap-free portrait matting (512×512)', modelPath: 'https://huggingface.co/litert-community/MODNet-LiteRT/resolve/main/modnet.tflite', tags: ['vision', 'matting'] },
  inputSpecs: [s('input', 'float32', [1, 3, 512, 512], 'RGB [-1,1] NCHW')],
  outputSpecs: [s('alpha', 'float32', [1, 1, 512, 512], 'Soft alpha matte [0,1]')],
  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const img = values['image'] as ImageData
    if (!img) throw new Error('Image data not provided')
    const [_, C, H, W] = this.inputSpecs[0].shape
    const resized = resizeImageData(img, W, H)
    const t = normalizeAndFormatImageData(resized, [1, C, H, W], { dataFormat: 'NCHW', colorOrder: 'RGB', normalization: '-1-1' })
    return { input: t }
  },
  parseOutputs(o: Record<string, Tensor>) {
    const [_, _c, H, W] = this.outputSpecs[0].shape
    return tensorToImageData(o.alpha, W, H).then(d => ({ alpha: d }))
  },
}

export const yolo26SegAdapter: ModelAdapter = {
  modelId: 'yolo26-seg',
  metadata: { name: 'YOLO26m Segmentation', description: 'EdgeFirst YOLO26 segmentation INT8', modelPath: '/models/yolo26-seg/yolo26m-seg-int8-smart.tflite', tags: ['vision', 'segmentation'] },
  disabled: true, // ponytail: no HF LiteRT mirror — needs locating (see /needs-locating)
  inputSpecs: [s('input', 'float32', [1, 640, 640, 3], 'RGB NHWC')],
  outputSpecs: [s('output', 'float32', [1, 116, 8400], 'Detection + mask coeffs')],
  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const img = values['image'] as ImageData
    if (!img) throw new Error('Image data not provided')
    const [N, H, W, C] = this.inputSpecs[0].shape
    const resized = resizeImageData(img, W, H)
    const t = normalizeAndFormatImageData(resized, [N, H, W, C], { dataFormat: 'NHWC', colorOrder: 'RGB', normalization: '0-1' })
    return { input: t }
  },
  parseOutputs(o: Record<string, Tensor>) { return Promise.resolve({ output: o.output }) },
}

export const yolo11SegAdapter: ModelAdapter = {
  modelId: 'yolo11-seg',
  metadata: { name: 'YOLO11m Segmentation', description: 'EdgeFirst YOLO11 segmentation INT8', modelPath: '/models/yolo11-seg/yolo11m-seg-int8-smart.tflite', tags: ['vision', 'segmentation'] },
  disabled: true, // ponytail: no HF LiteRT mirror — needs locating
  inputSpecs: [s('input', 'float32', [1, 640, 640, 3], 'RGB NHWC')],
  outputSpecs: [s('output', 'float32', [1, 116, 8400], 'Detection + mask coeffs')],
  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const img = values['image'] as ImageData
    if (!img) throw new Error('Image data not provided')
    const [N, H, W, C] = this.inputSpecs[0].shape
    const resized = resizeImageData(img, W, H)
    const t = normalizeAndFormatImageData(resized, [N, H, W, C], { dataFormat: 'NHWC', colorOrder: 'RGB', normalization: '0-1' })
    return { input: t }
  },
  parseOutputs(o: Record<string, Tensor>) { return Promise.resolve({ output: o.output }) },
}

export const yolov8SegAdapter: ModelAdapter = {
  modelId: 'yolov8-seg',
  metadata: { name: 'YOLOv8m Segmentation', description: 'EdgeFirst YOLOv8 segmentation INT8', modelPath: '/models/yolov8-seg/yolov8m-seg-int8-smart.tflite', tags: ['vision', 'segmentation'] },
  disabled: true, // ponytail: no HF LiteRT mirror — needs locating
  inputSpecs: [s('input', 'float32', [1, 640, 640, 3], 'RGB NHWC')],
  outputSpecs: [s('output', 'float32', [1, 116, 8400], 'Detection + mask coeffs')],
  prepareInputs(values: Record<string, any>): Record<string, Tensor> {
    const img = values['image'] as ImageData
    if (!img) throw new Error('Image data not provided')
    const [N, H, W, C] = this.inputSpecs[0].shape
    const resized = resizeImageData(img, W, H)
    const t = normalizeAndFormatImageData(resized, [N, H, W, C], { dataFormat: 'NHWC', colorOrder: 'RGB', normalization: '0-1' })
    return { input: t }
  },
  parseOutputs(o: Record<string, Tensor>) { return Promise.resolve({ output: o.output }) },
}

export const clipsegAdapters: ModelAdapter[] = [
  {
    modelId: 'clipseg-text',
    metadata: { name: 'CLIPSeg — Text Encoder', description: 'Text encoder for text-prompted segmentation', modelPath: 'https://huggingface.co/litert-community/CLIPSeg-rd64-LiteRT/resolve/main/clipseg_text_fp16.tflite', tags: ['vision', 'segmentation'] },
    inputSpecs: [{ name: 'text', dtype: 'string', shape: [], description: 'Text prompt, e.g. "a dog"', constraints: { text: true } }],
    outputSpecs: [s('text_emb', 'float32', [1, 64], 'Text embedding')],
    prepareInputs(): Record<string, Tensor> {
      throw new Error('clipseg-text tokenizes inside run()')
    },
    async run(values, ctx) {
      const tokenizer = await loadClipTokenizer()
      const ids = tokenizer.encode(String(values['text'] ?? ''), 77)
      const result = await ctx.predict('main', { input: ctx.createTensor(Float32Array.from(ids), [1, 77]) })
      const embedding = Object.values(result)[0]
      const data = await embedding.data()
      return { text_emb: Array.from(data as Float32Array) }
    },
    parseOutputs(o: Record<string, Tensor>) { return Promise.resolve({ text_emb: o.text_emb }) },
  },
  {
    modelId: 'clipseg-vision',
    metadata: { name: 'CLIPSeg — Vision Encoder', description: 'Vision encoder for text-prompted segmentation', modelPath: 'https://huggingface.co/litert-community/CLIPSeg-rd64-LiteRT/resolve/main/clipseg_vision_fp16.tflite', tags: ['vision', 'segmentation'] },
    inputSpecs: [s('input', 'float32', [1, 3, 352, 352], 'RGB x/255 NCHW')],
    outputSpecs: [s('vision_emb', 'float32', [1, 64], 'Visual embedding')],
    prepareInputs(values: Record<string, any>): Record<string, Tensor> {
      const img = values['image'] as ImageData
      if (!img) throw new Error('Image data not provided for clipseg-vision')
      const [_, C, H, W] = this.inputSpecs[0].shape
      const resized = resizeImageData(img, W, H)
      const t = normalizeAndFormatImageData(resized, [1, C, H, W], { dataFormat: 'NCHW', colorOrder: 'RGB', normalization: '0-1' })
      return { input: t }
    },
    parseOutputs(o: Record<string, Tensor>) { return Promise.resolve({ vision_emb: o.vision_emb }) },
  },
  {
    modelId: 'clipseg-decoder',
    metadata: { name: 'CLIPSeg — Decoder', description: 'Decoder for text-prompted segmentation', modelPath: 'https://huggingface.co/litert-community/CLIPSeg-rd64-LiteRT/resolve/main/clipseg_decoder.tflite', tags: ['vision', 'segmentation'] },
    inputSpecs: [
      s('text_emb', 'float32', [1, 64], 'From CLIPSeg text encoder'),
      s('vision_emb', 'float32', [1, 64], 'From CLIPSeg vision encoder'),
    ],
    outputSpecs: [s('mask', 'float32', [1, 1, 352, 352], 'Segmentation mask')],
    isPipeline: true as const,
    prepareInputs(values: Record<string, any>): Record<string, Tensor> {
      const textEmb = values['text_emb'] as Tensor
      const visionEmb = values['vision_emb'] as Tensor
      if (!textEmb || !visionEmb) throw new Error('Both text_emb and vision_emb required for clipseg-decoder')
      return { text_emb: textEmb, vision_emb: visionEmb }
    },
    parseOutputs(o: Record<string, Tensor>) {
      const [_, _c, H, W] = this.outputSpecs[0].shape
      return tensorToImageData(o.mask, W, H).then(d => ({ mask: d }))
    },
  },
]

export const adapters13: ModelAdapter[] = [
  depth3Adapter, rtmposeFaceAdapter, rtmposeHandAdapter, rtmposeSAdapter, rtmwAdapter,
  yolactAdapter, ormbgAdapter, u2netSalientAdapter, modnetAdapter,
  yolo26SegAdapter, yolo11SegAdapter, yolov8SegAdapter, ...clipsegAdapters,
]
