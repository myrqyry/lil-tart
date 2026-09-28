import type { RuntimeOperation, RuntimePathGraphBackend } from './runtimePathProof'

export type TartGuideTone = 'idle' | 'working' | 'success' | 'warning' | 'error'

export type TartGuideAction = 'preflight'

export interface TartGuideBackendOverride {
  graph: string
  backend: string
}

export interface TartGuideSnapshot {
  selectedModelName: string | null
  operation: RuntimeOperation
  loaded: boolean
  progressPercent: number | null
  error: string | null
  requestedBackend: string | null
  resolvedBackend: string | null
  backendOverrides: readonly TartGuideBackendOverride[]
  runtimeGraphBackends: readonly RuntimePathGraphBackend[]
  fallbackCount: number
  preflightComplete: boolean
  pathProofAvailable: boolean
}

export interface TartGuideMessage {
  tone: TartGuideTone
  kicker: string
  title: string
  message: string
  action?: TartGuideAction
  actionLabel?: string
}

function backendLabel(value: string | null): string {
  return value ? value.toUpperCase() : 'the available backend'
}

function plural(value: number, singular: string, pluralForm = singular + 's'): string {
  return value === 1 ? singular : pluralForm
}

function overrideList(overrides: readonly TartGuideBackendOverride[]): string {
  return overrides
    .map(({ graph, backend }) => graph + ' → ' + backendLabel(backend))
    .join(', ')
}

export function getTartGuideMessage(snapshot: TartGuideSnapshot): TartGuideMessage {
  if (snapshot.error) {
    return {
      tone: 'error',
      kicker: 'Lil Tart noticed a problem',
      title: 'Something got a little singed.',
      message: snapshot.error,
    }
  }

  if (snapshot.operation === 'model-load') {
    const progress = snapshot.progressPercent === null
      ? ''
      : ' ' + snapshot.progressPercent + '%'
    return {
      tone: 'working',
      kicker: 'Model oven',
      title: snapshot.selectedModelName
        ? 'Warming up ' + snapshot.selectedModelName
        : 'Warming the oven',
      message: 'Fetching and compiling the model' + progress + '. I’ll keep an eye on the runtime while it gets ready.',
    }
  }

  if (snapshot.operation === 'preflight') {
    return {
      tone: 'working',
      kicker: 'Runtime check',
      title: 'Running preflight.',
      message: 'The model is already loaded. I’m checking compile + synthetic inference on the main graph now.',
    }
  }

  if (snapshot.operation === 'inference') {
    return {
      tone: 'working',
      kicker: 'Real inference',
      title: 'Running your input.',
      message: snapshot.backendOverrides.length > 0
        ? 'The pipeline is executing with explicit graph backend constraints (' + overrideList(snapshot.backendOverrides) + '). If it completes, the session proof will preserve each graph’s requested and resolved backend.'
        : 'The loaded graph path is executing now. If it completes, I’ll capture each emitted runtime event as session proof.',
    }
  }

  if (!snapshot.selectedModelName) {
    return {
      tone: 'idle',
      kicker: 'Lil Tart guide',
      title: 'Pick what you want to run.',
      message: 'Choose a model and I’ll follow the real runtime state: download, backend resolution, preflight, inference, and anything that goes sideways.',
    }
  }

  if (snapshot.loaded && snapshot.backendOverrides.length > 0) {
    const selected = backendLabel(snapshot.requestedBackend)
    const overrides = overrideList(snapshot.backendOverrides)
    const fallbackMessage = snapshot.fallbackCount > 0
      ? ' After those explicit graph requests, the runtime also took ' + snapshot.fallbackCount + ' ' + plural(snapshot.fallbackCount, 'fallback') + '; that fallback evidence is preserved per graph in the session proof.'
      : ' Those are direct correctness requests, not runtime fallbacks.'

    return {
      tone: 'warning',
      kicker: 'Correctness override',
      title: snapshot.backendOverrides.length === 1
        ? 'This model has a correctness-pinned graph.'
        : 'This model has correctness-pinned graphs.',
      message: 'You selected ' + selected + '. Lil Tart overrides ' + overrides + '.' + fallbackMessage,
      action: snapshot.preflightComplete ? undefined : 'preflight',
      actionLabel: snapshot.preflightComplete ? undefined : 'Run preflight',
    }
  }

  if (snapshot.loaded && snapshot.fallbackCount > 0) {
    const requested = backendLabel(snapshot.requestedBackend)
    const graphDetail = snapshot.runtimeGraphBackends.length > 1
      ? ' across ' + snapshot.runtimeGraphBackends.length + ' graph paths'
      : ''
    return {
      tone: 'warning',
      kicker: 'Runtime receipt',
      title: 'It works, but the runtime took a fallback.',
      message: snapshot.pathProofAvailable
        ? requested + ' was selected and the runtime recorded ' + snapshot.fallbackCount + ' ' + plural(snapshot.fallbackCount, 'fallback') + graphDetail + '. The session proof preserves each graph’s requested and resolved backend; a recipe must not hide it.'
        : requested + ' was selected and the runtime recorded ' + snapshot.fallbackCount + ' ' + plural(snapshot.fallbackCount, 'fallback') + '. It can still run, but this is worth checking before you ship it.',
      action: snapshot.preflightComplete ? undefined : 'preflight',
      actionLabel: snapshot.preflightComplete ? undefined : 'Run preflight',
    }
  }

  if (snapshot.pathProofAvailable) {
    const graphMessage = snapshot.runtimeGraphBackends.length > 1
      ? ' across ' + snapshot.runtimeGraphBackends.length + ' graph paths'
      : ''
    return {
      tone: 'success',
      kicker: 'Inference complete',
      title: 'That’s a real working path. ✨',
      message: 'The model just ran locally' + graphMessage + ', and I captured the requested and resolved backend for every inference graph. That proof can now feed an “add this to my app” recipe without guessing from UI state.',
    }
  }

  if (snapshot.preflightComplete) {
    return {
      tone: 'success',
      kicker: 'Preflight passed',
      title: 'The main graph is alive.',
      message: 'Compile + synthetic inference succeeded on the main graph via ' + backendLabel(snapshot.resolvedBackend) + '. Feed it real input next so multi-graph adapters can prove the rest of their runtime path too.',
    }
  }

  if (snapshot.loaded) {
    return {
      tone: 'success',
      kicker: 'Model ready',
      title: snapshot.selectedModelName + ' is warm.',
      message: 'The main graph resolved to ' + backendLabel(snapshot.resolvedBackend) + '. Run real input to capture the complete per-graph runtime path.',
      action: 'preflight',
      actionLabel: 'Run preflight',
    }
  }

  return {
    tone: 'idle',
    kicker: 'Lil Tart guide',
    title: snapshot.selectedModelName + ' is selected.',
    message: 'Load the model and I’ll stay with the runtime instead of giving you generic setup advice.',
  }
}
