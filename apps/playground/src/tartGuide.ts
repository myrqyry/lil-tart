export type TartGuideTone = 'idle' | 'working' | 'success' | 'warning' | 'error'

export type TartGuideAction = 'preflight'

export interface TartGuideSnapshot {
  selectedModelName: string | null
  loading: boolean
  loaded: boolean
  progressPercent: number | null
  error: string | null
  requestedBackend: string | null
  resolvedBackend: string | null
  fallbackCount: number
  preflightComplete: boolean
  inferenceComplete: boolean
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

function plural(value: number, singular: string, pluralForm = `${singular}s`): string {
  return value === 1 ? singular : pluralForm
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

  if (snapshot.loading) {
    const progress = snapshot.progressPercent === null
      ? ''
      : ` ${snapshot.progressPercent}%`
    return {
      tone: 'working',
      kicker: 'Model oven',
      title: snapshot.selectedModelName
        ? `Warming up ${snapshot.selectedModelName}`
        : 'Warming the oven',
      message: `Fetching and compiling the model${progress}. I’ll keep an eye on the runtime while it gets ready.`,
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

  if (snapshot.loaded && snapshot.fallbackCount > 0) {
    const requested = backendLabel(snapshot.requestedBackend)
    const resolved = backendLabel(snapshot.resolvedBackend)
    return {
      tone: 'warning',
      kicker: 'Runtime receipt',
      title: 'It works, but we took a fallback.',
      message: `You asked for ${requested} and landed on ${resolved} with ${snapshot.fallbackCount} ${plural(snapshot.fallbackCount, 'fallback')}. It can still run, but this is worth checking before you ship it.`,
      action: snapshot.preflightComplete ? undefined : 'preflight',
      actionLabel: snapshot.preflightComplete ? undefined : 'Run preflight',
    }
  }

  if (snapshot.inferenceComplete) {
    return {
      tone: 'success',
      kicker: 'Inference complete',
      title: 'That’s a real working path. ✨',
      message: `The model just ran locally on ${backendLabel(snapshot.resolvedBackend)}. This is the configuration we eventually want to turn into an “add this to my app” recipe.`,
    }
  }

  if (snapshot.preflightComplete) {
    return {
      tone: 'success',
      kicker: 'Preflight passed',
      title: 'The model is alive.',
      message: `Compile + synthetic inference succeeded on ${backendLabel(snapshot.resolvedBackend)}. Feed it real input next; we’ve already proven the runtime path works.`,
    }
  }

  if (snapshot.loaded) {
    return {
      tone: 'success',
      kicker: 'Model ready',
      title: `${snapshot.selectedModelName} is warm.`,
      message: `Resolved to ${backendLabel(snapshot.resolvedBackend)}. Run a preflight before real input so we can prove the graph and output path together.`,
      action: 'preflight',
      actionLabel: 'Run preflight',
    }
  }

  return {
    tone: 'idle',
    kicker: 'Lil Tart guide',
    title: `${snapshot.selectedModelName} is selected.`,
    message: 'Load the model and I’ll stay with the runtime instead of giving you generic setup advice.',
  }
}
