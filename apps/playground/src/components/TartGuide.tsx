import {
  CheckCircleRegular,
  CloseCircleRegular,
  InformationRegular,
  Loading3Regular,
  WarningRegular,
} from '@mingcute/react/core-regular'
import type { TartGuideMessage, TartGuideTone } from '../tartGuide'

interface TartGuideProps {
  guide: TartGuideMessage
  onRunPreflight?: () => void
}

const toneClasses: Record<TartGuideTone, string> = {
  idle: 'border-outline/50 bg-surface-container',
  working: 'border-secondary/50 bg-secondary-container/20',
  success: 'border-primary/50 bg-primary/5',
  warning: 'border-amber-500/50 bg-amber-500/5',
  error: 'border-error/60 bg-error-container/20',
}

function ToneIcon({ tone }: { tone: TartGuideTone }) {
  const className = 'h-5 w-5'
  if (tone === 'working') return <Loading3Regular className={className + ' animate-spin'} />
  if (tone === 'success') return <CheckCircleRegular className={className} />
  if (tone === 'warning') return <WarningRegular className={className} />
  if (tone === 'error') return <CloseCircleRegular className={className} />
  return <InformationRegular className={className} />
}

export default function TartGuide({ guide, onRunPreflight }: TartGuideProps) {
  return (
    <aside
      className={`runtime-guide w-full p-3.5 text-on-surface ${toneClasses[guide.tone]}`}
      aria-label="Runtime status"
    >
      <div className="flex items-start gap-3">
        <div className="runtime-guide__icon mt-0.5 shrink-0" aria-hidden="true">
          <ToneIcon tone={guide.tone} />
        </div>

        <div className="min-w-0 flex-1" aria-live="polite">
          <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
            <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-on-surface-muted">
              {guide.kicker}
            </p>
            <h2 className="text-[15px] font-semibold text-on-surface">{guide.title}</h2>
          </div>

          <p className="mt-1 text-[13px] leading-snug text-on-surface-variant">{guide.message}</p>

          {guide.action === 'preflight' && onRunPreflight && (
            <button
              type="button"
              onClick={onRunPreflight}
              className="expressive-primary-button mt-2 px-3.5 py-2 text-[13px] font-semibold text-on-primary"
            >
              {guide.actionLabel ?? 'Run preflight'}
            </button>
          )}
        </div>
      </div>
    </aside>
  )
}
