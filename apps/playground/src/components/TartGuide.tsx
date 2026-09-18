import { useState } from 'react'
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

function TartMascot({ tone, compact = false }: { tone: TartGuideTone; compact?: boolean }) {
  const isError = tone === 'error'
  const isHappy = tone === 'success'
  const isWorking = tone === 'working'
  const isWarning = tone === 'warning'

  return (
    <svg
      viewBox="0 0 120 120"
      role="img"
      aria-label="Lil Tart mascot"
      className={compact ? 'h-14 w-14' : 'h-20 w-20 shrink-0'}
    >
      <ellipse cx="60" cy="103" rx="34" ry="7" fill="currentColor" opacity="0.12" />

      {isWorking && (
        <g fill="none" stroke="currentColor" strokeLinecap="round" opacity="0.45">
          <path d="M45 22c-7-8 6-10 0-18" />
          <path d="M60 18c-7-8 6-10 0-18" />
          <path d="M75 22c-7-8 6-10 0-18" />
        </g>
      )}

      <path
        d="M31 43c2-13 13-23 29-23s27 10 29 23l6 43c1 9-6 17-15 17H40c-9 0-16-8-15-17z"
        fill="#D99A45"
        stroke="#6B4423"
        strokeWidth="4"
        strokeLinejoin="round"
      />
      <ellipse cx="60" cy="50" rx="29" ry="24" fill="#F2C66D" stroke="#6B4423" strokeWidth="4" />
      <ellipse cx="60" cy="50" rx="22" ry="17" fill={isError ? '#C94A4A' : isWarning ? '#D97706' : '#B83F5B'} />
      <ellipse cx="54" cy="44" rx="8" ry="5" fill="#FFFFFF" opacity="0.2" transform="rotate(-18 54 44)" />

      <g stroke="#3A2416" strokeWidth="3.5" strokeLinecap="round" fill="none">
        {isError ? (
          <>
            <path d="M49 47l6 6m0-6-6 6" />
            <path d="M65 47l6 6m0-6-6 6" />
          </>
        ) : isHappy ? (
          <>
            <path d="M48 51c3-5 7-5 10 0" />
            <path d="M62 51c3-5 7-5 10 0" />
          </>
        ) : (
          <>
            <path d="M53 48v3" />
            <path d="M67 48v3" />
          </>
        )}

        {isWarning ? (
          <path d="M54 61c4-3 8 3 12 0" />
        ) : isError ? (
          <path d="M53 64c5-5 9-5 14 0" />
        ) : (
          <path d="M53 59c4 5 10 5 14 0" />
        )}

        <path d="M29 67c-9 0-12 6-15 12" />
        <path d="M91 67c9 0 12 6 15 12" />
      </g>

      <circle cx="14" cy="80" r="4" fill="#F2C66D" stroke="#6B4423" strokeWidth="2" />
      <circle cx="106" cy="80" r="4" fill="#F2C66D" stroke="#6B4423" strokeWidth="2" />
    </svg>
  )
}

export default function TartGuide({ guide, onRunPreflight }: TartGuideProps) {
  const [collapsed, setCollapsed] = useState(false)

  if (collapsed) {
    return (
      <button
        type="button"
        onClick={() => setCollapsed(false)}
        className="fixed bottom-4 right-4 z-50 rounded-full border border-outline/50 bg-surface-container-high p-1.5 text-on-surface shadow-lg transition-transform hover:scale-105"
        aria-label="Open Lil Tart guide"
        title="Open Lil Tart guide"
      >
        <TartMascot tone={guide.tone} compact />
      </button>
    )
  }

  return (
    <aside
      className={`fixed bottom-4 right-4 z-50 w-[min(24rem,calc(100vw-2rem))] rounded-2xl border p-3 text-on-surface shadow-xl backdrop-blur-md ${toneClasses[guide.tone]}`}
      aria-label="Lil Tart guide"
    >
      <div className="flex items-start gap-3">
        <TartMascot tone={guide.tone} />

        <div className="min-w-0 flex-1 pt-1" aria-live="polite">
          <div className="flex items-start justify-between gap-2">
            <div>
              <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-on-surface-variant">
                {guide.kicker}
              </p>
              <h2 className="mt-1 text-sm font-semibold text-on-surface">{guide.title}</h2>
            </div>
            <button
              type="button"
              onClick={() => setCollapsed(true)}
              className="rounded-full px-2 py-1 text-xs text-on-surface-variant transition-colors hover:bg-surface-container-high hover:text-on-surface"
              aria-label="Minimize Lil Tart guide"
              title="Minimize"
            >
              −
            </button>
          </div>

          <p className="mt-1.5 text-xs leading-relaxed text-on-surface-variant">{guide.message}</p>

          {guide.action === 'preflight' && onRunPreflight && (
            <button
              type="button"
              onClick={onRunPreflight}
              className="mt-3 rounded-full bg-primary px-4 py-2 text-xs font-semibold text-on-primary shadow-sm transition-transform hover:scale-[1.02] active:scale-[0.98]"
            >
              {guide.actionLabel ?? 'Run preflight'}
            </button>
          )}
        </div>
      </div>
    </aside>
  )
}
