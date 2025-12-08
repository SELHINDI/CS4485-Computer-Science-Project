export function Skeleton({ className = '' }: { className?: string }) {
  return <div className={`animate-pulse rounded-md bg-neutral-200/70 dark:bg-neutral-800 ${className}`} />
}

export function ErrorState({ message, onRetry }: { message?: string; onRetry?: () => void }) {
  return (
    <div className="card p-4 text-sm">
      <div className="text-red-600">{message || 'Something went wrong.'}</div>
      {onRetry && <button className="mt-2 px-3 py-1 rounded-md border" onClick={onRetry}>Retry</button>}
    </div>
  )
}

export function EmptyState({ title = 'No data', description }: { title?: string; description?: string }) {
  return (
    <div className="card p-6 text-sm">
      <div className="font-medium">{title}</div>
      {description && <div className="mt-1 text-neutral-500">{description}</div>}
    </div>
  )
}


