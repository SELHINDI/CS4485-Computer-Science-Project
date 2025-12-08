import { ArrowDownRight, ArrowUpRight, Copy } from "lucide-react"
import { useState } from "react"

type Props = {
  title: string
  value: string
  delta?: number
  spark?: number[]
}

export function MetricCard({ title, value, delta, spark }: Props) {
  const [copied, setCopied] = useState(false)
  const sign = delta === undefined ? 0 : delta

  const copy = async () => {
    await navigator.clipboard.writeText(value)
    setCopied(true)
    setTimeout(() => setCopied(false), 1200)
  }

  return (
    <div className="card p-4">
      <div className="flex items-center justify-between">
        <div className="text-sm text-neutral-600 dark:text-neutral-300">{title}</div>
        <button aria-label="Copy value" className="p-1 rounded hover:bg-neutral-100 dark:hover:bg-neutral-800" onClick={copy}>
          <Copy size={14} />
        </button>
      </div>
      <div className="mt-2 flex items-end gap-2">
        <div className="text-3xl font-semibold">{value}</div>
        {delta !== undefined && (
          <div className={`flex items-center text-sm ${sign >= 0 ? 'text-green-600' : 'text-red-500'}`}>
            {sign >= 0 ? <ArrowUpRight size={16} /> : <ArrowDownRight size={16} />}
            <span className="ml-0.5">{Math.abs(delta!).toFixed(2)}</span>
          </div>
        )}
      </div>
      {spark && spark.length > 1 && (
        <div className="mt-3 h-10">
          <Sparkline data={spark} />
        </div>
      )}
      {copied && <div className="mt-2 text-xs text-neutral-500">Copied</div>}
    </div>
  )
}

function Sparkline({ data }: { data: number[] }) {
  const w = 200
  const h = 40
  const min = Math.min(...data)
  const max = Math.max(...data)
  const path = data.map((v, i) => {
    const x = (i / (data.length - 1)) * (w - 4) + 2
    const y = h - ((v - min) / (max - min || 1)) * (h - 4) - 2
    return `${i === 0 ? 'M' : 'L'}${x},${y}`
  }).join(' ')
  return (
    <svg width="100%" height="100%" viewBox={`0 0 ${w} ${h}`} aria-label="sparkline">
      <path d={path} fill="none" className="stroke-primary" strokeWidth="2" />
    </svg>
  )
}


