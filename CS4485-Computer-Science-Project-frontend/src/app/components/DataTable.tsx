import fileDownload from 'js-file-download'
import { useMemo, useState } from 'react'

type Column<T> = { key: keyof T; header: string; render?: (row: T) => React.ReactNode }

type Props<T> = {
  data: T[]
  columns: Column<T>[]
  pageSize?: number
  filename?: string
}

export function DataTable<T extends Record<string, any>>({ data, columns, pageSize = 10, filename = 'export.csv' }: Props<T>) {
  const [page, setPage] = useState(0)
  const pages = Math.ceil(data.length / pageSize)
  const start = page * pageSize
  const pageRows = useMemo(() => data.slice(start, start + pageSize), [data, start, pageSize])

  function exportCSV() {
    const header = columns.map(c => c.header).join(',')
    const rows = data.map(row => columns.map(c => JSON.stringify(row[c.key])).join(',')).join('\n')
    const csv = header + '\n' + rows
    fileDownload(new Blob([csv], { type: 'text/csv;charset=utf-8;' }), filename)
  }

  return (
    <div className="card p-4">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left">
              {columns.map(col => (
                <th key={String(col.key)} className="py-2 pr-4 font-medium text-neutral-600 dark:text-neutral-300">{col.header}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {pageRows.map((row, i) => (
              <tr key={i} className="border-t border-neutral-200/60 dark:border-neutral-800">
                {columns.map(col => (
                  <td key={String(col.key)} className="py-2 pr-4">{col.render ? col.render(row) : String(row[col.key])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-3 flex items-center justify-between">
        <div className="text-xs text-neutral-500">Page {page + 1} / {pages || 1}</div>
        <div className="flex items-center gap-2">
          <button className="px-3 py-1 rounded-md border" onClick={() => setPage(p => Math.max(0, p - 1))} disabled={page === 0}>Prev</button>
          <button className="px-3 py-1 rounded-md border" onClick={() => setPage(p => Math.min(pages - 1, p + 1))} disabled={page >= pages - 1}>Next</button>
          <button className="px-3 py-1 rounded-md border" onClick={exportCSV}>Export CSV</button>
        </div>
      </div>
    </div>
  )
}


