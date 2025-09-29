import { describe, it, expect } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useOneStep } from './useData'

function wrapper({ children }: { children: React.ReactNode }) {
  const client = new QueryClient()
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>
}

describe('useOneStep', () => {
  it('fetches one-step prediction', async () => {
    const { result } = renderHook(() => useOneStep('US', '2024-12-31'), { wrapper })
    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(result.current.data?.country).toBe('US')
    expect(result.current.data?.horizon).toBe(1)
  })
})


