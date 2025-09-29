import { describe, it, expect } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { MetricCard } from './MetricCard'

describe('MetricCard', () => {
  it('renders title and value and copies', async () => {
    Object.assign(navigator, { clipboard: { writeText: async () => {} } })
    render(<MetricCard title="Next Quarter GDP" value="20,834.2" delta={0.5} spark={[1,2,3]} />)
    expect(screen.getByText('Next Quarter GDP')).toBeInTheDocument()
    expect(screen.getByText('20,834.2')).toBeInTheDocument()
    const btn = screen.getByRole('button', { name: /copy/i })
    fireEvent.click(btn)
  })
})


