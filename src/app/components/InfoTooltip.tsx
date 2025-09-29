import * as Tooltip from '@radix-ui/react-tooltip'

export function InfoTooltip({ content }: { content: string }) {
  return (
    <Tooltip.Provider delayDuration={200}>
      <Tooltip.Root>
        <Tooltip.Trigger asChild>
          <button aria-label="Info" className="inline-flex items-center justify-center w-5 h-5 rounded-full border text-xs">i</button>
        </Tooltip.Trigger>
        <Tooltip.Portal>
          <Tooltip.Content className="text-xs card p-2" sideOffset={6}>
            {content}
            <Tooltip.Arrow />
          </Tooltip.Content>
        </Tooltip.Portal>
      </Tooltip.Root>
    </Tooltip.Provider>
  )
}


