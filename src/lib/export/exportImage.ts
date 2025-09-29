import * as htmlToImage from 'html-to-image'

export async function exportNodeToPng(node: HTMLElement, filename: string) {
  const dataUrl = await htmlToImage.toPng(node, { pixelRatio: 2, backgroundColor: 'transparent' })
  const link = document.createElement('a')
  link.download = filename
  link.href = dataUrl
  link.click()
}


