import { Metadata } from 'next'

export const metadata: Metadata = {
  title: '반도체 - KSS',
  description: '반도체 설계부터 제조까지 - 칩의 모든 것',
}

export default function SemiconductorLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return <>{children}</>
}
