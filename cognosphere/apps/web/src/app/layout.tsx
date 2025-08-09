import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import { Providers } from './providers'
import '@/styles/globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Cognosphere - Knowledge Simulator Platform',
  description: 'Interactive learning environment for ontology and semantic web technologies',
  keywords: ['ontology', 'semantic web', 'RDF', 'OWL', 'SPARQL', 'knowledge graph', 'learning platform'],
  authors: [{ name: 'Cognosphere Team' }],
  openGraph: {
    title: 'Cognosphere - Knowledge Simulator Platform',
    description: 'Interactive learning environment for ontology and semantic web technologies',
    type: 'website',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="ko" suppressHydrationWarning>
      <body className={inter.className}>
        <Providers>{children}</Providers>
      </body>
    </html>
  )
}