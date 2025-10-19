/**
 * Deduplicator
 * Checks for existing papers in database to avoid duplicates
 */

import { PrismaClient } from '../../../../node_modules/@prisma/client'
import logger from '../utils/logger'
import { ArXivPaper } from './arxiv-api'

const prisma = new PrismaClient()

/**
 * Filter out papers that already exist in the database
 */
export async function filterNewPapers(papers: ArXivPaper[]): Promise<ArXivPaper[]> {
  try {
    // Extract all ArXiv IDs
    const arxivIds = papers.map((p) => p.arxivId)

    logger.debug(`Checking ${arxivIds.length} papers for duplicates`)

    // Query existing papers
    const existingPapers = await prisma.arXiv_Paper.findMany({
      where: {
        arxivId: {
          in: arxivIds,
        },
      },
      select: {
        arxivId: true,
      },
    })

    // Create set of existing IDs for fast lookup
    const existingIds = new Set(existingPapers.map((p: { arxivId: string }) => p.arxivId))

    // Filter out existing papers
    const newPapers = papers.filter((p) => !existingIds.has(p.arxivId))

    logger.info(`Found ${newPapers.length} new papers (${existingIds.size} already exist)`)

    return newPapers
  } catch (error) {
    logger.error('Failed to check for duplicate papers', {
      error: error instanceof Error ? error.message : String(error),
    })
    throw error
  }
}

/**
 * Check if a specific paper exists in the database
 */
export async function paperExists(arxivId: string): Promise<boolean> {
  try {
    const paper = await prisma.arXiv_Paper.findUnique({
      where: { arxivId },
    })

    return paper !== null
  } catch (error) {
    logger.error(`Failed to check if paper ${arxivId} exists`, {
      error: error instanceof Error ? error.message : String(error),
    })
    return false
  }
}

/**
 * Get papers by status
 */
export async function getPapersByStatus(status: string, limit: number = 50) {
  try {
    const papers = await prisma.arXiv_Paper.findMany({
      where: {
        status: status as any,
      },
      take: limit,
      orderBy: {
        createdAt: 'desc',
      },
    })

    return papers
  } catch (error) {
    logger.error(`Failed to get papers with status ${status}`, {
      error: error instanceof Error ? error.message : String(error),
    })
    return []
  }
}

/**
 * Cleanup: disconnect Prisma client
 */
export async function cleanup() {
  await prisma.$disconnect()
}
