/**
 * ArXiv Crawler
 * Main entry point for crawling papers from ArXiv
 */

import { PrismaClient } from '../../../../node_modules/@prisma/client'
import logger from '../utils/logger'
import { config } from '../utils/config'
import { fetchArXivPapers } from './arxiv-api'
import { filterNewPapers, cleanup } from './deduplicator'

const prisma = new PrismaClient()

/**
 * Main crawler function
 */
export async function runCrawler() {
  const startTime = Date.now()

  try {
    logger.info('🚀 Starting ArXiv crawler')

    // Step 1: Fetch papers from ArXiv API
    logger.info('📡 Fetching papers from ArXiv API...')
    const papers = await fetchArXivPapers()

    if (papers.length === 0) {
      logger.warn('⚠️  No papers fetched from ArXiv')
      return { success: true, newPapers: 0, totalFetched: 0 }
    }

    logger.info(`✅ Fetched ${papers.length} papers from ArXiv`)

    // Step 2: Filter out existing papers
    logger.info('🔍 Checking for duplicates...')
    const newPapers = await filterNewPapers(papers)

    if (newPapers.length === 0) {
      logger.info('✅ No new papers to process')
      return { success: true, newPapers: 0, totalFetched: papers.length }
    }

    logger.info(`📝 Found ${newPapers.length} new papers to save`)

    // Step 3: Save new papers to database
    logger.info('💾 Saving papers to database...')

    let savedCount = 0
    let errorCount = 0

    for (const paper of newPapers) {
      try {
        // Create paper in database
        const createdPaper = await prisma.arXiv_Paper.create({
          data: {
            arxivId: paper.arxivId,
            title: paper.title,
            authors: paper.authors,
            abstract: paper.abstract,
            categories: paper.categories,
            publishedDate: paper.publishedDate,
            pdfUrl: paper.pdfUrl,
            status: 'CRAWLED',
          },
        })

        // Log processing success using the database ID
        await prisma.arXiv_ProcessingLog.create({
          data: {
            paperId: createdPaper.id,
            stage: 'crawler',
            status: 'success',
            message: 'Paper successfully crawled and saved',
          },
        })

        savedCount++
        logger.debug(`✅ Saved paper: ${paper.arxivId}`)
      } catch (error) {
        errorCount++
        const errorMessage = error instanceof Error ? error.message : String(error)
        logger.error(`❌ Failed to save paper ${paper.arxivId}`, { error: errorMessage })

        // Try to find the paper if it exists to log the error
        try {
          const existingPaper = await prisma.arXiv_Paper.findUnique({
            where: { arxivId: paper.arxivId },
          })

          if (existingPaper) {
            await prisma.arXiv_ProcessingLog.create({
              data: {
                paperId: existingPaper.id,
                stage: 'crawler',
                status: 'error',
                message: 'Failed to save paper',
                errorStack: errorMessage,
              },
            })
          }
        } catch (logError) {
          logger.error('Failed to log processing error', { error: logError })
        }
      }
    }

    // Step 4: Summary
    const duration = ((Date.now() - startTime) / 1000).toFixed(2)

    logger.info('🎉 Crawler completed', {
      duration: `${duration}s`,
      totalFetched: papers.length,
      newPapers: newPapers.length,
      saved: savedCount,
      errors: errorCount,
    })

    return {
      success: true,
      totalFetched: papers.length,
      newPapers: newPapers.length,
      saved: savedCount,
      errors: errorCount,
      duration: parseFloat(duration),
    }
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error)
    logger.error('❌ Crawler failed', { error: errorMessage })

    return {
      success: false,
      error: errorMessage,
    }
  } finally {
    await prisma.$disconnect()
    await cleanup()
  }
}

/**
 * Run crawler if executed directly
 */
if (require.main === module) {
  runCrawler()
    .then((result) => {
      if (result.success) {
        logger.info('✅ Crawler execution completed successfully')
        process.exit(0)
      } else {
        logger.error('❌ Crawler execution failed')
        process.exit(1)
      }
    })
    .catch((error) => {
      logger.error('💥 Unexpected error', {
        error: error instanceof Error ? error.message : String(error),
      })
      process.exit(1)
    })
}
