/**
 * Paper Summarizer
 * Generates multi-level summaries using OpenAI GPT-4
 */

import { PrismaClient } from '../../../../node_modules/@prisma/client'
import logger from '../utils/logger'
import { generateJsonCompletion, calculateCost, estimateTokens } from './openai-client'
import {
  SYSTEM_PROMPT,
  generateSummarizationPrompt,
  validateSummarizationResult,
} from './prompts'

const prisma = new PrismaClient()

/**
 * Summarization result interface
 */
interface SummarizationResult {
  summaryShort: string
  summaryMedium: string
  summaryLong: string
  keywords: string[]
  relatedModules: string[]
}

/**
 * Summarize a single paper
 */
export async function summarizePaper(paperId: string): Promise<boolean> {
  const startTime = Date.now()

  try {
    // Fetch paper from database
    const paper = await prisma.arXiv_Paper.findUnique({
      where: { id: paperId },
    })

    if (!paper) {
      logger.error(`Paper ${paperId} not found`)
      return false
    }

    if (paper.status !== 'CRAWLED') {
      logger.warn(`Paper ${paper.arxivId} is not in CRAWLED status (${paper.status})`)
      return false
    }

    logger.info(`📝 Summarizing paper: ${paper.arxivId}`)
    logger.debug(`Title: ${paper.title}`)

    // Generate summarization prompt
    const userPrompt = generateSummarizationPrompt(
      paper.title,
      paper.abstract,
      paper.authors,
      paper.categories
    )

    // Estimate cost
    const estimatedInputTokens = estimateTokens(SYSTEM_PROMPT + userPrompt)
    const estimatedOutputTokens = 800 // Approximate for summaries
    const estimatedCost = calculateCost(estimatedInputTokens, estimatedOutputTokens)

    logger.debug('Estimated cost', {
      inputTokens: estimatedInputTokens,
      outputTokens: estimatedOutputTokens,
      cost: `$${estimatedCost.toFixed(4)}`,
    })

    // Generate summaries using OpenAI
    const result = await generateJsonCompletion<SummarizationResult>(
      SYSTEM_PROMPT,
      userPrompt,
      0.7,
      1500
    )

    // Validate result
    const validation = validateSummarizationResult(result)
    if (!validation.valid) {
      logger.error('Invalid summarization result', { errors: validation.errors })
      throw new Error(`Validation failed: ${validation.errors.join(', ')}`)
    }

    // Update paper in database
    await prisma.arXiv_Paper.update({
      where: { id: paperId },
      data: {
        summaryShort: result.summaryShort,
        summaryMedium: result.summaryMedium,
        summaryLong: result.summaryLong,
        keywords: result.keywords,
        relatedModules: result.relatedModules,
        status: 'SUMMARIZED',
        summarizedAt: new Date(),
      },
    })

    // Log success
    await prisma.arXiv_ProcessingLog.create({
      data: {
        paperId,
        stage: 'summarizer',
        status: 'success',
        message: 'Paper successfully summarized',
        duration: Date.now() - startTime,
      },
    })

    const duration = ((Date.now() - startTime) / 1000).toFixed(2)
    logger.info(`✅ Summarized paper ${paper.arxivId} in ${duration}s`)

    return true
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error)
    logger.error(`Failed to summarize paper ${paperId}`, { error: errorMessage })

    // Log error
    try {
      await prisma.arXiv_ProcessingLog.create({
        data: {
          paperId,
          stage: 'summarizer',
          status: 'error',
          message: 'Failed to summarize paper',
          errorStack: errorMessage,
          duration: Date.now() - startTime,
        },
      })

      // Update paper status to FAILED
      await prisma.arXiv_Paper.update({
        where: { id: paperId },
        data: {
          status: 'FAILED',
          errorMessage: errorMessage,
        },
      })
    } catch (logError) {
      logger.error('Failed to log summarization error', { error: logError })
    }

    return false
  }
}

/**
 * Main summarizer function - process all crawled papers
 */
export async function runSummarizer(limit: number = 50) {
  const startTime = Date.now()

  try {
    logger.info('🤖 Starting Paper Summarizer')

    // Fetch papers with CRAWLED status
    const papers = await prisma.arXiv_Paper.findMany({
      where: {
        status: 'CRAWLED',
      },
      take: limit,
      orderBy: {
        createdAt: 'desc',
      },
    })

    if (papers.length === 0) {
      logger.info('✅ No papers to summarize')
      return { success: true, processed: 0, succeeded: 0, failed: 0 }
    }

    logger.info(`📚 Found ${papers.length} papers to summarize`)

    let succeeded = 0
    let failed = 0
    let totalCost = 0

    // Process papers sequentially to avoid rate limits
    for (const paper of papers) {
      try {
        const success = await summarizePaper(paper.id)

        if (success) {
          succeeded++
          // Rough cost estimate per paper: ~$0.015
          totalCost += 0.015
        } else {
          failed++
        }

        // Rate limiting: wait 1 second between requests
        await new Promise((resolve) => setTimeout(resolve, 1000))
      } catch (error) {
        failed++
        logger.error(`Error processing paper ${paper.arxivId}`, { error })
      }
    }

    const duration = ((Date.now() - startTime) / 1000).toFixed(2)

    logger.info('🎉 Summarizer completed', {
      duration: `${duration}s`,
      total: papers.length,
      succeeded,
      failed,
      estimatedCost: `$${totalCost.toFixed(3)}`,
    })

    return {
      success: true,
      processed: papers.length,
      succeeded,
      failed,
      duration: parseFloat(duration),
      estimatedCost: totalCost,
    }
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error)
    logger.error('❌ Summarizer failed', { error: errorMessage })

    return {
      success: false,
      error: errorMessage,
    }
  } finally {
    await prisma.$disconnect()
  }
}

/**
 * Run summarizer if executed directly
 */
if (require.main === module) {
  runSummarizer()
    .then((result) => {
      if (result.success) {
        logger.info('✅ Summarizer execution completed successfully')
        process.exit(0)
      } else {
        logger.error('❌ Summarizer execution failed')
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
