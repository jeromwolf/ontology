/**
 * Notifier
 * Sends notifications about new papers to Discord and Slack
 */

import { PrismaClient } from '../../../../node_modules/@prisma/client'
import logger from '../utils/logger'
import { sendDiscordNotification, sendPipelineSummary as sendDiscordSummary } from './discord'
import { sendSlackNotification, sendPipelineSummary as sendSlackSummary } from './slack'

const prisma = new PrismaClient()

interface PaperNotification {
  arxivId: string
  title: string
  authors: string[]
  summaryShort: string
  keywords: string[]
  pdfUrl: string
}

/**
 * Get papers that need notification
 */
async function getPapersForNotification(): Promise<PaperNotification[]> {
  try {
    // Get papers with MDX_GENERATED status that haven't been published yet
    const papers = await prisma.arXiv_Paper.findMany({
      where: {
        status: 'MDX_GENERATED',
      },
      orderBy: {
        createdAt: 'desc',
      },
      take: 10, // Limit to 10 papers per notification
    })

    return papers.map((paper) => ({
      arxivId: paper.arxivId,
      title: paper.title,
      authors: paper.authors,
      summaryShort: paper.summaryShort || '',
      keywords: paper.keywords || [],
      pdfUrl: paper.pdfUrl,
    }))
  } catch (error) {
    logger.error('Failed to fetch papers for notification', { error })
    return []
  }
}

/**
 * Mark papers as published
 */
async function markPapersAsPublished(arxivIds: string[]): Promise<void> {
  try {
    await prisma.arXiv_Paper.updateMany({
      where: {
        arxivId: {
          in: arxivIds,
        },
      },
      data: {
        status: 'PUBLISHED',
        publishedAt: new Date(),
      },
    })

    logger.info(`Marked ${arxivIds.length} papers as PUBLISHED`)
  } catch (error) {
    logger.error('Failed to mark papers as published', { error })
  }
}

/**
 * Send notifications for new papers
 */
export async function notifyNewPapers(papers?: PaperNotification[]): Promise<boolean> {
  try {
    // If papers not provided, fetch from database
    const papersToNotify = papers || (await getPapersForNotification())

    if (papersToNotify.length === 0) {
      logger.info('✅ No new papers to notify')
      return true
    }

    logger.info(`📢 Notifying about ${papersToNotify.length} new papers`)

    // Send to Discord
    const discordSent = await sendDiscordNotification(papersToNotify)

    // Send to Slack
    const slackSent = await sendSlackNotification(papersToNotify)

    // Mark as published if at least one notification was sent
    if (discordSent || slackSent) {
      const arxivIds = papersToNotify.map((p) => p.arxivId)
      await markPapersAsPublished(arxivIds)
    }

    const sentTo: string[] = []
    if (discordSent) sentTo.push('Discord')
    if (slackSent) sentTo.push('Slack')

    if (sentTo.length > 0) {
      logger.info(`✅ Notifications sent to: ${sentTo.join(', ')}`)
      return true
    } else {
      logger.warn('⚠️  No notifications were sent (webhooks not configured)')
      return false
    }
  } catch (error) {
    logger.error('Failed to send notifications', { error })
    return false
  }
}

/**
 * Send pipeline completion summary
 */
export async function notifyPipelineCompletion(summary: {
  crawled: number
  summarized: number
  generated: number
  duration: number
  cost: number
}): Promise<boolean> {
  try {
    logger.info('📊 Sending pipeline completion summary')

    // Send to Discord
    const discordSent = await sendDiscordSummary(summary)

    // Send to Slack
    const slackSent = await sendSlackSummary(summary)

    const sentTo: string[] = []
    if (discordSent) sentTo.push('Discord')
    if (slackSent) sentTo.push('Slack')

    if (sentTo.length > 0) {
      logger.info(`✅ Summary sent to: ${sentTo.join(', ')}`)
      return true
    } else {
      logger.debug('No summary notifications sent (webhooks not configured)')
      return false
    }
  } catch (error) {
    logger.error('Failed to send pipeline summary', { error })
    return false
  }
}

/**
 * Main notifier function
 */
export async function runNotifier() {
  const startTime = Date.now()

  try {
    logger.info('📢 Starting Notifier')

    // Get papers to notify
    const papers = await getPapersForNotification()

    if (papers.length === 0) {
      logger.info('✅ No papers to notify about')
      return {
        success: true,
        notified: 0,
      }
    }

    // Send notifications
    const success = await notifyNewPapers(papers)

    const duration = ((Date.now() - startTime) / 1000).toFixed(2)

    logger.info('🎉 Notifier completed', {
      duration: `${duration}s`,
      papers: papers.length,
      success,
    })

    return {
      success,
      notified: papers.length,
      duration: parseFloat(duration),
    }
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error)
    logger.error('❌ Notifier failed', { error: errorMessage })

    return {
      success: false,
      error: errorMessage,
    }
  } finally {
    await prisma.$disconnect()
  }
}

/**
 * Run notifier if executed directly
 */
if (require.main === module) {
  runNotifier()
    .then((result) => {
      if (result.success) {
        logger.info('✅ Notifier execution completed')
        process.exit(0)
      } else {
        logger.error('❌ Notifier execution failed')
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
