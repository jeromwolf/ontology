/**
 * Slack Notifier
 * Sends notifications to Slack via webhook
 */

import axios from 'axios'
import logger from '../utils/logger'
import { config } from '../utils/config'

interface PaperNotification {
  arxivId: string
  title: string
  authors: string[]
  summaryShort: string
  keywords: string[]
  pdfUrl: string
}

/**
 * Send notification to Slack
 */
export async function sendSlackNotification(papers: PaperNotification[]): Promise<boolean> {
  if (!config.SLACK_WEBHOOK_URL || config.SLACK_WEBHOOK_URL.includes('...')) {
    logger.debug('Slack webhook not configured, skipping')
    return false
  }

  try {
    logger.info(`📤 Sending Slack notification for ${papers.length} papers`)

    // Create blocks for each paper
    const blocks: any[] = [
      {
        type: 'header',
        text: {
          type: 'plain_text',
          text:
            papers.length === 1
              ? '🆕 새로운 AI 논문'
              : `🆕 ${papers.length}개의 새로운 AI 논문`,
          emoji: true,
        },
      },
      {
        type: 'divider',
      },
    ]

    // Add each paper as a section
    papers.forEach((paper) => {
      blocks.push({
        type: 'section',
        text: {
          type: 'mrkdwn',
          text: `*<https://arxiv.org/abs/${paper.arxivId}|${paper.title}>*\n${paper.summaryShort}`,
        },
      })

      blocks.push({
        type: 'context',
        elements: [
          {
            type: 'mrkdwn',
            text: `📄 \`${paper.arxivId}\` | 👥 ${paper.authors.slice(0, 2).join(', ')}${paper.authors.length > 2 ? ' 외' : ''}`,
          },
        ],
      })

      blocks.push({
        type: 'context',
        elements: [
          {
            type: 'mrkdwn',
            text: `🔑 ${paper.keywords.slice(0, 5).join(' · ')}`,
          },
        ],
      })

      blocks.push({
        type: 'divider',
      })
    })

    // Add footer
    blocks.push({
      type: 'context',
      elements: [
        {
          type: 'mrkdwn',
          text: '🤖 KSS ArXiv Monitor | <https://kss.ai.kr|kss.ai.kr>',
        },
      ],
    })

    // Send webhook
    await axios.post(config.SLACK_WEBHOOK_URL, {
      blocks: blocks.slice(0, 50), // Slack limit: 50 blocks per message
    })

    logger.info('✅ Slack notification sent successfully')
    return true
  } catch (error) {
    logger.error('Failed to send Slack notification', {
      error: error instanceof Error ? error.message : String(error),
    })
    return false
  }
}

/**
 * Send simple text notification to Slack
 */
export async function sendSlackMessage(message: string): Promise<boolean> {
  if (!config.SLACK_WEBHOOK_URL || config.SLACK_WEBHOOK_URL.includes('...')) {
    logger.debug('Slack webhook not configured, skipping')
    return false
  }

  try {
    await axios.post(config.SLACK_WEBHOOK_URL, {
      text: message,
    })

    logger.debug('Slack message sent')
    return true
  } catch (error) {
    logger.error('Failed to send Slack message', { error })
    return false
  }
}

/**
 * Send pipeline completion summary to Slack
 */
export async function sendPipelineSummary(summary: {
  crawled: number
  summarized: number
  generated: number
  duration: number
  cost: number
}): Promise<boolean> {
  if (!config.SLACK_WEBHOOK_URL || config.SLACK_WEBHOOK_URL.includes('...')) {
    logger.debug('Slack webhook not configured, skipping')
    return false
  }

  try {
    const blocks: any[] = [
      {
        type: 'header',
        text: {
          type: 'plain_text',
          text: '🎉 ArXiv Monitor 파이프라인 완료',
          emoji: true,
        },
      },
      {
        type: 'section',
        fields: [
          {
            type: 'mrkdwn',
            text: `*📡 수집된 논문*\n${summary.crawled}개`,
          },
          {
            type: 'mrkdwn',
            text: `*🤖 요약 생성*\n${summary.summarized}개`,
          },
          {
            type: 'mrkdwn',
            text: `*📝 MDX 생성*\n${summary.generated}개`,
          },
          {
            type: 'mrkdwn',
            text: `*⏱️ 소요 시간*\n${summary.duration.toFixed(1)}초`,
          },
          {
            type: 'mrkdwn',
            text: `*💰 비용*\n$${summary.cost.toFixed(3)}`,
          },
          {
            type: 'mrkdwn',
            text: `*🌐 플랫폼*\n<https://kss.ai.kr|kss.ai.kr>`,
          },
        ],
      },
      {
        type: 'context',
        elements: [
          {
            type: 'mrkdwn',
            text: `🤖 KSS ArXiv Monitor | ${new Date().toLocaleString('ko-KR')}`,
          },
        ],
      },
    ]

    await axios.post(config.SLACK_WEBHOOK_URL, {
      blocks,
    })

    logger.info('✅ Pipeline summary sent to Slack')
    return true
  } catch (error) {
    logger.error('Failed to send pipeline summary to Slack', { error })
    return false
  }
}
