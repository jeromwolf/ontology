/**
 * Discord Notifier
 * Sends notifications to Discord via webhook
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
 * Send notification to Discord
 */
export async function sendDiscordNotification(papers: PaperNotification[]): Promise<boolean> {
  if (!config.DISCORD_WEBHOOK_URL || config.DISCORD_WEBHOOK_URL.includes('...')) {
    logger.debug('Discord webhook not configured, skipping')
    return false
  }

  try {
    logger.info(`📤 Sending Discord notification for ${papers.length} papers`)

    // Create embed for each paper
    const embeds = papers.map((paper) => ({
      title: paper.title,
      url: `https://arxiv.org/abs/${paper.arxivId}`,
      description: paper.summaryShort,
      color: 0x00d4ff, // KSS blue color
      fields: [
        {
          name: '📄 ArXiv ID',
          value: `[${paper.arxivId}](${paper.pdfUrl})`,
          inline: true,
        },
        {
          name: '👥 저자',
          value: paper.authors.slice(0, 3).join(', ') + (paper.authors.length > 3 ? ' 외' : ''),
          inline: true,
        },
        {
          name: '🔑 키워드',
          value: paper.keywords.join(', '),
          inline: false,
        },
      ],
      timestamp: new Date().toISOString(),
      footer: {
        text: 'KSS ArXiv Monitor',
      },
    }))

    // Send webhook
    await axios.post(config.DISCORD_WEBHOOK_URL, {
      username: 'KSS ArXiv Monitor',
      avatar_url: 'https://arxiv.org/static/browse/0.3.4/images/arxiv-logo-fb.png',
      content:
        papers.length === 1
          ? `🆕 **새로운 AI 논문이 추가되었습니다!**`
          : `🆕 **${papers.length}개의 새로운 AI 논문이 추가되었습니다!**`,
      embeds: embeds.slice(0, 10), // Discord limit: 10 embeds per message
    })

    logger.info('✅ Discord notification sent successfully')
    return true
  } catch (error) {
    logger.error('Failed to send Discord notification', {
      error: error instanceof Error ? error.message : String(error),
    })
    return false
  }
}

/**
 * Send simple text notification to Discord
 */
export async function sendDiscordMessage(message: string): Promise<boolean> {
  if (!config.DISCORD_WEBHOOK_URL || config.DISCORD_WEBHOOK_URL.includes('...')) {
    logger.debug('Discord webhook not configured, skipping')
    return false
  }

  try {
    await axios.post(config.DISCORD_WEBHOOK_URL, {
      username: 'KSS ArXiv Monitor',
      content: message,
    })

    logger.debug('Discord message sent')
    return true
  } catch (error) {
    logger.error('Failed to send Discord message', { error })
    return false
  }
}

/**
 * Send pipeline completion summary to Discord
 */
export async function sendPipelineSummary(summary: {
  crawled: number
  summarized: number
  generated: number
  duration: number
  cost: number
}): Promise<boolean> {
  if (!config.DISCORD_WEBHOOK_URL || config.DISCORD_WEBHOOK_URL.includes('...')) {
    logger.debug('Discord webhook not configured, skipping')
    return false
  }

  try {
    const embed = {
      title: '🎉 ArXiv Monitor 파이프라인 완료',
      color: 0x00ff00, // Green
      fields: [
        {
          name: '📡 수집된 논문',
          value: `${summary.crawled}개`,
          inline: true,
        },
        {
          name: '🤖 요약 생성',
          value: `${summary.summarized}개`,
          inline: true,
        },
        {
          name: '📝 MDX 생성',
          value: `${summary.generated}개`,
          inline: true,
        },
        {
          name: '⏱️ 소요 시간',
          value: `${summary.duration.toFixed(1)}초`,
          inline: true,
        },
        {
          name: '💰 비용',
          value: `$${summary.cost.toFixed(3)}`,
          inline: true,
        },
        {
          name: '🌐 플랫폼',
          value: '[kss.ai.kr](https://kss.ai.kr)',
          inline: true,
        },
      ],
      timestamp: new Date().toISOString(),
      footer: {
        text: 'KSS ArXiv Monitor',
      },
    }

    await axios.post(config.DISCORD_WEBHOOK_URL, {
      username: 'KSS ArXiv Monitor',
      embeds: [embed],
    })

    logger.info('✅ Pipeline summary sent to Discord')
    return true
  } catch (error) {
    logger.error('Failed to send pipeline summary to Discord', { error })
    return false
  }
}
