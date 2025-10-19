/**
 * ArXiv Monitor - Main Pipeline
 * Orchestrates the full workflow: Crawler → Summarizer → Generator → Notifier
 */

import logger from './utils/logger'
import { runCrawler } from './crawler'
import { runSummarizer } from './summarizer'
import { runGenerator } from './generator'
import { notifyNewPapers, notifyPipelineCompletion } from './notifier'

/**
 * Main pipeline execution
 */
async function runPipeline() {
  const startTime = Date.now()

  logger.info('🚀 Starting ArXiv Monitor Pipeline')
  logger.info('━'.repeat(60))

  try {
    // Step 1: Crawl new papers from ArXiv
    logger.info('📡 Step 1/3: Crawling papers from ArXiv...')
    const crawlerResult = await runCrawler()

    if (!crawlerResult.success) {
      logger.error('❌ Crawler failed, stopping pipeline')
      return {
        success: false,
        stage: 'crawler',
        error: crawlerResult.error,
      }
    }

    logger.info(`✅ Crawler completed: ${crawlerResult.newPapers} new papers`)
    logger.info('━'.repeat(60))

    // If no new papers, skip subsequent steps
    if (crawlerResult.newPapers === 0) {
      logger.info('✅ No new papers to process, pipeline complete')
      return {
        success: true,
        stage: 'crawler',
        newPapers: 0,
      }
    }

    // Step 2: Summarize papers using LLM
    logger.info('🤖 Step 2/3: Summarizing papers with GPT-4...')
    const summarizerResult = await runSummarizer()

    if (!summarizerResult.success) {
      logger.error('❌ Summarizer failed, stopping pipeline')
      return {
        success: false,
        stage: 'summarizer',
        error: summarizerResult.error,
      }
    }

    logger.info(
      `✅ Summarizer completed: ${summarizerResult.succeeded}/${summarizerResult.processed} papers`
    )
    logger.info(`💰 Estimated cost: $${summarizerResult.estimatedCost?.toFixed(3) || '0.000'}`)
    logger.info('━'.repeat(60))

    // If no papers were summarized, skip MDX generation
    if (summarizerResult.succeeded === 0) {
      logger.warn('⚠️  No papers were summarized, skipping MDX generation')
      return {
        success: true,
        stage: 'summarizer',
        summarized: 0,
      }
    }

    // Step 3: Generate MDX files
    logger.info('📝 Step 3/3: Generating MDX files...')
    const generatorResult = await runGenerator(50, false) // autoCommit disabled by default

    if (!generatorResult.success) {
      logger.error('❌ Generator failed')
      return {
        success: false,
        stage: 'generator',
        error: generatorResult.error,
      }
    }

    logger.info(
      `✅ Generator completed: ${generatorResult.succeeded ?? 0}/${generatorResult.processed ?? 0} MDX files`
    )
    logger.info('━'.repeat(60))

    // Step 4: Send notifications
    if ((generatorResult.succeeded ?? 0) > 0) {
      logger.info('📢 Step 4/4: Sending notifications...')
      const notifierSuccess = await notifyNewPapers()

      if (notifierSuccess) {
        logger.info('✅ Notifications sent successfully')
      } else {
        logger.warn('⚠️  Notifications not sent (webhooks not configured)')
      }
      logger.info('━'.repeat(60))
    }

    // Pipeline summary
    const totalDuration = ((Date.now() - startTime) / 1000).toFixed(2)

    logger.info('🎉 Pipeline completed successfully!')
    logger.info('📊 Summary:')
    logger.info(`  - New papers crawled: ${crawlerResult.newPapers}`)
    logger.info(`  - Papers summarized: ${summarizerResult.succeeded}`)
    logger.info(`  - MDX files generated: ${generatorResult.succeeded ?? 0}`)
    logger.info(`  - Total duration: ${totalDuration}s`)
    logger.info(`  - Estimated cost: $${summarizerResult.estimatedCost?.toFixed(3) || '0.000'}`)

    // Send pipeline completion summary
    await notifyPipelineCompletion({
      crawled: crawlerResult.newPapers ?? 0,
      summarized: summarizerResult.succeeded ?? 0,
      generated: generatorResult.succeeded ?? 0,
      duration: parseFloat(totalDuration),
      cost: summarizerResult.estimatedCost || 0,
    })

    return {
      success: true,
      crawled: crawlerResult.newPapers,
      summarized: summarizerResult.succeeded,
      generated: generatorResult.succeeded,
      duration: parseFloat(totalDuration),
      cost: summarizerResult.estimatedCost || 0,
    }
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error)
    logger.error('💥 Pipeline failed with unexpected error', { error: errorMessage })

    return {
      success: false,
      error: errorMessage,
    }
  }
}

/**
 * Run pipeline if executed directly
 */
if (require.main === module) {
  runPipeline()
    .then((result) => {
      if (result.success) {
        logger.info('✅ Pipeline execution completed')
        process.exit(0)
      } else {
        logger.error(`❌ Pipeline failed at stage: ${result.stage || 'unknown'}`)
        process.exit(1)
      }
    })
    .catch((error) => {
      logger.error('💥 Fatal error', {
        error: error instanceof Error ? error.message : String(error),
      })
      process.exit(1)
    })
}

export { runPipeline }
