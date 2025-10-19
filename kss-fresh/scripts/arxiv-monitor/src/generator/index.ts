/**
 * MDX Generator
 * Generates MDX files from summarized papers and commits them to Git
 */

import { PrismaClient } from '../../../../node_modules/@prisma/client'
import fs from 'fs/promises'
import path from 'path'
import logger from '../utils/logger'
import { generateMDX, generateFilename, generateDirectoryPath } from './mdx-template'
import {
  validateGitConfig,
  batchCommitPapers,
  hasUncommittedChanges,
} from './git-operations'

const prisma = new PrismaClient()

// Base directory for generated papers (relative to kss-fresh root)
const PAPERS_BASE_DIR = 'src/app/papers'

/**
 * Generate MDX file for a single paper
 */
export async function generatePaperMDX(paperId: string): Promise<boolean> {
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

    if (paper.status !== 'SUMMARIZED') {
      logger.warn(`Paper ${paper.arxivId} is not in SUMMARIZED status (${paper.status})`)
      return false
    }

    logger.info(`📝 Generating MDX for paper: ${paper.arxivId}`)

    // Generate MDX content
    const mdxContent = generateMDX({
      arxivId: paper.arxivId,
      title: paper.title,
      authors: paper.authors,
      abstract: paper.abstract,
      categories: paper.categories,
      publishedDate: paper.publishedDate,
      pdfUrl: paper.pdfUrl,
      summaryShort: paper.summaryShort || '',
      summaryMedium: paper.summaryMedium || '',
      summaryLong: paper.summaryLong || '',
      keywords: paper.keywords || [],
      relatedModules: paper.relatedModules || [],
    })

    // Generate file path
    const dirPath = generateDirectoryPath(paper.publishedDate)
    const filename = generateFilename(paper.arxivId)
    const fullDirPath = path.join(__dirname, '../../../../', PAPERS_BASE_DIR, dirPath)
    const fullFilePath = path.join(fullDirPath, filename)

    // Create directory if it doesn't exist
    await fs.mkdir(fullDirPath, { recursive: true })

    // Write MDX file
    await fs.writeFile(fullFilePath, mdxContent, 'utf-8')

    logger.info(`✅ Generated MDX file: ${path.join(PAPERS_BASE_DIR, dirPath, filename)}`)

    // Update paper status
    await prisma.arXiv_Paper.update({
      where: { id: paperId },
      data: {
        status: 'MDX_GENERATED',
        mdxGeneratedAt: new Date(),
      },
    })

    // Log success
    await prisma.arXiv_ProcessingLog.create({
      data: {
        paperId,
        stage: 'generator',
        status: 'success',
        message: `MDX file generated: ${path.join(dirPath, filename)}`,
        duration: Date.now() - startTime,
      },
    })

    const duration = ((Date.now() - startTime) / 1000).toFixed(2)
    logger.info(`✅ Generated MDX for ${paper.arxivId} in ${duration}s`)

    return true
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error)
    logger.error(`Failed to generate MDX for paper ${paperId}`, { error: errorMessage })

    // Log error
    try {
      await prisma.arXiv_ProcessingLog.create({
        data: {
          paperId,
          stage: 'generator',
          status: 'error',
          message: 'Failed to generate MDX file',
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
      logger.error('Failed to log generation error', { error: logError })
    }

    return false
  }
}

/**
 * Main generator function - process all summarized papers
 */
export async function runGenerator(limit: number = 50, autoCommit: boolean = false) {
  const startTime = Date.now()

  try {
    logger.info('📝 Starting MDX Generator')

    // Validate Git configuration if autoCommit is enabled
    if (autoCommit) {
      const isValidGit = await validateGitConfig()
      if (!isValidGit) {
        logger.warn('Git configuration invalid. Proceeding without auto-commit.')
        autoCommit = false
      }
    }

    // Fetch papers with SUMMARIZED status
    const papers = await prisma.arXiv_Paper.findMany({
      where: {
        status: 'SUMMARIZED',
      },
      take: limit,
      orderBy: {
        createdAt: 'desc',
      },
    })

    if (papers.length === 0) {
      logger.info('✅ No papers to generate MDX for')
      return { success: true, processed: 0, succeeded: 0, failed: 0 }
    }

    logger.info(`📚 Found ${papers.length} papers to generate MDX for`)

    let succeeded = 0
    let failed = 0
    const generatedPaths: string[] = []
    const generatedIds: string[] = []

    // Process papers sequentially
    for (const paper of papers) {
      const success = await generatePaperMDX(paper.id)

      if (success) {
        succeeded++
        const dirPath = generateDirectoryPath(paper.publishedDate)
        const filename = generateFilename(paper.arxivId)
        generatedPaths.push(path.join(PAPERS_BASE_DIR, dirPath, filename))
        generatedIds.push(paper.arxivId)
      } else {
        failed++
      }
    }

    // Auto-commit if enabled and there are generated files
    if (autoCommit && generatedPaths.length > 0) {
      logger.info('🔄 Committing generated files to Git...')
      try {
        const committed = await batchCommitPapers(generatedPaths, generatedIds)
        if (committed) {
          logger.info('✅ Successfully committed generated files')
        } else {
          logger.warn('⚠️  Failed to commit generated files')
        }
      } catch (error) {
        logger.error('Failed to commit files', { error })
      }
    }

    const duration = ((Date.now() - startTime) / 1000).toFixed(2)

    logger.info('🎉 Generator completed', {
      duration: `${duration}s`,
      total: papers.length,
      succeeded,
      failed,
      committed: autoCommit && generatedPaths.length > 0,
    })

    return {
      success: true,
      processed: papers.length,
      succeeded,
      failed,
      duration: parseFloat(duration),
      generatedPaths,
    }
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error)
    logger.error('❌ Generator failed', { error: errorMessage })

    return {
      success: false,
      error: errorMessage,
    }
  } finally {
    await prisma.$disconnect()
  }
}

/**
 * Run generator if executed directly
 */
if (require.main === module) {
  // Check if --commit flag is passed
  const autoCommit = process.argv.includes('--commit')

  runGenerator(50, autoCommit)
    .then((result) => {
      if (result.success) {
        logger.info('✅ Generator execution completed successfully')
        process.exit(0)
      } else {
        logger.error('❌ Generator execution failed')
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
