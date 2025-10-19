/**
 * Configuration management
 * Loads and validates environment variables
 */

import dotenv from 'dotenv'
import { z } from 'zod'
import path from 'path'

// Load .env file
dotenv.config({ path: path.join(__dirname, '../../.env') })

// Define configuration schema
const configSchema = z.object({
  // Database
  DATABASE_URL: z.string().url(),

  // OpenAI API
  OPENAI_API_KEY: z.string().min(1),

  // Discord Webhook (optional)
  DISCORD_WEBHOOK_URL: z.string().url().optional(),

  // Slack Webhook (optional)
  SLACK_WEBHOOK_URL: z.string().url().optional(),

  // ArXiv API Settings
  ARXIV_MAX_RESULTS: z.coerce.number().default(50),
  ARXIV_CATEGORIES: z.string().default('cs.AI,cs.LG,cs.CL,cs.CV,cs.RO'),

  // Logging
  LOG_LEVEL: z.enum(['error', 'warn', 'info', 'debug']).default('info'),
  LOG_FILE: z.string().default('./logs/arxiv-monitor.log'),

  // GitHub Settings (for auto-commit)
  GITHUB_TOKEN: z.string().optional(),
  GITHUB_REPO: z.string().default('jeromwolf/ontology'),
  GITHUB_BRANCH: z.string().default('main'),
})

// Parse and validate configuration
const parseConfig = () => {
  try {
    const config = configSchema.parse({
      DATABASE_URL: process.env.DATABASE_URL,
      OPENAI_API_KEY: process.env.OPENAI_API_KEY,
      DISCORD_WEBHOOK_URL: process.env.DISCORD_WEBHOOK_URL,
      SLACK_WEBHOOK_URL: process.env.SLACK_WEBHOOK_URL,
      ARXIV_MAX_RESULTS: process.env.ARXIV_MAX_RESULTS,
      ARXIV_CATEGORIES: process.env.ARXIV_CATEGORIES,
      LOG_LEVEL: process.env.LOG_LEVEL,
      LOG_FILE: process.env.LOG_FILE,
      GITHUB_TOKEN: process.env.GITHUB_TOKEN,
      GITHUB_REPO: process.env.GITHUB_REPO,
      GITHUB_BRANCH: process.env.GITHUB_BRANCH,
    })

    return config
  } catch (error) {
    if (error instanceof z.ZodError) {
      console.error('Configuration validation error:')
      error.errors.forEach((err) => {
        console.error(`  - ${err.path.join('.')}: ${err.message}`)
      })
      process.exit(1)
    }
    throw error
  }
}

// Export validated configuration
export const config = parseConfig()

// Export ArXiv categories as array
export const arxivCategories = config.ARXIV_CATEGORIES.split(',').map((c) => c.trim())
