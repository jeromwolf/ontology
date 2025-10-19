/**
 * OpenAI Client
 * Handles interactions with OpenAI API for paper summarization
 */

import OpenAI from 'openai'
import logger from '../utils/logger'
import { config } from '../utils/config'

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: config.OPENAI_API_KEY,
})

/**
 * Generate completion using GPT-4-turbo
 */
export async function generateCompletion(
  systemPrompt: string,
  userPrompt: string,
  temperature: number = 0.7,
  maxTokens: number = 1000
): Promise<string> {
  try {
    logger.debug('Generating OpenAI completion', {
      systemPromptLength: systemPrompt.length,
      userPromptLength: userPrompt.length,
      temperature,
      maxTokens,
    })

    const completion = await openai.chat.completions.create({
      model: 'gpt-4-turbo-preview',
      messages: [
        {
          role: 'system',
          content: systemPrompt,
        },
        {
          role: 'user',
          content: userPrompt,
        },
      ],
      temperature,
      max_tokens: maxTokens,
    })

    const content = completion.choices[0]?.message?.content

    if (!content) {
      throw new Error('No content returned from OpenAI')
    }

    // Log token usage for cost tracking
    logger.debug('OpenAI completion generated', {
      promptTokens: completion.usage?.prompt_tokens,
      completionTokens: completion.usage?.completion_tokens,
      totalTokens: completion.usage?.total_tokens,
    })

    return content.trim()
  } catch (error) {
    logger.error('Failed to generate OpenAI completion', {
      error: error instanceof Error ? error.message : String(error),
    })
    throw error
  }
}

/**
 * Generate JSON completion using GPT-4-turbo with response_format
 */
export async function generateJsonCompletion<T = any>(
  systemPrompt: string,
  userPrompt: string,
  temperature: number = 0.7,
  maxTokens: number = 1000
): Promise<T> {
  try {
    logger.debug('Generating OpenAI JSON completion', {
      systemPromptLength: systemPrompt.length,
      userPromptLength: userPrompt.length,
    })

    const completion = await openai.chat.completions.create({
      model: 'gpt-4-turbo-preview',
      messages: [
        {
          role: 'system',
          content: systemPrompt,
        },
        {
          role: 'user',
          content: userPrompt,
        },
      ],
      temperature,
      max_tokens: maxTokens,
      response_format: { type: 'json_object' },
    })

    const content = completion.choices[0]?.message?.content

    if (!content) {
      throw new Error('No content returned from OpenAI')
    }

    // Parse JSON response
    const parsed = JSON.parse(content)

    // Log token usage
    logger.debug('OpenAI JSON completion generated', {
      promptTokens: completion.usage?.prompt_tokens,
      completionTokens: completion.usage?.completion_tokens,
      totalTokens: completion.usage?.total_tokens,
    })

    return parsed as T
  } catch (error) {
    logger.error('Failed to generate OpenAI JSON completion', {
      error: error instanceof Error ? error.message : String(error),
    })
    throw error
  }
}

/**
 * Estimate token count (approximate)
 */
export function estimateTokens(text: string): number {
  // Rough estimate: ~4 characters per token
  return Math.ceil(text.length / 4)
}

/**
 * Calculate approximate cost for GPT-4-turbo
 * Input: $10 per 1M tokens
 * Output: $30 per 1M tokens
 */
export function calculateCost(inputTokens: number, outputTokens: number): number {
  const inputCost = (inputTokens / 1_000_000) * 10
  const outputCost = (outputTokens / 1_000_000) * 30
  return inputCost + outputCost
}
