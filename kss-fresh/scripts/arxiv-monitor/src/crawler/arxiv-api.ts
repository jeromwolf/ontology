/**
 * ArXiv API Client
 * Fetches papers from ArXiv API and parses XML responses
 */

import axios from 'axios'
import { XMLParser } from 'fast-xml-parser'
import logger from '../utils/logger'
import { arxivCategories, config } from '../utils/config'

// ArXiv API base URL
const ARXIV_API_URL = 'https://export.arxiv.org/api/query'

// Types
export interface ArXivPaper {
  arxivId: string
  title: string
  authors: string[]
  abstract: string
  categories: string[]
  publishedDate: Date
  pdfUrl: string
}

interface ArXivEntry {
  id: string
  title: string
  summary: string
  author: { name: string } | { name: string }[]
  category: { '@_term': string } | { '@_term': string }[]
  published: string
  link: { '@_href': string; '@_type'?: string }[]
}

// Initialize XML parser
const parser = new XMLParser({
  ignoreAttributes: false,
  attributeNamePrefix: '@_',
})

/**
 * Fetch papers from ArXiv API
 */
export async function fetchArXivPapers(
  categories: string[] = arxivCategories,
  maxResults: number = config.ARXIV_MAX_RESULTS
): Promise<ArXivPaper[]> {
  try {
    // Build search query - ArXiv requires parentheses for OR operations
    const categoryQuery = categories.length === 1
      ? `cat:${categories[0]}`
      : `(${categories.map((cat) => `cat:${cat}`).join(' OR ')})`

    const params = {
      search_query: categoryQuery,
      start: 0,
      max_results: maxResults,
      sortBy: 'submittedDate',
      sortOrder: 'descending',
    }

    logger.info('Fetching papers from ArXiv', { categories, maxResults })

    // Make API request
    const response = await axios.get(ARXIV_API_URL, {
      params,
      timeout: 30000,
    })

    // Parse XML response
    const parsed = parser.parse(response.data)

    // Debug: log the parsed response structure
    logger.debug('Parsed ArXiv response:', {
      hasFeed: !!parsed.feed,
      feedKeys: parsed.feed ? Object.keys(parsed.feed) : [],
      hasEntry: parsed.feed ? !!parsed.feed.entry : false
    })

    // Extract entries
    const feed = parsed.feed
    if (!feed || !feed.entry) {
      logger.warn('No papers found in ArXiv response')
      return []
    }

    // Ensure entries is an array
    const entries: ArXivEntry[] = Array.isArray(feed.entry) ? feed.entry : [feed.entry]

    // Map entries to our paper format
    const papers = entries.map((entry) => parsePaper(entry))

    logger.info(`Successfully fetched ${papers.length} papers from ArXiv`)

    return papers
  } catch (error) {
    logger.error('Failed to fetch papers from ArXiv', {
      error: error instanceof Error ? error.message : String(error),
    })
    throw error
  }
}

/**
 * Parse single ArXiv entry to our paper format
 */
function parsePaper(entry: ArXivEntry): ArXivPaper {
  // Extract ArXiv ID from URL
  const arxivId = entry.id.split('/abs/')[1] || entry.id

  // Extract title (remove newlines and extra spaces)
  const title = entry.title.replace(/\n/g, ' ').replace(/\s+/g, ' ').trim()

  // Extract authors (handle single or multiple)
  const authorArray = Array.isArray(entry.author) ? entry.author : [entry.author]
  const authors = authorArray.map((a) => a.name)

  // Extract abstract
  const abstract = entry.summary.replace(/\n/g, ' ').replace(/\s+/g, ' ').trim()

  // Extract categories
  const categoryArray = Array.isArray(entry.category) ? entry.category : [entry.category]
  const categories = categoryArray.map((c) => c['@_term'])

  // Extract published date
  const publishedDate = new Date(entry.published)

  // Extract PDF URL
  const pdfLink = entry.link.find((link) => link['@_type'] === 'application/pdf')
  const pdfUrl = pdfLink
    ? pdfLink['@_href']
    : `https://arxiv.org/pdf/${arxivId}.pdf`

  return {
    arxivId,
    title,
    authors,
    abstract,
    categories,
    publishedDate,
    pdfUrl,
  }
}

/**
 * Fetch a specific paper by ArXiv ID
 */
export async function fetchPaperById(arxivId: string): Promise<ArXivPaper | null> {
  try {
    const params = {
      id_list: arxivId,
    }

    logger.debug(`Fetching paper ${arxivId} from ArXiv`)

    const response = await axios.get(ARXIV_API_URL, {
      params,
      timeout: 10000,
    })

    const parsed = parser.parse(response.data)
    const feed = parsed.feed

    if (!feed || !feed.entry) {
      logger.warn(`Paper ${arxivId} not found in ArXiv`)
      return null
    }

    const entry: ArXivEntry = Array.isArray(feed.entry) ? feed.entry[0] : feed.entry

    return parsePaper(entry)
  } catch (error) {
    logger.error(`Failed to fetch paper ${arxivId}`, {
      error: error instanceof Error ? error.message : String(error),
    })
    return null
  }
}
