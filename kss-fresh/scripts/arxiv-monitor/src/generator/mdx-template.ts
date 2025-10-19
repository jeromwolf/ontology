/**
 * MDX Template Generator
 * Creates MDX files in KSS platform format
 */

interface PaperData {
  arxivId: string
  title: string
  authors: string[]
  abstract: string
  categories: string[]
  publishedDate: Date
  pdfUrl: string
  summaryShort: string
  summaryMedium: string
  summaryLong: string
  keywords: string[]
  relatedModules: string[]
}

/**
 * Generate MDX frontmatter
 */
function generateFrontmatter(paper: PaperData): string {
  const publishedDateStr = paper.publishedDate.toISOString().split('T')[0]

  return `---
title: "${paper.title.replace(/"/g, '\\"')}"
arxivId: "${paper.arxivId}"
authors: [${paper.authors.map((a) => `"${a.replace(/"/g, '\\"')}"`).join(', ')}]
publishedDate: "${publishedDateStr}"
categories: [${paper.categories.map((c) => `"${c}"`).join(', ')}]
keywords: [${paper.keywords.map((k) => `"${k}"`).join(', ')}]
relatedModules: [${paper.relatedModules.map((m) => `"${m}"`).join(', ')}]
pdfUrl: "${paper.pdfUrl}"
---`
}

/**
 * Generate paper overview section
 */
function generateOverview(paper: PaperData): string {
  return `
## 📋 논문 개요

<div className="paper-overview">
  <div className="overview-item">
    <strong>ArXiv ID:</strong> <a href="${paper.pdfUrl}" target="_blank" rel="noopener">${paper.arxivId}</a>
  </div>
  <div className="overview-item">
    <strong>발행일:</strong> ${paper.publishedDate.toLocaleDateString('ko-KR')}
  </div>
  <div className="overview-item">
    <strong>카테고리:</strong> ${paper.categories.join(', ')}
  </div>
</div>

### 📌 한 줄 요약
${paper.summaryShort}
`
}

/**
 * Generate authors section
 */
function generateAuthors(authors: string[]): string {
  return `
## 👥 저자

${authors.map((author, idx) => `${idx + 1}. ${author}`).join('\n')}
`
}

/**
 * Generate abstract section
 */
function generateAbstract(abstract: string): string {
  return `
## 📄 초록 (Abstract)

${abstract}
`
}

/**
 * Generate summary sections
 */
function generateSummaries(paper: PaperData): string {
  return `
## 🔍 요약

### 📝 상세 요약
${paper.summaryLong}

### 💡 핵심 내용
${paper.summaryMedium}
`
}

/**
 * Generate keywords section
 */
function generateKeywords(keywords: string[]): string {
  return `
## 🔑 키워드

${keywords.map((keyword) => `- **${keyword}**`).join('\n')}
`
}

/**
 * Generate related modules section
 */
function generateRelatedModules(modules: string[]): string {
  const moduleLinks = modules
    .map((moduleId) => {
      const moduleName = moduleId
        .split('-')
        .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
        .join(' ')
      return `- [${moduleName}](/modules/${moduleId})`
    })
    .join('\n')

  return `
## 🔗 관련 모듈

이 논문과 관련된 KSS 학습 모듈:

${moduleLinks}
`
}

/**
 * Generate footer with links
 */
function generateFooter(paper: PaperData): string {
  return `
---

## 📚 참고 자료

- **ArXiv 원문**: [${paper.arxivId}](${paper.pdfUrl})
- **ArXiv 페이지**: [https://arxiv.org/abs/${paper.arxivId}](https://arxiv.org/abs/${paper.arxivId})

---

<div className="paper-footer">
  <p>이 요약은 KSS ArXiv Monitor에 의해 자동 생성되었습니다.</p>
  <p>생성일: ${new Date().toLocaleDateString('ko-KR')}</p>
</div>
`
}

/**
 * Generate complete MDX content
 */
export function generateMDX(paper: PaperData): string {
  const sections = [
    generateFrontmatter(paper),
    '',
    `# ${paper.title}`,
    '',
    generateOverview(paper),
    generateAuthors(paper.authors),
    generateAbstract(paper.abstract),
    generateSummaries(paper),
    generateKeywords(paper.keywords),
    generateRelatedModules(paper.relatedModules),
    generateFooter(paper),
  ]

  return sections.join('\n')
}

/**
 * Generate filename for MDX file
 */
export function generateFilename(arxivId: string): string {
  // Remove version suffix (e.g., "2510.08569v1" -> "2510.08569")
  const cleanId = arxivId.replace(/v\d+$/, '')
  return `${cleanId}.mdx`
}

/**
 * Generate directory path for paper
 */
export function generateDirectoryPath(publishedDate: Date): string {
  const year = publishedDate.getFullYear()
  const month = String(publishedDate.getMonth() + 1).padStart(2, '0')
  return `papers/${year}/${month}`
}
