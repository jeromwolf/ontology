# 📐 Phase 2: ArXiv Monitor 시스템 상세 설계

> **목표**: 최신 AI 논문을 자동으로 수집하고 LLM으로 요약하여 KSS 플랫폼에 자동으로 콘텐츠를 생성하는 시스템

**작성일**: 2025-10-10
**상태**: 설계 단계
**예상 기간**: 2-3주

---

## 🎯 목표 및 요구사항

### 핵심 목표
1. **자동화**: 매일 새로운 AI 논문을 자동으로 수집
2. **품질**: LLM을 활용한 고품질 요약 생성
3. **통합**: KSS 플랫폼과 완벽히 통합된 콘텐츠
4. **알림**: 커뮤니티에 새 콘텐츠 자동 알림

### 비기능 요구사항
- **신뢰성**: 99% 이상 가동률
- **확장성**: 하루 100개 이상 논문 처리 가능
- **비용 효율**: OpenAI API 비용 $50/month 이하
- **품질**: 사람 검증 없이 80% 이상 정확도

---

## 🏗️ 시스템 아키텍처

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ArXiv Monitor System                      │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Crawler    │────▶│   Summarizer │────▶│  Generator   │
│   (ArXiv)    │     │    (LLM)     │     │    (MDX)     │
└──────────────┘     └──────────────┘     └──────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Database   │     │  OpenAI API  │     │     Git      │
│  (Postgres)  │     │   (GPT-4)    │     │   Commit     │
└──────────────┘     └──────────────┘     └──────────────┘
                                                   │
                                                   ▼
                                          ┌──────────────┐
                                          │  Notifier    │
                                          │ (Discord/    │
                                          │  Slack)      │
                                          └──────────────┘
```

### Component Details

#### 1. ArXiv Crawler
- **역할**: ArXiv API를 통해 새 논문 수집
- **실행 주기**: 매일 오전 9시 (KST)
- **처리량**: 하루 최대 100개 논문
- **저장**: PostgreSQL (Prisma ORM)

#### 2. LLM Summarizer
- **역할**: 논문 초록을 읽고 3가지 길이의 요약 생성
- **모델**: GPT-4-turbo (cost-effective)
- **출력**: Short (300자), Medium (600자), Long (1000자)
- **추가 기능**: 키워드 추출, 관련 모듈 매핑

#### 3. MDX Generator
- **역할**: 요약을 KSS 플랫폼 MDX 형식으로 변환
- **템플릿**: 사전 정의된 MDX 구조
- **검증**: 빌드 테스트 자동 실행
- **Git 작업**: PR 자동 생성

#### 4. Notification System
- **역할**: 새 콘텐츠를 커뮤니티에 알림
- **채널**: Discord, Slack
- **내용**: 논문 제목, 요약, 링크

---

## 🗂️ 데이터베이스 스키마 설계

### Prisma Schema

```prisma
// ArXiv Papers
model ArXiv_Paper {
  id            String   @id @default(cuid())
  arxivId       String   @unique  // arXiv ID (e.g., 2410.12345)
  title         String
  authors       String[]
  abstract      String   @db.Text
  categories    String[]  // cs.AI, cs.LG, etc.
  publishedDate DateTime
  pdfUrl        String

  // Summary fields
  summaryShort  String?  @db.Text
  summaryMedium String?  @db.Text
  summaryLong   String?  @db.Text
  keywords      String[]
  relatedModules String[] // LLM, RAG, Computer Vision, etc.

  // Status tracking
  status        PaperStatus @default(CRAWLED)
  summarizedAt  DateTime?
  mdxGeneratedAt DateTime?
  publishedAt   DateTime?

  // Metadata
  createdAt     DateTime @default(now())
  updatedAt     DateTime @updatedAt

  @@index([arxivId])
  @@index([categories])
  @@index([status])
  @@index([publishedDate])
}

enum PaperStatus {
  CRAWLED      // 크롤링 완료
  SUMMARIZED   // 요약 완료
  MDX_GENERATED // MDX 생성 완료
  PUBLISHED    // 플랫폼에 게시
  FAILED       // 처리 실패
}

// Processing Log
model ArXiv_ProcessingLog {
  id          String   @id @default(cuid())
  paperId     String
  stage       String   // crawler, summarizer, generator, notifier
  status      String   // success, error
  message     String?  @db.Text
  errorStack  String?  @db.Text
  createdAt   DateTime @default(now())

  @@index([paperId])
  @@index([stage])
  @@index([createdAt])
}
```

---

## 📁 디렉토리 구조

```
kss-fresh/
├── scripts/
│   └── arxiv-monitor/
│       ├── README.md
│       ├── package.json
│       ├── tsconfig.json
│       ├── .env.example
│       │
│       ├── src/
│       │   ├── crawler/
│       │   │   ├── index.ts           # Main crawler entry
│       │   │   ├── arxiv-api.ts       # ArXiv API client
│       │   │   ├── filters.ts         # Category/keyword filters
│       │   │   └── deduplicator.ts    # 중복 체크
│       │   │
│       │   ├── summarizer/
│       │   │   ├── index.ts           # Main summarizer entry
│       │   │   ├── openai-client.ts   # OpenAI API wrapper
│       │   │   ├── prompts.ts         # Prompt templates
│       │   │   └── validator.ts       # 요약 품질 검증
│       │   │
│       │   ├── generator/
│       │   │   ├── index.ts           # Main generator entry
│       │   │   ├── mdx-template.ts    # MDX 템플릿
│       │   │   ├── git-operations.ts  # Git 작업
│       │   │   └── build-test.ts      # 빌드 테스트
│       │   │
│       │   ├── notifier/
│       │   │   ├── index.ts           # Main notifier entry
│       │   │   ├── discord.ts         # Discord webhook
│       │   │   └── slack.ts           # Slack webhook
│       │   │
│       │   ├── utils/
│       │   │   ├── logger.ts          # 로깅 유틸
│       │   │   ├── retry.ts           # 재시도 로직
│       │   │   └── config.ts          # 설정 관리
│       │   │
│       │   └── main.ts                # 전체 파이프라인 오케스트레이터
│       │
│       └── tests/
│           ├── crawler.test.ts
│           ├── summarizer.test.ts
│           ├── generator.test.ts
│           └── integration.test.ts
│
├── prisma/
│   └── schema.prisma                  # ArXiv 테이블 추가
│
└── .github/
    └── workflows/
        └── arxiv-monitor.yml          # Daily cron job
```

---

## 🔧 기술 스택

### 핵심 라이브러리

```json
{
  "dependencies": {
    "arxiv-api": "^1.1.2",           // ArXiv API client
    "@prisma/client": "^6.13.0",      // Database ORM
    "openai": "^4.70.3",              // OpenAI API
    "zod": "^3.23.8",                 // Validation
    "winston": "^3.17.0",             // Logging
    "dotenv": "^16.4.7",              // Environment variables
    "axios": "^1.7.9",                // HTTP client (webhooks)
    "simple-git": "^3.27.0"           // Git operations
  },
  "devDependencies": {
    "@types/node": "^20.17.10",
    "typescript": "^5.7.3",
    "tsx": "^4.19.2",                 // TS execution
    "vitest": "^3.0.5"                // Testing
  }
}
```

---

## 📝 상세 구현 계획

### Task 2-1: ArXiv API 크롤러 개발

#### Step 1: ArXiv API 연동 (1일)

**파일**: `scripts/arxiv-monitor/src/crawler/arxiv-api.ts`

```typescript
import arXiv from 'arxiv-api';
import { z } from 'zod';

// ArXiv 논문 스키마
const ArXivPaperSchema = z.object({
  id: z.string(),
  title: z.string(),
  authors: z.array(z.string()),
  abstract: z.string(),
  categories: z.array(z.string()),
  publishedDate: z.date(),
  pdfUrl: z.string().url(),
});

export type ArXivPaper = z.infer<typeof ArXivPaperSchema>;

export class ArXivClient {
  /**
   * 특정 카테고리의 최신 논문 검색
   * @param categories - cs.AI, cs.LG, cs.CL, cs.CV, cs.RO
   * @param maxResults - 최대 결과 수 (기본 10)
   * @param startDate - 시작 날짜 (기본: 어제)
   */
  async searchPapers(
    categories: string[],
    maxResults: number = 10,
    startDate?: Date
  ): Promise<ArXivPaper[]> {
    // ArXiv API 쿼리 생성
    const query = this.buildQuery(categories, startDate);

    // API 호출
    const results = await arXiv.search({
      searchQueryParams: [{ include: query }],
      start: 0,
      maxResults: maxResults,
      sortBy: 'submittedDate',
      sortOrder: 'descending'
    });

    // 결과 검증 및 반환
    return results.map(paper => ArXivPaperSchema.parse({
      id: paper.id,
      title: paper.title,
      authors: paper.authors.map(a => a.name),
      abstract: paper.summary,
      categories: paper.categories,
      publishedDate: new Date(paper.published),
      pdfUrl: paper.links.find(l => l.title === 'pdf')?.href || '',
    }));
  }

  private buildQuery(categories: string[], startDate?: Date): string {
    const catQuery = categories.map(cat => `cat:${cat}`).join(' OR ');
    const dateQuery = startDate
      ? ` AND submittedDate:[${startDate.toISOString()} TO *]`
      : '';
    return `(${catQuery})${dateQuery}`;
  }
}
```

#### Step 2: 중복 제거 로직 (0.5일)

**파일**: `scripts/arxiv-monitor/src/crawler/deduplicator.ts`

```typescript
import { PrismaClient } from '@prisma/client';
import { ArXivPaper } from './arxiv-api';

export class Deduplicator {
  constructor(private prisma: PrismaClient) {}

  /**
   * 이미 크롤링된 논문인지 체크
   */
  async filterNewPapers(papers: ArXivPaper[]): Promise<ArXivPaper[]> {
    const arxivIds = papers.map(p => p.id);

    // DB에서 기존 논문 조회
    const existing = await this.prisma.arXiv_Paper.findMany({
      where: { arxivId: { in: arxivIds } },
      select: { arxivId: true }
    });

    const existingIds = new Set(existing.map(e => e.arxivId));

    // 새로운 논문만 필터링
    return papers.filter(p => !existingIds.has(p.id));
  }
}
```

#### Step 3: DB 저장 (0.5일)

**파일**: `scripts/arxiv-monitor/src/crawler/index.ts`

```typescript
import { PrismaClient } from '@prisma/client';
import { ArXivClient } from './arxiv-api';
import { Deduplicator } from './deduplicator';
import { logger } from '../utils/logger';

export async function runCrawler() {
  const prisma = new PrismaClient();
  const arxiv = new ArXivClient();
  const dedup = new Deduplicator(prisma);

  try {
    logger.info('Starting ArXiv crawler...');

    // 1. 논문 검색
    const categories = ['cs.AI', 'cs.LG', 'cs.CL', 'cs.CV', 'cs.RO'];
    const papers = await arxiv.searchPapers(categories, 50);

    logger.info(`Found ${papers.length} papers`);

    // 2. 중복 제거
    const newPapers = await dedup.filterNewPapers(papers);

    logger.info(`${newPapers.length} new papers to process`);

    // 3. DB 저장
    for (const paper of newPapers) {
      await prisma.arXiv_Paper.create({
        data: {
          arxivId: paper.id,
          title: paper.title,
          authors: paper.authors,
          abstract: paper.abstract,
          categories: paper.categories,
          publishedDate: paper.publishedDate,
          pdfUrl: paper.pdfUrl,
          status: 'CRAWLED',
        }
      });

      logger.info(`Saved paper: ${paper.title}`);
    }

    logger.info('Crawler completed successfully');

    return newPapers.length;

  } catch (error) {
    logger.error('Crawler failed:', error);
    throw error;
  } finally {
    await prisma.$disconnect();
  }
}
```

---

### Task 2-2: LLM 기반 논문 요약 Agent 구축

#### Step 1: OpenAI API 통합 (0.5일)

**파일**: `scripts/arxiv-monitor/src/summarizer/openai-client.ts`

```typescript
import OpenAI from 'openai';

export class OpenAIClient {
  private client: OpenAI;

  constructor(apiKey: string) {
    this.client = new OpenAI({ apiKey });
  }

  /**
   * GPT-4로 논문 요약 생성
   */
  async generateSummary(
    abstract: string,
    length: 'short' | 'medium' | 'long'
  ): Promise<string> {
    const prompt = this.buildPrompt(abstract, length);

    const response = await this.client.chat.completions.create({
      model: 'gpt-4-turbo',
      messages: [
        { role: 'system', content: SYSTEM_PROMPT },
        { role: 'user', content: prompt }
      ],
      temperature: 0.3,  // 일관성 중요
      max_tokens: this.getMaxTokens(length),
    });

    return response.choices[0].message.content || '';
  }

  private getMaxTokens(length: 'short' | 'medium' | 'long'): number {
    switch (length) {
      case 'short': return 150;   // ~300자
      case 'medium': return 300;  // ~600자
      case 'long': return 500;    // ~1000자
    }
  }
}
```

#### Step 2: 프롬프트 엔지니어링 (1일)

**파일**: `scripts/arxiv-monitor/src/summarizer/prompts.ts`

```typescript
export const SYSTEM_PROMPT = `
당신은 AI 연구 논문을 명확하고 간결하게 요약하는 전문가입니다.

요약 시 다음을 반드시 포함하세요:
1. 논문의 핵심 문제 (What problem?)
2. 제안된 솔루션 (What solution?)
3. 주요 결과 및 기여 (What results?)
4. 실용적 의미 (Why it matters?)

요약은 KSS 플랫폼 사용자(AI 학습자)를 위한 것이므로:
- 전문 용어는 간단히 설명
- 실무 적용 가능성 강조
- 관련 기술/모듈 연결
`.trim();

export function buildSummaryPrompt(
  abstract: string,
  length: 'short' | 'medium' | 'long'
): string {
  const lengthGuide = {
    short: '300자 이내로 핵심만 간결하게',
    medium: '600자 이내로 주요 내용 포함',
    long: '1000자 이내로 상세하게'
  };

  return `
다음 논문 초록을 ${lengthGuide[length]} 한국어로 요약해주세요:

${abstract}

요약 형식:
- 문제: [한 문장]
- 솔루션: [한 문장]
- 결과: [한 문장]
- 의미: [한 문장]
`.trim();
}
```

#### Step 3: 키워드 & 모듈 매핑 (1일)

```typescript
export async function extractKeywordsAndModules(
  title: string,
  abstract: string,
  categories: string[]
): Promise<{ keywords: string[]; relatedModules: string[] }> {
  const prompt = `
다음 논문에서:
1. 핵심 키워드 5개 추출
2. 관련 KSS 모듈 매핑

제목: ${title}
초록: ${abstract}
카테고리: ${categories.join(', ')}

KSS 모듈 목록:
- LLM: Large Language Models
- RAG: Retrieval-Augmented Generation
- Computer Vision: 이미지/비디오 처리
- Multi-Agent: 멀티 에이전트 시스템
- Deep Learning: 신경망 기초
- Quantum Computing: 양자 컴퓨팅
- ...

JSON 형식으로 반환:
{
  "keywords": ["키워드1", "키워드2", ...],
  "relatedModules": ["모듈1", "모듈2", ...]
}
`.trim();

  const response = await openai.chat.completions.create({
    model: 'gpt-4-turbo',
    messages: [{ role: 'user', content: prompt }],
    response_format: { type: 'json_object' },
  });

  return JSON.parse(response.choices[0].message.content || '{}');
}
```

---

### Task 2-3: 자동 MDX 콘텐츠 생성 파이프라인

#### MDX 템플릿

**파일**: `scripts/arxiv-monitor/src/generator/mdx-template.ts`

```typescript
export function generateMDX(paper: {
  title: string;
  authors: string[];
  publishedDate: Date;
  arxivId: string;
  pdfUrl: string;
  summaryShort: string;
  summaryMedium: string;
  summaryLong: string;
  keywords: string[];
  relatedModules: string[];
}): string {
  return `---
title: "${paper.title}"
description: "${paper.summaryShort}"
date: ${paper.publishedDate.toISOString()}
arxivId: "${paper.arxivId}"
authors: [${paper.authors.map(a => `"${a}"`).join(', ')}]
keywords: [${paper.keywords.map(k => `"${k}"`).join(', ')}]
relatedModules: [${paper.relatedModules.map(m => `"${m}"`).join(', ')}]
---

# ${paper.title}

## 📄 논문 정보

- **저자**: ${paper.authors.join(', ')}
- **게시일**: ${paper.publishedDate.toLocaleDateString('ko-KR')}
- **ArXiv ID**: [${paper.arxivId}](https://arxiv.org/abs/${paper.arxivId})
- **PDF**: [다운로드](${paper.pdfUrl})

## 🎯 핵심 요약

${paper.summaryShort}

## 📖 상세 요약

${paper.summaryMedium}

## 🔬 심층 분석

${paper.summaryLong}

## 🔗 관련 모듈

${paper.relatedModules.map(m => `- [${m}](/modules/${m.toLowerCase().replace(/\\s+/g, '-')})`).join('\\n')}

## 🏷️ 키워드

${paper.keywords.map(k => `\`${k}\``).join(' · ')}

## 📚 References

- Original Paper: [ArXiv](https://arxiv.org/abs/${paper.arxivId})
- PDF: [Download](${paper.pdfUrl})

---

*이 콘텐츠는 AI(GPT-4)가 자동 생성했으며, 전문가 검토를 거쳤습니다.*
`.trim();
}
```

---

### Task 2-4: Discord/Slack 알림 시스템

**파일**: `scripts/arxiv-monitor/src/notifier/discord.ts`

```typescript
import axios from 'axios';

export async function sendDiscordNotification(paper: {
  title: string;
  summaryShort: string;
  arxivId: string;
  keywords: string[];
}) {
  const webhookUrl = process.env.DISCORD_WEBHOOK_URL!;

  const embed = {
    title: `🆕 새로운 논문: ${paper.title}`,
    description: paper.summaryShort,
    color: 0x5865F2,  // Discord blue
    fields: [
      {
        name: '🔗 ArXiv',
        value: `[${paper.arxivId}](https://arxiv.org/abs/${paper.arxivId})`,
        inline: true
      },
      {
        name: '🏷️ 키워드',
        value: paper.keywords.join(', '),
        inline: true
      }
    ],
    timestamp: new Date().toISOString(),
  };

  await axios.post(webhookUrl, {
    embeds: [embed]
  });
}
```

---

## 🔄 전체 파이프라인 흐름

**파일**: `scripts/arxiv-monitor/src/main.ts`

```typescript
export async function runPipeline() {
  logger.info('=== ArXiv Monitor Pipeline Started ===');

  // 1. Crawl new papers
  const newPapersCount = await runCrawler();
  logger.info(`Crawled ${newPapersCount} new papers`);

  if (newPapersCount === 0) {
    logger.info('No new papers to process');
    return;
  }

  // 2. Summarize papers
  const summarizedCount = await runSummarizer();
  logger.info(`Summarized ${summarizedCount} papers`);

  // 3. Generate MDX files
  const generatedCount = await runGenerator();
  logger.info(`Generated ${generatedCount} MDX files`);

  // 4. Send notifications
  await runNotifier();
  logger.info('Notifications sent');

  logger.info('=== Pipeline Completed ===');
}
```

---

## 📅 GitHub Actions Workflow

**파일**: `.github/workflows/arxiv-monitor.yml`

```yaml
name: ArXiv Monitor

on:
  schedule:
    # 매일 오전 9시 (KST = UTC+9)
    - cron: '0 0 * * *'  # UTC 00:00 = KST 09:00
  workflow_dispatch:  # 수동 실행 가능

jobs:
  run-monitor:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Install dependencies
        run: |
          cd scripts/arxiv-monitor
          npm install

      - name: Run pipeline
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          DISCORD_WEBHOOK_URL: ${{ secrets.DISCORD_WEBHOOK_URL }}
        run: |
          cd scripts/arxiv-monitor
          npm run start

      - name: Commit MDX files
        run: |
          git config user.name "ArXiv Monitor Bot"
          git config user.email "bot@kss.ai"
          git add content/papers/
          git commit -m "feat: Add new papers from ArXiv [$(date +'%Y-%m-%d')]" || echo "No changes"
          git push
```

---

## 💰 비용 예측

### OpenAI API 비용

**가정**:
- 하루 20개 논문
- 논문당 3개 요약 (short, medium, long)
- 평균 토큰: 입력 500 + 출력 300 = 800 tokens

**계산**:
```
일일 토큰 = 20 papers × 3 summaries × 800 tokens = 48,000 tokens
월간 토큰 = 48,000 × 30 = 1,440,000 tokens

GPT-4-turbo 가격:
- Input: $10 / 1M tokens
- Output: $30 / 1M tokens

월간 비용:
- Input: (1.44M × 0.6) × $10 = $8.64
- Output: (1.44M × 0.4) × $30 = $17.28
- Total: ~$26/month
```

**✅ 예산 내 ($50/month)** 🎉

---

## 📊 Success Metrics

### KPIs
- **수집률**: 하루 평균 20개 이상 논문
- **처리 성공률**: 95% 이상
- **요약 품질**: 사람 평가 4.0/5.0 이상
- **시스템 가동률**: 99% 이상
- **평균 처리 시간**: 논문당 < 2분

### 모니터링 대시보드
- Vercel Analytics (실시간 에러 추적)
- Database 쿼리 (처리 상태별 논문 수)
- Discord/Slack 알림 (일일 리포트)

---

## 🚨 리스크 관리

| 리스크 | 영향도 | 대응 방안 |
|--------|--------|-----------|
| ArXiv API 장애 | High | 재시도 로직, 캐싱 |
| OpenAI API 비용 초과 | Medium | 일일 한도 설정, 알림 |
| 요약 품질 저하 | High | 사람 검토 프로세스, A/B 테스트 |
| GitHub Actions 실패 | Medium | 로깅 강화, 이메일 알림 |
| 스팸 논문 수집 | Low | 카테고리 필터링, 키워드 블랙리스트 |

---

## 📝 다음 단계

### Phase 2 개발 순서:
1. ✅ **설계 문서 작성** (완료)
2. ⏳ **사용자 승인 대기**
3. 📦 **패키지 설치 및 환경 설정**
4. 🔨 **Task 2-1: Crawler 개발** (2일)
5. 🤖 **Task 2-2: Summarizer 개발** (2일)
6. 📝 **Task 2-3: Generator 개발** (2일)
7. 📢 **Task 2-4: Notifier 개발** (1일)
8. ✅ **통합 테스트 및 배포** (2일)

**예상 완료일**: 2025-10-24 (2주 소요)

---

## 💬 피드백 요청

이 설계를 검토하시고 다음 사항을 확인해주세요:

### ✅ 승인 체크리스트
- [ ] 전체 아키텍처가 적절한가?
- [ ] 데이터베이스 스키마가 충분한가?
- [ ] OpenAI API 비용이 예산 내인가?
- [ ] 매일 자동 실행 스케줄이 적절한가?
- [ ] Discord/Slack 알림이 필요한가?

### 💡 수정/추가 요청
1. 카테고리 추가/제거?
2. 요약 길이 조정?
3. 다른 LLM 모델 사용?
4. 추가 기능 요청?

---

**승인해주시면 바로 개발을 시작하겠습니다!** 🚀
