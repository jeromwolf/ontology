/**
 * Prompt templates for paper summarization
 */

export const SYSTEM_PROMPT = `당신은 AI 연구 논문을 분석하고 요약하는 전문가입니다.
논문의 핵심 내용을 명확하고 이해하기 쉽게 전달하는 것이 목표입니다.

주요 역할:
1. 논문의 핵심 아이디어와 기여를 파악
2. 기술적 세부사항을 일반인도 이해할 수 있도록 설명
3. 실용적 가치와 응용 분야 강조
4. 관련 키워드 및 모듈 식별`

/**
 * Generate user prompt for 3-level summarization
 */
export function generateSummarizationPrompt(
  title: string,
  abstract: string,
  authors: string[],
  categories: string[]
): string {
  return `다음 AI 논문을 3가지 길이로 요약하고 메타데이터를 추출해주세요.

**논문 정보:**
- 제목: ${title}
- 저자: ${authors.join(', ')}
- 카테고리: ${categories.join(', ')}

**초록:**
${abstract}

---

다음 JSON 형식으로 응답해주세요:

{
  "summaryShort": "2-3문장으로 핵심만 요약 (100-150자)",
  "summaryMedium": "1개 문단으로 주요 내용 요약 (300-500자, 문제정의/접근방법/결과 포함)",
  "summaryLong": "2-3개 문단으로 상세 요약 (800-1200자, 배경/방법론/실험결과/의의 포함)",
  "keywords": ["키워드1", "키워드2", "키워드3", "키워드4", "키워드5"],
  "relatedModules": ["module-slug-1", "module-slug-2"]
}

**관련 모듈 가이드** (가장 관련 있는 1-3개 선택):
- "ontology": 지식 그래프, 시맨틱 웹, RDF, OWL
- "llm": 대규모 언어 모델, GPT, Transformer, Fine-tuning
- "rag": 검색 증강 생성, 벡터 데이터베이스, 임베딩
- "multi-agent": 멀티 에이전트 시스템, 협업 AI
- "computer-vision": 이미지 인식, 객체 탐지, 세그멘테이션
- "deep-learning": 신경망, CNN, RNN, 딥러닝 아키텍처
- "quantum-computing": 양자 컴퓨팅, 양자 알고리즘
- "reinforcement-learning": 강화학습, Q-learning, 정책 최적화
- "natural-language-processing": NLP, 텍스트 분석, 언어 이해
- "ai-security": AI 보안, 적대적 공격, 프라이버시
- "autonomous-mobility": 자율주행, 로봇 내비게이션
- "smart-factory": 스마트 제조, Industry 4.0, IoT
- "stock-analysis": 금융 AI, 주가 예측, 알고리즘 트레이딩
- "bioinformatics": 바이오인포매틱스, 유전체 분석
- "data-science": 데이터 분석, 머신러닝 파이프라인

**요약 작성 가이드:**
1. **Short**: 핵심 문제와 해결책을 한 줄로
2. **Medium**: 문제정의 → 제안 방법 → 주요 결과 구조
3. **Long**: 연구 배경 → 방법론 상세 → 실험 및 검증 → 의의와 한계

**키워드**: 논문의 핵심 기술/방법론 중심으로 5개 선택`
}

/**
 * System prompt for keyword extraction only
 */
export const KEYWORD_EXTRACTION_SYSTEM = `당신은 AI 논문에서 핵심 키워드를 추출하는 전문가입니다.
논문의 기술적 핵심과 응용 분야를 나타내는 키워드를 선정합니다.`

/**
 * Generate prompt for keyword extraction
 */
export function generateKeywordPrompt(title: string, abstract: string): string {
  return `다음 논문에서 핵심 키워드 5개를 추출해주세요.

제목: ${title}
초록: ${abstract}

JSON 형식으로 응답:
{
  "keywords": ["키워드1", "키워드2", "키워드3", "키워드4", "키워드5"]
}

키워드는 기술/방법론 중심으로 선택하세요 (예: "Transformer", "Few-shot Learning", "Graph Neural Network")`
}

/**
 * System prompt for module mapping
 */
export const MODULE_MAPPING_SYSTEM = `당신은 AI 논문을 KSS 플랫폼의 학습 모듈과 연결하는 전문가입니다.
논문의 주제와 가장 관련 있는 모듈을 식별합니다.`

/**
 * Generate prompt for module mapping
 */
export function generateModuleMappingPrompt(
  title: string,
  abstract: string,
  keywords: string[]
): string {
  return `다음 논문과 가장 관련 있는 KSS 모듈을 1-3개 선택해주세요.

제목: ${title}
초록: ${abstract}
키워드: ${keywords.join(', ')}

**사용 가능한 모듈:**
- ontology, llm, rag, multi-agent, computer-vision, deep-learning
- quantum-computing, reinforcement-learning, natural-language-processing
- ai-security, autonomous-mobility, smart-factory, stock-analysis
- bioinformatics, data-science

JSON 형식으로 응답:
{
  "relatedModules": ["module-id-1", "module-id-2"],
  "reasoning": "선택 이유를 간단히 설명"
}`
}

/**
 * Validation function for summarization result
 */
export function validateSummarizationResult(result: any): {
  valid: boolean
  errors: string[]
} {
  const errors: string[] = []

  if (!result.summaryShort || result.summaryShort.length < 50) {
    errors.push('summaryShort is too short (min 50 characters)')
  }
  if (!result.summaryMedium || result.summaryMedium.length < 200) {
    errors.push('summaryMedium is too short (min 200 characters)')
  }
  if (!result.summaryLong || result.summaryLong.length < 500) {
    errors.push('summaryLong is too short (min 500 characters)')
  }
  if (!Array.isArray(result.keywords) || result.keywords.length < 3) {
    errors.push('keywords must be an array with at least 3 items')
  }
  if (!Array.isArray(result.relatedModules) || result.relatedModules.length === 0) {
    errors.push('relatedModules must be a non-empty array')
  }

  return {
    valid: errors.length === 0,
    errors,
  }
}
