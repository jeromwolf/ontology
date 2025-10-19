# 📚 ArXiv Monitor System

자동으로 최신 AI 논문을 수집하고, LLM으로 요약하여, KSS 플랫폼에 MDX 콘텐츠를 생성하는 시스템

---

## 🎯 기능

1. **ArXiv Crawler**: 매일 새로운 AI 논문 자동 수집
2. **LLM Summarizer**: GPT-4로 3가지 길이의 요약 생성
3. **MDX Generator**: KSS 플랫폼 형식으로 자동 변환
4. **Notifier**: Discord/Slack으로 새 논문 알림

---

## 🚀 빠른 시작

### 1. 의존성 설치
```bash
cd scripts/arxiv-monitor
npm install
```

### 2. 환경 변수 설정
```bash
cp .env.example .env
# .env 파일 수정 (DATABASE_URL, OPENAI_API_KEY 등)
```

### 3. 실행
```bash
# ✅ 전체 파이프라인 실행 (크롤러 → 요약 → MDX 생성)
npm run dev

# 개별 컴포넌트 실행
npm run crawler      # ✅ 크롤러만 실행
npm run summarizer   # ✅ 요약만 실행 (GPT-4 사용, 비용 발생)
npm run generator    # ✅ MDX 생성만 실행

# Git 자동 커밋 옵션
npm run generator -- --commit  # MDX 생성 후 자동 Git 커밋
```

---

## 📁 프로젝트 구조

```
scripts/arxiv-monitor/
├── src/
│   ├── crawler/          # ArXiv API 크롤러
│   │   ├── index.ts
│   │   ├── arxiv-api.ts
│   │   └── deduplicator.ts
│   ├── summarizer/       # LLM 요약 생성기
│   │   ├── index.ts
│   │   ├── openai-client.ts
│   │   └── prompts.ts
│   ├── generator/        # MDX 파일 생성기
│   │   ├── index.ts
│   │   ├── mdx-template.ts
│   │   └── git-operations.ts
│   ├── notifier/         # 알림 시스템
│   │   ├── index.ts
│   │   ├── discord.ts
│   │   └── slack.ts
│   ├── utils/            # 유틸리티
│   │   ├── logger.ts
│   │   ├── config.ts
│   │   └── retry.ts
│   └── main.ts           # 메인 파이프라인
├── tests/                # 테스트
├── package.json
├── tsconfig.json
└── README.md
```

---

## ⚙️ 환경 변수

### 필수
- `DATABASE_URL`: PostgreSQL 연결 문자열
- `OPENAI_API_KEY`: OpenAI API 키

### 선택
- `DISCORD_WEBHOOK_URL`: Discord 알림
- `SLACK_WEBHOOK_URL`: Slack 알림
- `ARXIV_MAX_RESULTS`: 최대 논문 수 (기본: 50)
- `ARXIV_CATEGORIES`: 수집할 카테고리 (기본: cs.AI,cs.LG,cs.CL,cs.CV,cs.RO)

---

## 🧪 테스트

```bash
# 모든 테스트 실행
npm test

# Watch 모드
npm run test:watch
```

---

## 📊 비용

### OpenAI API (GPT-4-turbo)
- 하루 20개 논문 × 3개 요약 = 60개 요약
- 월간 약 1,800개 요약
- **예상 비용**: ~$26/month

---

## 🔄 자동 실행 (GitHub Actions)

`.github/workflows/arxiv-monitor.yml` 파일로 매일 자동 실행 설정 가능

---

## 📝 로그

로그는 `./logs/arxiv-monitor.log` 파일에 저장됩니다.

```bash
# 실시간 로그 보기
tail -f logs/arxiv-monitor.log
```

---

## 🐛 트러블슈팅

### ArXiv API 연결 실패
- 네트워크 연결 확인
- ArXiv API 상태 확인: https://status.arxiv.org/

### OpenAI API 오류
- API 키 확인
- 사용량 제한 확인: https://platform.openai.com/usage

### 데이터베이스 연결 실패
- DATABASE_URL 확인
- Prisma 클라이언트 재생성: `npx prisma generate`

---

**Created**: 2025-10-10
**Version**: 1.0.0
**Author**: KSS Team
