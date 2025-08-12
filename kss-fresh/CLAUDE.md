# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Korean-language educational platform called KSS (Knowledge Space Simulator) - a next-generation learning platform that simulates and experiences complex technical concepts. Currently focused on Ontology education with 16 chapters of comprehensive content.

## Project Structure

The project has evolved through multiple iterations:
- `index.html` - Original single-page ontology education site
- `kss-standalone/` - Current active Next.js 14 project
- `cognosphere/` - Future monorepo structure (planned)
- `chapters/` - Original HTML content files

## Current Focus: kss-standalone

### Technical Stack
- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS + custom CSS modules
- **UI Components**: Radix UI, Lucide Icons
- **Visualization**: D3.js (planned)
- **Font**: Inter + Noto Sans KR

### Key Features Implemented
1. **Learning Experience**
   - 16 chapters of ontology content
   - Dark mode support
   - Progress tracking (localStorage)
   - Table of Contents with scroll tracking
   - Responsive design

2. **UI Components**
   - Sidebar navigation with chapter numbers
   - Progress tracker
   - Dark mode toggle
   - Enhanced code blocks

### Development Commands
```bash
cd kss-standalone
npm install
npm run dev   # Development server
npm run build # Production build
npm start     # Production server
```

## Architecture Decisions

### Hybrid Approach
1. **Phase 1**: Minimal structure design (3 days) ✅
2. **Phase 2**: Ontology MVP development (2-3 weeks) - IN PROGRESS
3. **Phase 3**: Structure expansion (1-2 months)

### Development Methodology
- **A2A (Agent to Agent)**: Divide large tasks into independent agents
- **Task Master MCP**: Complex task division and management
- **Microservices**: Future scalability preparation

## Important Context

### Vision
- Building a platform like Jensen Huang's COSMOS for Physical AI
- Aiming for a large-scale platform with multiple domain simulators
- Starting with ontology, expanding to LLM, Quantum Computing, RAG simulators

### Next Steps
1. RDF Triple visual editor
2. SPARQL query playground
3. Real-time inference visualization
4. 3D knowledge graphs
5. YouTube content generation with Remotion

### GitHub Repository
https://github.com/jeromwolf/kss-simulator

## ⚠️ CRITICAL: Module Structure Guidelines

### 🚨 MUST-FOLLOW RULES for All Modules:
1. **NEVER create ChapterContent.tsx files larger than 1000 lines**
2. **ALWAYS split chapters into separate component files**
3. **Each chapter = One file** in `/components/chapters/` directory
4. **Use dynamic imports** for performance optimization
5. **Share common components** (code blocks, alerts, tooltips)

### 🔍 파일 크기 체크 방법:
```bash
# 단일 파일 체크
wc -l src/app/modules/[module]/components/ChapterContent.tsx

# 전체 모듈 체크
npm run check:sizes

# 린트와 함께 체크
npm run check:all
```

### 📋 Module Refactoring Priority List:
| Module | Current Size | Priority | Status |
|--------|--------------|----------|---------|
| **Smart Factory** | 8,113 lines | 🔴 CRITICAL | ✅ Completed |
| **Quantum Computing** | 916 lines | 🟡 HIGH | Pending |
| **LLM** | 853 lines | 🟡 HIGH | Pending |
| **RAG** | 793 lines | 🟡 HIGH | Pending |
| **Computer Vision** | 712 lines | 🟡 HIGH | Pending |

### ✅ Correct Module Structure Example:
```
/app/modules/[module-name]/
├── components/
│   ├── chapters/
│   │   ├── Chapter1.tsx (< 500 lines)
│   │   ├── Chapter2.tsx (< 500 lines)
│   │   └── ...
│   ├── ChapterContent.tsx (< 200 lines - router only)
│   ├── simulators/
│   │   └── [Reusable simulator components]
│   └── common/
│       └── [Shared UI components]
└── simulators/
    └── [simulator-name]/
        └── page.tsx (thin wrapper using components)
```

## Session Notes
- Last updated: 2025-08-08 (Session 22 - Chapter Fixes)
- Main working directory: `/Users/kelly/Desktop/Space/project/Ontology/kss-fresh`
- Content preservation: Keep original HTML structure while enhancing styles
- Focus on learning experience over pure technical implementation
- **CRITICAL ISSUE**: Smart Factory module refactoring completed ✅

### 🚨 MUST-FOLLOW RULES for All Modules:
1. **NEVER create ChapterContent.tsx files larger than 1000 lines**
2. **ALWAYS split chapters into separate component files**
3. **Each chapter = One file** in `/components/chapters/` directory
4. **Use dynamic imports with { ssr: false }** for performance
5. **Test build after every major change**
6. **Never use HTML strings - only React components**

### 🎯 Smart Factory 리팩토링 성공 사례 (2025-08-07)
**문제**: 8,113줄 거대 파일로 인한 수정 불가능 상태
**해결**: 16개 독립 챕터 컴포넌트로 완전 분리 (98.7% 감소)
**결과**: 
- ChapterContent.tsx: 8,113줄 → 107줄 
- 각 챕터: 평균 500줄 이하의 관리 가능한 크기
- 빌드 성공, 모든 챕터 정상 작동
- 유지보수성 대폭 향상

**핵심 패턴**:
```typescript
// ChapterContent.tsx (메인 라우터)
const Chapter1 = dynamic(() => import('./chapters/Chapter1'), { ssr: false })
const Chapter2 = dynamic(() => import('./chapters/Chapter2'), { ssr: false })

export default function ChapterContent({ chapterId }: { chapterId: string }) {
  const getChapterComponent = () => {
    switch (chapterId) {
      case 'chapter-slug': return <Chapter1 />
      // ...
    }
  }
}
```

**다음 리팩토링 대상** (큰 파일 순):
1. Quantum Computing (916 lines) 🎯 NEXT
2. LLM (853 lines) 
3. RAG (793 lines)
4. Computer Vision (712 lines)

### Current Session Status (2025-08-09)
- **Session 23**: LLM 모듈 날짜 수정 및 리팩토링 준비

**🎯 완료된 작업**:
1. **LLM 모듈 타임라인 정확성 개선**:
   - o1: 2025년 9월 → 2024년 9월으로 수정
   - o3: 2025년 12월 → 2024년 12월으로 수정
   - GPT-5: 2025년 8월 유지 (사용자 제공 정보)
   - 타임라인 연대순 재정렬 완료

2. **파일 크기 자동 체크 시스템 구축**:
   - `scripts/check-file-sizes.sh` 스크립트 생성
   - `npm run check:sizes` 명령어 추가
   - Pre-commit hook 생성 (1000줄 초과 방지)
   - 11개 모듈 1000줄 초과 발견 (Ontology 3733줄 최대)

3. **LLM 모듈 리팩토링 필요성 확인**:
   - ChapterContent.tsx 1023줄로 긴급 리팩토링 필요
   - Smart Factory 패턴 적용 예정

**🎯 다음 작업**:
- GitHub 푸시 후 LLM 모듈 리팩토링 진행

### Previous Session 22 (2025-08-08)
- **Smart Factory 챕터 재구성 및 버그 수정**

**🎯 완료된 작업**:
1. **Chapter 12 & 13 내용 교체**: 잘못된 매핑 수정
   - Chapter12: 이제 OT 보안 & 국제 표준 내용
   - Chapter13: 이제 스마트팩토리 구현 방법론 내용
   - ChapterContent.tsx 매핑도 수정 완료

2. **Chapter14.tsx 완전 수정**: 시스템 아키텍처 설계 콘텐츠 구현
   - 스마트팩토리 5계층 참조 아키텍처 (Level 1-5)
   - 클라우드 vs 온프레미스 vs 하이브리드 비교
   - 데이터 레이크 & 웨어하우스 아키텍처
   - 마이크로서비스 아키텍처 (서비스 분해, 통신 패턴, 컨테이너화)
   - 실시간 모니터링 4계층 아키텍처 (수집→저장→분석→시각화)
   - 217줄의 풍부한 React 컴포넌트로 구현

3. **GitHub 푸시 완료**: 모든 변경사항 커밋 및 업로드

**🔧 기술적 성과**:
- Chapter14 컴파일 오류 완전 해결
- TypeScript 타입 안전성 유지
- 모든 16개 챕터 정상 작동 확인
- README 업데이트 완료

**⚠️ 중요한 교훈**:
1. 지속적인 컴파일 오류 시 파일 완전 재작성이 효과적
2. 동적 색상 클래스는 Tailwind에서 지원하지 않으므로 주의
3. 챕터 컴포넌트 이름과 파일명 일치 중요

**🎯 다음 우선순위**:
- **8월 14일 발표 준비 우선** - 리팩토링은 발표 이후 진행
- **Quantum Computing 모듈 리팩토링 (916 lines)** - 8.14 이후 목표

### 🔴 중요: 다음 세션 시작 시 필수 확인사항
1. **작업 디렉토리**: `/Users/kelly/Desktop/Space/project/Ontology/kss-fresh`
2. **현재 상태**: 
   - Smart Factory 16개 챕터 리팩토링 완료 ✅
   - Chapter 12, 13, 14 내용 수정 완료 ✅
   - 모든 챕터 정상 작동 확인 ✅
3. **발표 일정**: 8월 14일 발표 준비 중 (리팩토링 작업 보류)
4. **다음 작업**: 발표 준비 우선, 이후 Quantum Computing 모듈 리팩토링

### Current Session Status (2025-08-13)
- **Session 29**: Mermaid 다이어그램 에디터 완성 및 오류 수정

**🎯 완료된 작업**:
1. **전문급 Mermaid 다이어그램 에디터 구현**:
   - System Design 모듈에 핵심 시뮬레이터 추가
   - 6개 실무 템플릿 제공 (마이크로서비스, CI/CD, DB 샤딩 등)
   - 실시간 에디팅, 히스토리 관리, 키보드 단축키 지원
   - 고급 내보내기 기능 (SVG, PNG, 코드 공유)

2. **공간 최적화 UI 컴포넌트 시스템 구축**:
   - ResponsiveCanvas: 30% 공간 효율성 향상
   - AdaptiveLayout: 4가지 레이아웃 모드
   - CollapsibleControls: 섹션별 접을 수 있는 제어판
   - SpaceOptimizedButton: 6가지 변형의 컴팩트 버튼
   - MermaidEditor: 전문급 코드 에디터
   - MermaidPreview: 실시간 미리보기

3. **오류 메시지 제거 및 UX 개선**:
   - "Syntax error in text" 메시지 반복 표시 문제 해결
   - suppressErrorRendering: true 설정으로 오류 렌더링 비활성화
   - DOM에서 오류 텍스트 요소 자동 제거
   - 깔끔한 미리보기 화면 구현

4. **기술적 성과**:
   - 28개 중복 UI 패턴을 6개 컴포넌트로 통합
   - Mermaid 11.9.0 (MIT 라이선스) 상업적 사용 가능
   - WCAG 2.1 AA 접근성 준수
   - ResizeObserver 기반 반응형 캔버스

### 💡 세션 연결 방법
새 세션 시작 시 다음과 같이 요청하세요:
```
"CLAUDE.md 파일 확인하고 작업 진행해줘. 
특히 Session 29의 Mermaid 에디터 완성과 
공간 최적화 UI 컴포넌트 시스템 구축 상황을 참고해줘."
```

### Previous Session 21 (2025-08-07)
- **Smart Factory 리팩토링 완료**: ChapterContent.tsx 8,113줄 → 107줄 (98.7% 감소)
- **16개 챕터 완전 분리**: 각 챕터 평균 500줄 이하
- **8개 시뮬레이터 구현**: Enhanced 4개 + 신규 4개
- **스마트 팩토리 생태계 맵**: 21개 구성요소 통합 시각화
- **시나리오 모드**: 장비고장, AI최적화, 품질위기 3개 시나리오

### Previous Session 16 (2025-08-05)
- **Computer Vision Module** 완전 구현:
  - Homepage에서 확인 가능 (id: 'computer-vision')
  - Teal-Cyan 테마 색상 (from-teal-500 to-cyan-600)
  - AI/ML 카테고리, 중급 난이도, 20시간 과정
  - 320명 수강생, 4.9 평점, 활성 상태
  - 완전한 파일 구조: metadata.ts, layout.tsx, ChapterContent.tsx
  - 5개 시뮬레이터 컴포넌트: ObjectDetectionLab, FaceRecognitionSystem, ImageEnhancementStudio, PoseEstimationTracker, TwoDToThreeDConverter
  - 5개 전용 시뮬레이터 페이지 완성
- **GraphRAG Explorer** RAG 모듈에 완전 구현:
  - Neo4j 스타일 지식 그래프 시각화
  - 엔티티/관계 추출 시뮬레이션
  - Force-directed 레이아웃, 파티클 효과
  - 전체화면 모드, 줌/팬 기능
  - 커뮤니티 감지, 인터랙티브 쿼리
  - Canvas 기반 고성능 렌더링
- **YouTube Summarizer** System Management Tools에 추가:
  - lilys.ai 스타일 YouTube 동영상 요약 도구
  - URL 입력으로 동영상 자동 분석
  - AI 기반 핵심 내용 요약 및 타임스탬프
  - 전체 스크립트 추출, 상세 분석 (난이도, 감정, 주제)
  - 요약 복사, JSON 다운로드, 공유 기능
  - Red-Orange 그라데이션 테마
- **Crypto Prediction Markets** Web3 모듈에 추가:
  - 블록체인 기반 암호화폐 가격 예측 시장 시뮬레이터
  - BTC, ETH, SOL, ADA, AVAX 5개 암호화폐 지원
  - 집단지성 기반 확률 계산, 실시간 시장 업데이트
  - YES/NO 토큰 거래, 포지션 관리, P&L 추적
  - 새로운 예측 시장 생성 기능
  - Chainlink 오라클 연동 시뮬레이션
  - 사용자 잔액 관리 및 거래 실행 시스템
- **Platform Status**:
  - 20+ active modules (Computer Vision 포함)
  - System Management Tools 5개 (YouTube Summarizer 포함)
  - 100+ chapters total
  - 50+ interactive simulators