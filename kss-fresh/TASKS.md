# 📋 KSS Project Tasks

> **프로젝트 비전**: AI 시대의 지식 확장 전략 플랫폼 - Wikipedia처럼 커뮤니티 + AI가 함께 만드는 지식

---

## 🎯 Overall Progress

| Phase | Status | Progress | Completion Date |
|-------|--------|----------|-----------------|
| **Phase 1** | ✅ Completed | 4/4 (100%) | 2025-10-10 |
| **Phase 2** | ⏳ Pending | 0/4 (0%) | TBD |
| **Phase 3** | ⏳ Pending | 0/4 (0%) | TBD |

---

## ✅ Phase 1: GitHub Discussions & 커뮤니티 구축 (COMPLETED)

**목표**: 커뮤니티 참여 기반 구축 및 기여 가이드라인 확립

**완료일**: 2025-10-10

### Tasks
- [x] **1-1. GitHub Discussions 활성화 및 카테고리 설정**
  - 8개 카테고리 설계 완료
  - GITHUB_DISCUSSIONS_SETUP.md 작성 (493줄)
  - 4개 초기 Discussion 주제 템플릿 제공
  - GitHub Actions 자동화 워크플로우 작성

- [x] **1-2. CONTRIBUTING.md 작성 (기여 가이드라인)**
  - 504줄 완성
  - 6가지 기여 방법 명시
  - AI 검증 4단계 프로세스 확립
  - 3-tier 기여자 시스템 설계
  - MIT License 선택 근거 문서화

- [x] **1-3. CODE_OF_CONDUCT.md 작성 (행동 강령)**
  - 293줄 완성
  - "존중과 포용" 핵심 가치 반영
  - 3단계 경고 시스템 (Warning → Temporary Ban → Permanent Ban)
  - 신고자 보호 정책 수립

- [x] **1-4. 커뮤니티 온보딩 페이지 제작 (/contribute)**
  - 529줄 React 컴포넌트 완성
  - 인터랙티브 UI (6가지 기여 타입 카드)
  - 4단계 Quick Start 가이드
  - 완전 반응형 + 다크모드 지원

### Deliverables
- ✅ CONTRIBUTING.md (504줄)
- ✅ CODE_OF_CONDUCT.md (293줄)
- ✅ GITHUB_DISCUSSIONS_SETUP.md (493줄)
- ✅ /contribute page (529줄)
- ✅ Git commit & documentation

### Success Metrics
- ✅ 모든 문서 작성 완료
- ✅ 온보딩 페이지 빌드 성공
- ✅ GitHub 저장소에 커밋 완료

---

## ⏳ Phase 2: ArXiv Monitor 시스템 구축 (PENDING)

**목표**: 최신 AI 논문을 자동으로 수집하고 LLM으로 요약하여 MDX 콘텐츠 생성

**예상 기간**: 2-3주

### Tasks
- [ ] **2-1. ArXiv API 크롤러 개발**
  - [ ] ArXiv API 연동 라이브러리 설치
  - [ ] 카테고리별 논문 수집 스크립트 (cs.AI, cs.LG, cs.CL, cs.CV, cs.RO)
  - [ ] 매일 자동 실행 스케줄러 (cron/GitHub Actions)
  - [ ] 논문 메타데이터 파싱 (제목, 저자, 초록, PDF 링크)
  - [ ] 데이터베이스 저장 (Prisma + PostgreSQL)
  - [ ] 중복 논문 필터링

- [ ] **2-2. LLM 기반 논문 요약 Agent 구축**
  - [ ] OpenAI API 통합 (GPT-4)
  - [ ] 프롬프트 엔지니어링 (논문 요약 템플릿)
  - [ ] 3가지 요약 길이 (Short/Medium/Long)
  - [ ] 핵심 개념 추출 (Key Concepts)
  - [ ] 관련 모듈 매핑 (LLM, Computer Vision, RAG 등)
  - [ ] 요약 품질 검증 시스템

- [ ] **2-3. 자동 MDX 콘텐츠 생성 파이프라인**
  - [ ] MDX 템플릿 설계
  - [ ] 논문 → MDX 변환 로직
  - [ ] 코드 예제 자동 생성 (선택적)
  - [ ] References 섹션 자동 추가
  - [ ] Git commit & PR 자동 생성
  - [ ] 리뷰 요청 알림

- [ ] **2-4. Discord/Slack 알림 시스템 연동**
  - [ ] Webhook 설정
  - [ ] 새 논문 알림 메시지 포맷
  - [ ] 요약 결과 자동 전송
  - [ ] 에러 알림 시스템
  - [ ] 주간 리포트 자동 생성

### Deliverables
- [ ] `/scripts/arxiv-crawler/` (크롤러 스크립트)
- [ ] `/scripts/llm-summarizer/` (요약 Agent)
- [ ] `/scripts/mdx-generator/` (MDX 생성기)
- [ ] `/scripts/notification/` (알림 시스템)
- [ ] GitHub Actions workflow
- [ ] 관련 문서 (README, 설정 가이드)

### Success Metrics
- [ ] 매일 새 논문 10개 이상 수집
- [ ] 요약 품질 90% 이상
- [ ] MDX 빌드 에러 0%
- [ ] Discord/Slack 알림 100% 전송

### Dependencies
- OpenAI API 키
- ArXiv API access
- Discord/Slack Webhook URL
- Prisma DB 설정

---

## ⏳ Phase 3: Contribution System 개발 (PENDING)

**목표**: 커뮤니티 기여자가 직접 콘텐츠를 작성/수정할 수 있는 시스템

**예상 기간**: 3-4주

### Tasks
- [ ] **3-1. MDX 에디터 컴포넌트 개발**
  - [ ] Monaco Editor 통합
  - [ ] MDX 문법 하이라이팅
  - [ ] 실시간 미리보기
  - [ ] 자동완성 (컴포넌트, 프롭)
  - [ ] 이미지 업로드
  - [ ] 코드 스니펫 라이브러리

- [ ] **3-2. Git 기반 콘텐츠 관리 시스템 (CMS)**
  - [ ] 웹 UI에서 파일 생성/수정
  - [ ] Git commit 자동화
  - [ ] Branch 전략 (feature/contribution-*)
  - [ ] PR 자동 생성
  - [ ] 버전 히스토리 UI
  - [ ] 충돌 해결 도구

- [ ] **3-3. AI 자동 리뷰 시스템**
  - [ ] 사실 확인 (fact-checking) Agent
  - [ ] 논문 인용 검증
  - [ ] 코드 실행 테스트
  - [ ] 타입 에러 체크
  - [ ] 파일 크기 검증 (< 500줄)
  - [ ] 스타일 가이드 준수 체크
  - [ ] 리뷰 결과 코멘트 자동 생성

- [ ] **3-4. 기여자 프로필 & 배지 시스템**
  - [ ] 사용자 프로필 페이지
  - [ ] 기여 통계 (PR 수, 리뷰 수, 챕터 수)
  - [ ] 배지 시스템 (🌱 Contributor, 🎓 Expert, 🏆 Maintainer)
  - [ ] 특별 배지 (🤖 AI Pioneer, 🎮 Simulator Master)
  - [ ] 기여자 랭킹 (선택적)
  - [ ] 활동 타임라인

### Deliverables
- [ ] `/src/components/mdx-editor/` (에디터 컴포넌트)
- [ ] `/src/app/cms/` (CMS 페이지)
- [ ] `/scripts/ai-review/` (AI 리뷰 Agent)
- [ ] `/src/app/profile/[userId]/` (프로필 페이지)
- [ ] Database schema 확장 (User, Badge, Contribution)
- [ ] API routes (/api/cms/*, /api/profile/*)

### Success Metrics
- [ ] MDX 에디터 응답 속도 < 100ms
- [ ] Git 작업 성공률 > 95%
- [ ] AI 리뷰 정확도 > 85%
- [ ] 기여자 만족도 > 4.0/5.0

### Dependencies
- NextAuth.js (인증)
- Monaco Editor
- OpenAI API (AI 리뷰)
- GitHub API (Git 작업)

---

## 📊 Project Timeline

```
2025-10-10: Phase 1 완료 ✅
2025-10-11 ~ 2025-10-31: Phase 2 개발 (예정)
2025-11-01 ~ 2025-11-30: Phase 3 개발 (예정)
2025-12-01: 전체 시스템 통합 테스트 (예정)
```

---

## 🚀 Next Actions

### Immediate (Phase 2 시작 전 준비)
- [ ] OpenAI API 키 발급 및 설정
- [ ] ArXiv API 테스트
- [ ] Discord Webhook 설정
- [ ] Prisma schema 설계 (Paper, Summary 테이블)
- [ ] Phase 2 상세 설계 문서 작성

### Short-term (Phase 2 개발)
- [ ] 2-1 태스크 시작
- [ ] 크롤러 프로토타입 구현
- [ ] 첫 논문 수집 테스트

### Long-term (Phase 3 이후)
- [ ] 커뮤니티 베타 테스트
- [ ] 피드백 수집 및 개선
- [ ] 공식 런칭 준비
- [ ] YouTube 콘텐츠 제작 (Remotion)

---

## 📝 Notes

### Design Principles
1. **천천히 차분히**: 거대한 목표를 위해 철저하게 기획/분석/설계/개발
2. **검증 필수**: AI 생성 콘텐츠도 반드시 사람이 검증
3. **커뮤니티 우선**: 사용자 경험과 기여자 만족도 최우선
4. **확장 가능**: 모든 시스템이 대규모로 확장 가능하도록 설계

### Key Decisions
- **라이선스**: MIT License (교육 플랫폼에 최적)
- **기여자 시스템**: 3-tier (간결성 우선)
- **AI 역할**: 콘텐츠 생성 보조 + 자동 리뷰
- **Git 전략**: Feature branch + PR 기반

### Risks & Mitigation
| Risk | Mitigation |
|------|-----------|
| AI 요약 품질 저하 | 사람 리뷰 필수, 프롬프트 최적화 |
| 커뮤니티 참여 부족 | 온보딩 강화, 인센티브 설계 |
| 시스템 복잡도 증가 | 모듈화, 문서화 철저 |
| API 비용 증가 | 캐싱, 배치 처리, 비용 모니터링 |

---

## 📞 Contact & Support

- **GitHub Discussions**: [질문하기](https://github.com/jeromwolf/ontology/discussions)
- **Issues**: [버그 리포트](https://github.com/jeromwolf/ontology/issues)
- **Email**: support@kss.ai (coming soon)

---

**Last updated**: 2025-10-10
**Next review**: Phase 2 시작 시
**Maintained by**: KSS Team
