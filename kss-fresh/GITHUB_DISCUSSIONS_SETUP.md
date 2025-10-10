# 🗨️ GitHub Discussions 설정 가이드

## 📋 목차
1. [GitHub Discussions 활성화 방법](#1-github-discussions-활성화-방법)
2. [카테고리 구조](#2-카테고리-구조)
3. [초기 Discussion 주제](#3-초기-discussion-주제)
4. [모더레이션 가이드](#4-모더레이션-가이드)
5. [자동화 설정](#5-자동화-설정)

---

## 1. GitHub Discussions 활성화 방법

### Step 1: Repository 설정 접근
1. GitHub에서 `https://github.com/jeromwolf/ontology` 저장소 접속
2. 상단 메뉴에서 **Settings** 클릭
3. 왼쪽 사이드바 **Features** 섹션에서 **Discussions** 찾기

### Step 2: Discussions 활성화
1. **Set up discussions** 버튼 클릭
2. 기본 카테고리가 자동 생성됨:
   - 📣 Announcements
   - 💬 General
   - 💡 Ideas
   - 🙏 Q&A
   - 🙌 Show and tell

### Step 3: 카테고리 커스터마이징
1. Discussions 탭으로 이동
2. 오른쪽 상단 **⚙️ (톱니바퀴 아이콘)** 클릭 → **Edit categories** 선택
3. 아래 권장 카테고리 구조로 수정

---

## 2. 카테고리 구조

KSS 프로젝트에 최적화된 **8개 카테고리** 구조:

### 📣 **Announcements** (공지사항)
- **Format**: Announcement (관리자만 생성 가능)
- **설명**: 프로젝트 업데이트, 새로운 모듈 출시, 중요 변경사항
- **예시 주제**:
  - "🚀 Python Programming 모듈 출시!"
  - "📊 주간 통계: 기여자 50명 돌파"
  - "🎯 2025 Q1 로드맵"

### 💡 **Ideas & Proposals** (아이디어 & 제안)
- **Format**: Open discussion
- **설명**: 새로운 모듈, 기능, 개선사항 제안
- **예시 주제**:
  - "🧠 Neuroscience AI 모듈 추가 제안"
  - "🎨 3D 시뮬레이터에 VR 지원 추가"
  - "🌐 다국어 지원 (영어, 한국어, 일본어)"

### ❓ **Q&A** (질문 & 답변)
- **Format**: Q&A (답변 선택 가능)
- **설명**: 기술적 질문, 사용법, 트러블슈팅
- **예시 주제**:
  - "시뮬레이터 로컬에서 실행 안 됨 (ENOENT 에러)"
  - "MDX 파일에 LaTeX 수식 추가하는 방법?"
  - "Chapter 파일 크기 500줄 넘을 때 분리 방법"

### 🎓 **Learning & Resources** (학습 자료)
- **Format**: Open discussion
- **설명**: 외부 학습 자료 공유, 논문 리뷰, 강의 추천
- **예시 주제**:
  - "📚 추천 논문: Attention Is All You Need 완벽 분석"
  - "🎥 Andrew Ng의 Deep Learning Specialization 후기"
  - "🔬 arXiv 최신 논문: Llama 4 발표"

### 🛠️ **Development** (개발 논의)
- **Format**: Open discussion
- **설명**: 아키텍처, 코드 리뷰, 성능 최적화, 기술 스택
- **예시 주제**:
  - "⚡ 시뮬레이터 렌더링 성능 개선 방법"
  - "🏗️ 모듈 파일 구조 Best Practice"
  - "🔄 ArXiv Monitor 시스템 설계 논의"

### 🎉 **Show & Tell** (작업 공유)
- **Format**: Open discussion
- **설명**: 완성한 작업 자랑, 프로젝트 쇼케이스
- **예시 주제**:
  - "🎮 제가 만든 Quantum Circuit 시뮬레이터!"
  - "📊 RAG 모듈 완전 리팩토링 완료 후기"
  - "🌟 첫 PR 병합되었어요!"

### 🌍 **Community** (커뮤니티)
- **Format**: Open discussion
- **설명**: 자기소개, 네트워킹, 이벤트, 오프라인 모임
- **예시 주제**:
  - "👋 안녕하세요, AI 엔지니어 지망생입니다"
  - "🤝 서울 지역 스터디 모임 참여자 모집"
  - "🎤 KSS 컨트리뷰터 밋업 (11월 15일)"

### 🐛 **Bug Reports & Support** (버그 리포트)
- **Format**: Q&A
- **설명**: 버그 발견, 에러 보고, 긴급 지원 요청
- **예시 주제**:
  - "🔴 Ontology 모듈 Chapter 3 렌더링 실패"
  - "⚠️ 빌드 시 TypeScript 에러 발생"
  - "💥 3D 그래프 노드 클릭 시 앱 크래시"

---

## 3. 초기 Discussion 주제

커뮤니티 활성화를 위한 **첫 Discussion 주제 4개**:

### 📣 Announcements
```markdown
Title: 🎉 KSS 커뮤니티 GitHub Discussions 오픈!

안녕하세요, KSS 커뮤니티 여러분!

**Knowledge Space Simulator**의 공식 커뮤니티 포럼이 GitHub Discussions로 오픈되었습니다.

## 🌟 여기서 할 수 있는 것
- 💡 새로운 모듈 아이디어 제안
- ❓ 기술적 질문 & 답변
- 🛠️ 개발 논의 & 코드 리뷰
- 📚 학습 자료 공유
- 🎉 작업 결과물 자랑

## 📖 시작 가이드
1. [CONTRIBUTING.md](./CONTRIBUTING.md) 읽기
2. [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md) 확인
3. [/contribute](https://kss.ai/contribute) 페이지 방문

## 🎯 첫 번째 작업
- Community 카테고리에서 **자기소개** 남기기
- Ideas 카테고리에서 **원하는 모듈 제안**하기
- Q&A 카테고리에서 **궁금한 점 질문**하기

함께 AI 시대의 지식을 만들어갑시다! 🚀

---
*질문이 있으시면 언제든지 물어보세요!*
```

### 🌍 Community
```markdown
Title: 👋 자기소개 스레드 - 여러분을 소개해주세요!

KSS 커뮤니티에 오신 것을 환영합니다!

간단히 자기소개를 남겨주세요:
- **이름/닉네임**:
- **관심 분야**: (AI, 양자컴퓨팅, 블록체인 등)
- **현재 하는 일**: (학생, 개발자, 연구자 등)
- **KSS에서 기여하고 싶은 분야**: (챕터 작성, 시뮬레이터 개발 등)
- **좋아하는 KSS 모듈**:

## 🌟 예시
> **이름**: Kelly
> **관심 분야**: AI, Ontology, Knowledge Graphs
> **현재 하는 일**: KSS Founder
> **기여하고 싶은 분야**: 전체 플랫폼 설계, 커뮤니티 운영
> **좋아하는 모듈**: LLM, Multi-Agent, Ontology

여러분의 이야기를 들려주세요! 🙌
```

### 💡 Ideas & Proposals
```markdown
Title: 💭 원하는 모듈 투표 - 다음에 무엇을 만들까요?

다음 개발할 모듈을 커뮤니티가 결정합니다!

## 🗳️ 후보 모듈 (댓글로 👍 투표)
1. **AI Infrastructure** - MLOps, 모델 배포, 모니터링
2. **Cloud Computing** - AWS, GCP, Azure 실습
3. **Creative AI** - Stable Diffusion, DALL-E, Midjourney
4. **Robotics** - ROS, 로봇 제어, 시뮬레이션
5. **AI for Finance** - 알고리즘 트레이딩, 리스크 관리
6. **Natural Language Processing** - BERT, GPT, 토크나이징

## 📝 새로운 제안
위 목록에 없는 모듈을 제안하고 싶으시면 **댓글로 남겨주세요!**

가장 많은 투표를 받은 모듈을 우선 개발합니다. 🚀
```

### 🛠️ Development
```markdown
Title: 🏗️ ArXiv Monitor 시스템 설계 논의 (Phase 2 시작)

Phase 2: **ArXiv Monitor 시스템** 개발을 시작합니다.

## 🎯 목표
최신 AI 논문을 자동으로 수집하고, LLM으로 요약하여, MDX 콘텐츠를 생성하는 시스템

## 💡 현재 계획
1. **ArXiv API 크롤러**: 매일 새 논문 수집
2. **LLM 요약 Agent**: GPT-4로 논문 핵심 내용 추출
3. **MDX 생성 파이프라인**: 챕터 형식으로 자동 변환
4. **Discord/Slack 알림**: 새 논문 자동 공지

## ❓ 논의 주제
1. 어떤 논문 카테고리를 추적할까요?
   - cs.AI, cs.LG, cs.CL, cs.CV, cs.RO?
2. 요약 길이는?
   - Short (300 words), Medium (600 words), Long (1000+ words)?
3. 어떤 형식으로 제공?
   - 챕터, References, 별도 페이지?

여러분의 의견을 들려주세요! 💬
```

---

## 4. 모더레이션 가이드

### 모더레이터 역할
- **Kelly (Founder)** - 전체 관리, 전략적 결정
- **Maintainer 팀** - 일상 모더레이션, 분쟁 해결
- **Expert 기여자** - 기술 질문 답변, 신규 회원 멘토링

### 모더레이션 원칙
1. **빠른 응답**: Q&A는 24시간 내 답변 목표
2. **존중과 포용**: CODE_OF_CONDUCT.md 엄격 적용
3. **건설적 피드백**: 비판이 아닌 개선 제안
4. **투명성**: 모든 결정 공개 및 설명

### 문제 상황 처리
| 상황 | 조치 |
|------|------|
| **경미한 규칙 위반** (예: 무례한 언어) | DM으로 경고, Discussion 수정 요청 |
| **반복 위반** (예: 스팸, Trolling) | Discussion 잠금, 7일 임시 정지 |
| **심각한 위반** (예: 괴롭힘, 차별) | Discussion 삭제, 영구 퇴출 |

---

## 5. 자동화 설정

### GitHub Actions Workflow 추천

**파일**: `.github/workflows/discussions-automation.yml`

```yaml
name: Discussions Automation

on:
  discussion:
    types: [created]
  discussion_comment:
    types: [created]

jobs:
  auto-label:
    runs-on: ubuntu-latest
    steps:
      - name: Auto-label new discussions
        uses: actions/github-script@v6
        with:
          script: |
            const discussion = context.payload.discussion;
            const labels = [];

            // Auto-label based on category
            if (discussion.category.name === 'Q&A') {
              labels.push('question');
            } else if (discussion.category.name === 'Bug Reports & Support') {
              labels.push('bug', 'needs-triage');
            }

            if (labels.length > 0) {
              await github.rest.issues.addLabels({
                owner: context.repo.owner,
                repo: context.repo.repo,
                issue_number: discussion.number,
                labels: labels
              });
            }

  welcome-message:
    runs-on: ubuntu-latest
    steps:
      - name: Welcome new contributors
        uses: actions/github-script@v6
        with:
          script: |
            const discussion = context.payload.discussion;
            const author = discussion.user.login;

            // Check if first-time contributor
            const discussions = await github.rest.search.issuesAndPullRequests({
              q: `repo:${context.repo.owner}/${context.repo.repo} author:${author} type:discussions`
            });

            if (discussions.data.total_count === 1) {
              await github.rest.discussions.createComment({
                owner: context.repo.owner,
                repo: context.repo.repo,
                discussion_number: discussion.number,
                body: `👋 @${author}님, KSS 커뮤니티에 첫 Discussion을 작성해주셔서 감사합니다!\n\n다음을 확인해보세요:\n- [CONTRIBUTING.md](./CONTRIBUTING.md)\n- [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md)\n- [/contribute](https://kss.ai/contribute)`
              });
            }
```

### Discord/Slack 연동 (Phase 2에서 구현 예정)

**Webhook 설정 미리보기**:
```javascript
// .github/workflows/notify-discord.yml
- name: Send Discord notification
  run: |
    curl -X POST ${{ secrets.DISCORD_WEBHOOK_URL }} \
      -H "Content-Type: application/json" \
      -d '{
        "embeds": [{
          "title": "🆕 New Discussion",
          "description": "${{ github.event.discussion.title }}",
          "url": "${{ github.event.discussion.html_url }}",
          "color": 5814783,
          "author": {
            "name": "${{ github.event.discussion.user.login }}"
          }
        }]
      }'
```

---

## 6. 성공 지표 (KPIs)

### 1개월 목표
- ✅ **활성 멤버**: 50명 이상
- ✅ **Discussion 수**: 30개 이상
- ✅ **Q&A 답변률**: 80% 이상 (24시간 내)
- ✅ **기여자 전환율**: 20% (Discussion 참여 → PR 제출)

### 3개월 목표
- ✅ **활성 멤버**: 200명 이상
- ✅ **Discussion 수**: 150개 이상
- ✅ **Expert 등급**: 10명 이상
- ✅ **신규 모듈**: 커뮤니티 제안으로 3개 이상 추가

---

## 7. 체크리스트

Phase 1 완료를 위한 최종 체크리스트:

- [ ] GitHub Discussions 활성화
- [ ] 8개 카테고리 설정 (위 구조대로)
- [ ] 4개 초기 Discussion 작성 (Announcements, Community, Ideas, Development)
- [ ] Welcome message automation 설정
- [ ] CONTRIBUTING.md에 Discussions 링크 추가
- [ ] /contribute 페이지에 Discussions 섹션 추가
- [ ] Discord/Slack 채널에 공지

---

## 📞 문의

질문이나 도움이 필요하시면:
- **GitHub Issues**: 기술적 버그
- **GitHub Discussions**: 일반 질문, 제안
- **Email**: conduct@kss.ai (예정)

---

**Last updated**: 2025-10-10
**Version**: 1.0.0
**Author**: KSS Team

---

**다음 단계**: Phase 2 - ArXiv Monitor 시스템 구축 🚀
