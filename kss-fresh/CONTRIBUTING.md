# 🤝 Contributing to KSS (Knowledge Space Simulator)

Welcome to the KSS community! We're building the **Wikipedia of Interactive AI Education** - a platform where knowledge grows through collective intelligence and AI collaboration.

## 🌟 Our Vision

KSS는 **AI 시대의 지식 확장 전략 플랫폼**입니다:

> "확장 가능한 사고(Scalable Thinking)와 지식을 습득하기 위한 Interactive Learning Ecosystem"

**핵심 가치**:
- **확장성**: 커뮤니티 + AI가 함께 지식을 무한 확장
- **인터랙티브**: 모든 개념을 직접 만지고 실험 가능
- **실시간**: 최신 연구가 자동으로 반영
- **집단지성**: 위키피디아처럼 함께 만드는 지식

---

## 📚 Ways to Contribute

### 1. 📝 **Create New Content**

#### A. Write a Chapter
- **What**: Explain an AI concept in depth (Beginner/Intermediate/Advanced)
- **Format**: MDX (Markdown + React components)
- **Length**: 300-1000 lines
- **Requirements**:
  - Clear learning objectives
  - Code examples
  - Interactive elements
  - References (papers, docs, GitHub)

**Example Structure**:
```mdx
---
title: "Understanding Transformers"
description: "Deep dive into attention mechanisms"
difficulty: intermediate
duration: 2h
tags: [nlp, deep-learning, transformers]
---

## Learning Objectives
- Understand self-attention mechanism
- Implement multi-head attention
- Build a simple transformer

## Beginner Section
... (text + diagrams)

<AttentionVisualizer />

## Intermediate Section
... (code examples)

## Advanced Section
... (research papers)

## References
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- ...
```

#### B. Build a Simulator
- **What**: Interactive visualization of AI concepts
- **Tech**: React + TypeScript + Canvas/SVG/Three.js
- **Must Have**:
  - User-controllable parameters
  - Real-time visualization
  - Step-by-step explanation
  - Responsive design

**Use Our UI Components**:
```tsx
import { ResponsiveCanvas, AdaptiveLayout, CollapsibleControls }
  from '@/components/ui';

export function TransformerSimulator() {
  return (
    <AdaptiveLayout
      visualization={<AttentionHeatmap />}
      controls={
        <CollapsibleControls>
          <Slider label="Num Heads" min={1} max={12} />
          <Slider label="Sequence Length" min={4} max={128} />
        </CollapsibleControls>
      }
    />
  );
}
```

#### C. Improve Existing Content
- Fix typos or errors
- Update outdated information
- Add better examples
- Improve explanations

#### D. Add References
- Curate authoritative sources
- Link to latest papers
- Add GitHub implementations
- Connect related concepts

---

### 2. 🐛 **Report Issues**

Found a bug? Inaccurate content? [Create an issue](https://github.com/jeromwolf/ontology/issues/new/choose)

**Good Issue Format**:
```markdown
**Type**: Bug / Content Error / Feature Request

**Location**: /modules/llm/chapter-3

**Description**:
The explanation of RLHF is outdated (uses 2020 paper).

**Suggested Fix**:
Should reference InstructGPT (2022) and DPO (2023).

**References**:
- https://arxiv.org/abs/2203.02155
- https://arxiv.org/abs/2305.18290
```

---

### 3. 🔍 **Review Contributions**

Help maintain quality by reviewing others' work:
- Check factual accuracy
- Test simulators
- Verify code examples
- Suggest improvements

**Reviewer Checklist**:
- [ ] Content is accurate (fact-checked against sources)
- [ ] Code examples run without errors
- [ ] Simulators work on desktop + mobile
- [ ] Writing is clear and pedagogical
- [ ] References are authoritative
- [ ] Difficulty level is appropriate

---

## 🛠️ **How to Contribute**

### Prerequisites
```bash
# Clone the repository
git clone https://github.com/jeromwolf/ontology.git
cd ontology/kss-fresh

# Install dependencies
npm install

# Run development server
npm run dev
```

### Step-by-Step Guide

#### Option A: GitHub UI (No Coding Required)
1. Browse to the file you want to edit on GitHub
2. Click the **pencil icon** (✏️) to edit
3. Make your changes in the web editor
4. Scroll down and click **"Propose changes"**
5. Create a pull request

#### Option B: Local Development (For Simulators/Code)
```bash
# 1. Create a new branch
git checkout -b feature/your-contribution-name

# 2. Make your changes
# - Add files to src/app/modules/[module]/
# - Create simulators in components/
# - Update metadata.ts

# 3. Test locally
npm run dev
# Visit http://localhost:3000 to test

# 4. Build test (REQUIRED!)
npm run build
# Ensure no errors

# 5. Commit with descriptive message
git add .
git commit -m "feat: Add Transformer Attention Visualizer

- Implements multi-head attention simulator
- Includes 3 pre-built examples
- Mobile responsive design
- Tested on Chrome/Safari/Firefox"

# 6. Push to your fork
git push origin feature/your-contribution-name

# 7. Open Pull Request on GitHub
# Go to https://github.com/jeromwolf/ontology
# Click "Compare & pull request"
```

---

## 📋 **Contribution Guidelines**

### Code Style
- **TypeScript**: All new code must be TypeScript
- **Formatting**: Use Prettier (auto-format on save)
- **Naming**:
  - Components: `PascalCase` (e.g., `AttentionVisualizer`)
  - Files: `kebab-case` (e.g., `attention-visualizer.tsx`)
  - Variables: `camelCase`

### File Structure
```
src/app/modules/[module-name]/
├── components/
│   ├── chapters/
│   │   ├── Chapter1.tsx       (< 500 lines!)
│   │   └── Chapter2.tsx
│   └── simulators/
│       └── MySimulator.tsx
├── simulators/
│   └── [simulator-id]/
│       └── page.tsx
└── metadata.ts
```

### File Size Limits ⚠️

**왜 파일 크기를 제한하나요?**

큰 파일의 문제점:
- ⚠️ **사이드 이펙트 발생 위험 증가**
- 🐛 **버그 추적 어려움**
- 🔄 **병합 충돌 빈번**
- 📉 **유지보수성 감소**
- 🧠 **인지 부하 증가** (이해하기 어려움)

**해결책: 기능별 컴포넌트 분리**
- ✅ Single Responsibility Principle (한 파일 = 한 책임)
- ✅ 독립적 테스트 가능
- ✅ 재사용성 향상
- ✅ 병렬 작업 가능

**제한 기준**:
- **Chapter files**: < 500 lines (split if larger)
- **Simulator components**: < 800 lines
- **ChapterContent.tsx**: < 200 lines (router only!)

**예시**:
```
❌ Bad: 하나의 거대 파일
MySimulator.tsx (2000 lines)
  - UI + 로직 + 데이터 + 유틸리티 모두 포함

✅ Good: 기능별 분리
/my-simulator/
  ├── MySimulator.tsx (200 lines - 메인 컴포넌트)
  ├── SimulatorCanvas.tsx (300 lines - 시각화)
  ├── ControlPanel.tsx (150 lines - UI 컨트롤)
  ├── SimulatorLogic.ts (200 lines - 비즈니스 로직)
  └── utils.ts (100 lines - 유틸리티)
```

### Commit Message Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature (chapter, simulator)
- `fix`: Bug fix
- `docs`: Documentation
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance

**Example**:
```
feat(llm): Add RLHF Interactive Trainer

- Implements PPO algorithm visualization
- Shows reward model training in real-time
- Includes 3 difficulty modes
- References latest InstructGPT paper

Closes #123
```

---

## 🤖 **AI-Assisted Contributions**

### Using AI to Generate Content

We **encourage** using AI (Claude, GPT-4, Gemini) to:
- Draft initial chapter outlines
- Generate code examples
- Create quiz questions
- Suggest references

**⚠️ 중요: 검증은 필수입니다!**

> "사람도 실수하고 AI도 실수할 수 있습니다. 최대한 실수를 방지해야 합니다."

**필수 검증 프로세스**:

#### 1️⃣ **사실 확인 (Fact Checking)**
- [ ] 주요 주장을 **신뢰할 수 있는 소스**와 대조
  - 원본 논문 (arXiv, 학술지)
  - 공식 문서 (PyTorch, TensorFlow docs)
  - 신뢰할 수 있는 교과서
- [ ] 논문 인용이 정확한지 확인 (제목, 저자, 년도)
- [ ] 수식과 알고리즘이 원본과 일치하는지 확인

#### 2️⃣ **코드 검증 (Code Verification)**
- [ ] **실제로 실행**해서 작동 확인
- [ ] 엣지 케이스 테스트 (빈 입력, 큰 값, 음수 등)
- [ ] 브라우저 호환성 (Chrome, Safari, Firefox)
- [ ] 모바일 반응형 테스트
- [ ] TypeScript 타입 에러 없음
- [ ] Build 통과 (`npm run build`)

#### 3️⃣ **인간 리뷰 (Human Review)**
- [ ] 최소 **2명 이상**의 리뷰어 승인 필요
- [ ] Expert 등급 이상 **1명 포함**
- [ ] 리뷰어가 실제로 코드 실행 확인

#### 4️⃣ **AI 어시스턴트 명시**
PR Description에 반드시 포함:
```markdown
## AI Assistance
- Used GPT-4 to generate initial chapter outline
- Claude helped refactor simulator code

## Verification
- ✅ All content verified against original papers:
  - DDPM: https://arxiv.org/abs/2006.11239 (Checked: Fig 2, Eq 1-5)
  - DDIM: https://arxiv.org/abs/2010.02502 (Checked: Algorithm 1)
- ✅ Code tested on Chrome 120, Safari 17, Firefox 121
- ✅ Mobile tested on iPhone 15 Pro, Galaxy S24
- ✅ Build passes without errors
- ✅ Reviewed by @expert-reviewer-1, @expert-reviewer-2

## Test Evidence
[Screenshots or video of working simulator]
```

**자동 거부 사유**:
- ❌ 검증 없이 AI가 생성한 내용 그대로 제출
- ❌ 코드가 실행 안 되거나 에러 발생
- ❌ 논문 인용 오류 (잘못된 제목, 링크)
- ❌ 리뷰어 승인 없이 병합 시도

---

## 🏆 **Contributor Levels**

심플하고 명확한 3단계 시스템:

| Level | Requirements | Badge | Permissions |
|-------|-------------|-------|-------------|
| **🌱 Contributor** | 1+ merged PR | 🌱 | - Submit PRs<br>- Participate in discussions |
| **🎓 Expert** | 10+ merged PRs<br>+ High quality (90%+ approval) | 🎓 | - Review others' PRs<br>- Approve minor changes<br>- Mentor newcomers |
| **🏆 Maintainer** | 50+ merged PRs<br>+ Community leadership<br>+ Technical excellence | 🏆 | - Merge permissions<br>- Repository admin<br>- Strategic decisions |

**등급 업그레이드 기준**:
- **Quality over Quantity** (양보다 질!)
- 다른 기여자 도움 (Mentorship)
- 커뮤니티 활동 (Discussions, Reviews)
- 코드 품질 & 문서화

**Special Badges** (추가 성취):
- 🤖 **AI Pioneer**: 10+ AI-verified contributions
- 🎮 **Simulator Master**: 10+ interactive simulators
- 📚 **Knowledge Architect**: 50+ concept connections
- 🔍 **Code Reviewer**: 30+ quality reviews
- 🌍 **Translator**: Multi-language support

---

## ❓ **FAQ**

### Q: I'm not an AI expert. Can I still contribute?
**A**: Absolutely! We need:
- Technical writers (improve clarity)
- Designers (better visualizations)
- Translators (multi-language support)
- Testers (find bugs)
- Curators (organize references)

### Q: How long does review take?
**A**:
- Small fixes: 1-2 days
- New chapters: 3-7 days
- Major features: 1-2 weeks

### Q: Can I contribute in languages other than English?
**A**: Yes! We're building multi-language support. Korean (한국어) is fully supported.

### Q: What if my PR is rejected?
**A**: We'll provide detailed feedback. Common reasons:
- Factual errors
- Poor code quality
- Duplicate content
- Out of scope

You can always revise and resubmit!

### Q: Do I retain copyright?
**A**: You retain copyright. By contributing, you grant KSS a license to use, modify, and distribute your work under MIT License.

---

## 📞 **Get Help**

- **GitHub Discussions**: [Ask questions](https://github.com/jeromwolf/ontology/discussions)
  - 💬 새로운 사용자? [Discussions 설정 가이드](GITHUB_DISCUSSIONS_SETUP.md) 참고
  - 8개 카테고리: Announcements, Ideas, Q&A, Learning, Development, Show & Tell, Community, Bug Reports
- **Discord**: Join our community (link TBD)
- **Email**: support@kss.ai (coming soon)

---

## 🙏 **Code of Conduct**

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

**TL;DR**:
- Be respectful and inclusive
- Focus on constructive feedback
- No harassment or discrimination
- Assume good intentions

---

## 🎯 **Quick Start for Beginners**

New to open source? Start here:

1. **Read an existing chapter**: Understand our style
2. **Find a "good first issue"**: [Browse issues](https://github.com/jeromwolf/ontology/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
3. **Make a small fix**: Fix a typo or improve wording
4. **Ask for feedback**: We're here to help!

**First PR Checklist**:
- [ ] Read this guide
- [ ] Installed dependencies (`npm install`)
- [ ] Made your changes
- [ ] Tested locally (`npm run dev`)
- [ ] Build passed (`npm run build`)
- [ ] Opened PR with clear description

---

## 📜 **License**

By contributing, you agree that your contributions will be licensed under the **MIT License**.

### Why MIT License?

**MIT License**는 가장 널리 사용되는 오픈소스 라이선스입니다:
- ✅ **상업적 사용 가능** - 기업도 자유롭게 사용
- ✅ **수정 가능** - 누구나 개선 가능
- ✅ **재배포 가능** - Fork & 공유 자유
- ✅ **간결함** - 법적 복잡성 최소화

**다른 라이선스 옵션**:

| License | 특징 | 장점 | 단점 | 추천 여부 |
|---------|-----|------|------|----------|
| **MIT** | 매우 허용적 | 간단, 널리 사용됨 | 파생물이 closed-source 가능 | ✅ **현재 선택** |
| **Apache 2.0** | MIT + 특허 보호 | 특허 명시, 대기업 선호 | 조금 복잡 | ⭐ 고려 가능 |
| **GPL v3** | Copyleft (강력한 오픈소스) | 파생물도 오픈소스 강제 | 상업적 사용 제한적 | ❌ 교육용에는 과함 |
| **CC BY-SA** | 크리에이티브 커먼즈 | 교육 콘텐츠에 적합 | 코드에는 부적합 | ❌ 코드 포함 프로젝트엔 안 맞음 |

**권장: MIT License 유지**
- 교육 플랫폼에 가장 적합
- 기여자 유입 쉬움 (진입 장벽 낮음)
- 기업 도입 용이 (상업적 활용 가능)

**궁금하신 점**:
- MIT License에 대해 더 알고 싶으시면: https://choosealicense.com/licenses/mit/
- 라이선스 변경 제안: GitHub Discussions에서 논의

---

**Thank you for helping build the future of AI education! 🚀**

Together, we're creating a knowledge commons that will educate millions.

---

*Last updated: 2025-10-10*
*Questions? Open a [discussion](https://github.com/jeromwolf/ontology/discussions/new)*
