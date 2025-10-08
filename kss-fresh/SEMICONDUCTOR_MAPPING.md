# 반도체 모듈 전체 매핑 문서

## 📚 ChapterId → Chapter 파일 매핑

| ChapterId | Chapter 파일 | 제목 |
|-----------|-------------|------|
| `basics` | Chapter1.tsx | Chapter 1: 반도체 기초 |
| `design` | Chapter2.tsx | Chapter 2: 디지털 회로 설계 |
| `lithography` | Chapter3.tsx | Chapter 3: 포토리소그래피 |
| `fabrication` | Chapter4.tsx | Chapter 4: 반도체 제조 공정 |
| `advanced` | Chapter5.tsx | Chapter 5: 첨단 반도체 기술 |
| `ai-chips` | Chapter6.tsx | Chapter 6: AI 반도체 설계 |
| `memory` | Chapter7.tsx | Chapter 7: 차세대 메모리 |
| `future` | Chapter8.tsx | Chapter 8: 미래 반도체 기술 |
| `image-display` | Chapter9.tsx | Chapter 9: 이미지센서 & 디스플레이 반도체 |

---

## 📘 초급 과정 (beginnerCurriculum.ts)

| 번호 | Module ID | 제목 | chapterId | 실제 링크 | 이동할 Chapter |
|------|-----------|------|-----------|-----------|----------------|
| 1 | basics-1 | 1. 반도체란 무엇인가? | `basics` | `/modules/semiconductor/basics` | Chapter 1: 반도체 기초 |
| 2 | basics-2 | 2. 도핑과 PN 접합 | `basics` | `/modules/semiconductor/basics` | Chapter 1: 반도체 기초 |
| 3 | basics-3 | 3. 다이오드의 원리 | `basics` | `/modules/semiconductor/basics` | Chapter 1: 반도체 기초 |
| 4 | basics-4 | 4. 트랜지스터 기초 | `design` | `/modules/semiconductor/design` | Chapter 2: 디지털 회로 설계 |
| 5 | basics-5 | 5. 디지털 논리 회로 | `design` | `/modules/semiconductor/design` | Chapter 2: 디지털 회로 설계 |

**초급 과정 요약:**
- 모듈 1-3 → `basics` → Chapter 1
- 모듈 4-5 → `design` → Chapter 2

---

## 📙 중급 과정 (intermediateCurriculum.ts)

| 번호 | Module ID | 제목 | chapterId | 실제 링크 | 이동할 Chapter |
|------|-----------|------|-----------|-----------|----------------|
| 1 | inter-1 | 1. 포토리소그래피 | `lithography` | `/modules/semiconductor/lithography` | Chapter 3: 포토리소그래피 |
| 2 | inter-2 | 2. 박막 증착 기술 | `fabrication` | `/modules/semiconductor/fabrication` | Chapter 4: 반도체 제조 공정 |
| 3 | inter-3 | 3. 에칭 공정 | `fabrication` | `/modules/semiconductor/fabrication` | Chapter 4: 반도체 제조 공정 |
| 4 | inter-4 | 4. 이온주입 & CMP | `fabrication` | `/modules/semiconductor/fabrication` | Chapter 4: 반도체 제조 공정 |
| 5 | inter-5 | 5. 웨이퍼 제조 & 패키징 | `fabrication` | `/modules/semiconductor/fabrication` | Chapter 4: 반도체 제조 공정 |
| 6 | inter-6 | 6. 첨단 제조 기술 | `advanced` | `/modules/semiconductor/advanced` | Chapter 5: 첨단 반도체 기술 |

**중급 과정 요약:**
- 모듈 1 → `lithography` → Chapter 3
- 모듈 2-5 → `fabrication` → Chapter 4
- 모듈 6 → `advanced` → Chapter 5

---

## 📕 고급 과정 (advancedCurriculum.ts)

| 번호 | Module ID | 제목 | chapterId | 실제 링크 | 이동할 Chapter |
|------|-----------|------|-----------|-----------|----------------|
| 1 | adv-1 | 1. FinFET & GAA 기술 | `advanced` | `/modules/semiconductor/advanced` | Chapter 5: 첨단 반도체 기술 |
| 2 | adv-2 | 2. 3D 적층 기술 | `advanced` | `/modules/semiconductor/advanced` | Chapter 5: 첨단 반도체 기술 |
| 3 | adv-3 | 3. AI 반도체 아키텍처 | `ai-chips` | `/modules/semiconductor/ai-chips` | Chapter 6: AI 반도체 설계 |
| 4 | adv-4 | 4. 메모리 반도체 | `memory` | `/modules/semiconductor/memory` | Chapter 7: 차세대 메모리 |
| 5 | adv-5 | 5. 미래 반도체 기술 | `future` | `/modules/semiconductor/future` | Chapter 8: 미래 반도체 기술 |
| 6 | adv-6 | 6. 이미지센서 & 디스플레이 반도체 | `image-display` | `/modules/semiconductor/image-display` | Chapter 9: 이미지센서 & 디스플레이 반도체 |

**고급 과정 요약:**
- 모듈 1-2 → `advanced` → Chapter 5
- 모듈 3 → `ai-chips` → Chapter 6
- 모듈 4 → `memory` → Chapter 7
- 모듈 5 → `future` → Chapter 8
- 모듈 6 → `image-display` → Chapter 9

---

## 🔍 검증 포인트

### 사용자가 "2. 도핑과 PN 접합" 클릭 시:
1. **커리큘럼 파일**: `beginnerCurriculum.ts`
2. **Module ID**: `basics-2`
3. **chapterId**: `basics`
4. **생성될 링크**: `/modules/semiconductor/basics`
5. **표시될 Chapter**: `Chapter1.tsx` → "Chapter 1: 반도체 기초"

### 만약 Chapter 4로 간다면:
- URL이 `/modules/semiconductor/fabrication`인지 확인
- 브라우저 개발자 도구에서 실제 클릭한 링크 확인
- 캐시 문제일 가능성

---

## 📝 파일 위치

```
src/data/semiconductor/
├── beginnerCurriculum.ts      # 초급 5개 모듈
├── intermediateCurriculum.ts  # 중급 6개 모듈
└── advancedCurriculum.ts      # 고급 6개 모듈

src/app/modules/semiconductor/
├── page.tsx                   # 메인 페이지 (학습 시작 버튼)
└── components/
    ├── ChapterContent.tsx     # chapterId → Chapter 컴포넌트 매핑
    └── chapters/
        ├── Chapter1.tsx       # basics
        ├── Chapter2.tsx       # design
        ├── Chapter3.tsx       # lithography
        ├── Chapter4.tsx       # fabrication
        ├── Chapter5.tsx       # advanced
        ├── Chapter6.tsx       # ai-chips
        ├── Chapter7.tsx       # memory
        ├── Chapter8.tsx       # future
        └── Chapter9.tsx       # image-display
```

---

## ✅ 코드 확인 필요 사항

### 1. page.tsx의 Link 생성 코드 (Line 165):
```tsx
<Link
  href={`/modules/semiconductor/${(module as any).chapterId || 'basics'}`}
  className="..."
>
  학습 시작 →
</Link>
```

### 2. ChapterContent.tsx의 Switch 문 (Line 35-50):
```tsx
switch (chapterId) {
  case 'basics':
    return <Chapter1 />
  case 'design':
    return <Chapter2 />
  case 'lithography':
    return <Chapter3 />
  case 'fabrication':
    return <Chapter4 />
  // ...
}
```

### 3. 확인 방법:
브라우저에서 "2. 도핑과 PN 접합" 우클릭 → "링크 복사" → URL 확인
- 예상: `http://localhost:3000/modules/semiconductor/basics`
- 잘못된 경우: `http://localhost:3000/modules/semiconductor/fabrication`

