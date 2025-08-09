# KSS 플랫폼 인증 시스템 설계 문서

## 📋 개요

KSS(Knowledge Space Simulator) 플랫폼의 사용자 인증 및 권한 관리 시스템 설계 문서입니다. 
본 시스템은 NextAuth.js를 기반으로 구축되며, 역할 기반 접근 제어(RBAC)를 구현합니다.

## 🎯 목표

1. **보안성**: 안전한 사용자 인증 및 세션 관리
2. **확장성**: 다양한 인증 방식 지원 (OAuth, Email, Credentials)
3. **유연성**: 역할 기반 권한 관리로 세밀한 접근 제어
4. **사용자 경험**: 간편한 로그인 및 학습 진도 추적
5. **성능**: 효율적인 세션 관리 및 캐싱

## 🏗️ 아키텍처

### 기술 스택

```
Frontend:
├── Next.js 14 (App Router)
├── NextAuth.js v5
├── React Hook Form
└── Zod (Validation)

Backend:
├── Next.js API Routes
├── Prisma ORM
├── PostgreSQL
└── Redis (Session Store)

Security:
├── JWT Tokens
├── bcrypt (Password Hashing)
├── CSRF Protection
└── Rate Limiting
```

## 👥 사용자 역할 (User Roles)

### 1. Guest (비회원)
- 홈페이지 접근
- 무료 콘텐츠 미리보기
- 시뮬레이터 데모 체험 (제한적)
- 회원가입 유도 페이지

### 2. Student (학생)
- 모든 학습 콘텐츠 접근
- 시뮬레이터 전체 기능 사용
- 학습 진도 저장 및 추적
- 퀴즈 및 과제 제출
- 토론 게시판 참여
- 개인 대시보드

### 3. Premium Student (프리미엄 학생)
- Student 권한 + 추가 혜택
- AI 멘토 무제한 사용
- 고급 시뮬레이터 접근
- 1:1 튜터링 예약
- 수료증 발급
- 소스코드 다운로드

### 4. Instructor (강사)
- 코스 콘텐츠 생성/편집
- 학생 진도 모니터링
- 과제 평가 및 피드백
- 라이브 세션 진행
- 통계 대시보드

### 5. Admin (관리자)
- 전체 시스템 관리
- 사용자 관리 (생성/수정/삭제)
- 콘텐츠 승인/거부
- 시스템 설정 변경
- 분석 대시보드
- 결제 관리

## 🔐 접근 제어 매트릭스

| 리소스 | Guest | Student | Premium | Instructor | Admin |
|--------|-------|---------|---------|------------|-------|
| 홈페이지 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 학습 콘텐츠 | 미리보기 | ✅ | ✅ | ✅ | ✅ |
| 기본 시뮬레이터 | 데모 | ✅ | ✅ | ✅ | ✅ |
| 고급 시뮬레이터 | ❌ | ❌ | ✅ | ✅ | ✅ |
| AI 멘토 | ❌ | 일일 5회 | 무제한 | ✅ | ✅ |
| 학습 진도 저장 | ❌ | ✅ | ✅ | ✅ | ✅ |
| 퀴즈/과제 | ❌ | ✅ | ✅ | ✅ | ✅ |
| 토론 게시판 | 읽기 | ✅ | ✅ | ✅ | ✅ |
| 콘텐츠 생성 | ❌ | ❌ | ❌ | ✅ | ✅ |
| 콘텐츠 편집 | ❌ | ❌ | ❌ | 본인 것만 | ✅ |
| 사용자 관리 | ❌ | ❌ | ❌ | ❌ | ✅ |
| 시스템 설정 | ❌ | ❌ | ❌ | ❌ | ✅ |
| 분석 대시보드 | ❌ | 개인 | 개인 | 코스별 | 전체 |

## 💾 데이터베이스 스키마

### User 테이블
```prisma
model User {
  id            String    @id @default(cuid())
  email         String    @unique
  password      String?   // OAuth users may not have password
  name          String?
  image         String?
  role          Role      @default(STUDENT)
  emailVerified DateTime?
  createdAt     DateTime  @default(now())
  updatedAt     DateTime  @updatedAt
  
  // Relations
  accounts      Account[]
  sessions      Session[]
  profile       Profile?
  enrollments   Enrollment[]
  progress      Progress[]
  submissions   Submission[]
  
  @@index([email])
  @@index([role])
}

enum Role {
  GUEST
  STUDENT
  PREMIUM_STUDENT
  INSTRUCTOR
  ADMIN
}
```

### Profile 테이블
```prisma
model Profile {
  id              String   @id @default(cuid())
  userId          String   @unique
  bio             String?
  phone           String?
  organization    String?
  learningGoals   String?
  preferredLang   String   @default("ko")
  timezone        String   @default("Asia/Seoul")
  notifications   Boolean  @default(true)
  
  user            User     @relation(fields: [userId], references: [id])
  
  @@index([userId])
}
```

### Session 테이블
```prisma
model Session {
  id           String   @id @default(cuid())
  sessionToken String   @unique
  userId       String
  expires      DateTime
  
  user         User     @relation(fields: [userId], references: [id], onDelete: Cascade)
  
  @@index([userId])
  @@index([sessionToken])
}
```

### Account 테이블 (OAuth)
```prisma
model Account {
  id                String  @id @default(cuid())
  userId            String
  type              String
  provider          String
  providerAccountId String
  refresh_token     String? @db.Text
  access_token      String? @db.Text
  expires_at        Int?
  token_type        String?
  scope             String?
  id_token          String? @db.Text
  session_state     String?
  
  user              User    @relation(fields: [userId], references: [id], onDelete: Cascade)
  
  @@unique([provider, providerAccountId])
  @@index([userId])
}
```

### Progress 테이블 (학습 진도)
```prisma
model Progress {
  id          String   @id @default(cuid())
  userId      String
  moduleId    String
  chapterId   String
  completed   Boolean  @default(false)
  progress    Int      @default(0) // 0-100
  timeSpent   Int      @default(0) // seconds
  lastAccess  DateTime @default(now())
  createdAt   DateTime @default(now())
  updatedAt   DateTime @updatedAt
  
  user        User     @relation(fields: [userId], references: [id])
  
  @@unique([userId, moduleId, chapterId])
  @@index([userId])
  @@index([moduleId])
}
```

## 🔑 인증 플로우

### 1. 회원가입 플로우
```mermaid
graph LR
    A[회원가입 페이지] --> B{인증 방식 선택}
    B --> C[이메일/비밀번호]
    B --> D[소셜 로그인]
    C --> E[이메일 검증]
    D --> F[OAuth 인증]
    E --> G[프로필 생성]
    F --> G
    G --> H[대시보드 리다이렉트]
```

### 2. 로그인 플로우
```mermaid
graph LR
    A[로그인 페이지] --> B{인증 방식}
    B --> C[Credentials]
    B --> D[OAuth]
    C --> E[비밀번호 검증]
    D --> F[Provider 인증]
    E --> G[JWT 생성]
    F --> G
    G --> H[세션 생성]
    H --> I[대시보드]
```

## 🛠️ 구현 계획

### Phase 1: 기본 인증 (1주)
- [ ] NextAuth.js 설정
- [ ] PostgreSQL + Prisma 설정
- [ ] User, Session, Account 모델 생성
- [ ] 이메일/비밀번호 로그인
- [ ] 회원가입 페이지
- [ ] 로그인/로그아웃 기능

### Phase 2: OAuth 통합 (1주)
- [ ] Google OAuth
- [ ] GitHub OAuth
- [ ] Kakao OAuth
- [ ] Naver OAuth
- [ ] OAuth 계정 연동

### Phase 3: 역할 기반 접근 제어 (1주)
- [ ] 역할 정의 (Guest, Student, Premium, Instructor, Admin)
- [ ] 미들웨어 구현
- [ ] 권한 체크 HOC/Hook
- [ ] 보호된 라우트 설정
- [ ] 권한별 UI 조건부 렌더링

### Phase 4: 사용자 프로필 (3일)
- [ ] 프로필 페이지
- [ ] 프로필 편집
- [ ] 아바타 업로드
- [ ] 비밀번호 변경
- [ ] 계정 설정

### Phase 5: 학습 진도 추적 (1주)
- [ ] Progress 모델 구현
- [ ] 진도 저장 API
- [ ] 진도 대시보드
- [ ] 학습 통계
- [ ] 수료증 생성

### Phase 6: 보안 강화 (3일)
- [ ] Rate Limiting
- [ ] CSRF 보호
- [ ] 2FA (Two-Factor Authentication)
- [ ] 세션 타임아웃
- [ ] 보안 로그

### Phase 7: 관리자 기능 (1주)
- [ ] 관리자 대시보드
- [ ] 사용자 관리 (CRUD)
- [ ] 역할 변경
- [ ] 활동 로그
- [ ] 시스템 설정

## 📁 파일 구조

```
src/
├── app/
│   ├── api/
│   │   └── auth/
│   │       └── [...nextauth]/
│   │           └── route.ts
│   ├── auth/
│   │   ├── login/
│   │   │   └── page.tsx
│   │   ├── register/
│   │   │   └── page.tsx
│   │   ├── forgot-password/
│   │   │   └── page.tsx
│   │   └── verify-email/
│   │       └── page.tsx
│   ├── dashboard/
│   │   ├── page.tsx
│   │   ├── profile/
│   │   └── settings/
│   └── admin/
│       ├── users/
│       ├── content/
│       └── analytics/
├── lib/
│   ├── auth.ts
│   ├── prisma.ts
│   └── auth-options.ts
├── middleware.ts
├── components/
│   ├── auth/
│   │   ├── LoginForm.tsx
│   │   ├── RegisterForm.tsx
│   │   ├── SocialLogins.tsx
│   │   └── AuthGuard.tsx
│   └── dashboard/
│       ├── UserProfile.tsx
│       └── ProgressTracker.tsx
└── types/
    └── auth.ts
```

## 🔧 환경 변수

```env
# Database
DATABASE_URL="postgresql://user:password@localhost:5432/kss"

# NextAuth
NEXTAUTH_URL="http://localhost:3000"
NEXTAUTH_SECRET="your-secret-key"

# OAuth Providers
GOOGLE_CLIENT_ID=""
GOOGLE_CLIENT_SECRET=""
GITHUB_CLIENT_ID=""
GITHUB_CLIENT_SECRET=""
KAKAO_CLIENT_ID=""
KAKAO_CLIENT_SECRET=""
NAVER_CLIENT_ID=""
NAVER_CLIENT_SECRET=""

# Redis (optional)
REDIS_URL="redis://localhost:6379"

# Email Service
EMAIL_SERVER_HOST=""
EMAIL_SERVER_PORT=""
EMAIL_SERVER_USER=""
EMAIL_SERVER_PASSWORD=""
EMAIL_FROM=""
```

## 🚀 실행 명령어

```bash
# Prisma 설정
npx prisma init
npx prisma migrate dev --name init
npx prisma generate

# 개발 서버 실행
npm run dev

# Prisma Studio (DB 관리 GUI)
npx prisma studio
```

## 📊 모니터링 및 분석

### 추적할 메트릭
- 일일 활성 사용자 (DAU)
- 월간 활성 사용자 (MAU)
- 회원가입 전환율
- 로그인 성공/실패율
- 세션 지속 시간
- 역할별 사용자 분포

### 보안 모니터링
- 로그인 시도 실패
- 비정상적인 접근 패턴
- 권한 에스컬레이션 시도
- API Rate Limit 초과

## 📝 주의사항

1. **비밀번호 정책**
   - 최소 8자 이상
   - 대소문자, 숫자, 특수문자 포함
   - 이전 비밀번호 재사용 금지

2. **세션 관리**
   - 기본 세션 시간: 30일
   - 민감한 작업 시 재인증 요구
   - 다중 디바이스 세션 관리

3. **개인정보 보호**
   - GDPR/KISA 가이드라인 준수
   - 개인정보 암호화 저장
   - 사용자 데이터 삭제 요청 처리

## 🔗 참고 자료

- [NextAuth.js Documentation](https://next-auth.js.org/)
- [Prisma Documentation](https://www.prisma.io/docs)
- [OAuth 2.0 Specification](https://oauth.net/2/)
- [OWASP Authentication Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)

## 📅 타임라인

- **Week 1-2**: 기본 인증 시스템 구현
- **Week 3**: OAuth 통합 및 역할 기반 접근 제어
- **Week 4**: 프로필 및 학습 진도 시스템
- **Week 5**: 보안 강화 및 관리자 기능
- **Week 6**: 테스트 및 배포

---

작성일: 2025-08-04
작성자: KSS Development Team
버전: 1.0.0