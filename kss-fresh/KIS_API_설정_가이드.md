# 한국투자증권(KIS) API 설정 가이드 📈

KIS API를 사용하여 실제 주식 데이터를 받아오는 방법을 단계별로 설명합니다.

## 📋 목차
1. [KIS API 계정 생성](#1-kis-api-계정-생성)
2. [API 키 발급](#2-api-키-발급)
3. [프로젝트 환경변수 설정](#3-프로젝트-환경변수-설정)
4. [API 연결 테스트](#4-api-연결-테스트)
5. [주의사항](#5-주의사항)

---

## 1. KIS API 계정 생성

### Step 1: 한국투자증권 홈페이지 접속
1. [한국투자증권 OpenAPI 포털](https://apiportal.koreainvestment.com) 접속
2. 우측 상단 **"회원가입"** 클릭

### Step 2: 회원가입
1. **일반회원**으로 가입 (무료)
2. 이메일 인증 완료
3. 휴대폰 본인인증 진행

### Step 3: 로그인
- 가입한 계정으로 로그인

---

## 2. API 키 발급

### Step 1: 마이페이지 접속
1. 로그인 후 우측 상단 프로필 클릭
2. **"마이페이지"** 선택

### Step 2: 앱 등록
1. **"나의 앱"** 탭 클릭
2. **"앱 등록"** 버튼 클릭
3. 앱 정보 입력:
   ```
   앱 이름: KSS Trading Chart (또는 원하는 이름)
   앱 설명: 주식 차트 및 시세 조회용 앱
   사용 용도: 개인 사용
   ```
4. **"등록"** 클릭

### Step 3: API 키 확인
앱 등록 완료 후 다음 정보가 표시됩니다:
- **App Key**: (32자리 영문+숫자)
- **App Secret**: (180자리 영문+숫자)

⚠️ **중요**: 이 키들을 안전한 곳에 복사해두세요!

---

## 3. 프로젝트 환경변수 설정

### Step 1: .env.local 파일 생성
프로젝트 루트 디렉토리(`kss-fresh`)에서:

```bash
# 터미널에서 실행
cd /Users/kelly/Desktop/Space/project/Ontology/kss-fresh
touch .env.local
```

### Step 2: 환경변수 추가
`.env.local` 파일을 열고 다음 내용 추가:

```env
# KIS API 설정
NEXT_PUBLIC_KIS_APP_KEY=여기에_발급받은_App_Key_입력
NEXT_PUBLIC_KIS_APP_SECRET=여기에_발급받은_App_Secret_입력

# 예시 (실제 키로 교체하세요)
# NEXT_PUBLIC_KIS_APP_KEY=PSxxx...xxx
# NEXT_PUBLIC_KIS_APP_SECRET=1nxxx...xxx
```

### Step 3: 개발 서버 재시작
환경변수가 적용되도록 서버를 재시작합니다:

```bash
# 서버 중지 (Ctrl + C)
# 다시 시작
npm run dev
```

---

## 4. API 연결 테스트

### Step 1: Pro Trading Chart 접속
브라우저에서 접속:
```
http://localhost:3000/modules/stock-analysis/tools/pro-trading-chart
```

### Step 2: KIS 토큰 상태 확인
우측 사이드바의 **"KIS API 상태"** 패널에서:

1. **토큰 갱신** 버튼 클릭
2. 상태가 **"정상"**으로 변경되는지 확인
3. **"연결됨"** 표시 확인

### Step 3: 실제 데이터 확인
- 차트가 실제 주식 데이터로 업데이트됩니다
- 호가창에 실시간 매수/매도 호가가 표시됩니다

---

## 5. 주의사항

### 📌 API 사용 제한
- **일일 요청 한도**: 100,000건
- **초당 요청 한도**: 20건
- **토큰 유효기간**: 24시간 (자동 갱신됨)

### 🔒 보안 주의사항
1. **절대 GitHub에 키 커밋 금지**
   - `.env.local`은 `.gitignore`에 포함되어 있음
   - 실수로 커밋하면 즉시 키 재발급 필요

2. **프로덕션 환경**
   - Vercel/Netlify 등에서는 환경변수 설정 페이지에서 추가
   - 서버사이드에서만 사용하는 민감한 작업은 API Route 활용

### 🚨 문제 해결

#### 토큰 생성 실패
```
오류: KIS API 인증 정보가 설정되지 않았습니다.
```
**해결**: 
- `.env.local` 파일 확인
- 키가 올바르게 입력되었는지 확인
- 서버 재시작

#### API 연결 실패
```
오류: 401 Unauthorized
```
**해결**:
- App Key/Secret이 정확한지 확인
- KIS 포털에서 앱 상태가 "활성"인지 확인

#### 실시간 데이터 미수신
```
오류: 503 Service Unavailable
```
**해결**:
- 주식시장 운영시간 확인 (평일 09:00-15:30)
- API 서버 상태 확인

---

## 🎯 다음 단계

API 설정이 완료되면:

1. **실시간 호가 연동**: WebSocket으로 실시간 체결가 수신
2. **차트 고도화**: 볼린저밴드, MACD 등 지표 추가
3. **주문 기능**: 모의투자 API 연동 (별도 신청 필요)

---

## 📞 지원

### KIS OpenAPI 고객센터
- 전화: 1544-5000
- 이메일: openapi@koreainvestment.com
- 운영시간: 평일 09:00-18:00

### 개발자 커뮤니티
- [KIS Developers 포럼](https://apiportal.koreainvestment.com/community)
- 다른 개발자들의 질문과 답변 확인 가능

---

## 🔗 유용한 링크

- [API 문서](https://apiportal.koreainvestment.com/apiservice/apiservice-domestic-stock)
- [샘플 코드](https://github.com/koreainvestment/open-trading-api)
- [API 테스트 도구](https://apiportal.koreainvestment.com/apiservice/oauth2#tab_2)

---

**작성일**: 2024-01-19  
**작성자**: KSS Development Team