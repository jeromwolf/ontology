# Database Schema Strategy for KSS Platform

## 테이블 명명 규칙

### 1. 공통 테이블 (프리픽스 없음)
- User, Profile, Session
- Notification, ContentUpdate
- Progress, Enrollment

### 2. 모듈별 테이블 (모듈_엔티티 형식)

#### Stock Analysis Module
- Stock_Symbol      // 종목 마스터
- Stock_Quote       // 시세
- Stock_Financial   // 재무제표
- Stock_Portfolio   // 포트폴리오
- Stock_Transaction // 거래내역

#### AI/ML Module  
- AI_Model          // 모델 정보
- AI_Dataset        // 데이터셋
- AI_Training       // 학습 이력
- AI_Prediction     // 예측 결과

#### Ontology Module
- Onto_Entity       // 엔티티
- Onto_Relation     // 관계
- Onto_Triple       // 트리플
- Onto_Schema       // 스키마

#### Bioinformatics Module
- Bio_Sequence      // 시퀀스
- Bio_Gene          // 유전자
- Bio_Protein       // 단백질
- Bio_Analysis      // 분석결과

#### Smart Factory Module
- Factory_Sensor    // 센서 데이터
- Factory_Machine   // 기계 정보
- Factory_Production // 생산 데이터
- Factory_Quality   // 품질 데이터

## 관계 설계 원칙

### 1. 모듈 내부 관계
```prisma
// 강한 결합 - Foreign Key 사용
model Stock_Quote {
  stockId String
  stock   Stock_Symbol @relation(...)
}
```

### 2. 모듈 간 관계
```prisma
// 약한 결합 - ID 참조만
model AI_Prediction {
  targetType String // "stock", "bio", etc
  targetId   String // 해당 모듈의 ID
  
  @@index([targetType, targetId])
}
```

### 3. 공통 테이블과의 관계
```prisma
// 모든 모듈이 User와 연결
model Stock_Portfolio {
  userId String
  user   User @relation(...)
}
```

## 성능 최적화 전략

### 1. 인덱스 설계
- 각 모듈별 주요 쿼리 패턴에 맞춤
- 복합 인덱스 적극 활용

### 2. 파티셔닝 (대용량 데이터)
- Stock_Quote: 날짜별 파티션
- Factory_Sensor: 시간별 파티션

### 3. 캐싱 전략
- Redis로 실시간 데이터 캐싱
- 정적 데이터는 메모리 캐싱

## 마이그레이션 전략

### Phase 1: 핵심 모듈 (현재)
- Stock Analysis
- User/Auth

### Phase 2: AI/ML 모듈
- AI 관련 테이블 추가

### Phase 3: 도메인별 확장
- 각 모듈별 순차 추가