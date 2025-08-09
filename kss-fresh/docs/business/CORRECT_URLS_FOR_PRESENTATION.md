# 🎯 프레젠테이션용 정확한 URL 목록

## ✅ 포트 번호: 3000

---

## 🏠 메인 페이지
- **홈페이지**: http://localhost:3000

---

## 📘 온톨로지 모듈
- **모듈 메인**: http://localhost:3000/modules/ontology
- **3D Knowledge Graph**: http://localhost:3000/3d-graph
- **RDF Editor**: http://localhost:3000/rdf-editor
- **SPARQL Playground**: http://localhost:3000/sparql-playground

---

## 💎 주식투자 모듈
- **모듈 메인**: http://localhost:3000/modules/stock-analysis
- **시뮬레이터 목록**: http://localhost:3000/modules/stock-analysis/simulators
- **Chart Analyzer**: http://localhost:3000/modules/stock-analysis/simulators/chart-analyzer
- **Real-time Dashboard**: http://localhost:3000/modules/stock-analysis/simulators/real-time-dashboard
- **Portfolio Optimizer**: http://localhost:3000/modules/stock-analysis/simulators/portfolio-optimizer
- **AI Mentor**: http://localhost:3000/modules/stock-analysis/simulators/ai-mentor

---

## 🏭 스마트팩토리 모듈
- **모듈 메인**: http://localhost:3000/modules/smart-factory

### ✅ 구현된 시뮬레이터 (4개)
1. **Digital Twin Factory** (핵심)
   - http://localhost:3000/modules/smart-factory/simulators/digital-twin-factory

2. **Predictive Maintenance Lab** (핵심)
   - http://localhost:3000/modules/smart-factory/simulators/predictive-maintenance-lab
   - ⚠️ 주의: "predictive-analytics-lab"가 아니라 "predictive-maintenance-lab"입니다!

3. **Production Line Monitor**
   - http://localhost:3000/modules/smart-factory/simulators/production-line-monitor

4. **Quality Control Vision**
   - http://localhost:3000/modules/smart-factory/simulators/quality-control-vision

---

## 🚀 프레젠테이션 추천 경로

### 시나리오 1: 핵심만 빠르게 (각 모듈 2개씩)
1. 홈페이지 → 온톨로지 → 3D Graph → SPARQL
2. 홈페이지 → 주식투자 → Chart Analyzer → Real-time Dashboard  
3. 홈페이지 → 스마트팩토리 → Digital Twin → Predictive Maintenance

### 시나리오 2: 전체 둘러보기
1. 각 모듈 메인 페이지 먼저 보여주기
2. 각 모듈별 시뮬레이터 2-3개씩 시연
3. 마지막에 홈페이지로 돌아와서 전체 구조 재확인

---

## 📌 자주 하는 실수
- ✅ 포트 3000 사용
- ❌ predictive-analytics-lab → ✅ predictive-maintenance-lab
- ❌ /modules/ontology/simulators/3d-graph → ✅ /3d-graph (루트 경로)

---

## 🛠️ 문제 해결
만약 페이지가 안 열린다면:
1. 개발 서버 확인: `npm run dev`
2. 포트 확인: 3000번인지 확인
3. URL 오타 확인: 위 목록과 정확히 일치하는지 확인