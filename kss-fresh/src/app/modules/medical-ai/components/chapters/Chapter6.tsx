import React from 'react';
import { Dna, Brain, Activity, Code, TrendingUp, Shield, Database, Zap } from 'lucide-react';
import References from '../References';

export default function Chapter6() {
  return (
    <div className="space-y-8">
      {/* 헤더 */}
      <div>
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
          정밀 의료 (Precision Medicine)
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-300">
          유전체, 라이프스타일 데이터로 최적화된 개인 맞춤 치료
        </p>
      </div>

      {/* 정밀 의료 핵심 요소 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Dna className="w-7 h-7 text-pink-600" />
          정밀 의료의 4가지 핵심 데이터
        </h2>

        <div className="grid md:grid-cols-2 gap-6">
          {/* 유전체 데이터 */}
          <div className="bg-gradient-to-br from-pink-50 to-pink-100 dark:from-pink-900/20 dark:to-pink-800/20 p-6 rounded-lg border-2 border-pink-300">
            <Dna className="w-12 h-12 text-pink-600 mb-4" />
            <h3 className="text-xl font-bold mb-3 text-pink-900 dark:text-pink-300">
              1. 유전체 데이터 (Genomic Data)
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-4">
              개인의 DNA, RNA, 단백질 정보 분석
            </p>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg mb-4">
              <p className="text-sm font-semibold mb-2">주요 오믹스:</p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• WGS (Whole Genome Sequencing) - 전체 유전체</li>
                <li>• WES (Whole Exome Sequencing) - 코딩 영역</li>
                <li>• RNA-seq (유전자 발현 프로파일)</li>
                <li>• Proteomics (단백질 발현)</li>
                <li>• Metabolomics (대사체 분석)</li>
              </ul>
            </div>
            <div className="bg-pink-900/10 dark:bg-pink-900/30 p-3 rounded text-xs">
              <p className="font-semibold text-pink-900 dark:text-pink-300 mb-1">비용 변화:</p>
              <p className="text-gray-700 dark:text-gray-300">
                2003년 $30억 → 2024년 $300 (10만 배 감소, Illumina NovaSeq X)
              </p>
            </div>
          </div>

          {/* 임상 데이터 */}
          <div className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 p-6 rounded-lg border-2 border-blue-300">
            <Database className="w-12 h-12 text-blue-600 mb-4" />
            <h3 className="text-xl font-bold mb-3 text-blue-900 dark:text-blue-300">
              2. 전자의무기록 (EHR)
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-4">
              병력, 검사 결과, 투약 이력 통합
            </p>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg mb-4">
              <p className="text-sm font-semibold mb-2">EHR 데이터:</p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• 진단 코드 (ICD-10, SNOMED CT)</li>
                <li>• 처방 기록 (RxNorm)</li>
                <li>• 검사 결과 (LOINC)</li>
                <li>• 영상 판독 (DICOM)</li>
                <li>• 가족력 및 사회력</li>
              </ul>
            </div>
            <div className="bg-blue-900/10 dark:bg-blue-900/30 p-3 rounded text-xs">
              <p className="font-semibold text-blue-900 dark:text-blue-300 mb-1">규모:</p>
              <p className="text-gray-700 dark:text-gray-300">
                Epic Cosmos: 2.5억 환자 EHR (미국 인구 75%)
              </p>
            </div>
          </div>

          {/* 웨어러블 데이터 */}
          <div className="bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 p-6 rounded-lg border-2 border-green-300">
            <Activity className="w-12 h-12 text-green-600 mb-4" />
            <h3 className="text-xl font-bold mb-3 text-green-900 dark:text-green-300">
              3. 웨어러블 & IoMT 데이터
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-4">
              실시간 생체 신호 및 라이프스타일 추적
            </p>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg mb-4">
              <p className="text-sm font-semibold mb-2">측정 항목:</p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• 심박수, 심박변이도 (Apple Watch, Fitbit)</li>
                <li>• 혈당 (Continuous Glucose Monitoring)</li>
                <li>• 수면 패턴 (Sleep Stages)</li>
                <li>• 운동량 (Steps, Calories)</li>
                <li>• 심전도 (ECG), 혈압</li>
              </ul>
            </div>
            <div className="bg-green-900/10 dark:bg-green-900/30 p-3 rounded text-xs">
              <p className="font-semibold text-green-900 dark:text-green-300 mb-1">혁신:</p>
              <p className="text-gray-700 dark:text-gray-300">
                Apple Heart Study: 40만 참가자, 심방세동 탐지 민감도 97.5%
              </p>
            </div>
          </div>

          {/* 환경 & 행동 데이터 */}
          <div className="bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 p-6 rounded-lg border-2 border-purple-300">
            <Brain className="w-12 h-12 text-purple-600 mb-4" />
            <h3 className="text-xl font-bold mb-3 text-purple-900 dark:text-purple-300">
              4. 환경 & 행동 데이터
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-4">
              외부 요인과 생활 습관 분석
            </p>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg mb-4">
              <p className="text-sm font-semibold mb-2">데이터 소스:</p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• 대기오염 (PM2.5, 오존)</li>
                <li>• 식습관 (영양 섭취 패턴)</li>
                <li>• 흡연, 음주 이력</li>
                <li>• 스트레스 수준 (Cortisol)</li>
                <li>• 사회경제적 지표 (SDOH)</li>
              </ul>
            </div>
            <div className="bg-purple-900/10 dark:bg-purple-900/30 p-3 rounded text-xs">
              <p className="font-semibold text-purple-900 dark:text-purple-300 mb-1">통합 플랫폼:</p>
              <p className="text-gray-700 dark:text-gray-300">
                All of Us (NIH): 유전체 + EHR + 웨어러블 통합 (100만 명)
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 실전 코드 - 유전체 변이 분석 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Code className="w-7 h-7 text-indigo-600" />
          실전 코드: 유전체 변이 해석 및 약물 반응 예측
        </h2>

        <div className="space-y-6">
          {/* VCF 파일 분석 */}
          <div>
            <h3 className="font-bold text-lg mb-3 text-blue-900 dark:text-blue-300">
              1. VCF 파일 파싱 및 병원성 변이 예측 (PyVCF + scikit-learn)
            </h3>
            <div className="bg-slate-900 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm text-gray-100">
                <code>{`import vcf  # PyVCF
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import requests

# VCF 파일 로드 (Whole Genome Sequencing 결과)
vcf_reader = vcf.Reader(filename='patient_001_WGS.vcf.gz')

# 변이 정보 추출
variants = []
for record in vcf_reader:
    variant = {
        'chromosome': record.CHROM,
        'position': record.POS,
        'ref': record.REF,
        'alt': str(record.ALT[0]),
        'quality': record.QUAL,
        'gene': record.INFO.get('GENE', ['Unknown'])[0],
        'variant_type': record.var_type  # SNP, INDEL
    }
    variants.append(variant)

df = pd.DataFrame(variants)
print(f"총 변이 수: {len(df)}")

# ClinVar API로 병원성 확인
def query_clinvar(chrom, pos, ref, alt):
    """ClinVar 데이터베이스에서 변이 임상 의미 조회"""
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        'db': 'clinvar',
        'term': f'{chrom}[chr] AND {pos}[chrpos]',
        'retmode': 'json'
    }
    response = requests.get(url, params=params)
    data = response.json()

    # 간단한 예시 (실제로는 eSummary API 추가 호출 필요)
    if 'esearchresult' in data and 'count' in data['esearchresult']:
        if int(data['esearchresult']['count']) > 0:
            return 'Pathogenic'  # 실제로는 상세 정보 파싱 필요
    return 'Benign'

# 병원성 변이 필터링 (예시: BRCA1/BRCA2 유전자)
cancer_genes = ['BRCA1', 'BRCA2', 'TP53', 'PTEN', 'APC']
df_cancer = df[df['gene'].isin(cancer_genes)]

print(f"\\n암 관련 유전자 변이: {len(df_cancer)}개")
for idx, row in df_cancer.head(5).iterrows():
    print(f"{row['gene']}: {row['chromosome']}:{row['position']} {row['ref']}>{row['alt']}")

# 약물유전체학 (Pharmacogenomics) - CYP2D6 변이 분석
def predict_drug_response(genotype):
    """
    CYP2D6 유전자형 기반 약물 대사 능력 예측
    *1/*1: Normal Metabolizer
    *1/*4: Intermediate Metabolizer
    *4/*4: Poor Metabolizer
    """
    metabolizer_map = {
        '*1/*1': 'Normal Metabolizer',
        '*1/*4': 'Intermediate Metabolizer',
        '*4/*4': 'Poor Metabolizer',
        '*1/*2': 'Ultrarapid Metabolizer'
    }

    metabolizer = metabolizer_map.get(genotype, 'Unknown')

    # 약물 용량 추천
    if metabolizer == 'Poor Metabolizer':
        recommendation = "Codeine 효과 없음 → Morphine 직접 투여 권장"
    elif metabolizer == 'Ultrarapid Metabolizer':
        recommendation = "Codeine 과다 대사 위험 → 용량 50% 감소"
    else:
        recommendation = "표준 용량 사용 가능"

    return {
        'genotype': genotype,
        'metabolizer_status': metabolizer,
        'drug_recommendation': recommendation
    }

# 사용 예시
patient_cyp2d6 = '*1/*4'
result = predict_drug_response(patient_cyp2d6)

print(f"\\n💊 약물 유전체 분석:")
print(f"CYP2D6 유전자형: {result['genotype']}")
print(f"대사 능력: {result['metabolizer_status']}")
print(f"권장사항: {result['drug_recommendation']}")`}</code>
              </pre>
            </div>
          </div>

          {/* 다중 오믹스 통합 분석 */}
          <div>
            <h3 className="font-bold text-lg mb-3 text-green-900 dark:text-green-300">
              2. Multi-Omics 통합 분석 (암 환자 예후 예측)
            </h3>
            <div className="bg-slate-900 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm text-gray-100">
                <code>{`import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# TCGA (The Cancer Genome Atlas) 데이터 로드 예시
# 실제로는 GDC Data Portal에서 다운로드
data = {
    # Genomics (Mutation Load)
    'TMB': [12.5, 8.3, 15.7, 6.2, 20.1],  # Tumor Mutation Burden
    'MSI_status': [0, 1, 0, 0, 1],  # Microsatellite Instability

    # Transcriptomics (Gene Expression)
    'PD-L1_expression': [2.3, 5.7, 1.2, 0.8, 6.1],
    'CD8_T_cell_score': [3.2, 7.5, 2.1, 1.5, 8.3],

    # Proteomics
    'HER2_protein': [0.2, 0.5, 3.2, 0.3, 0.1],
    'EGFR_protein': [1.5, 2.3, 0.8, 4.5, 1.2],

    # Clinical Data
    'age': [65, 58, 72, 61, 69],
    'stage': [2, 3, 1, 4, 3],

    # Outcome (1: 5년 생존, 0: 사망)
    'survival_5y': [1, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

# 특징 선택
features = ['TMB', 'MSI_status', 'PD-L1_expression', 'CD8_T_cell_score',
            'HER2_protein', 'EGFR_protein', 'age', 'stage']
X = df[features]
y = df['survival_5y']

# 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 모델 학습 (실제로는 더 큰 데이터셋 사용)
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# 새 환자 예후 예측
new_patient = {
    'TMB': 18.5,
    'MSI_status': 1,
    'PD-L1_expression': 7.2,
    'CD8_T_cell_score': 6.5,
    'HER2_protein': 0.4,
    'EGFR_protein': 1.8,
    'age': 63,
    'stage': 2
}

new_patient_df = pd.DataFrame([new_patient])
new_patient_scaled = scaler.transform(new_patient_df)

survival_prob = model.predict_proba(new_patient_scaled)[0, 1]

print(f"\\n📊 다중 오믹스 기반 예후 예측:")
print(f"5년 생존 확률: {survival_prob:.1%}")

# 치료 전략 추천
if new_patient['PD-L1_expression'] > 5 and new_patient['TMB'] > 10:
    print("\\n💉 추천 치료: 면역항암제 (Pembrolizumab)")
    print("근거: 높은 PD-L1 발현 + 높은 TMB (면역치료 반응률 65%+)")
elif new_patient['HER2_protein'] > 2:
    print("\\n💉 추천 치료: HER2 표적치료 (Trastuzumab)")
    print("근거: HER2 과발현 (표적치료 반응률 80%+)")
else:
    print("\\n💉 추천 치료: 표준 화학요법")

# 특성 중요도
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\\n중요 바이오마커 순위:")
print(feature_importance)`}</code>
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* 2024-2025 최신 동향 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <TrendingUp className="w-7 h-7 text-orange-600" />
          2024-2025 정밀 의료 혁신 동향
        </h2>

        <div className="space-y-4">
          <div className="border-l-4 border-blue-500 bg-blue-50 dark:bg-blue-900/20 p-4 rounded-r-lg">
            <h3 className="font-bold text-lg mb-2 text-blue-900 dark:text-blue-300">
              1. Polygenic Risk Scores (PRS) 임상 적용
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              수백만 유전 변이 결합으로 질병 위험도 예측 (2024 FDA 승인 검토 중)
            </p>
            <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
              <li>• <strong>관상동맥질환 PRS:</strong> 유전적 고위험군 식별, 예방적 스타틴 투여</li>
              <li>• <strong>유방암 PRS:</strong> BRCA 음성 환자도 고위험 판별 (AUC 0.68)</li>
              <li>• <strong>당뇨병 PRS:</strong> 조기 개입으로 발병 5년 지연 (UK Biobank 연구)</li>
            </ul>
          </div>

          <div className="border-l-4 border-green-500 bg-green-50 dark:bg-green-900/20 p-4 rounded-r-lg">
            <h3 className="font-bold text-lg mb-2 text-green-900 dark:text-green-300">
              2. Single-Cell Multi-Omics
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              단일 세포 수준에서 유전체, 전사체, 후성유전체 동시 분석
            </p>
            <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
              <li>• <strong>10x Genomics Multiome:</strong> 단일 세포 RNA-seq + ATAC-seq</li>
              <li>• <strong>암 이질성 분석:</strong> 종양 내 세포 아형 정밀 분류, 약물 저항성 예측</li>
              <li>• <strong>Human Cell Atlas:</strong> 370억+ 세포 데이터, 질병 세포 지도</li>
            </ul>
          </div>

          <div className="border-l-4 border-purple-500 bg-purple-50 dark:bg-purple-900/20 p-4 rounded-r-lg">
            <h3 className="font-bold text-lg mb-2 text-purple-900 dark:text-purple-300">
              3. Digital Twins for Precision Medicine
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              환자의 디지털 복제본으로 치료 효과 사전 시뮬레이션
            </p>
            <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
              <li>• <strong>Dassault Systèmes Living Heart:</strong> 심장 3D 모델, 수술 시뮬레이션</li>
              <li>• <strong>Siemens Healthineers:</strong> 환자별 방사선 치료 최적화</li>
              <li>• <strong>Aitia (스탠포드):</strong> 질병 진행 예측 디지털 트윈 (Nature 2024)</li>
            </ul>
          </div>

          <div className="border-l-4 border-pink-500 bg-pink-50 dark:bg-pink-900/20 p-4 rounded-r-lg">
            <h3 className="font-bold text-lg mb-2 text-pink-900 dark:text-pink-300">
              4. AI-Powered Drug Matching
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              환자 오믹스 프로파일 기반 최적 약물 조합 추천
            </p>
            <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
              <li>• <strong>Tempus AI:</strong> 300만 암 환자 데이터 → 개인화 치료 추천 (FDA Breakthrough)</li>
              <li>• <strong>Foundation Medicine:</strong> NGS 기반 companion diagnostics (70+ 암 종류)</li>
              <li>• <strong>Guardant360:</strong> 혈액 기반 암 유전자 검사, 표적치료 매칭</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 정밀 의료 통계 */}
      <section className="bg-gradient-to-r from-pink-600 to-purple-600 rounded-xl p-6 shadow-lg text-white">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <Shield className="w-7 h-7" />
          정밀 의료 시장 & 임상 성과 (2024)
        </h2>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="text-4xl font-bold mb-2">$217B</p>
            <p className="text-sm opacity-90">2028 정밀 의료 시장 규모</p>
            <p className="text-xs mt-2 opacity-75">출처: Allied Market Research</p>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="text-4xl font-bold mb-2">$300</p>
            <p className="text-sm opacity-90">2024 전장유전체 분석 비용</p>
            <p className="text-xs mt-2 opacity-75">출처: Illumina NovaSeq X</p>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="text-4xl font-bold mb-2">65%</p>
            <p className="text-sm opacity-90">PD-L1 고발현 환자 면역치료 반응률</p>
            <p className="text-xs mt-2 opacity-75">출처: KEYNOTE-024 Trial</p>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="text-4xl font-bold mb-2">100만</p>
            <p className="text-sm opacity-90">NIH All of Us 참가자 수 (2024)</p>
            <p className="text-xs mt-2 opacity-75">출처: NIH All of Us Research</p>
          </div>
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 핵심 데이터베이스 & 바이오뱅크',
            icon: 'docs' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'All of Us Research Program (NIH)',
                url: 'https://allofus.nih.gov/',
                description: '100만 명 유전체 + EHR + 웨어러블 데이터 통합'
              },
              {
                title: 'UK Biobank',
                url: 'https://www.ukbiobank.ac.uk/',
                description: '50만 명 WGS + 영상 + 임상 데이터'
              },
              {
                title: 'TCGA (The Cancer Genome Atlas)',
                url: 'https://www.cancer.gov/tcga',
                description: '33개 암 종류, 2.5 페타바이트 오믹스 데이터'
              },
              {
                title: 'ClinVar (NCBI)',
                url: 'https://www.ncbi.nlm.nih.gov/clinvar/',
                description: '230만+ 유전 변이 임상 의미 데이터베이스'
              },
            ]
          },
          {
            title: '🔬 최신 연구 논문 (2023-2024)',
            icon: 'research' as const,
            color: 'border-pink-500',
            items: [
              {
                title: 'Polygenic Risk Scores for CVD (Nature 2024)',
                url: 'https://www.nature.com/articles/s41586-024-07156-0',
                description: 'PRS 기반 관상동맥질환 조기 예측, AUC 0.81'
              },
              {
                title: 'Single-Cell Multi-Omics (Cell 2024)',
                url: 'https://www.cell.com/cell/fulltext/S0092-8674(24)00123-4',
                description: '10x Genomics: 단일 세포 RNA+ATAC-seq 동시 분석'
              },
              {
                title: 'Digital Twin for Precision Medicine (Nature Medicine 2024)',
                url: 'https://www.nature.com/articles/s41591-024-02867-w',
                description: 'Aitia: AI 기반 환자 디지털 트윈, 치료 반응 예측'
              },
              {
                title: 'Foundation Medicine Comprehensive Genomic Profiling (JCO 2024)',
                url: 'https://ascopubs.org/doi/full/10.1200/JCO.23.01234',
                description: 'NGS 기반 companion diagnostics, 표적치료 매칭'
              },
            ]
          },
          {
            title: '🛠️ 실전 도구 & 플랫폼',
            icon: 'tools' as const,
            color: 'border-green-500',
            items: [
              {
                title: 'PyVCF',
                url: 'https://pyvcf.readthedocs.io/',
                description: 'VCF 파일 파싱 파이썬 라이브러리'
              },
              {
                title: 'ANNOVAR',
                url: 'https://annovar.openbioinformatics.org/',
                description: '유전 변이 주석 도구 (병원성 예측, 유전자 매핑)'
              },
              {
                title: 'VarSome',
                url: 'https://varsome.com/',
                description: '변이 해석 통합 플랫폼 (ClinVar, gnomAD 통합)'
              },
              {
                title: 'Scanpy',
                url: 'https://scanpy.readthedocs.io/',
                description: '단일 세포 RNA-seq 분석 파이썬 라이브러리'
              },
              {
                title: 'PharmGKB',
                url: 'https://www.pharmgkb.org/',
                description: '약물유전체학 데이터베이스 (유전자-약물 상호작용)'
              },
            ]
          },
          {
            title: '📖 임상 검사 & 규제',
            icon: 'docs' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'Foundation Medicine FoundationOne CDx',
                url: 'https://www.foundationmedicine.com/test/foundationone-cdx',
                description: 'FDA 승인 종합 유전자 검사 (324개 유전자, 모든 고형암)'
              },
              {
                title: 'Tempus xT',
                url: 'https://www.tempus.com/oncology/xt/',
                description: 'NGS 기반 암 유전자 검사 + AI 치료 추천'
              },
              {
                title: 'FDA Guidance on Next-Generation Sequencing',
                url: 'https://www.fda.gov/regulatory-information/search-fda-guidance-documents/considerations-design-pivotal-clinical-study-and-associated-in-vitro-companion-diagnostic-test',
                description: 'NGS 기반 companion diagnostics 승인 가이드라인'
              },
            ]
          },
        ]}
      />

      {/* 요약 */}
      <section className="bg-gradient-to-br from-gray-100 to-gray-200 dark:from-gray-700 dark:to-gray-800 rounded-xl p-6">
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          🎯 핵심 요약
        </h2>
        <ul className="space-y-2 text-gray-700 dark:text-gray-300">
          <li className="flex items-start gap-2">
            <span className="text-pink-600 font-bold">•</span>
            <span>정밀 의료 4대 데이터: <strong>유전체, EHR, 웨어러블, 환경/행동</strong></span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-pink-600 font-bold">•</span>
            <span>핵심 기술: <strong>PRS (질병 위험 예측), Single-Cell Omics, Digital Twins</strong></span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-pink-600 font-bold">•</span>
            <span><strong>2024 혁신</strong>: WGS $300 (10만배 감소), PRS 임상 적용, AI 약물 매칭</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-pink-600 font-bold">•</span>
            <span>PD-L1 고발현 환자 면역치료 반응률 <strong>65%+</strong> (맞춤형 치료 효과)</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-pink-600 font-bold">•</span>
            <span>필수 도구: <strong>PyVCF, ANNOVAR, Scanpy, PharmGKB</strong></span>
          </li>
        </ul>
      </section>
    </div>
  );
}
