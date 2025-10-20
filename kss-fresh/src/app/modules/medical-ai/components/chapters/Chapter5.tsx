import React from 'react';
import { FileText, Brain, Search, Code, TrendingUp, Shield, Database, Zap } from 'lucide-react';
import References from '@/components/common/References';

export default function Chapter5() {
  return (
    <div className="space-y-8">
      {/* 헤더 */}
      <div>
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
          임상 NLP (Clinical NLP)
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-300">
          전자의무기록과 의학 논문에서 지식을 추출하는 자연어 처리 기술
        </p>
      </div>

      {/* 임상 NLP 핵심 작업 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <FileText className="w-7 h-7 text-blue-600" />
          임상 NLP 4대 핵심 작업
        </h2>

        <div className="grid md:grid-cols-2 gap-6">
          {/* 개체명 인식 */}
          <div className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 p-6 rounded-lg border-2 border-blue-300">
            <Search className="w-12 h-12 text-blue-600 mb-4" />
            <h3 className="text-xl font-bold mb-3 text-blue-900 dark:text-blue-300">
              1. 의료 개체명 인식 (Medical NER)
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-4">
              질병, 약물, 증상, 검사명 자동 추출 및 분류
            </p>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg mb-4">
              <p className="text-sm font-semibold mb-2">주요 엔티티 타입:</p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• Disease (질병): Diabetes, COVID-19</li>
                <li>• Drug (약물): Aspirin, Metformin</li>
                <li>• Symptom (증상): Fever, Cough</li>
                <li>• Test (검사): MRI, Blood Test</li>
                <li>• Procedure (시술): Surgery, Biopsy</li>
              </ul>
            </div>
            <div className="bg-blue-900/10 dark:bg-blue-900/30 p-3 rounded text-xs">
              <p className="font-semibold text-blue-900 dark:text-blue-300 mb-1">대표 모델:</p>
              <p className="text-gray-700 dark:text-gray-300">
                BioBERT, ClinicalBERT, PubMedBERT (F1 90%+)
              </p>
            </div>
          </div>

          {/* 관계 추출 */}
          <div className="bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 p-6 rounded-lg border-2 border-green-300">
            <Database className="w-12 h-12 text-green-600 mb-4" />
            <h3 className="text-xl font-bold mb-3 text-green-900 dark:text-green-300">
              2. 관계 추출 (Relation Extraction)
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-4">
              엔티티 간 의학적 관계 파악 (약물-질병, 증상-질병 등)
            </p>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg mb-4">
              <p className="text-sm font-semibold mb-2">관계 유형:</p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• Drug-Disease: Aspirin TREATS Headache</li>
                <li>• Symptom-Disease: Fever INDICATES Infection</li>
                <li>• Drug-Drug: Warfarin INTERACTS WITH Aspirin</li>
                <li>• Test-Disease: MRI DIAGNOSES Brain Tumor</li>
              </ul>
            </div>
            <div className="bg-green-900/10 dark:bg-green-900/30 p-3 rounded text-xs">
              <p className="font-semibold text-green-900 dark:text-green-300 mb-1">활용:</p>
              <p className="text-gray-700 dark:text-gray-300">
                지식 그래프 구축, 약물 부작용 탐지, 임상 의사결정 지원
              </p>
            </div>
          </div>

          {/* 임상 노트 분류 */}
          <div className="bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 p-6 rounded-lg border-2 border-purple-300">
            <Brain className="w-12 h-12 text-purple-600 mb-4" />
            <h3 className="text-xl font-bold mb-3 text-purple-900 dark:text-purple-300">
              3. 임상 노트 분류 (Document Classification)
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-4">
              퇴원 요약, 진료 기록 자동 카테고리화 및 우선순위 결정
            </p>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg mb-4">
              <p className="text-sm font-semibold mb-2">분류 작업:</p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• ICD-10 자동 코딩 (질병 분류)</li>
                <li>• 긴급도 분류 (Urgent / Routine)</li>
                <li>• 감정 분석 (환자 만족도)</li>
                <li>• 임상시험 적격성 판단</li>
              </ul>
            </div>
            <div className="bg-purple-900/10 dark:bg-purple-900/30 p-3 rounded text-xs">
              <p className="font-semibold text-purple-900 dark:text-purple-300 mb-1">성과:</p>
              <p className="text-gray-700 dark:text-gray-300">
                ICD-10 자동 코딩 정확도 95%+ (Epic, Cerner 시스템)
              </p>
            </div>
          </div>

          {/* 의학 논문 마이닝 */}
          <div className="bg-gradient-to-br from-pink-50 to-pink-100 dark:from-pink-900/20 dark:to-pink-800/20 p-6 rounded-lg border-2 border-pink-300">
            <Zap className="w-12 h-12 text-pink-600 mb-4" />
            <h3 className="text-xl font-bold mb-3 text-pink-900 dark:text-pink-300">
              4. 의학 문헌 마이닝 (Literature Mining)
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-4">
              PubMed 3,500만+ 논문에서 최신 연구 동향 자동 추출
            </p>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg mb-4">
              <p className="text-sm font-semibold mb-2">핵심 기술:</p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• 논문 요약 생성 (Abstractive Summarization)</li>
                <li>• 임상 질문 답변 (Question Answering)</li>
                <li>• 약물-질병 연관성 발굴</li>
                <li>• 체계적 문헌 고찰 자동화</li>
              </ul>
            </div>
            <div className="bg-pink-900/10 dark:bg-pink-900/30 p-3 rounded text-xs">
              <p className="font-semibold text-pink-900 dark:text-pink-300 mb-1">대표 시스템:</p>
              <p className="text-gray-700 dark:text-gray-300">
                PubMedGPT, BioGPT, SciBERT (논문 분석 특화 LLM)
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 실전 코드 - BioBERT NER */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Code className="w-7 h-7 text-indigo-600" />
          실전 코드: BioBERT 의료 개체명 인식
        </h2>

        <div className="space-y-6">
          {/* BioBERT NER */}
          <div>
            <h3 className="font-bold text-lg mb-3 text-blue-900 dark:text-blue-300">
              1. BioBERT로 EHR에서 질병/약물 추출 (Hugging Face)
            </h3>
            <div className="bg-slate-900 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm text-gray-100">
                <code>{`from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import torch

# BioBERT NER 모델 로드 (dmis-lab/biobert-base-cased-v1.2)
model_name = "dmis-lab/biobert-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# BC5CDR 데이터셋 Fine-tuned 모델 (Chemical & Disease Recognition)
ner_model = AutoModelForTokenClassification.from_pretrained(
    "alvaroalon2/biobert_chemical_ner"
)

# NER 파이프라인 생성
ner_pipeline = pipeline(
    "ner",
    model=ner_model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"  # 서브워드 병합
)

# 임상 노트 예시
clinical_note = """
Patient presents with acute chest pain and shortness of breath.
ECG shows ST-segment elevation. Administered aspirin 325mg and
nitroglycerin sublingually. Diagnosis: Acute Myocardial Infarction.
Plan: Emergency PCI, start dual antiplatelet therapy with clopidogrel 75mg.
"""

# 개체명 추출
entities = ner_pipeline(clinical_note)

print("\\n🔍 추출된 의료 개체명:\\n")
for entity in entities:
    print(f"{entity['word']:20} | {entity['entity_group']:10} | Score: {entity['score']:.3f}")

# 출력 예시:
# chest pain           | DISEASE    | Score: 0.987
# aspirin              | CHEMICAL   | Score: 0.995
# nitroglycerin        | CHEMICAL   | Score: 0.991
# Myocardial Infarction| DISEASE    | Score: 0.988
# clopidogrel          | CHEMICAL   | Score: 0.993

# 구조화된 결과 생성
def extract_medical_entities(text):
    entities = ner_pipeline(text)

    diseases = [e['word'] for e in entities if e['entity_group'] == 'DISEASE']
    chemicals = [e['word'] for e in entities if e['entity_group'] == 'CHEMICAL']

    return {
        'diseases': list(set(diseases)),
        'medications': list(set(chemicals)),
        'total_entities': len(entities)
    }

result = extract_medical_entities(clinical_note)
print(f"\\n질병: {result['diseases']}")
print(f"약물: {result['medications']}")
print(f"총 개체 수: {result['total_entities']}")`}</code>
              </pre>
            </div>
          </div>

          {/* 관계 추출 */}
          <div>
            <h3 className="font-bold text-lg mb-3 text-green-900 dark:text-green-300">
              2. 약물-질병 관계 추출 (Relation Extraction)
            </h3>
            <div className="bg-slate-900 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm text-gray-100">
                <code>{`from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 관계 추출 모델 (GAD 데이터셋 학습: Gene-Disease 관계)
model_name = "allenai/biomed_roberta_base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    "allenai/dsp_roberta_base_dapt_biomed_tapt_chemprot_4169"
)

def extract_relation(entity1, entity2, sentence):
    """
    entity1: 약물명
    entity2: 질병명
    sentence: 문장
    Returns: 관계 타입 (TREATS, CAUSES, NONE)
    """
    # 엔티티 마크업
    marked_sentence = sentence.replace(
        entity1, f"[E1]{entity1}[/E1]"
    ).replace(
        entity2, f"[E2]{entity2}[/E2]"
    )

    inputs = tokenizer(marked_sentence, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()

    relation_map = {
        0: "NONE",
        1: "TREATS",
        2: "CAUSES",
        3: "PREVENTS"
    }

    return {
        'entity1': entity1,
        'entity2': entity2,
        'relation': relation_map.get(predicted_class, "UNKNOWN"),
        'confidence': probs[0][predicted_class].item()
    }

# 사용 예시
sentences = [
    "Aspirin treats headache and reduces fever.",
    "Smoking causes lung cancer and heart disease.",
    "Metformin is prescribed for diabetes management."
]

print("\\n🔗 약물-질병 관계 추출:\\n")
for sent in sentences:
    # 간단한 예시 (실제로는 NER 결과 활용)
    if "Aspirin" in sent:
        result = extract_relation("Aspirin", "headache", sent)
    elif "Smoking" in sent:
        result = extract_relation("Smoking", "lung cancer", sent)
    elif "Metformin" in sent:
        result = extract_relation("Metformin", "diabetes", sent)

    print(f"{result['entity1']} {result['relation']} {result['entity2']}")
    print(f"Confidence: {result['confidence']:.2%}\\n")`}</code>
              </pre>
            </div>
          </div>

          {/* ICD-10 자동 코딩 */}
          <div>
            <h3 className="font-bold text-lg mb-3 text-purple-900 dark:text-purple-300">
              3. ICD-10 자동 코딩 (Extreme Multi-Label Classification)
            </h3>
            <div className="bg-slate-900 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm text-gray-100">
                <code>{`from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

# MIMIC-III ICD-10 코딩 모델 (50개 가장 빈번한 코드)
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ICD-10 코드 매핑 (Top 50)
icd10_codes = {
    0: "I50.9 - Heart Failure, Unspecified",
    1: "J44.1 - COPD with Acute Exacerbation",
    2: "I25.10 - Atherosclerotic Heart Disease",
    3: "E11.9 - Type 2 Diabetes Mellitus",
    4: "N18.9 - Chronic Kidney Disease",
    5: "J18.9 - Pneumonia, Unspecified",
    # ... (50개 코드)
}

def predict_icd10_codes(discharge_summary, top_k=5):
    """
    discharge_summary: 퇴원 요약문
    top_k: 상위 k개 코드 반환
    """
    inputs = tokenizer(
        discharge_summary,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True
    )

    # 다중 레이블 분류 (Sigmoid 활성화)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits)  # Multi-label

    # 상위 k개 코드 추출
    top_probs, top_indices = torch.topk(probs[0], k=top_k)

    results = []
    for prob, idx in zip(top_probs, top_indices):
        if prob > 0.5:  # 임계값
            results.append({
                'icd10_code': icd10_codes.get(idx.item(), "Unknown"),
                'probability': prob.item()
            })

    return results

# 퇴원 요약문 예시
discharge_summary = """
85-year-old male with history of hypertension and diabetes presented
with acute dyspnea and peripheral edema. Chest X-ray revealed pulmonary
congestion. Echocardiogram showed reduced ejection fraction (35%).
Diagnosed with acute decompensated heart failure. Treated with diuretics
and ACE inhibitors. Patient stabilized and discharged on day 5.
"""

codes = predict_icd10_codes(discharge_summary)

print("\\n📋 자동 생성된 ICD-10 코드:\\n")
for i, code in enumerate(codes, 1):
    print(f"{i}. {code['icd10_code']}")
    print(f"   확률: {code['probability']:.1%}\\n")`}</code>
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* 2024-2025 최신 동향 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <TrendingUp className="w-7 h-7 text-orange-600" />
          2024-2025 임상 NLP 혁신 동향
        </h2>

        <div className="space-y-4">
          <div className="border-l-4 border-blue-500 bg-blue-50 dark:bg-blue-900/20 p-4 rounded-r-lg">
            <h3 className="font-bold text-lg mb-2 text-blue-900 dark:text-blue-300">
              1. Large Language Models for Medicine
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              대규모 의료 텍스트 학습 LLM의 등장 (2024)
            </p>
            <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
              <li>• <strong>Med-PaLM 2 (Google):</strong> PubMed + 의학 교과서 학습, USMLE 85.4%</li>
              <li>• <strong>GatorTron (UF Health):</strong> 900억 파라미터, 900억 토큰 EHR 학습</li>
              <li>• <strong>BioGPT-Large:</strong> 1,500만 PubMed 논문 학습, 논문 요약 SOTA</li>
            </ul>
          </div>

          <div className="border-l-4 border-green-500 bg-green-50 dark:bg-green-900/20 p-4 rounded-r-lg">
            <h3 className="font-bold text-lg mb-2 text-green-900 dark:text-green-300">
              2. Zero-Shot & Few-Shot Learning
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              라벨 없는 데이터로 새로운 의료 작업 수행
            </p>
            <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
              <li>• <strong>Prompt Engineering:</strong> "환자가 당뇨병 진단을 받았습니까?" → Yes/No 분류</li>
              <li>• <strong>In-Context Learning:</strong> 5-10개 예제만으로 ICD-10 코딩 학습</li>
              <li>• <strong>Chain-of-Thought:</strong> 단계별 추론으로 진단 정확도 향상</li>
            </ul>
          </div>

          <div className="border-l-4 border-purple-500 bg-purple-50 dark:bg-purple-900/20 p-4 rounded-r-lg">
            <h3 className="font-bold text-lg mb-2 text-purple-900 dark:text-purple-300">
              3. Multimodal Clinical AI
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              텍스트 + 영상 + 유전체 데이터 통합 분석
            </p>
            <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
              <li>• <strong>PLIP (PathLang):</strong> 병리 이미지 + 진단 보고서 결합</li>
              <li>• <strong>MedCLIP:</strong> X-ray 이미지 + 방사선 리포트 매칭</li>
              <li>• <strong>ClinicalGPT-Vision:</strong> EHR 텍스트 + 의료 영상 동시 분석</li>
            </ul>
          </div>

          <div className="border-l-4 border-pink-500 bg-pink-50 dark:bg-pink-900/20 p-4 rounded-r-lg">
            <h3 className="font-bold text-lg mb-2 text-pink-900 dark:text-pink-300">
              4. Real-Time EHR Monitoring AI
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              실시간 임상 노트 분석 및 조기 경보 시스템
            </p>
            <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
              <li>• <strong>Epic AI:</strong> 진료 노트 입력과 동시에 패혈증 위험도 알림</li>
              <li>• <strong>Cerner HealtheIntent:</strong> 10만+ 환자 EHR 실시간 분석</li>
              <li>• <strong>Microsoft Healthcare NLP:</strong> Azure Health Text Analytics API</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 임상 NLP 통계 */}
      <section className="bg-gradient-to-r from-blue-600 to-indigo-600 rounded-xl p-6 shadow-lg text-white">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <Shield className="w-7 h-7" />
          임상 NLP 시장 & 성능 (2024)
        </h2>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="text-4xl font-bold mb-2">$43B</p>
            <p className="text-sm opacity-90">2030 Healthcare NLP 시장 규모</p>
            <p className="text-xs mt-2 opacity-75">출처: Grand View Research</p>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="text-4xl font-bold mb-2">95%</p>
            <p className="text-sm opacity-90">BioBERT 의료 개체명 인식 F1 Score</p>
            <p className="text-xs mt-2 opacity-75">출처: BC5CDR Benchmark</p>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="text-4xl font-bold mb-2">70%</p>
            <p className="text-sm opacity-90">ICD-10 자동 코딩 채택 병원 비율</p>
            <p className="text-xs mt-2 opacity-75">출처: HIMSS 2024 Survey</p>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="text-4xl font-bold mb-2">3,500만+</p>
            <p className="text-sm opacity-90">PubMed 의학 논문 수 (2024)</p>
            <p className="text-xs mt-2 opacity-75">출처: NIH NLM</p>
          </div>
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 핵심 데이터셋 & 벤치마크',
            icon: 'docs' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'MIMIC-III Clinical Notes',
                url: 'https://physionet.org/content/mimiciii/',
                description: '200만+ 임상 노트 (퇴원 요약, 방사선 리포트, 간호 기록)'
              },
              {
                title: 'n2c2 (National NLP Clinical Challenges)',
                url: 'https://n2c2.dbmi.hms.harvard.edu/',
                description: '임상 NLP 벤치마크 (NER, 관계 추출, 시간 정보)'
              },
              {
                title: 'BC5CDR (BioCreative V Chemical-Disease)',
                url: 'https://biocreative.bioinformatics.udel.edu/tasks/biocreative-v/track-3-cdr/',
                description: '1,500 PubMed 초록, 화학물질-질병 개체명 + 관계'
              },
              {
                title: 'PubMed Database',
                url: 'https://pubmed.ncbi.nlm.nih.gov/',
                description: '3,500만+ 생의학 논문 초록 및 전문'
              },
            ]
          },
          {
            title: '🔬 최신 연구 논문 (2023-2024)',
            icon: 'research' as const,
            color: 'border-pink-500',
            items: [
              {
                title: 'GatorTron: Large Clinical Language Model (NPJ Digital Medicine 2024)',
                url: 'https://www.nature.com/articles/s41746-024-01045-4',
                description: '900억 파라미터 EHR 특화 LLM, 20+ NLP 작업 SOTA'
              },
              {
                title: 'Med-PaLM 2: Medical Question Answering (Nature 2024)',
                url: 'https://www.nature.com/articles/s41586-023-06291-2',
                description: 'USMLE 85.4% 정답률, 9개국 의사 평가 통과'
              },
              {
                title: 'PLIP: PathLang Image-Text Model (CVPR 2024)',
                url: 'https://arxiv.org/abs/2305.04175',
                description: '병리 이미지 + 진단 텍스트 멀티모달 학습'
              },
              {
                title: 'Zero-Shot Clinical NLP (EMNLP 2024)',
                url: 'https://arxiv.org/abs/2310.12345',
                description: 'Prompt Engineering으로 ICD-10 코딩 F1 92%'
              },
            ]
          },
          {
            title: '🛠️ 실전 프레임워크 & 도구',
            icon: 'tools' as const,
            color: 'border-green-500',
            items: [
              {
                title: 'BioBERT (DMIS Lab)',
                url: 'https://github.com/dmis-lab/biobert',
                description: 'PubMed + PMC 사전학습 BERT, 의료 NER F1 95%+'
              },
              {
                title: 'ClinicalBERT (MIT)',
                url: 'https://github.com/EmilyAlsentzer/clinicalBERT',
                description: 'MIMIC-III 임상 노트 학습, EHR 텍스트 분석 최적'
              },
              {
                title: 'scispaCy',
                url: 'https://allenai.github.io/scispacy/',
                description: '생의학 텍스트 처리 spaCy 확장 (NER, 관계 추출)'
              },
              {
                title: 'MedCAT (King\'s College London)',
                url: 'https://github.com/CogStack/MedCAT',
                description: '임상 개념 주석 도구, UMLS/SNOMED CT 통합'
              },
              {
                title: 'Azure Health Text Analytics',
                url: 'https://azure.microsoft.com/en-us/products/ai-services/text-analytics-for-health',
                description: 'Microsoft 클라우드 의료 NLP API (HIPAA 준수)'
              },
            ]
          },
          {
            title: '📖 표준 온톨로지 & 용어집',
            icon: 'docs' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'UMLS (Unified Medical Language System)',
                url: 'https://www.nlm.nih.gov/research/umls/',
                description: '400만+ 의학 개념 통합 온톨로지 (NIH)'
              },
              {
                title: 'SNOMED CT',
                url: 'https://www.snomed.org/',
                description: '35만+ 임상 용어 표준, 전 세계 EHR 시스템 채택'
              },
              {
                title: 'ICD-10 (International Classification of Diseases)',
                url: 'https://www.who.int/standards/classifications/classification-of-diseases',
                description: 'WHO 질병 분류 체계, 72,000+ 코드'
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
            <span className="text-blue-600 font-bold">•</span>
            <span>임상 NLP 4대 작업: <strong>개체명 인식, 관계 추출, 문서 분류, 문헌 마이닝</strong></span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-600 font-bold">•</span>
            <span>핵심 모델: <strong>BioBERT (NER F1 95%), GatorTron (900억 파라미터), Med-PaLM 2</strong></span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-600 font-bold">•</span>
            <span><strong>2024 트렌드</strong>: LLM (Med-PaLM 2), Zero-Shot Learning, Multimodal AI</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-600 font-bold">•</span>
            <span>ICD-10 자동 코딩 정확도 <strong>95%+</strong>, 70% 병원 채택 (Epic, Cerner)</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-600 font-bold">•</span>
            <span>필수 도구: <strong>BioBERT, scispaCy, MedCAT, Azure Health Text Analytics</strong></span>
          </li>
        </ul>
      </section>
    </div>
  );
}
