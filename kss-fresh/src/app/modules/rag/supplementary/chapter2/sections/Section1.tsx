'use client'

import { UserX } from 'lucide-react'

export default function Section1() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-red-100 dark:bg-red-900/20 flex items-center justify-center">
          <UserX className="text-red-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">2.1 PII 탐지 및 마스킹</h2>
          <p className="text-gray-600 dark:text-gray-400">개인식별정보 자동 감지 및 보호</p>
        </div>
      </div>

      <div className="space-y-6">
        <div className="bg-red-50 dark:bg-red-900/20 p-6 rounded-xl">
          <h3 className="font-bold text-red-800 dark:text-red-200 mb-3">PII 유형과 위험도</h3>
          <div className="space-y-2 text-red-700 dark:text-red-300">
            <p>🔴 <strong>고위험</strong>: 주민등록번호, 신용카드번호, 의료기록</p>
            <p>🟡 <strong>중위험</strong>: 이름, 전화번호, 이메일, 주소</p>
            <p>🟢 <strong>저위험</strong>: 나이, 직업, 지역명</p>
          </div>
        </div>

        <div className="bg-gray-50 dark:bg-gray-900 p-6 rounded-xl">
          <h3 className="font-bold text-gray-800 dark:text-gray-200 mb-4">실시간 PII 탐지 시스템</h3>
          <pre className="bg-black text-green-400 p-4 rounded-lg overflow-x-auto">
            <code>{`import re
from typing import Dict, List, Tuple
import spacy
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
import hashlib

class PIIProtectionSystem:
    def __init__(self):
        # Presidio 초기화
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

        # 한국 특화 패턴
        self.korean_patterns = {
            'korean_rrn': r'\\d{6}-[1-4]\\d{6}',  # 주민등록번호
            'korean_phone': r'(010|011|016|017|018|019)-?\\d{3,4}-?\\d{4}',
            'korean_card': r'\\d{4}-?\\d{4}-?\\d{4}-?\\d{4}',
            'korean_account': r'\\d{3}-\\d{2}-\\d{6,}',  # 계좌번호
            'korean_passport': r'[A-Z]\\d{8}',  # 여권번호
        }

        # 마스킹 설정
        self.masking_config = {
            'PERSON': 'replace',
            'EMAIL_ADDRESS': 'hash',
            'PHONE_NUMBER': 'mask',
            'CREDIT_CARD': 'encrypt',
            'KOREAN_RRN': 'remove'
        }

    def detect_pii(self, text: str, language: str = 'ko') -> List[Dict]:
        """PII 탐지"""
        # Presidio 기본 탐지
        results = self.analyzer.analyze(
            text=text,
            language=language,
            entities=[
                "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
                "CREDIT_CARD", "IBAN_CODE", "IP_ADDRESS"
            ]
        )

        # 한국 특화 패턴 탐지
        korean_results = self._detect_korean_patterns(text)

        # 결과 통합
        all_results = self._merge_results(results, korean_results)

        return all_results

    def _detect_korean_patterns(self, text: str) -> List[Dict]:
        """한국 특화 패턴 탐지"""
        results = []

        for pattern_name, pattern in self.korean_patterns.items():
            for match in re.finditer(pattern, text):
                results.append({
                    'entity_type': pattern_name.upper(),
                    'start': match.start(),
                    'end': match.end(),
                    'score': 0.95,  # 패턴 매칭은 높은 신뢰도
                    'text': match.group()
                })

        return results

    def mask_pii(self, text: str, detected_pii: List[Dict]) -> Tuple[str, Dict]:
        """PII 마스킹"""
        masked_text = text
        masking_map = {}

        # 뒤에서부터 처리 (인덱스 변경 방지)
        for pii in sorted(detected_pii, key=lambda x: x['start'], reverse=True):
            entity_type = pii['entity_type']
            original = text[pii['start']:pii['end']]

            # 마스킹 방법 선택
            method = self.masking_config.get(entity_type, 'mask')
            masked_value = self._apply_masking(original, entity_type, method)

            # 텍스트 교체
            masked_text = masked_text[:pii['start']] + masked_value + masked_text[pii['end']:]

            # 매핑 저장 (복원용)
            masking_map[masked_value] = {
                'original': original,
                'type': entity_type,
                'method': method
            }

        return masked_text, masking_map

    def _apply_masking(self, text: str, entity_type: str, method: str) -> str:
        """마스킹 적용"""
        if method == 'remove':
            return '[REMOVED]'

        elif method == 'mask':
            # 앞 2글자만 보이고 나머지는 마스킹
            if len(text) > 2:
                return text[:2] + '*' * (len(text) - 2)
            return '*' * len(text)

        elif method == 'hash':
            # 일방향 해시 (복원 불가)
            hash_obj = hashlib.sha256(text.encode())
            return f"[HASH:{hash_obj.hexdigest()[:8]}]"

        elif method == 'encrypt':
            # 양방향 암호화 (복원 가능) - 실제로는 적절한 암호화 사용
            return f"[ENC:{text[:4]}...{text[-4:]}]"

        elif method == 'replace':
            # 엔티티 타입으로 대체
            return f"[{entity_type}]"

        return '[MASKED]'

    def create_safe_context(self, documents: List[str]) -> List[Dict]:
        """RAG 컨텍스트를 위한 안전한 문서 생성"""
        safe_documents = []

        for doc_id, doc in enumerate(documents):
            # PII 탐지
            pii_list = self.detect_pii(doc)

            # 마스킹 적용
            masked_doc, mapping = self.mask_pii(doc, pii_list)

            safe_documents.append({
                'doc_id': doc_id,
                'original_length': len(doc),
                'masked_text': masked_doc,
                'pii_count': len(pii_list),
                'pii_types': list(set(p['entity_type'] for p in pii_list)),
                'masking_map': mapping
            })

        return safe_documents

# 실제 사용 예제
pii_system = PIIProtectionSystem()

# 위험한 텍스트 예제
dangerous_text = """
고객명: 김철수
주민등록번호: 880101-1234567
전화번호: 010-1234-5678
이메일: kim@example.com
신용카드: 1234-5678-9012-3456
주소: 서울시 강남구 테헤란로 123

김철수님의 최근 구매 내역을 확인했습니다.
"""

# PII 탐지
detected = pii_system.detect_pii(dangerous_text)
print(f"탐지된 PII: {len(detected)}개")

# 마스킹 적용
safe_text, mapping = pii_system.mask_pii(dangerous_text, detected)
print(f"\\n안전한 텍스트:\\n{safe_text}")

# RAG 컨텍스트 생성
safe_contexts = pii_system.create_safe_context([dangerous_text])
print(f"\\nRAG용 안전한 컨텍스트 생성 완료")`}</code>
          </pre>
        </div>
      </div>
    </section>
  )
}
