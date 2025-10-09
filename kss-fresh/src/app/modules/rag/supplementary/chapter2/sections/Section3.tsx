'use client'

import { Lock } from 'lucide-react'

export default function Section3() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-blue-100 dark:bg-blue-900/20 flex items-center justify-center">
          <Lock className="text-blue-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">2.3 데이터 접근 제어</h2>
          <p className="text-gray-600 dark:text-gray-400">사용자별 권한 기반 문서 필터링</p>
        </div>
      </div>

      <div className="space-y-6">
        <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl">
          <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-3">접근 제어 레벨</h3>
          <div className="space-y-2 text-blue-700 dark:text-blue-300">
            <p>🔐 <strong>Level 5</strong>: 최고 기밀 (C-Level만 접근)</p>
            <p>🔒 <strong>Level 4</strong>: 기밀 (부서장 이상)</p>
            <p>🔓 <strong>Level 3</strong>: 내부용 (정직원)</p>
            <p>🗝️ <strong>Level 2</strong>: 제한적 공개 (계약직 포함)</p>
            <p>🌐 <strong>Level 1</strong>: 공개 (모든 사용자)</p>
          </div>
        </div>

        <div className="bg-gray-50 dark:bg-gray-900 p-6 rounded-xl">
          <h3 className="font-bold text-gray-800 dark:text-gray-200 mb-4">권한 기반 RAG 시스템</h3>
          <pre className="bg-black text-green-400 p-4 rounded-lg overflow-x-auto">
            <code>{`from typing import List, Dict, Set
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class User:
    id: str
    name: str
    department: str
    role: str
    access_level: int
    special_permissions: Set[str]

@dataclass
class Document:
    id: str
    content: str
    metadata: Dict
    access_level: int
    required_permissions: Set[str]
    owner_department: str

class SecureRAGSystem:
    def __init__(self):
        self.documents = []
        self.access_logs = []

        # 역할별 기본 접근 레벨
        self.role_access_levels = {
            'CEO': 5,
            'CTO': 5,
            'Director': 4,
            'Manager': 3,
            'Employee': 3,
            'Contractor': 2,
            'Guest': 1
        }

    def add_document(self, document: Document):
        """문서 추가 (메타데이터 포함)"""
        self.documents.append(document)

    def search_documents(self, query: str, user: User, log_access: bool = True) -> List[Document]:
        """사용자 권한에 따른 문서 검색"""
        # 1. 사용자 권한 확인
        user_access_level = self._get_user_access_level(user)

        # 2. 접근 가능한 문서 필터링
        accessible_docs = []

        for doc in self.documents:
            if self._can_access_document(user, doc, user_access_level):
                accessible_docs.append(doc)
            else:
                # 접근 거부 로깅
                if log_access:
                    self._log_access_denied(user, doc, "Insufficient access level")

        # 3. 접근 로깅
        if log_access and accessible_docs:
            self._log_access(user, query, accessible_docs)

        # 4. 쿼리와 관련된 문서만 반환 (실제로는 벡터 검색)
        relevant_docs = self._filter_relevant(query, accessible_docs)

        return relevant_docs

    def _get_user_access_level(self, user: User) -> int:
        """사용자의 유효 접근 레벨 계산"""
        base_level = self.role_access_levels.get(user.role, 1)

        # 특별 권한으로 레벨 상승 가능
        if 'security_clearance' in user.special_permissions:
            base_level = min(base_level + 1, 5)

        return max(base_level, user.access_level)

    def _can_access_document(self, user: User, doc: Document, user_access_level: int) -> bool:
        """문서 접근 가능 여부 확인"""
        # 1. 접근 레벨 확인
        if user_access_level < doc.access_level:
            return False

        # 2. 특별 권한 확인
        if doc.required_permissions:
            if not doc.required_permissions.issubset(user.special_permissions):
                return False

        # 3. 부서별 접근 제한
        if doc.metadata.get('department_only'):
            if user.department != doc.owner_department:
                # 크로스 부서 권한 확인
                if 'cross_department_access' not in user.special_permissions:
                    return False

        return True

    def _filter_relevant(self, query: str, documents: List[Document]) -> List[Document]:
        """관련성 기반 필터링 (간단한 예제)"""
        # 실제로는 벡터 유사도 검색
        relevant = []
        query_lower = query.lower()

        for doc in documents:
            if any(word in doc.content.lower() for word in query_lower.split()):
                relevant.append(doc)

        return relevant

    def _log_access(self, user: User, query: str, documents: List[Document]):
        """접근 로그 기록"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user.id,
            'user_name': user.name,
            'department': user.department,
            'query': query,
            'accessed_documents': [doc.id for doc in documents],
            'document_count': len(documents)
        }

        self.access_logs.append(log_entry)

        # 실시간 모니터링을 위한 알림 (예: 높은 레벨 문서 접근시)
        high_level_docs = [doc for doc in documents if doc.access_level >= 4]
        if high_level_docs:
            self._alert_high_level_access(user, high_level_docs)

    def _log_access_denied(self, user: User, doc: Document, reason: str):
        """접근 거부 로그"""
        denied_log = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user.id,
            'document_id': doc.id,
            'reason': reason,
            'user_level': user.access_level,
            'required_level': doc.access_level
        }

        # 의심스러운 활동 감지
        recent_denials = self._get_recent_denials(user.id, minutes=5)
        if len(recent_denials) >= 3:
            self._alert_suspicious_activity(user, recent_denials)

    def _alert_high_level_access(self, user: User, documents: List[Document]):
        """고레벨 문서 접근 알림"""
        print(f"⚠️ HIGH LEVEL ACCESS ALERT")
        print(f"User: {user.name} ({user.role})")
        print(f"Documents: {[doc.id for doc in documents]}")
        # 실제로는 Slack, Email 등으로 알림

    def _alert_suspicious_activity(self, user: User, denials: List[Dict]):
        """의심스러운 활동 알림"""
        print(f"🚨 SUSPICIOUS ACTIVITY DETECTED")
        print(f"User: {user.name} attempted {len(denials)} unauthorized accesses")
        # 실제로는 보안팀에 즉시 알림

    def _get_recent_denials(self, user_id: str, minutes: int) -> List[Dict]:
        """최근 접근 거부 내역 조회"""
        # 실제로는 데이터베이스에서 조회
        return []  # 예제를 위한 빈 리스트

    def generate_audit_report(self, start_date: datetime, end_date: datetime) -> Dict:
        """감사 보고서 생성"""
        report = {
            'period': f"{start_date.date()} to {end_date.date()}",
            'total_accesses': len(self.access_logs),
            'unique_users': len(set(log['user_id'] for log in self.access_logs)),
            'high_level_accesses': 0,
            'denied_attempts': 0,
            'most_accessed_documents': {},
            'user_activity': {}
        }

        # 상세 분석 로직...

        return report

# 실제 사용 예제
rag_system = SecureRAGSystem()

# 문서 추가
rag_system.add_document(Document(
    id="DOC001",
    content="2024년 회사 재무제표: 매출 1조원, 영업이익 1000억원",
    metadata={'department_only': True, 'year': 2024},
    access_level=4,
    required_permissions={'financial_data'},
    owner_department="Finance"
))

rag_system.add_document(Document(
    id="DOC002",
    content="신제품 개발 로드맵: AI 기반 추천 시스템",
    metadata={'project': 'AI_REC_2024'},
    access_level=3,
    required_permissions=set(),
    owner_department="Engineering"
))

rag_system.add_document(Document(
    id="DOC003",
    content="회사 휴가 정책: 연차 15일, 여름휴가 5일",
    metadata={'policy_type': 'HR'},
    access_level=1,
    required_permissions=set(),
    owner_department="HR"
))

# 사용자 정의
ceo = User(
    id="USR001",
    name="김대표",
    department="Executive",
    role="CEO",
    access_level=5,
    special_permissions={'financial_data', 'cross_department_access'}
)

employee = User(
    id="USR002",
    name="박직원",
    department="Engineering",
    role="Employee",
    access_level=3,
    special_permissions=set()
)

contractor = User(
    id="USR003",
    name="이계약",
    department="Engineering",
    role="Contractor",
    access_level=2,
    special_permissions=set()
)

# 검색 테스트
print("🔍 권한별 검색 결과\\n")

for user in [ceo, employee, contractor]:
    print(f"{user.name} ({user.role}) 검색 결과:")
    results = rag_system.search_documents("회사 정보", user)
    for doc in results:
        print(f"  - {doc.id}: {doc.content[:30]}...")
    print()

# GDPR 준수 데이터 삭제 기능
class GDPRCompliantRAG(SecureRAGSystem):
    def delete_user_data(self, user_id: str) -> Dict:
        """사용자 요청에 따른 데이터 삭제 (GDPR Article 17)"""
        deletion_report = {
            'user_id': user_id,
            'deleted_logs': 0,
            'anonymized_logs': 0,
            'timestamp': datetime.now().isoformat()
        }

        # 1. 액세스 로그에서 사용자 정보 익명화
        for log in self.access_logs:
            if log['user_id'] == user_id:
                log['user_id'] = f"ANON_{hash(user_id)}"
                log['user_name'] = "[DELETED]"
                deletion_report['anonymized_logs'] += 1

        # 2. 사용자가 생성한 문서 삭제
        self.documents = [
            doc for doc in self.documents
            if doc.metadata.get('created_by') != user_id
        ]

        return deletion_report`}</code>
          </pre>
        </div>
      </div>
    </section>
  )
}
