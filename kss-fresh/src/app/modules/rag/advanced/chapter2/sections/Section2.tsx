import { GitBranch } from 'lucide-react'

export default function Section2() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-indigo-100 dark:bg-indigo-900/20 flex items-center justify-center">
          <GitBranch className="text-indigo-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">2.2 Multi-Agent 아키텍처 설계</h2>
          <p className="text-gray-600 dark:text-gray-400">효율적인 에이전트 협력을 위한 시스템 구조</p>
        </div>
      </div>

      <div className="space-y-6">
        <div className="bg-indigo-50 dark:bg-indigo-900/20 p-6 rounded-xl border border-indigo-200 dark:border-indigo-700">
          <h3 className="font-bold text-indigo-800 dark:text-indigo-200 mb-4">핵심 아키텍처 패턴</h3>

          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border mb-4">
            <h4 className="font-medium text-gray-900 dark:text-white mb-3">1. 계층적 분산 (Hierarchical Distribution)</h4>
            <pre className="text-xs overflow-x-auto bg-gray-100 dark:bg-gray-700 p-3 rounded">
{`                    ┌─────────────────┐
                    │  Orchestrator   │
                    │     Agent       │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
    ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
    │ Medical     │  │ Legal       │  │ Technical   │
    │ Specialist  │  │ Specialist  │  │ Specialist  │
    │ Agent       │  │ Agent       │  │ Agent       │
    └─────────────┘  └─────────────┘  └─────────────┘
           │                 │                 │
    ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
    │ Cardiology  │  │ Contract    │  │ Software    │
    │ Sub-Agent   │  │ Sub-Agent   │  │ Sub-Agent   │
    └─────────────┘  └─────────────┘  └─────────────┘`}
            </pre>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
            <h4 className="font-medium text-gray-900 dark:text-white mb-3">2. 파이프라인 기반 (Pipeline-based)</h4>
            <pre className="text-xs overflow-x-auto bg-gray-100 dark:bg-gray-700 p-3 rounded">
{`Query → Analysis → Routing → Parallel Processing → Synthesis → Validation → Response

        ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
        │ Agent A     │    │ Agent B     │    │ Agent C     │
        │ (Research)  │    │ (Analysis)  │    │ (Synthesis) │
        └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
               │                  │                  │
               ▼                  ▼                  ▼
        ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
        │ Results A   │    │ Results B   │    │ Final       │
        │             │    │             │    │ Answer      │
        └─────────────┘    └─────────────┘    └─────────────┘`}
            </pre>
          </div>
        </div>

        <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl border border-green-200 dark:border-green-700">
          <h3 className="font-bold text-green-800 dark:text-green-200 mb-4">에이전트 간 통신 프로토콜</h3>

          <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg overflow-hidden border border-slate-200 dark:border-slate-700">
            <pre className="text-sm text-slate-800 dark:text-slate-200 overflow-x-auto max-h-96 overflow-y-auto font-mono">
{`from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import asyncio
import json
from datetime import datetime

class MessageType(Enum):
    QUERY = "query"
    RESPONSE = "response"
    REQUEST_HELP = "request_help"
    PROVIDE_CONTEXT = "provide_context"
    VALIDATE = "validate"
    ERROR = "error"

@dataclass
class AgentMessage:
    sender: str
    receiver: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: datetime
    conversation_id: str
    priority: int = 1  # 1=low, 2=medium, 3=high

class MessageBus:
    def __init__(self):
        self.subscribers: Dict[str, List[callable]] = {}
        self.message_history: List[AgentMessage] = []

    def subscribe(self, agent_id: str, handler: callable):
        if agent_id not in self.subscribers:
            self.subscribers[agent_id] = []
        self.subscribers[agent_id].append(handler)

    async def publish(self, message: AgentMessage):
        self.message_history.append(message)

        # 특정 수신자에게 메시지 전달
        if message.receiver in self.subscribers:
            tasks = []
            for handler in self.subscribers[message.receiver]:
                tasks.append(handler(message))
            await asyncio.gather(*tasks)

    def get_conversation(self, conversation_id: str) -> List[AgentMessage]:
        return [msg for msg in self.message_history
                if msg.conversation_id == conversation_id]

class BaseAgent:
    def __init__(self, agent_id: str, message_bus: MessageBus):
        self.agent_id = agent_id
        self.message_bus = message_bus
        self.knowledge_base = None
        self.capabilities = []
        self.active_conversations = set()

        # 메시지 핸들러 등록
        message_bus.subscribe(agent_id, self.handle_message)

    async def handle_message(self, message: AgentMessage):
        """메시지 처리 메인 로직"""
        self.active_conversations.add(message.conversation_id)

        try:
            if message.message_type == MessageType.QUERY:
                await self.process_query(message)
            elif message.message_type == MessageType.REQUEST_HELP:
                await self.provide_help(message)
            elif message.message_type == MessageType.VALIDATE:
                await self.validate_response(message)
        except Exception as e:
            await self.send_error(message, str(e))

    async def process_query(self, message: AgentMessage):
        """쿼리 처리 - 각 에이전트에서 구현"""
        raise NotImplementedError

    async def provide_help(self, message: AgentMessage):
        """다른 에이전트 도움 요청 처리"""
        raise NotImplementedError

    async def validate_response(self, message: AgentMessage):
        """응답 검증"""
        raise NotImplementedError

    async def send_message(self, receiver: str, message_type: MessageType,
                          content: Dict[str, Any], conversation_id: str,
                          priority: int = 1):
        """메시지 전송"""
        message = AgentMessage(
            sender=self.agent_id,
            receiver=receiver,
            message_type=message_type,
            content=content,
            timestamp=datetime.now(),
            conversation_id=conversation_id,
            priority=priority
        )
        await self.message_bus.publish(message)

    async def send_error(self, original_message: AgentMessage, error: str):
        """에러 메시지 전송"""
        await self.send_message(
            receiver=original_message.sender,
            message_type=MessageType.ERROR,
            content={"error": error, "original_message": original_message.content},
            conversation_id=original_message.conversation_id
        )

class OrchestratorAgent(BaseAgent):
    def __init__(self, agent_id: str, message_bus: MessageBus):
        super().__init__(agent_id, message_bus)
        self.specialist_agents = {}
        self.query_routing_rules = {}

    def register_specialist(self, agent_id: str, capabilities: List[str]):
        """전문 에이전트 등록"""
        self.specialist_agents[agent_id] = capabilities
        for capability in capabilities:
            if capability not in self.query_routing_rules:
                self.query_routing_rules[capability] = []
            self.query_routing_rules[capability].append(agent_id)

    async def process_query(self, message: AgentMessage):
        """쿼리를 분석하고 적절한 전문 에이전트들에게 라우팅"""
        query = message.content.get('query', '')
        conversation_id = message.conversation_id

        # 1. 쿼리 분석 - 필요한 도메인들 식별
        required_domains = await self.analyze_query_domains(query)

        # 2. 각 도메인별 전문 에이전트들에게 병렬 요청
        specialist_tasks = []
        for domain in required_domains:
            if domain in self.query_routing_rules:
                for agent_id in self.query_routing_rules[domain]:
                    task = self.send_message(
                        receiver=agent_id,
                        message_type=MessageType.QUERY,
                        content={"query": query, "domain_focus": domain},
                        conversation_id=conversation_id,
                        priority=2
                    )
                    specialist_tasks.append(task)

        await asyncio.gather(*specialist_tasks)

        # 3. 응답 수집 대기 로직은 별도 구현
        await self.wait_and_synthesize_responses(conversation_id, len(specialist_tasks))

    async def analyze_query_domains(self, query: str) -> List[str]:
        """쿼리에서 필요한 도메인들을 추출"""
        # 실제로는 LLM 또는 분류 모델 사용
        domains = []

        # 간단한 키워드 기반 예시
        if any(word in query.lower() for word in ['heart', 'cardiac', 'chest pain']):
            domains.append('cardiology')
        if any(word in query.lower() for word in ['legal', 'contract', 'law']):
            domains.append('legal')
        if any(word in query.lower() for word in ['code', 'programming', 'software']):
            domains.append('technical')

        return domains if domains else ['general']

    async def wait_and_synthesize_responses(self, conversation_id: str, expected_responses: int):
        """응답들을 수집하고 통합"""
        # 타임아웃과 함께 응답 대기
        timeout = 30  # 30초
        collected_responses = []

        start_time = datetime.now()
        while len(collected_responses) < expected_responses:
            if (datetime.now() - start_time).seconds > timeout:
                break

            # 메시지 히스토리에서 응답 수집
            conversation_messages = self.message_bus.get_conversation(conversation_id)
            responses = [msg for msg in conversation_messages
                        if msg.message_type == MessageType.RESPONSE
                        and msg.receiver == self.agent_id]

            if len(responses) > len(collected_responses):
                collected_responses = responses

            await asyncio.sleep(0.1)

        # 응답 통합
        synthesized_answer = await self.synthesize_responses(collected_responses)

        # 최종 답변 전송 (원래 질문자에게)
        original_query_message = next(
            (msg for msg in conversation_messages
             if msg.message_type == MessageType.QUERY), None
        )

        if original_query_message:
            await self.send_message(
                receiver=original_query_message.sender,
                message_type=MessageType.RESPONSE,
                content={"answer": synthesized_answer, "sources": [r.sender for r in collected_responses]},
                conversation_id=conversation_id
            )

    async def synthesize_responses(self, responses: List[AgentMessage]) -> str:
        """여러 에이전트 응답을 통합하여 최종 답변 생성"""
        if not responses:
            return "죄송합니다. 적절한 답변을 생성할 수 없습니다."

        # 실제로는 LLM을 사용하여 응답 통합
        combined_content = []
        for response in responses:
            agent_name = response.sender
            content = response.content.get('answer', '')
            combined_content.append(f"**{agent_name}의 관점**: {content}")

        return "\\n\\n".join(combined_content)

class SpecialistRAGAgent(BaseAgent):
    def __init__(self, agent_id: str, message_bus: MessageBus,
                 domain: str, vector_db, llm_client):
        super().__init__(agent_id, message_bus)
        self.domain = domain
        self.vector_db = vector_db
        self.llm_client = llm_client
        self.capabilities = [domain]

    async def process_query(self, message: AgentMessage):
        """도메인별 특화된 RAG 처리"""
        query = message.content.get('query', '')
        domain_focus = message.content.get('domain_focus', self.domain)
        conversation_id = message.conversation_id

        try:
            # 1. 도메인 특화 검색
            relevant_docs = await self.search_domain_knowledge(query, domain_focus)

            # 2. 도메인별 특화 프롬프트로 답변 생성
            answer = await self.generate_domain_answer(query, relevant_docs, domain_focus)

            # 3. 신뢰도 점수 계산
            confidence = await self.calculate_confidence(query, answer, relevant_docs)

            # 4. 응답 전송
            await self.send_message(
                receiver=message.sender,
                message_type=MessageType.RESPONSE,
                content={
                    "answer": answer,
                    "confidence": confidence,
                    "domain": self.domain,
                    "sources": [doc['id'] for doc in relevant_docs]
                },
                conversation_id=conversation_id
            )

        except Exception as e:
            await self.send_error(message, f"Domain processing error: {str(e)}")

    async def search_domain_knowledge(self, query: str, domain_focus: str) -> List[Dict]:
        """도메인별 특화 검색"""
        # 도메인별 필터링과 가중치 적용
        search_query = f"{domain_focus}: {query}"

        # 벡터 데이터베이스에서 검색
        results = await self.vector_db.search(
            query=search_query,
            filters={"domain": self.domain},
            top_k=5
        )

        return results

    async def generate_domain_answer(self, query: str, docs: List[Dict], domain: str) -> str:
        """도메인별 특화 답변 생성"""
        context = "\\n\\n".join([doc['content'] for doc in docs])

        domain_prompt = f"""
        당신은 {domain} 분야의 전문가입니다.
        제공된 컨텍스트를 바탕으로 질문에 대해 전문적이고 정확한 답변을 제공하세요.

        질문: {query}

        관련 자료:
        {context}

        {domain} 전문가로서의 답변:
        """

        response = await self.llm_client.generate(domain_prompt)
        return response

    async def calculate_confidence(self, query: str, answer: str, docs: List[Dict]) -> float:
        """답변 신뢰도 계산"""
        # 간단한 신뢰도 계산 로직
        # 실제로는 더 복잡한 신뢰도 모델 사용

        if not docs:
            return 0.0

        # 검색 결과 점수 기반
        avg_score = sum(doc.get('score', 0) for doc in docs) / len(docs)

        # 답변 길이 기반 (너무 짧으면 신뢰도 낮음)
        length_factor = min(len(answer) / 100, 1.0)

        return min(avg_score * length_factor, 1.0)

    async def provide_help(self, message: AgentMessage):
        """다른 에이전트의 도움 요청 처리"""
        request_type = message.content.get('help_type', '')
        context = message.content.get('context', '')

        if request_type == 'domain_expertise':
            # 도메인 전문 지식 제공
            expertise = await self.provide_domain_expertise(context)

            await self.send_message(
                receiver=message.sender,
                message_type=MessageType.PROVIDE_CONTEXT,
                content={"expertise": expertise, "domain": self.domain},
                conversation_id=message.conversation_id
            )

# 사용 예시
async def setup_multi_agent_rag():
    # 메시지 버스 초기화
    message_bus = MessageBus()

    # Orchestrator 에이전트 생성
    orchestrator = OrchestratorAgent("orchestrator", message_bus)

    # 전문 에이전트들 생성
    medical_agent = SpecialistRAGAgent("medical_agent", message_bus, "medical", medical_vector_db, llm_client)
    legal_agent = SpecialistRAGAgent("legal_agent", message_bus, "legal", legal_vector_db, llm_client)
    technical_agent = SpecialistRAGAgent("technical_agent", message_bus, "technical", tech_vector_db, llm_client)

    # 전문 에이전트들을 orchestrator에 등록
    orchestrator.register_specialist("medical_agent", ["medical", "cardiology", "health"])
    orchestrator.register_specialist("legal_agent", ["legal", "contract", "law"])
    orchestrator.register_specialist("technical_agent", ["technical", "programming", "software"])

    return orchestrator, message_bus`}
            </pre>
          </div>
        </div>
      </div>
    </section>
  )
}
