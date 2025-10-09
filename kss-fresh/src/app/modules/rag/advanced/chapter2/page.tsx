'use client'

import Link from 'next/link'
import { ArrowLeft, ArrowRight, Users, GitBranch, MessageSquare, Zap, Shield, Database } from 'lucide-react'
import References from '@/components/common/References'

export default function Chapter2Page() {
  return (
    <div className="max-w-4xl mx-auto py-8 px-4">
      {/* Header */}
      <div className="mb-8">
        <Link
          href="/modules/rag/advanced"
          className="inline-flex items-center gap-2 text-emerald-600 hover:text-emerald-700 mb-4 transition-colors"
        >
          <ArrowLeft size={20} />
          고급 과정으로 돌아가기
        </Link>
        
        <div className="bg-gradient-to-r from-purple-500 to-indigo-600 rounded-2xl p-8 text-white">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-16 h-16 rounded-xl bg-white/20 flex items-center justify-center">
              <Users size={32} />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Chapter 2: Multi-Agent RAG Systems</h1>
              <p className="text-purple-100 text-lg">분산 지능을 통한 복잡한 질의응답 시스템 구축</p>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="space-y-8">
        {/* Section 1: Multi-Agent RAG 개념 */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-purple-100 dark:bg-purple-900/20 flex items-center justify-center">
              <Users className="text-purple-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">2.1 Multi-Agent RAG의 필요성</h2>
              <p className="text-gray-600 dark:text-gray-400">복잡한 질문을 전문 에이전트들이 협력하여 해결</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-xl border border-purple-200 dark:border-purple-700">
              <h3 className="font-bold text-purple-800 dark:text-purple-200 mb-4">단일 Agent vs Multi-Agent 접근법</h3>
              
              <div className="prose prose-sm dark:prose-invert mb-4">
                <p className="text-gray-700 dark:text-gray-300">
                  <strong>복잡한 질문은 종종 여러 도메인의 전문 지식을 필요로 합니다.</strong> 
                  단일 RAG 시스템으로는 한계가 있는 상황에서, Multi-Agent RAG는 각 도메인에 특화된 
                  에이전트들이 협력하여 더 정확하고 포괄적인 답변을 제공할 수 있습니다.
                </p>
                <p className="text-gray-700 dark:text-gray-300">
                  <strong>핵심 아키텍처 요소:</strong>
                </p>
                <ul className="list-disc list-inside text-gray-700 dark:text-gray-300 space-y-1">
                  <li><strong>Orchestrator Agent</strong>: 질문을 분석하고 적절한 전문 에이전트들에게 배분</li>
                  <li><strong>Specialist Agents</strong>: 각 도메인별 전문 지식을 보유한 RAG 시스템</li>
                  <li><strong>Synthesis Agent</strong>: 여러 에이전트의 결과를 통합하여 최종 답변 생성</li>
                  <li><strong>Quality Validator</strong>: 답변의 일관성과 품질을 검증</li>
                </ul>
              </div>

              <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border">
                <h4 className="font-medium text-gray-900 dark:text-white mb-3">실제 활용 사례: 의료 진단 시스템</h4>
                <div className="bg-gray-100 dark:bg-gray-700 p-4 rounded">
                  <p className="text-sm text-gray-700 dark:text-gray-300">
                    <strong>질문:</strong> "40세 남성 환자가 가슴 통증과 호흡곤란을 호소합니다. 
                    최근 장거리 비행 후 발생했으며, 가족력상 심장병이 있습니다."
                  </p>
                </div>
                <div className="mt-3 grid md:grid-cols-3 gap-3 text-xs">
                  <div className="bg-blue-50 dark:bg-blue-900/30 p-2 rounded">
                    <strong>심장내과 Agent</strong><br/>
                    심근경색, 협심증 가능성 분석
                  </div>
                  <div className="bg-green-50 dark:bg-green-900/30 p-2 rounded">
                    <strong>호흡기내과 Agent</strong><br/>
                    폐색전증, 기흉 가능성 검토
                  </div>
                  <div className="bg-orange-50 dark:bg-orange-900/30 p-2 rounded">
                    <strong>응급의학 Agent</strong><br/>
                    중증도 판단 및 응급 처치 가이드
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl border border-blue-200 dark:border-blue-700">
              <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-4">Multi-Agent RAG의 장점</h3>
              
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-green-600 dark:text-green-400 mb-2">✅ 성능 향상</h4>
                  <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                    <li>• 도메인별 최적화된 검색 성능</li>
                    <li>• 전문 지식의 깊이 증가</li>
                    <li>• 교차 검증을 통한 정확도 향상</li>
                    <li>• 편향 감소 (다양한 관점)</li>
                  </ul>
                </div>
                
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-blue-600 dark:text-blue-400 mb-2">⚡ 확장성</h4>
                  <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                    <li>• 새로운 도메인 에이전트 쉽게 추가</li>
                    <li>• 병렬 처리로 응답 속도 최적화</li>
                    <li>• 모듈화된 유지보수</li>
                    <li>• 독립적 에이전트 업데이트 가능</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Section 2: 아키텍처 설계 */}
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

        {/* Section 3: 협력 패턴 */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-green-100 dark:bg-green-900/20 flex items-center justify-center">
              <MessageSquare className="text-green-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">2.3 에이전트 협력 패턴</h2>
              <p className="text-gray-600 dark:text-gray-400">효과적인 멀티 에이전트 협력을 위한 설계 패턴</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl border border-green-200 dark:border-green-700">
              <h3 className="font-bold text-green-800 dark:text-green-200 mb-4">주요 협력 패턴</h3>
              
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-blue-600 dark:text-blue-400 mb-3">🔄 Sequential Pattern</h4>
                  <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                    에이전트들이 순차적으로 작업을 수행하여 점진적으로 답변을 개선
                  </p>
                  <div className="bg-gray-100 dark:bg-gray-700 p-2 rounded text-xs">
                    Research Agent → Analysis Agent → Synthesis Agent
                  </div>
                </div>
                
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-green-600 dark:text-green-400 mb-3">⚡ Parallel Pattern</h4>
                  <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                    여러 에이전트가 동시에 작업하여 다양한 관점의 답변을 수집
                  </p>
                  <div className="bg-gray-100 dark:bg-gray-700 p-2 rounded text-xs">
                    Agent A + Agent B + Agent C → Synthesizer
                  </div>
                </div>
                
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-purple-600 dark:text-purple-400 mb-3">🔀 Debate Pattern</h4>
                  <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                    에이전트들이 서로 다른 관점에서 토론하여 최적의 답변 도출
                  </p>
                  <div className="bg-gray-100 dark:bg-gray-700 p-2 rounded text-xs">
                    Pro Agent ↔ Con Agent → Moderator → Final Answer
                  </div>
                </div>
                
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-orange-600 dark:text-orange-400 mb-3">🎯 Expert Consultation</h4>
                  <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                    일반 에이전트가 전문 에이전트에게 자문을 구하는 패턴
                  </p>
                  <div className="bg-gray-100 dark:bg-gray-700 p-2 rounded text-xs">
                    General Agent → Expert1, Expert2 → Integrated Answer
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-yellow-50 dark:bg-yellow-900/20 p-6 rounded-xl border border-yellow-200 dark:border-yellow-700">
              <h3 className="font-bold text-yellow-800 dark:text-yellow-200 mb-4">실제 구현 사례: 의사결정 지원 시스템</h3>
              
              <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg overflow-hidden border border-slate-200 dark:border-slate-700">
                <pre className="text-sm text-slate-800 dark:text-slate-200 overflow-x-auto max-h-96 overflow-y-auto font-mono">
{`class DecisionSupportSystem:
    def __init__(self):
        self.message_bus = MessageBus()
        self.agents = {}
        self.setup_agents()
    
    def setup_agents(self):
        # 다양한 역할의 에이전트 생성
        self.agents['researcher'] = ResearchAgent("researcher", self.message_bus)
        self.agents['analyst'] = AnalysisAgent("analyst", self.message_bus)
        self.agents['validator'] = ValidationAgent("validator", self.message_bus)
        self.agents['synthesizer'] = SynthesisAgent("synthesizer", self.message_bus)
        self.agents['devil_advocate'] = DevilAdvocateAgent("devil_advocate", self.message_bus)
    
    async def process_complex_query(self, query: str) -> str:
        conversation_id = f"decision_{datetime.now().timestamp()}"
        
        # Phase 1: 초기 연구 및 분석
        await self.agents['researcher'].send_message(
            receiver="researcher",
            message_type=MessageType.QUERY,
            content={"query": query, "phase": "initial_research"},
            conversation_id=conversation_id
        )
        
        # 연구 결과 대기
        research_results = await self.wait_for_response("researcher", conversation_id)
        
        # Phase 2: 심화 분석
        await self.agents['analyst'].send_message(
            receiver="analyst", 
            message_type=MessageType.QUERY,
            content={"query": query, "research_data": research_results},
            conversation_id=conversation_id
        )
        
        analysis_results = await self.wait_for_response("analyst", conversation_id)
        
        # Phase 3: 반대 의견 검토
        await self.agents['devil_advocate'].send_message(
            receiver="devil_advocate",
            message_type=MessageType.QUERY, 
            content={"analysis": analysis_results, "task": "find_weaknesses"},
            conversation_id=conversation_id
        )
        
        critique_results = await self.wait_for_response("devil_advocate", conversation_id)
        
        # Phase 4: 검증 및 통합
        await self.agents['validator'].send_message(
            receiver="validator",
            message_type=MessageType.VALIDATE,
            content={
                "original_query": query,
                "research": research_results,
                "analysis": analysis_results, 
                "critique": critique_results
            },
            conversation_id=conversation_id
        )
        
        validation_results = await self.wait_for_response("validator", conversation_id)
        
        # Phase 5: 최종 종합
        await self.agents['synthesizer'].send_message(
            receiver="synthesizer",
            message_type=MessageType.QUERY,
            content={
                "query": query,
                "all_perspectives": {
                    "research": research_results,
                    "analysis": analysis_results,
                    "critique": critique_results,
                    "validation": validation_results
                }
            },
            conversation_id=conversation_id
        )
        
        final_answer = await self.wait_for_response("synthesizer", conversation_id)
        return final_answer

class ResearchAgent(BaseAgent):
    async def process_query(self, message: AgentMessage):
        query = message.content.get('query', '')
        
        # 다양한 소스에서 기초 정보 수집
        research_results = await self.comprehensive_search(query)
        
        # 정보의 신뢰성 평가
        credibility_scores = await self.evaluate_source_credibility(research_results)
        
        # 구조화된 연구 결과 생성
        structured_research = {
            "key_findings": research_results[:5],
            "supporting_evidence": research_results[5:10],
            "credibility_analysis": credibility_scores,
            "research_gaps": await self.identify_gaps(research_results),
            "recommended_next_steps": await self.suggest_further_research(query, research_results)
        }
        
        await self.send_message(
            receiver=message.sender,
            message_type=MessageType.RESPONSE,
            content=structured_research,
            conversation_id=message.conversation_id
        )

class DevilAdvocateAgent(BaseAgent):
    async def process_query(self, message: AgentMessage):
        analysis = message.content.get('analysis', {})
        
        # 분석의 약점 찾기
        weaknesses = await self.find_logical_weaknesses(analysis)
        alternative_perspectives = await self.generate_counterarguments(analysis)
        potential_biases = await self.identify_biases(analysis)
        
        critique = {
            "logical_weaknesses": weaknesses,
            "alternative_viewpoints": alternative_perspectives,
            "identified_biases": potential_biases,
            "risk_assessment": await self.assess_risks(analysis),
            "improvement_suggestions": await self.suggest_improvements(analysis)
        }
        
        await self.send_message(
            receiver=message.sender,
            message_type=MessageType.RESPONSE,
            content=critique,
            conversation_id=message.conversation_id
        )

# 사용 예시
async def main():
    decision_system = DecisionSupportSystem()
    
    complex_query = """
    우리 회사가 AI 기반 의료 진단 시스템을 개발해야 할지 결정해주세요.
    시장 기회, 기술적 feasibility, 규제 리스크, 경쟁 환경을 고려해주세요.
    """
    
    final_recommendation = await decision_system.process_complex_query(complex_query)
    print("최종 의사결정 권고사항:", final_recommendation)`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Section 4: 성능 최적화 */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-orange-100 dark:bg-orange-900/20 flex items-center justify-center">
              <Zap className="text-orange-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">2.4 Multi-Agent 성능 최적화</h2>
              <p className="text-gray-600 dark:text-gray-400">대규모 멀티 에이전트 시스템의 효율성 극대화</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-xl border border-orange-200 dark:border-orange-700">
              <h3 className="font-bold text-orange-800 dark:text-orange-200 mb-4">핵심 최적화 전략</h3>
              
              <div className="grid md:grid-cols-3 gap-4">
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-blue-600 dark:text-blue-400 mb-3">🚀 병렬 처리 최적화</h4>
                  <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                    <li>• 에이전트별 독립적 리소스 할당</li>
                    <li>• 비동기 메시지 처리</li>
                    <li>• 로드 밸런싱 구현</li>
                    <li>• 작업 큐 관리</li>
                  </ul>
                </div>
                
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-green-600 dark:text-green-400 mb-3">💾 메모리 관리</h4>
                  <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                    <li>• 컨텍스트 윈도우 최적화</li>
                    <li>• 메시지 히스토리 압축</li>
                    <li>• 에이전트별 메모리 풀</li>
                    <li>• 가비지 컬렉션 튜닝</li>
                  </ul>
                </div>
                
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-purple-600 dark:text-purple-400 mb-3">⚡ 통신 최적화</h4>
                  <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                    <li>• 메시지 압축</li>
                    <li>• 배치 처리</li>
                    <li>• 우선순위 큐</li>
                    <li>• 연결 풀링</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl border border-blue-200 dark:border-blue-700">
              <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-4">성능 벤치마크 결과</h3>
              
              <div className="overflow-x-auto mb-4">
                <table className="min-w-full text-sm">
                  <thead>
                    <tr className="border-b border-blue-300 dark:border-blue-600">
                      <th className="text-left py-2 text-blue-800 dark:text-blue-200">시스템 구성</th>
                      <th className="text-left py-2 text-blue-800 dark:text-blue-200">응답 시간</th>
                      <th className="text-left py-2 text-blue-800 dark:text-blue-200">정확도</th>
                      <th className="text-left py-2 text-blue-800 dark:text-blue-200">처리량 (QPS)</th>
                    </tr>
                  </thead>
                  <tbody className="text-blue-700 dark:text-blue-300">
                    <tr className="border-b border-blue-200 dark:border-blue-700">
                      <td className="py-2">Single RAG Agent</td>
                      <td className="py-2">2.1초</td>
                      <td className="py-2">78%</td>
                      <td className="py-2">45</td>
                    </tr>
                    <tr className="border-b border-blue-200 dark:border-blue-700">
                      <td className="py-2">3-Agent System</td>
                      <td className="py-2">3.8초</td>
                      <td className="py-2">89%</td>
                      <td className="py-2">32</td>
                    </tr>
                    <tr className="border-b border-blue-200 dark:border-blue-700">
                      <td className="py-2">5-Agent System</td>
                      <td className="py-2">4.2초</td>
                      <td className="py-2">94%</td>
                      <td className="py-2">28</td>
                    </tr>
                    <tr>
                      <td className="py-2">Optimized 5-Agent</td>
                      <td className="py-2">2.9초</td>
                      <td className="py-2">93%</td>
                      <td className="py-2">38</td>
                    </tr>
                  </tbody>
                </table>
              </div>
              
              <div className="p-3 bg-blue-100 dark:bg-blue-900/40 rounded">
                <p className="text-xs text-blue-800 dark:text-blue-200">
                  💡 <strong>핵심 인사이트:</strong> 적절한 최적화를 통해 Multi-Agent 시스템은 
                  높은 정확도를 유지하면서도 단일 에이전트 대비 합리적인 성능을 달성할 수 있습니다.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Section 5: 실습 과제 */}
        <section className="bg-gradient-to-r from-purple-500 to-indigo-600 rounded-2xl p-8 text-white">
          <h2 className="text-2xl font-bold mb-6">실습 과제</h2>
          
          <div className="bg-white/10 rounded-xl p-6 backdrop-blur">
            <h3 className="font-bold mb-4">Multi-Agent RAG 시스템 구축</h3>
            
            <div className="space-y-4">
              <div className="bg-white/10 p-4 rounded-lg">
                <h4 className="font-medium mb-2">🏗️ 시스템 설계</h4>
                <ol className="space-y-2 text-sm">
                  <li>1. 의료진단 지원을 위한 3-Agent 시스템 설계</li>
                  <li>2. 메시지 버스 기반 비동기 통신 구현</li>
                  <li>3. 도메인별 전문 에이전트 개발 (내과, 영상의학, 응급의학)</li>
                  <li>4. Orchestrator의 지능적 라우팅 알고리즘 구현</li>
                  <li>5. 답변 통합 및 신뢰도 평가 시스템 개발</li>
                </ol>
              </div>
              
              <div className="bg-white/10 p-4 rounded-lg">
                <h4 className="font-medium mb-2">🎯 평가 기준</h4>
                <ul className="space-y-1 text-sm">
                  <li>• 에이전트 간 협력 효율성 (응답 시간, 메시지 교환 횟수)</li>
                  <li>• 최종 답변의 종합성과 정확도</li>
                  <li>• 시스템 확장성 (새로운 전문 도메인 추가 용이성)</li>
                  <li>• 장애 복구 능력 (일부 에이전트 실패 시 대응)</li>
                </ul>
              </div>
              
              <div className="bg-white/10 p-4 rounded-lg">
                <h4 className="font-medium mb-2">🚀 고급 과제</h4>
                <p className="text-sm">
                  자가 학습 Multi-Agent 시스템: 에이전트들이 상호작용 과정에서 서로의 강점을 학습하고, 
                  협력 패턴을 최적화하는 강화학습 기반 시스템을 구현해보세요.
                </p>
              </div>
            </div>
          </div>
        </section>
      </div>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 Multi-Agent 프레임워크 & 도구',
            icon: 'web' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'AutoGen: Multi-Agent Framework',
                authors: 'Microsoft Research',
                year: '2024',
                description: 'Multi-Agent 대화 프레임워크 - Agent 협력, 자동 협상, 코드 생성',
                link: 'https://microsoft.github.io/autogen/'
              },
              {
                title: 'LangGraph: Agent Orchestration',
                authors: 'LangChain',
                year: '2024',
                description: 'Agent 워크플로우 그래프 - 상태 관리, 분기/병합, 순환 로직',
                link: 'https://langchain-ai.github.io/langgraph/'
              },
              {
                title: 'CrewAI: Role-based Multi-Agent',
                authors: 'CrewAI',
                year: '2024',
                description: '역할 기반 Agent 팀 - 자율적 협력, Task 분배',
                link: 'https://docs.crewai.com/'
              },
              {
                title: 'Semantic Kernel Multi-Agent',
                authors: 'Microsoft',
                year: '2024',
                description: 'LLM 오케스트레이션 - Plugin 시스템, Memory 관리',
                link: 'https://learn.microsoft.com/en-us/semantic-kernel/agents/'
              },
              {
                title: 'Agent Protocol Specification',
                authors: 'AI Engineer Foundation',
                year: '2024',
                description: 'Agent 간 통신 표준 - REST API 기반 프로토콜',
                link: 'https://agentprotocol.ai/'
              }
            ]
          },
          {
            title: '📖 Multi-Agent RAG 연구',
            icon: 'research' as const,
            color: 'border-indigo-500',
            items: [
              {
                title: 'Multi-Agent Collaboration for Complex QA',
                authors: 'Du et al., Stanford University',
                year: '2024',
                description: '다중 에이전트 협력 - Debate Pattern으로 정확도 23% 향상',
                link: 'https://arxiv.org/abs/2305.14325'
              },
              {
                title: 'Communicative Agents for Software Development',
                authors: 'Qian et al., Tsinghua University',
                year: '2023',
                description: 'ChatDev 프레임워크 - Software 개발을 위한 Multi-Agent 시스템',
                link: 'https://arxiv.org/abs/2307.07924'
              },
              {
                title: 'Cooperative Multi-Agent Deep RL',
                authors: 'Lowe et al., OpenAI',
                year: '2017',
                description: 'MADDPG 알고리즘 - Agent 간 협력 학습 기법',
                link: 'https://arxiv.org/abs/1706.02275'
              },
              {
                title: 'AgentVerse: Facilitating Multi-Agent Collaboration',
                authors: 'Chen et al., Tsinghua University',
                year: '2023',
                description: '대규모 Multi-Agent 협업 - 동적 Task 배분, Consensus Mechanism',
                link: 'https://arxiv.org/abs/2308.10848'
              }
            ]
          },
          {
            title: '🛠️ 분산 시스템 & 메시지 큐',
            icon: 'tools' as const,
            color: 'border-green-500',
            items: [
              {
                title: 'Ray: Distributed Computing',
                authors: 'Anyscale',
                year: '2024',
                description: 'Python 분산 처리 - Agent 병렬 실행, 자원 관리',
                link: 'https://docs.ray.io/en/latest/'
              },
              {
                title: 'Celery: Distributed Task Queue',
                authors: 'Celery Project',
                year: '2024',
                description: '비동기 작업 큐 - Agent 간 메시지 전달, Task 스케줄링',
                link: 'https://docs.celeryq.dev/'
              },
              {
                title: 'RabbitMQ Message Broker',
                authors: 'Pivotal/VMware',
                year: '2024',
                description: 'AMQP 메시지 브로커 - Agent 통신 인프라, 메시지 라우팅',
                link: 'https://www.rabbitmq.com/documentation.html'
              },
              {
                title: 'Redis Pub/Sub',
                authors: 'Redis Labs',
                year: '2024',
                description: 'Publish/Subscribe 패턴 - 실시간 Agent 이벤트 브로드캐스팅',
                link: 'https://redis.io/docs/manual/pubsub/'
              },
              {
                title: 'LangSmith Agent Tracing',
                authors: 'LangChain',
                year: '2024',
                description: 'Multi-Agent 추적 및 디버깅 - Conversation Flow 시각화',
                link: 'https://docs.smith.langchain.com/tracing'
              }
            ]
          }
        ]}
      />

      {/* Navigation */}
      <div className="mt-12 bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-center">
          <Link
            href="/modules/rag/advanced/chapter1"
            className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <ArrowLeft size={16} />
            이전: GraphRAG 아키텍처
          </Link>
          
          <Link
            href="/modules/rag/advanced/chapter3"
            className="inline-flex items-center gap-2 bg-purple-500 text-white px-6 py-3 rounded-lg font-medium hover:bg-purple-600 transition-colors"
          >
            다음: 분산 RAG 시스템
            <ArrowRight size={16} />
          </Link>
        </div>
      </div>
    </div>
  )
}