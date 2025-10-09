import { MessageSquare } from 'lucide-react'

export default function Section3() {
  return (
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
  )
}
