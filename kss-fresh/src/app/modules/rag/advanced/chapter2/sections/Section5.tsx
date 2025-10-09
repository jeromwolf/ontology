import References from '@/components/common/References'

export default function Section5() {
  return (
    <>
      {/* 실습 과제 */}
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
    </>
  )
}
