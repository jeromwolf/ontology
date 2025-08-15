'use client';

export default function Chapter8() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          LangChain & AutoGen
        </h2>
        
        <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-6 mb-6">
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
            LangChain과 AutoGen을 활용하여 복잡한 AI 에이전트 시스템을 구축합니다.
            여러 AI 모델과 도구를 조합하여 자율적으로 작업을 수행하는 에이전트를 만들 수 있습니다.
          </p>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🔗 LangChain 프레임워크
        </h3>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700 mb-6">
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">핵심 컴포넌트</h4>
              <div className="space-y-2">
                <div className="bg-purple-50 dark:bg-purple-900/20 rounded p-3">
                  <h5 className="font-semibold text-purple-700 dark:text-purple-400 mb-1">Models</h5>
                  <p className="text-xs text-gray-600 dark:text-gray-400">LLM, Chat, Embeddings</p>
                </div>
                <div className="bg-blue-50 dark:bg-blue-900/20 rounded p-3">
                  <h5 className="font-semibold text-blue-700 dark:text-blue-400 mb-1">Prompts</h5>
                  <p className="text-xs text-gray-600 dark:text-gray-400">템플릿, 예시 선택기</p>
                </div>
                <div className="bg-green-50 dark:bg-green-900/20 rounded p-3">
                  <h5 className="font-semibold text-green-700 dark:text-green-400 mb-1">Memory</h5>
                  <p className="text-xs text-gray-600 dark:text-gray-400">대화 기록, 요약</p>
                </div>
                <div className="bg-orange-50 dark:bg-orange-900/20 rounded p-3">
                  <h5 className="font-semibold text-orange-700 dark:text-orange-400 mb-1">Chains</h5>
                  <p className="text-xs text-gray-600 dark:text-gray-400">순차/병렬 실행</p>
                </div>
              </div>
            </div>
            
            <div>
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">에이전트 타입</h4>
              <div className="space-y-2">
                <div className="bg-gray-50 dark:bg-gray-900 rounded p-3">
                  <h5 className="font-semibold text-gray-700 dark:text-gray-300 mb-1">ReAct</h5>
                  <p className="text-xs text-gray-600 dark:text-gray-400">추론과 행동 반복</p>
                </div>
                <div className="bg-gray-50 dark:bg-gray-900 rounded p-3">
                  <h5 className="font-semibold text-gray-700 dark:text-gray-300 mb-1">Self-Ask</h5>
                  <p className="text-xs text-gray-600 dark:text-gray-400">자가 질문 생성</p>
                </div>
                <div className="bg-gray-50 dark:bg-gray-900 rounded p-3">
                  <h5 className="font-semibold text-gray-700 dark:text-gray-300 mb-1">Plan-and-Execute</h5>
                  <p className="text-xs text-gray-600 dark:text-gray-400">계획 후 실행</p>
                </div>
                <div className="bg-gray-50 dark:bg-gray-900 rounded p-3">
                  <h5 className="font-semibold text-gray-700 dark:text-gray-300 mb-1">OpenAI Functions</h5>
                  <p className="text-xs text-gray-600 dark:text-gray-400">함수 호출 에이전트</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🤖 AutoGen 멀티 에이전트
        </h3>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            Microsoft의 AutoGen은 여러 AI 에이전트가 협업하는 시스템을 쉽게 구축할 수 있게 합니다.
          </p>
          
          <div className="bg-gray-900 rounded-lg p-4 mb-4">
            <pre className="text-green-400 font-mono text-xs overflow-x-auto">
{`import autogen

# 에이전트 설정
config_list = [{
    "model": "gpt-4",
    "api_key": "your-api-key"
}]

# 어시스턴트 에이전트
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={"config_list": config_list},
    system_message="You are a helpful AI assistant."
)

# 사용자 프록시 에이전트
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={"work_dir": "coding"}
)

# 대화 시작
user_proxy.initiate_chat(
    assistant,
    message="Create a snake game in Python"
)`}</pre>
          </div>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
              <h5 className="font-bold text-purple-700 dark:text-purple-400 mb-2">장점</h5>
              <ul className="space-y-1 text-xs text-gray-700 dark:text-gray-300">
                <li>• 자동 코드 실행</li>
                <li>• 에이전트 간 자율 대화</li>
                <li>• 복잡한 작업 분해</li>
                <li>• 피드백 루프</li>
              </ul>
            </div>
            
            <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
              <h5 className="font-bold text-blue-700 dark:text-blue-400 mb-2">활용 사례</h5>
              <ul className="space-y-1 text-xs text-gray-700 dark:text-gray-300">
                <li>• 코드 생성 및 디버깅</li>
                <li>• 데이터 분석</li>
                <li>• 연구 논문 작성</li>
                <li>• 프로젝트 계획</li>
              </ul>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}