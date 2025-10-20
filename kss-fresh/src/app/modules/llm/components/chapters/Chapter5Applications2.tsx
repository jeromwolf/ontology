'use client';

import References from '@/components/common/References';

export default function Chapter5Applications2() {
  return (
    <div className="space-y-8">
      {/* 페이지 헤더 */}
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold mb-4">Chapter 5-2: 질문 답변 & 대화 AI</h1>
        <p className="text-lg text-gray-600 dark:text-gray-400">
          Retrieval QA부터 Conversational AI까지 - 차세대 대화 시스템 구축
        </p>
      </div>

      <section>
        <h2 className="text-2xl font-bold text-indigo-800 dark:text-indigo-200 mb-4">
          질문 답변 시스템 (Question Answering)
        </h2>

        <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-xl p-6 mb-6">
          <h3 className="text-xl font-semibold text-indigo-700 dark:text-indigo-300 mb-4">
            QA 시스템의 3가지 유형
          </h3>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-bold text-indigo-600 dark:text-indigo-400 mb-2">
                1. Extractive QA
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                문서에서 정확한 답변 추출
              </p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• BERT, RoBERTa 기반</li>
                <li>• SQuAD 데이터셋</li>
                <li>• 정확도 높음</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-bold text-purple-600 dark:text-purple-400 mb-2">
                2. Generative QA
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                새로운 답변 생성
              </p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• GPT, T5 기반</li>
                <li>• 유연한 응답</li>
                <li>• 창의적 답변</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-bold text-blue-600 dark:text-blue-400 mb-2">
                3. Retrieval QA
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                검색 + 생성 결합 (RAG)
              </p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• 벡터 DB + LLM</li>
                <li>• 실시간 정보</li>
                <li>• 소스 추적 가능</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6 mb-6">
          <h3 className="text-xl font-semibold text-blue-700 dark:text-blue-300 mb-4">
            RAG 기반 QA 시스템 구축
          </h3>
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-bold text-blue-600 dark:text-blue-400 mb-2">
                LangChain RetrievalQA
              </h4>
              <pre className="bg-gray-50 dark:bg-gray-900 p-3 rounded text-sm overflow-x-auto">
{`from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI

# 벡터 스토어 로드
vectorstore = Pinecone.from_existing_index(
    index_name="my-docs",
    embedding=OpenAIEmbeddings()
)

# RetrievalQA 체인 구성
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4"),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(k=5),
    return_source_documents=True
)

# 질문 답변
result = qa_chain({"query": "RAG의 장점은?"})
print(result["result"])
print(result["source_documents"])`}
              </pre>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold text-indigo-800 dark:text-indigo-200 mb-4">
          대화형 AI (Conversational AI)
        </h2>

        <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-xl p-6 mb-6">
          <h3 className="text-xl font-semibold text-green-700 dark:text-green-300 mb-4">
            최신 챗봇 플랫폼
          </h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-bold text-green-600 dark:text-green-400 mb-2">
                ChatGPT & GPT-4o
              </h4>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• 멀티턴 대화 컨텍스트 유지</li>
                <li>• Function calling으로 외부 API 연동</li>
                <li>• Assistants API로 커스텀 챗봇 구축</li>
                <li>• Memory 관리 (최대 128K 토큰)</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-bold text-emerald-600 dark:text-emerald-400 mb-2">
                Claude 3 & Anthropic API
              </h4>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• Constitutional AI로 안전한 응답</li>
                <li>• 200K 토큰 장문 대화</li>
                <li>• System prompt 최적화</li>
                <li>• 사고 과정 추적 (Chain of Thought)</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-6 mb-6">
          <h3 className="text-xl font-semibold text-purple-700 dark:text-purple-300 mb-4">
            오픈소스 대화 프레임워크
          </h3>
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-bold text-purple-600 dark:text-purple-400 mb-2">
                Rasa Conversational AI
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                엔터프라이즈급 챗봇 개발 프레임워크
              </p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• NLU (자연어 이해) + Dialogue Management</li>
                <li>• Intent classification & Entity extraction</li>
                <li>• Custom actions & Forms</li>
                <li>• On-premise 배포 가능</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-bold text-pink-600 dark:text-pink-400 mb-2">
                LangChain Conversational Agents
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                LLM 기반 대화 에이전트 구축
              </p>
              <pre className="bg-gray-50 dark:bg-gray-900 p-3 rounded text-sm overflow-x-auto mt-2">
{`from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=ChatOpenAI(temperature=0.7),
    memory=memory,
    verbose=True
)

# 대화 시작
conversation.predict(input="안녕하세요")
conversation.predict(input="RAG에 대해 설명해주세요")`}
              </pre>
            </div>
          </div>
        </div>

        <div className="bg-orange-50 dark:bg-orange-900/20 rounded-xl p-6">
          <h3 className="text-xl font-semibold text-orange-700 dark:text-orange-300 mb-4">
            실전 챗봇 아키텍처
          </h3>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-2">
                핵심 컴포넌트
              </h4>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• Intent Classifier (의도 분류)</li>
                <li>• Entity Extractor (개체명 인식)</li>
                <li>• Dialogue Manager (대화 흐름 관리)</li>
                <li>• Response Generator (응답 생성)</li>
                <li>• Memory Store (대화 기록)</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-semibold text-amber-600 dark:text-amber-400 mb-2">
                고급 기능
              </h4>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• Multi-turn context tracking</li>
                <li>• Slot filling & Form validation</li>
                <li>• Fallback handling</li>
                <li>• Sentiment analysis</li>
                <li>• A/B testing</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* 학습 요약 */}
      <section className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-xl p-6 mt-8">
        <h2 className="text-xl font-bold mb-4 text-indigo-800 dark:text-indigo-200">
          📚 이 챕터에서 배운 것
        </h2>
        <ul className="space-y-2">
          <li className="flex items-start gap-2">
            <span className="text-indigo-600 dark:text-indigo-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              Extractive, Generative, Retrieval QA 시스템의 차이와 각각의 장단점
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-indigo-600 dark:text-indigo-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              LangChain RetrievalQA를 활용한 RAG 기반 질문 답변 시스템 구축
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-indigo-600 dark:text-indigo-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              ChatGPT Assistants API와 Claude API의 대화형 AI 구현 방법
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-indigo-600 dark:text-indigo-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              Rasa 프레임워크를 활용한 엔터프라이즈급 챗봇 아키텍처 설계
            </span>
          </li>
        </ul>
      </section>

      <References
        sections={[
          {
            title: '📚 핵심 논문 & 연구',
            icon: 'paper',
            color: 'border-indigo-500',
            items: [
              {
                title: 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks',
                authors: 'Patrick Lewis et al.',
                year: '2020',
                description: 'RAG 기반 질문 답변 시스템의 기초 논문',
                link: 'https://arxiv.org/abs/2005.11401'
              },
              {
                title: 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding',
                authors: 'Jacob Devlin et al.',
                year: '2018',
                description: 'Extractive QA의 기반이 된 BERT 모델',
                link: 'https://arxiv.org/abs/1810.04805'
              },
              {
                title: 'Dense Passage Retrieval for Open-Domain Question Answering',
                authors: 'Vladimir Karpukhin et al.',
                year: '2020',
                description: '밀집 벡터 검색 기반 QA 시스템',
                link: 'https://arxiv.org/abs/2004.04906'
              },
              {
                title: 'Constitutional AI: Harmlessness from AI Feedback',
                authors: 'Yuntao Bai et al. (Anthropic)',
                year: '2022',
                description: '안전한 대화형 AI 구축을 위한 Claude의 핵심 기술',
                link: 'https://arxiv.org/abs/2212.08073'
              }
            ]
          },
          {
            title: '🔬 벤치마크 & 데이터셋',
            icon: 'book',
            color: 'border-purple-500',
            items: [
              {
                title: 'SQuAD: 100,000+ Questions for Machine Comprehension of Text',
                authors: 'Pranav Rajpurkar et al.',
                year: '2016',
                description: 'Extractive QA의 표준 벤치마크 데이터셋',
                link: 'https://arxiv.org/abs/1606.05250'
              },
              {
                title: 'Natural Questions: A Benchmark for Question Answering Research',
                authors: 'Tom Kwiatkowski et al.',
                year: '2019',
                description: 'Google의 실제 검색 질의 기반 QA 데이터셋',
                link: 'https://ai.google.com/research/NaturalQuestions'
              },
              {
                title: 'MS MARCO: A Human Generated MAchine Reading COmprehension Dataset',
                authors: 'Payal Bajaj et al.',
                year: '2018',
                description: 'Microsoft의 대규모 검색 기반 QA 데이터셋',
                link: 'https://microsoft.github.io/msmarco/'
              }
            ]
          },
          {
            title: '🛠️ 프레임워크 & 도구',
            icon: 'web',
            color: 'border-blue-500',
            items: [
              {
                title: 'LangChain: RetrievalQA Documentation',
                description: 'RAG 기반 질문 답변 체인 구축 가이드',
                link: 'https://python.langchain.com/docs/use_cases/question_answering/'
              },
              {
                title: 'OpenAI Assistants API',
                description: '커스텀 챗봇 구축을 위한 OpenAI Assistants',
                link: 'https://platform.openai.com/docs/assistants/overview'
              },
              {
                title: 'Anthropic Claude API: Conversations',
                description: 'Claude를 활용한 대화형 AI 구현',
                link: 'https://docs.anthropic.com/claude/docs/intro-to-claude'
              },
              {
                title: 'Rasa Open Source Documentation',
                description: '엔터프라이즈급 챗봇 프레임워크',
                link: 'https://rasa.com/docs/rasa/'
              },
              {
                title: 'Hugging Face: Question Answering',
                description: 'BERT, RoBERTa 기반 QA 모델 구현',
                link: 'https://huggingface.co/tasks/question-answering'
              }
            ]
          },
          {
            title: '🏢 실전 응용 사례',
            icon: 'web',
            color: 'border-green-500',
            items: [
              {
                title: 'ChatGPT: Conversational AI Platform',
                description: 'OpenAI의 대화형 AI 서비스',
                link: 'https://chat.openai.com/'
              },
              {
                title: 'Perplexity AI: Conversational Search',
                description: 'RAG 기반 대화형 검색 엔진',
                link: 'https://www.perplexity.ai/'
              },
              {
                title: 'Notion AI: Document Q&A',
                description: '문서 기반 질문 답변 어시스턴트',
                link: 'https://www.notion.so/product/ai'
              },
              {
                title: 'GitHub Copilot Chat',
                description: '코드 이해 및 생성을 위한 대화형 AI',
                link: 'https://github.com/features/copilot'
              }
            ]
          }
        ]}
      />
    </div>
  );
}
