'use client';

import References from '@/components/common/References';

export default function Chapter5Applications1() {
  return (
    <div className="space-y-8">
      {/* 페이지 헤더 */}
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold mb-4">Chapter 5-1: 텍스트 생성 & 요약</h1>
        <p className="text-lg text-gray-600 dark:text-gray-400">
          LLM의 핵심 응용: 창의적 텍스트 생성부터 고급 문서 요약까지
        </p>
      </div>

      <section>
        <h2 className="text-2xl font-bold text-indigo-800 dark:text-indigo-200 mb-4">
          텍스트 생성의 혁명
        </h2>

        <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-xl p-6 mb-6">
          <h3 className="text-xl font-semibold text-indigo-700 dark:text-indigo-300 mb-4">
            최신 생성 모델 (2024-2025)
          </h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-bold text-indigo-600 dark:text-indigo-400 mb-2">GPT-4 Turbo & GPT-4o</h4>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• 128K 토큰 컨텍스트 윈도우</li>
                <li>• JSON mode, function calling</li>
                <li>• 창의적 글쓰기, 시나리오 작성</li>
                <li>• 다국어 번역 (100+ 언어)</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-bold text-purple-600 dark:text-purple-400 mb-2">Claude 3.5 Sonnet</h4>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• 200K 토큰 컨텍스트</li>
                <li>• 장문 문서 분석 & 요약</li>
                <li>• Constitutional AI 기반 안전성</li>
                <li>• 기술 문서 작성 특화</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6 mb-6">
          <h3 className="text-xl font-semibold text-blue-700 dark:text-blue-300 mb-4">
            문서 요약 기법
          </h3>
          <div className="space-y-4">
            <div className="border-l-4 border-blue-500 pl-4">
              <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">
                1. Extractive Summarization (추출적 요약)
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                원문에서 중요 문장을 선택하여 요약 생성. BERT, ROUGE 스코어 활용.
              </p>
            </div>
            <div className="border-l-4 border-indigo-500 pl-4">
              <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">
                2. Abstractive Summarization (생성적 요약)
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                원문 내용을 이해하고 새로운 문장으로 재구성. BART, T5, GPT 활용.
              </p>
            </div>
            <div className="border-l-4 border-purple-500 pl-4">
              <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">
                3. Hybrid Summarization (하이브리드)
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                추출과 생성을 결합한 2단계 접근법. Pegasus, ProphetNet 모델.
              </p>
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-xl p-6">
          <h3 className="text-xl font-semibold text-green-700 dark:text-green-300 mb-4">
            실전 응용 사례
          </h3>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">
                콘텐츠 마케팅
              </h4>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• 블로그 포스트 자동 생성</li>
                <li>• SEO 최적화 메타 설명</li>
                <li>• 소셜 미디어 캡션</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-semibold text-emerald-600 dark:text-emerald-400 mb-2">
                기업 문서
              </h4>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• 회의록 자동 요약</li>
                <li>• 보고서 초안 작성</li>
                <li>• 이메일 자동 응답</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-semibold text-teal-600 dark:text-teal-400 mb-2">
                연구 & 교육
              </h4>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• 논문 Abstract 생성</li>
                <li>• 강의 자료 요약</li>
                <li>• 문헌 리뷰 자동화</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold text-indigo-800 dark:text-indigo-200 mb-4">
          번역의 미래
        </h2>

        <div className="bg-gradient-to-r from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-xl p-6">
          <h3 className="text-xl font-semibold text-blue-700 dark:text-blue-300 mb-4">
            신경망 기계 번역 (NMT)
          </h3>
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-bold text-blue-600 dark:text-blue-400 mb-2">
                Google Translate Neural MT
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                Transformer 기반 다국어 번역 시스템 (133개 언어 지원)
              </p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• Zero-shot translation (직접 학습 없이 언어쌍 번역)</li>
                <li>• Context-aware translation (문맥 기반)</li>
                <li>• BLEU score 40+ 달성</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-bold text-cyan-600 dark:text-cyan-400 mb-2">
                DeepL & LLM 번역
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                고품질 번역 서비스와 GPT-4/Claude의 번역 능력
              </p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• 자연스러운 문체 번역 (Style preservation)</li>
                <li>• 전문 용어 처리 (Domain-specific)</li>
                <li>• 문화적 뉘앙스 반영</li>
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
              GPT-4 Turbo, Claude 3.5, Gemini 등 최신 생성 모델의 특징과 활용법
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-indigo-600 dark:text-indigo-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              Extractive vs Abstractive 요약 기법의 차이와 실무 적용 방법
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-indigo-600 dark:text-indigo-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              BART, T5 등 전문 요약 모델의 아키텍처와 성능 비교
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-indigo-600 dark:text-indigo-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              Neural Machine Translation의 최신 기술과 실전 번역 서비스
            </span>
          </li>
        </ul>
      </section>

      <References
        sections={[
          {
            title: '📚 핵심 논문 & 모델',
            icon: 'paper',
            color: 'border-indigo-500',
            items: [
              {
                title: 'GPT-4 Technical Report',
                authors: 'OpenAI',
                year: '2024',
                description: 'GPT-4 Turbo와 GPT-4o의 아키텍처 및 성능 분석',
                link: 'https://arxiv.org/abs/2303.08774'
              },
              {
                title: 'Introducing Claude 3.5 Sonnet',
                authors: 'Anthropic',
                year: '2024',
                description: '200K 컨텍스트와 고급 문서 분석 능력을 갖춘 Claude 3.5',
                link: 'https://www.anthropic.com/news/claude-3-5-sonnet'
              },
              {
                title: 'BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation',
                authors: 'Mike Lewis et al.',
                year: '2019',
                description: 'Meta AI의 텍스트 생성 및 요약 모델',
                link: 'https://arxiv.org/abs/1910.13461'
              },
              {
                title: 'Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)',
                authors: 'Colin Raffel et al.',
                year: '2020',
                description: 'Google의 통합 텍스트 변환 모델',
                link: 'https://arxiv.org/abs/1910.10683'
              },
              {
                title: 'Pegasus: Pre-training with Extracted Gap-sentences for Abstractive Summarization',
                authors: 'Jingqing Zhang et al.',
                year: '2020',
                description: 'Google Research의 고성능 요약 모델',
                link: 'https://arxiv.org/abs/1912.08777'
              }
            ]
          },
          {
            title: '🔬 번역 & 다국어 모델',
            icon: 'paper',
            color: 'border-purple-500',
            items: [
              {
                title: 'Google\'s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation',
                authors: 'Yonghui Wu et al.',
                year: '2016',
                description: 'Google Translate의 신경망 기반 번역 시스템',
                link: 'https://arxiv.org/abs/1609.08144'
              },
              {
                title: 'Attention Is All You Need',
                authors: 'Vaswani et al.',
                year: '2017',
                description: 'Transformer 아키텍처 - 현대 번역 시스템의 기초',
                link: 'https://arxiv.org/abs/1706.03762'
              },
              {
                title: 'Multilingual Denoising Pre-training for Neural Machine Translation',
                authors: 'Yinhan Liu et al.',
                year: '2020',
                description: 'mBART: 다국어 요약 및 번역 모델',
                link: 'https://arxiv.org/abs/2001.08210'
              }
            ]
          },
          {
            title: '🛠️ 실전 도구 & API',
            icon: 'web',
            color: 'border-blue-500',
            items: [
              {
                title: 'OpenAI GPT-4 API Documentation',
                description: '텍스트 생성, 요약, 번역을 위한 GPT-4 API 가이드',
                link: 'https://platform.openai.com/docs/guides/text-generation'
              },
              {
                title: 'Anthropic Claude API',
                description: '100K+ 토큰 장문 문서 요약에 최적화된 Claude API',
                link: 'https://docs.anthropic.com/claude/docs'
              },
              {
                title: 'Hugging Face Transformers: Summarization',
                description: 'BART, T5, Pegasus 등 요약 모델 구현 예제',
                link: 'https://huggingface.co/docs/transformers/tasks/summarization'
              },
              {
                title: 'Google Cloud Translation API',
                description: 'Neural MT 기반 실시간 번역 서비스',
                link: 'https://cloud.google.com/translate/docs'
              },
              {
                title: 'DeepL API Documentation',
                description: '고품질 전문 번역 API',
                link: 'https://www.deepl.com/docs-api'
              }
            ]
          },
          {
            title: '📊 벤치마크 & 평가',
            icon: 'book',
            color: 'border-green-500',
            items: [
              {
                title: 'ROUGE: A Package for Automatic Evaluation of Summaries',
                description: '요약 품질 평가를 위한 표준 메트릭',
                link: 'https://aclanthology.org/W04-1013/'
              },
              {
                title: 'CNN/Daily Mail Dataset',
                description: '뉴스 기사 요약 벤치마크 데이터셋',
                link: 'https://huggingface.co/datasets/cnn_dailymail'
              },
              {
                title: 'WMT (Workshop on Machine Translation)',
                description: '기계 번역 성능 평가 국제 대회',
                link: 'https://www.statmt.org/wmt24/'
              }
            ]
          }
        ]}
      />
    </div>
  );
}
