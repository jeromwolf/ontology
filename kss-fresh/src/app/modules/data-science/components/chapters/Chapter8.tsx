'use client';

import React from 'react';
import { BookOpen, MessageSquare, Brain, Code, Lightbulb } from 'lucide-react';
import References from '@/components/common/References';

interface Chapter8Props {
  onComplete?: () => void
}

export default function Chapter8({ onComplete }: Chapter8Props) {
  return (
    <div className="max-w-4xl mx-auto p-6 space-y-8">
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold text-primary">Chapter 8: 자연어 처리(NLP) 기초</h1>
        <p className="text-xl text-muted-foreground">
          텍스트 데이터를 이해하고 분석하는 AI 기술을 학습합니다
        </p>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 px-6 py-4">
          <h2 className="text-xl font-semibold flex items-center gap-2">
            <MessageSquare className="w-6 h-6 text-blue-600 dark:text-blue-400" />
            자연어 처리란?
          </h2>
        </div>
        <div className="p-6 space-y-4">
          <div className="space-y-2">
            <h3 className="text-lg font-semibold">정의</h3>
            <p className="text-gray-600 dark:text-gray-400">
              자연어 처리(Natural Language Processing)는 컴퓨터가 인간의 언어를 
              이해하고 해석하며 생성할 수 있도록 하는 AI의 한 분야입니다.
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4">
              <h4 className="font-semibold mb-2">주요 작업</h4>
              <ul className="space-y-2 list-disc list-inside text-sm">
                <li>텍스트 분류 (Text Classification)</li>
                <li>감성 분석 (Sentiment Analysis)</li>
                <li>개체명 인식 (NER)</li>
                <li>기계 번역 (Machine Translation)</li>
                <li>질의응답 (Question Answering)</li>
              </ul>
            </div>
            
            <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4">
              <h4 className="font-semibold mb-2">응용 분야</h4>
              <ul className="space-y-2 list-disc list-inside text-sm">
                <li>챗봇 & 가상 비서</li>
                <li>검색 엔진</li>
                <li>텍스트 요약</li>
                <li>콘텐츠 추천</li>
                <li>문서 분석</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
        <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 px-6 py-4">
          <h2 className="text-xl font-semibold flex items-center gap-2">
            <Brain className="w-6 h-6 text-purple-600 dark:text-purple-400" />
            텍스트 전처리
          </h2>
        </div>
        <div className="p-6 space-y-6">
          <div className="space-y-4">
            <div className="border-l-4 border-blue-500 pl-4">
              <h4 className="font-semibold">1. 토큰화 (Tokenization)</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                텍스트를 의미 있는 단위(토큰)로 분할합니다.
              </p>
              <pre className="bg-gray-100 dark:bg-gray-900 p-2 rounded text-xs mt-2">
                "나는 AI를 공부한다" → ["나는", "AI를", "공부한다"]
              </pre>
            </div>
            
            <div className="border-l-4 border-purple-500 pl-4">
              <h4 className="font-semibold">2. 정규화 (Normalization)</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                대소문자 통일, 특수문자 제거, 숫자 처리 등을 수행합니다.
              </p>
              <pre className="bg-gray-100 dark:bg-gray-900 p-2 rounded text-xs mt-2">
                "Hello, World! 123" → "hello world"
              </pre>
            </div>
            
            <div className="border-l-4 border-pink-500 pl-4">
              <h4 className="font-semibold">3. 불용어 제거 (Stopword Removal)</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                의미가 적은 단어들을 제거합니다.
              </p>
              <pre className="bg-gray-100 dark:bg-gray-900 p-2 rounded text-xs mt-2">
                ["나는", "매우", "좋은", "책을", "읽었다"] → ["좋은", "책", "읽었다"]
              </pre>
            </div>
            
            <div className="border-l-4 border-gray-500 pl-4">
              <h4 className="font-semibold">4. 어간 추출/표제어 추출</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                단어를 기본형으로 변환합니다.
              </p>
              <pre className="bg-gray-100 dark:bg-gray-900 p-2 rounded text-xs mt-2">
                "running", "runs", "ran" → "run"
              </pre>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
        <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 px-6 py-4">
          <h2 className="text-xl font-semibold flex items-center gap-2">
            <Code className="w-6 h-6 text-green-600 dark:text-green-400" />
            텍스트 표현 방법
          </h2>
        </div>
        <div className="p-6 space-y-6">
          <div className="space-y-4">
            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <h4 className="font-semibold mb-2">Bag of Words (BoW)</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                단어의 출현 빈도만을 고려하는 가장 단순한 방법입니다.
              </p>
              <pre className="bg-gray-100 dark:bg-gray-900 p-2 rounded text-xs overflow-x-auto">
{`문서1: "나는 사과를 좋아한다"
문서2: "나는 바나나를 좋아한다"

BoW: {나는: [1,1], 사과를: [1,0], 바나나를: [0,1], 좋아한다: [1,1]}`}</pre>
            </div>
            
            <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
              <h4 className="font-semibold mb-2">TF-IDF</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                단어의 중요도를 빈도와 희귀성을 고려하여 계산합니다.
              </p>
              <pre className="bg-gray-100 dark:bg-gray-900 p-2 rounded text-xs overflow-x-auto">
{`TF(단어 빈도) × IDF(역문서 빈도)
= (단어 출현 횟수 / 문서 내 전체 단어 수) × log(전체 문서 수 / 단어가 포함된 문서 수)`}</pre>
            </div>
            
            <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
              <h4 className="font-semibold mb-2">Word Embeddings</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                단어를 의미를 담은 밀집 벡터로 표현합니다.
              </p>
              <ul className="text-sm space-y-1 mt-2 list-disc list-inside">
                <li>Word2Vec: CBOW, Skip-gram</li>
                <li>GloVe: 전역 행렬 분해</li>
                <li>FastText: 서브워드 정보 활용</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
        <div className="bg-gradient-to-r from-indigo-50 to-blue-50 dark:from-indigo-900/20 dark:to-blue-900/20 px-6 py-4">
          <h2 className="text-xl font-semibold flex items-center gap-2">
            <BookOpen className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />
            Python으로 NLP 시작하기
          </h2>
        </div>
        <div className="p-6 space-y-4">
          <div>
            <h4 className="font-semibold mb-2">1. 기본 텍스트 처리</h4>
            <pre className="bg-gray-100 dark:bg-gray-900 p-4 rounded-lg text-sm overflow-x-auto">
{`import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Okt

# 한국어 형태소 분석
okt = Okt()
text = "나는 자연어 처리를 공부하고 있습니다."
tokens = okt.morphs(text)
print(tokens)  # ['나', '는', '자연어', '처리', '를', '공부', '하고', '있습니다', '.']

# 품사 태깅
pos_tags = okt.pos(text)
print(pos_tags)  # [('나', 'Noun'), ('는', 'Josa'), ...]`}</pre>
          </div>
          
          <div>
            <h4 className="font-semibold mb-2">2. 감성 분석 예제</h4>
            <pre className="bg-gray-100 dark:bg-gray-900 p-4 rounded-lg text-sm overflow-x-auto">
{`from transformers import pipeline

# 사전 학습된 감성 분석 모델 로드
sentiment_analyzer = pipeline("sentiment-analysis", 
                             model="nlptown/bert-base-multilingual-uncased-sentiment")

# 텍스트 감성 분석
reviews = [
    "이 제품은 정말 훌륭합니다!",
    "별로예요. 다시는 구매하지 않을 것 같아요.",
    "그저 그래요. 가격 대비 괜찮은 편입니다."
]

for review in reviews:
    result = sentiment_analyzer(review)
    print(f"텍스트: {review}")
    print(f"감성: {result[0]['label']}, 신뢰도: {result[0]['score']:.2f}\\n")`}</pre>
          </div>
          
          <div>
            <h4 className="font-semibold mb-2">3. 단어 임베딩 활용</h4>
            <pre className="bg-gray-100 dark:bg-gray-900 p-4 rounded-lg text-sm overflow-x-auto">
{`from gensim.models import Word2Vec
import numpy as np

# 문장 토큰화
sentences = [
    ["나는", "파이썬을", "좋아한다"],
    ["나는", "머신러닝을", "공부한다"],
    ["파이썬은", "프로그래밍", "언어이다"],
    ["머신러닝은", "인공지능의", "한", "분야이다"]
]

# Word2Vec 모델 학습
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# 단어 벡터 확인
vector = model.wv['파이썬']
print(f"'파이썬' 벡터 차원: {vector.shape}")

# 유사 단어 찾기
similar_words = model.wv.most_similar('파이썬', topn=3)
print("'파이썬'과 유사한 단어:", similar_words)`}</pre>
          </div>
        </div>
      </div>

      <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4 flex gap-3">
        <Lightbulb className="w-5 h-5 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
        <div>
          <strong>최신 트렌드:</strong> Transformer 기반 모델들이 NLP를 혁신하고 있습니다:
          <ul className="mt-2 space-y-1 list-disc list-inside">
            <li>BERT: 양방향 사전학습으로 문맥 이해 향상</li>
            <li>GPT: 강력한 텍스트 생성 능력</li>
            <li>T5: 모든 NLP 작업을 텍스트 생성으로 통합</li>
            <li>LLaMA, Claude: 대규모 언어 모델의 민주화</li>
          </ul>
        </div>
      </div>

      <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/10 dark:to-indigo-900/10 rounded-xl p-6 border-2 border-blue-200 dark:border-blue-800">
        <h3 className="text-xl font-semibold mb-2">실습 프로젝트: 뉴스 기사 분류기</h3>
        <p className="text-gray-600 dark:text-gray-400 mb-4">
          한국어 뉴스 기사를 카테고리별로 자동 분류하는 시스템을 만들어봅시다
        </p>
        <div className="space-y-4">
          <div className="space-y-3">
            <h4 className="font-semibold">프로젝트 단계:</h4>
            <ol className="space-y-2 list-decimal list-inside">
              <li>네이버 뉴스 API로 데이터 수집</li>
              <li>KoNLPy로 한국어 전처리</li>
              <li>TF-IDF 또는 BERT로 특징 추출</li>
              <li>다중 클래스 분류 모델 학습</li>
              <li>Flask/FastAPI로 웹 서비스 구축</li>
            </ol>
          </div>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4">
              <h5 className="font-semibold text-sm mb-1">입력</h5>
              <p className="text-xs text-gray-600 dark:text-gray-400">
                "삼성전자가 차세대 반도체 개발에 10조원을 투자한다고 발표했다..."
              </p>
            </div>
            <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4">
              <h5 className="font-semibold text-sm mb-1">출력</h5>
              <p className="text-xs">
                카테고리: IT/과학 (신뢰도: 92%)
              </p>
            </div>
          </div>
        </div>
      </div>

      <References
        sections={[
          {
            title: 'NLP Foundations',
            icon: 'paper',
            color: 'border-blue-500',
            items: [
              {
                title: 'Speech and Language Processing',
                authors: 'Daniel Jurafsky, James H. Martin',
                year: '2024',
                description: 'NLP 교과서의 표준 - 무료 온라인 제공 (3rd Edition Draft)',
                link: 'https://web.stanford.edu/~jurafsky/slp3/'
              },
              {
                title: 'Efficient Estimation of Word Representations',
                authors: 'Tomas Mikolov, et al.',
                year: '2013',
                description: 'Word2Vec - 단어 임베딩의 시작 (ICLR 2013)',
                link: 'https://arxiv.org/abs/1301.3781'
              },
              {
                title: 'GloVe: Global Vectors for Word Representation',
                authors: 'Jeffrey Pennington, Richard Socher, Christopher Manning',
                year: '2014',
                description: 'GloVe - 전역 벡터 임베딩 (EMNLP 2014)',
                link: 'https://aclanthology.org/D14-1162/'
              }
            ]
          },
          {
            title: 'Transformer & Modern NLP',
            icon: 'paper',
            color: 'border-purple-500',
            items: [
              {
                title: 'Attention Is All You Need',
                authors: 'Ashish Vaswani, et al.',
                year: '2017',
                description: 'Transformer 아키텍처 - NLP의 패러다임 전환 (NeurIPS 2017)',
                link: 'https://arxiv.org/abs/1706.03762'
              },
              {
                title: 'BERT: Pre-training of Deep Bidirectional Transformers',
                authors: 'Jacob Devlin, et al.',
                year: '2019',
                description: 'BERT - 양방향 사전학습의 혁명 (NAACL 2019)',
                link: 'https://arxiv.org/abs/1810.04805'
              },
              {
                title: 'Language Models are Few-Shot Learners',
                authors: 'Tom B. Brown, et al.',
                year: '2020',
                description: 'GPT-3 - 대규모 언어모델의 시작 (NeurIPS 2020)',
                link: 'https://arxiv.org/abs/2005.14165'
              }
            ]
          },
          {
            title: 'Specialized Tasks',
            icon: 'paper',
            color: 'border-green-500',
            items: [
              {
                title: 'Neural Machine Translation by Jointly Learning to Align',
                authors: 'Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio',
                year: '2015',
                description: 'Attention 메커니즘의 기원 - 기계번역 (ICLR 2015)',
                link: 'https://arxiv.org/abs/1409.0473'
              },
              {
                title: 'A Unified Architecture for Natural Language Processing',
                authors: 'Ronan Collobert, Jason Weston',
                year: '2008',
                description: '다중 NLP 태스크 통합 아키텍처 (ICML 2008)',
                link: 'https://ronan.collobert.com/pub/matos/2008_nlp_icml.pdf'
              }
            ]
          },
          {
            title: 'Libraries & Tools',
            icon: 'web',
            color: 'border-orange-500',
            items: [
              {
                title: 'Hugging Face Transformers',
                description: '사전학습된 모델 허브 - BERT, GPT, T5 등',
                link: 'https://huggingface.co/docs/transformers/index'
              },
              {
                title: 'spaCy',
                description: '산업용 NLP 라이브러리 - 빠르고 효율적',
                link: 'https://spacy.io/'
              },
              {
                title: 'NLTK',
                description: 'NLP 교육용 라이브러리 - 풍부한 리소스',
                link: 'https://www.nltk.org/'
              },
              {
                title: 'Gensim',
                description: 'Topic Modeling & Word Embeddings',
                link: 'https://radimrehurek.com/gensim/'
              },
              {
                title: 'Stanford CoreNLP',
                description: 'Stanford의 NLP 도구 모음',
                link: 'https://stanfordnlp.github.io/CoreNLP/'
              }
            ]
          }
        ]}
      />

      <div className="flex justify-between items-center pt-8">
        <p className="text-sm text-gray-600 dark:text-gray-400">
          NLP는 AI의 가장 활발한 연구 분야 중 하나입니다
        </p>
        {onComplete && (
          <button
            onClick={onComplete}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            다음 챕터로
          </button>
        )}
      </div>
    </div>
  )
}