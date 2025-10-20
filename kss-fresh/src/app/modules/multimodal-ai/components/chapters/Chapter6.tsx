'use client'

import React from 'react'
import { Layers, Search, Compass, Zap, BookOpen, Target, Map, Database } from 'lucide-react'

export default function Chapter6() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-purple-50 dark:from-gray-900 dark:to-purple-900">
      <div className="max-w-4xl mx-auto px-6 py-12">
        {/* Header */}
        <div className="mb-12">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-violet-500 to-purple-600 rounded-xl">
              <Map className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-violet-600 to-purple-600 bg-clip-text text-transparent">
                멀티모달 임베딩
              </h1>
              <p className="text-gray-600 dark:text-gray-400 mt-1">
                공통 임베딩 공간과 크로스모달 검색
              </p>
            </div>
          </div>
        </div>

        {/* Introduction */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-6 h-6 text-violet-600" />
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
              멀티모달 임베딩이란?
            </h2>
          </div>

          <div className="prose dark:prose-invert max-w-none">
            <p className="text-lg text-gray-700 dark:text-gray-300 leading-relaxed mb-6">
              멀티모달 임베딩은 서로 다른 모달리티(텍스트, 이미지, 오디오 등)를 동일한 벡터 공간에 매핑하는 기술입니다.
              이 공통 임베딩 공간(Common Embedding Space)에서는 의미론적으로 유사한 콘텐츠가
              모달리티에 관계없이 가까운 위치에 배치됩니다.
            </p>

            <div className="bg-gradient-to-r from-violet-50 to-purple-50 dark:from-violet-900/20 dark:to-purple-900/20 rounded-xl p-6 mb-6 border border-violet-200 dark:border-violet-800">
              <p className="text-violet-900 dark:text-violet-100 font-semibold mb-2">
                💡 핵심 아이디어
              </p>
              <p className="text-violet-800 dark:text-violet-200">
                "강아지 사진"과 "귀여운 강아지"라는 텍스트는 모달리티는 다르지만 의미가 유사합니다.
                멀티모달 임베딩은 이 둘을 512차원 벡터 공간의 <strong>가까운 위치</strong>에 배치하여,
                <strong>크로스모달 검색</strong>(텍스트로 이미지 찾기)과 <strong>제로샷 학습</strong>을 가능하게 합니다.
              </p>
            </div>
          </div>
        </section>

        {/* 공통 임베딩 공간의 특성 */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <Compass className="w-6 h-6 text-violet-600" />
            공통 임베딩 공간의 핵심 특성
          </h2>

          <div className="grid md:grid-cols-2 gap-6">
            {[
              {
                icon: <Layers className="w-8 h-8" />,
                property: 'Modality Alignment',
                title: '모달리티 정렬',
                description: '서로 다른 모달리티의 의미론적으로 동일한 콘텐츠가 가까이 위치',
                example: 'image("dog") ≈ text("a photo of a dog")',
                formula: 'cos(embedding_image, embedding_text) → 1',
                color: 'from-blue-500 to-cyan-500'
              },
              {
                icon: <Target className="w-8 h-8" />,
                property: 'Semantic Clustering',
                title: '의미론적 군집화',
                description: '유사한 의미의 콘텐츠가 클러스터를 형성',
                example: '강아지 관련 모든 이미지/텍스트가 공간의 한 영역에 모임',
                formula: '같은 클래스 → 작은 거리',
                color: 'from-purple-500 to-pink-500'
              },
              {
                icon: <Zap className="w-8 h-8" />,
                property: 'Transferability',
                title: '전이 가능성',
                description: '한 모달리티에서 학습한 지식을 다른 모달리티로 전이',
                example: '텍스트만으로 학습한 카테고리를 이미지 분류에 적용 (제로샷)',
                formula: 'Train(text) → Infer(image)',
                color: 'from-green-500 to-emerald-500'
              },
              {
                icon: <Database className="w-8 h-8" />,
                property: 'Metric Space',
                title: '메트릭 공간',
                description: '거리 메트릭(코사인 유사도, 유클리드 거리)이 의미론적 유사도를 반영',
                example: 'd(cat, dog) < d(cat, car)',
                formula: 'distance ∝ semantic dissimilarity',
                color: 'from-orange-500 to-red-500'
              }
            ].map((prop, idx) => (
              <div
                key={idx}
                className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg hover:shadow-xl transition-shadow"
              >
                <div className={`inline-flex p-3 rounded-lg bg-gradient-to-br ${prop.color} text-white mb-4`}>
                  {prop.icon}
                </div>
                <div className="mb-2">
                  <span className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    {prop.property}
                  </span>
                </div>
                <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                  {prop.title}
                </h3>
                <p className="text-gray-600 dark:text-gray-400 mb-4">
                  {prop.description}
                </p>
                <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-3 mb-3">
                  <p className="text-sm text-gray-700 dark:text-gray-300">
                    <span className="font-semibold">예시:</span> {prop.example}
                  </p>
                </div>
                <div className="bg-violet-50 dark:bg-violet-900/20 rounded-lg p-3 border-l-4 border-violet-500">
                  <p className="text-sm text-violet-900 dark:text-violet-100 font-mono">
                    {prop.formula}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Metric Learning */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            📏 Metric Learning (메트릭 러닝)
          </h2>

          <div className="space-y-6">
            <p className="text-gray-700 dark:text-gray-300">
              Metric Learning은 임베딩 공간에서 의미론적 유사도를 거리로 표현하도록 학습하는 방법입니다.
              매칭 쌍은 가깝게, 비매칭 쌍은 멀게 배치하는 것이 목표입니다.
            </p>

            <div className="bg-gray-50 dark:bg-gray-700/50 rounded-xl p-6">
              <h3 className="font-bold text-gray-900 dark:text-white mb-4">
                주요 Loss Functions
              </h3>

              <div className="space-y-6">
                {[
                  {
                    name: 'Contrastive Loss',
                    description: '쌍(pair) 기반 학습으로 매칭 쌍은 가깝게, 비매칭 쌍은 멀게',
                    formula: 'L = y·d² + (1-y)·max(0, margin - d)²',
                    explanation: 'y=1이면 거리 최소화, y=0이면 margin 이상으로 밀어냄',
                    usage: 'Siamese Networks, 초기 임베딩 학습',
                    color: 'blue'
                  },
                  {
                    name: 'Triplet Loss',
                    description: '(anchor, positive, negative) 삼중쌍 기반 학습',
                    formula: 'L = max(0, d(a,p) - d(a,n) + margin)',
                    explanation: 'anchor-positive는 가깝게, anchor-negative는 margin 이상 멀게',
                    usage: 'FaceNet, 이미지 검색',
                    color: 'purple'
                  },
                  {
                    name: 'N-Pair Loss',
                    description: '하나의 positive와 여러 negative를 동시에 고려',
                    formula: 'L = log(1 + Σ exp(d(a,p) - d(a,n_i)))',
                    explanation: '배치 내 모든 negative를 활용하여 학습 효율 향상',
                    usage: '대규모 검색 시스템',
                    color: 'green'
                  }
                ].map((loss, idx) => (
                  <div key={idx} className="bg-white dark:bg-gray-800 rounded-lg p-5 border-l-4" style={{ borderColor: `var(--${loss.color}-500)` }}>
                    <h4 className="font-bold text-gray-900 dark:text-white mb-2 flex items-center gap-2">
                      <span className={`w-8 h-8 rounded-full bg-${loss.color}-500 text-white flex items-center justify-center font-bold text-sm`}>
                        {idx + 1}
                      </span>
                      {loss.name}
                    </h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                      {loss.description}
                    </p>
                    <div className="bg-gray-900 rounded-lg p-3 mb-2 overflow-x-auto">
                      <code className="text-sm text-green-400">{loss.formula}</code>
                    </div>
                    <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                      <span className="font-semibold">설명:</span> {loss.explanation}
                    </p>
                    <p className="text-sm text-violet-900 dark:text-violet-100">
                      <span className="font-semibold">활용:</span> {loss.usage}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-violet-50 dark:bg-violet-900/20 rounded-xl p-6 border border-violet-200 dark:border-violet-800">
              <h4 className="font-bold text-violet-900 dark:text-violet-100 mb-3">
                💻 Triplet Loss 코드 예시
              </h4>
              <div className="bg-gray-900 rounded-lg p-6 overflow-x-auto">
                <pre className="text-sm text-gray-100">
{`import torch
import torch.nn.functional as F

def triplet_loss(anchor, positive, negative, margin=0.2):
    """
    anchor: [batch_size, embedding_dim]
    positive: [batch_size, embedding_dim]
    negative: [batch_size, embedding_dim]
    """
    # 거리 계산 (유클리드)
    d_ap = torch.sum((anchor - positive) ** 2, dim=1)  # anchor-positive 거리
    d_an = torch.sum((anchor - negative) ** 2, dim=1)  # anchor-negative 거리

    # Triplet Loss
    loss = torch.mean(torch.relu(d_ap - d_an + margin))
    return loss

# 사용 예시
anchor_emb = image_encoder(anchor_images)      # [32, 512]
positive_emb = text_encoder(positive_texts)    # [32, 512]
negative_emb = text_encoder(negative_texts)    # [32, 512]

loss = triplet_loss(anchor_emb, positive_emb, negative_emb, margin=0.2)`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* 크로스모달 검색 */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <Search className="w-6 h-6 text-violet-600" />
            크로스모달 검색 (Cross-Modal Retrieval)
          </h2>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <h3 className="font-bold text-gray-900 dark:text-white mb-4">
                크로스모달 검색의 종류
              </h3>

              <div className="grid md:grid-cols-2 gap-6">
                {[
                  {
                    type: 'Text → Image',
                    description: '텍스트 쿼리로 관련 이미지 검색',
                    example: 'Query: "sunset over ocean" → Top-K 이미지 반환',
                    method: '텍스트를 임베딩으로 변환 후 이미지 임베딩과 코사인 유사도 계산',
                    applications: 'Google Images, Pinterest, E-commerce',
                    color: 'from-blue-500 to-cyan-500'
                  },
                  {
                    type: 'Image → Text',
                    description: '이미지로 관련 텍스트/캡션 검색',
                    example: 'Query: [강아지 사진] → "A cute golden retriever playing in the park"',
                    method: '이미지를 임베딩으로 변환 후 텍스트 임베딩 데이터베이스에서 검색',
                    applications: 'Reverse Image Search, Image Captioning',
                    color: 'from-purple-500 to-pink-500'
                  },
                  {
                    type: 'Audio → Text/Image',
                    description: '음성 쿼리로 텍스트나 이미지 검색',
                    example: 'Query: [음성 "고양이"] → 고양이 관련 이미지/텍스트',
                    method: '음성을 텍스트로 변환(ASR) 후 텍스트 임베딩으로 검색',
                    applications: 'Voice Search, Smart Assistants',
                    color: 'from-green-500 to-emerald-500'
                  },
                  {
                    type: 'Video → Text',
                    description: '비디오 클립으로 관련 설명/캡션 검색',
                    example: 'Query: [비디오 클립] → "A person surfing on a big wave"',
                    method: '비디오 프레임을 평균 풀링하여 단일 임베딩 생성 후 검색',
                    applications: 'Video Understanding, Content Moderation',
                    color: 'from-orange-500 to-red-500'
                  }
                ].map((retrieval, idx) => (
                  <div key={idx} className="bg-gray-50 dark:bg-gray-700/50 rounded-xl p-5">
                    <div className={`inline-block px-3 py-1 rounded-full bg-gradient-to-r ${retrieval.color} text-white text-sm font-bold mb-3`}>
                      {retrieval.type}
                    </div>
                    <h4 className="font-bold text-gray-900 dark:text-white mb-2">
                      {retrieval.description}
                    </h4>
                    <div className="space-y-3">
                      <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                        <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">예시</p>
                        <p className="text-sm text-gray-700 dark:text-gray-300">{retrieval.example}</p>
                      </div>
                      <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                        <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">방법</p>
                        <p className="text-sm text-gray-700 dark:text-gray-300">{retrieval.method}</p>
                      </div>
                      <div className="bg-violet-50 dark:bg-violet-900/20 rounded-lg p-3">
                        <p className="text-xs text-violet-700 dark:text-violet-300 mb-1">응용</p>
                        <p className="text-sm text-violet-900 dark:text-violet-100">{retrieval.applications}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <h3 className="font-bold text-gray-900 dark:text-white mb-4">
                검색 시스템 구현 파이프라인
              </h3>
              <div className="space-y-4">
                {[
                  {
                    step: '1',
                    title: '오프라인 인덱싱',
                    desc: '모든 이미지/텍스트를 임베딩으로 변환하여 벡터 데이터베이스(FAISS, Milvus)에 저장'
                  },
                  {
                    step: '2',
                    title: '쿼리 임베딩',
                    desc: '사용자 쿼리(텍스트/이미지/음성)를 동일한 인코더로 임베딩 생성'
                  },
                  {
                    step: '3',
                    title: 'Similarity Search',
                    desc: '쿼리 임베딩과 데이터베이스 임베딩 간 코사인 유사도 계산 (ANN으로 고속화)'
                  },
                  {
                    step: '4',
                    title: 'Top-K Ranking',
                    desc: '유사도가 높은 상위 K개 결과를 반환 (보통 K=10~100)'
                  },
                  {
                    step: '5',
                    title: 'Re-ranking (Optional)',
                    desc: '더 복잡한 모델(Cross-Encoder)로 Top-K 결과를 재정렬하여 정확도 향상'
                  }
                ].map((stage, idx) => (
                  <div key={idx} className="flex gap-4 items-start">
                    <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center text-white font-bold">
                      {stage.step}
                    </div>
                    <div>
                      <p className="font-semibold text-gray-900 dark:text-white">{stage.title}</p>
                      <p className="text-sm text-gray-600 dark:text-gray-400">{stage.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* Zero-Shot Capabilities */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            🎯 제로샷 학습 (Zero-Shot Learning)
          </h2>

          <div className="space-y-6">
            <p className="text-gray-700 dark:text-gray-300">
              멀티모달 임베딩의 가장 강력한 능력은 <strong>제로샷 학습</strong>입니다.
              학습 시 보지 못한 카테고리도 텍스트 설명만으로 분류할 수 있습니다.
            </p>

            <div className="bg-gradient-to-r from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-xl p-6 border border-blue-200 dark:border-blue-800">
              <h3 className="font-bold text-blue-900 dark:text-blue-100 mb-4">
                CLIP의 제로샷 분류 방법
              </h3>
              <div className="space-y-4">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <p className="font-semibold text-gray-900 dark:text-white mb-2">1단계: 클래스 텍스트 임베딩 생성</p>
                  <div className="bg-gray-900 rounded p-3 overflow-x-auto">
                    <code className="text-sm text-green-400">
{`classes = ["cat", "dog", "car", "airplane"]
prompts = [f"a photo of a {c}" for c in classes]
text_embeddings = clip_text_encoder(prompts)  # [4, 512]`}
                    </code>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <p className="font-semibold text-gray-900 dark:text-white mb-2">2단계: 이미지 임베딩 생성</p>
                  <div className="bg-gray-900 rounded p-3 overflow-x-auto">
                    <code className="text-sm text-green-400">
{`image_embedding = clip_image_encoder(test_image)  # [1, 512]`}
                    </code>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <p className="font-semibold text-gray-900 dark:text-white mb-2">3단계: 유사도 계산 및 분류</p>
                  <div className="bg-gray-900 rounded p-3 overflow-x-auto">
                    <code className="text-sm text-green-400">
{`similarities = image_embedding @ text_embeddings.T  # [1, 4]
predicted_class = classes[similarities.argmax()]  # "dog"`}
                    </code>
                  </div>
                </div>
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-green-50 dark:bg-green-900/20 rounded-xl p-5 border border-green-200 dark:border-green-800">
                <h4 className="font-bold text-green-900 dark:text-green-100 mb-3">
                  ✅ 제로샷의 장점
                </h4>
                <ul className="space-y-2 text-green-800 dark:text-green-200">
                  <li className="flex gap-2">
                    <span>•</span>
                    <span>새로운 클래스에 대한 라벨 데이터 불필요</span>
                  </li>
                  <li className="flex gap-2">
                    <span>•</span>
                    <span>텍스트 설명만으로 즉시 분류 가능</span>
                  </li>
                  <li className="flex gap-2">
                    <span>•</span>
                    <span>Long-tail 카테고리 처리 용이</span>
                  </li>
                  <li className="flex gap-2">
                    <span>•</span>
                    <span>도메인 일반화 능력 우수</span>
                  </li>
                </ul>
              </div>

              <div className="bg-amber-50 dark:bg-amber-900/20 rounded-xl p-5 border border-amber-200 dark:border-amber-800">
                <h4 className="font-bold text-amber-900 dark:text-amber-100 mb-3">
                  ⚠️ 제로샷의 한계
                </h4>
                <ul className="space-y-2 text-amber-800 dark:text-amber-200">
                  <li className="flex gap-2">
                    <span>•</span>
                    <span>Fine-tuned 모델보다 정확도 낮음</span>
                  </li>
                  <li className="flex gap-2">
                    <span>•</span>
                    <span>프롬프트 설계에 성능이 크게 좌우됨</span>
                  </li>
                  <li className="flex gap-2">
                    <span>•</span>
                    <span>유사한 카테고리 간 혼동 가능</span>
                  </li>
                  <li className="flex gap-2">
                    <span>•</span>
                    <span>복잡한 추론 태스크에는 부적합</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* 실전 응용 */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            🚀 실전 응용 사례
          </h2>

          <div className="grid gap-6">
            {[
              {
                application: 'E-commerce 검색',
                description: '텍스트 또는 이미지로 상품 검색, 유사 상품 추천',
                tech: 'CLIP 임베딩 + FAISS 벡터 검색',
                impact: 'Amazon, Alibaba의 시각 검색 기능',
                metric: '검색 만족도 30% 향상, 구매 전환율 15% 증가',
                color: 'from-blue-500 to-cyan-500'
              },
              {
                application: '콘텐츠 추천',
                description: '사용자 선호도를 멀티모달로 학습하여 콘텐츠 추천',
                tech: '멀티모달 임베딩 + Collaborative Filtering',
                impact: 'YouTube, Netflix의 개인화 추천',
                metric: '시청 시간 20% 증가, 이탈률 감소',
                color: 'from-purple-500 to-pink-500'
              },
              {
                application: '의료 영상 검색',
                description: '텍스트 설명으로 유사한 의료 영상 검색 (유사 증례 찾기)',
                tech: 'Medical CLIP + Case-based Reasoning',
                impact: '방사선과, 병리과 진단 보조',
                metric: '진단 정확도 5% 향상, 시간 40% 단축',
                color: 'from-green-500 to-emerald-500'
              },
              {
                application: '저작권 검증',
                description: '멀티모달 유사도로 복제/표절 콘텐츠 탐지',
                tech: 'Perceptual Hashing + CLIP Similarity',
                impact: 'YouTube Content ID, Getty Images',
                metric: '허위 양성 10% 감소, 탐지율 95%',
                color: 'from-orange-500 to-red-500'
              }
            ].map((app, idx) => (
              <div
                key={idx}
                className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg hover:shadow-xl transition-shadow"
              >
                <div className="flex items-start gap-4">
                  <div className={`flex-shrink-0 w-12 h-12 rounded-lg bg-gradient-to-br ${app.color} flex items-center justify-center text-white font-bold text-xl`}>
                    {idx + 1}
                  </div>
                  <div className="flex-1">
                    <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                      {app.application}
                    </h3>
                    <p className="text-gray-600 dark:text-gray-400 mb-3">
                      {app.description}
                    </p>
                    <div className="space-y-2">
                      <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-3">
                        <p className="text-sm text-gray-700 dark:text-gray-300">
                          <span className="font-semibold">기술:</span> {app.tech}
                        </p>
                      </div>
                      <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3">
                        <p className="text-sm text-blue-900 dark:text-blue-100">
                          <span className="font-semibold">영향:</span> {app.impact}
                        </p>
                      </div>
                      <div className="bg-violet-50 dark:bg-violet-900/20 rounded-lg p-3 border-l-4 border-violet-500">
                        <p className="text-sm text-violet-900 dark:text-violet-100">
                          <span className="font-semibold">성과:</span> {app.metric}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* 학습 목표 요약 */}
        <section className="bg-gradient-to-br from-violet-600 to-purple-600 rounded-2xl p-8 text-white">
          <h2 className="text-2xl font-bold mb-6">📚 이 챕터에서 배운 내용</h2>
          <ul className="space-y-3">
            {[
              '공통 임베딩 공간의 4가지 핵심 특성 (정렬, 군집화, 전이성, 메트릭)',
              'Metric Learning의 Loss Functions (Contrastive, Triplet, N-Pair)',
              '크로스모달 검색 파이프라인 (인덱싱, 쿼리 임베딩, ANN, Re-ranking)',
              'CLIP의 제로샷 분류 방법과 장단점',
              '실전 응용: E-commerce, 콘텐츠 추천, 의료 영상, 저작권 검증',
              'FAISS, Milvus 등 벡터 데이터베이스 활용'
            ].map((item, idx) => (
              <li key={idx} className="flex items-start gap-3">
                <span className="text-violet-200 mt-1">✓</span>
                <span>{item}</span>
              </li>
            ))}
          </ul>

          <div className="mt-8 pt-6 border-t border-violet-400">
            <p className="text-violet-100">
              <span className="font-semibold">다음 챕터:</span> 실시간 멀티모달 AI를 학습합니다.
              저지연 파이프라인 구현, 최적화 기법, 엣지 디바이스 배포 전략을 살펴봅니다.
            </p>
          </div>
        </section>
      </div>
    </div>
  )
}
