'use client';

import References from '@/components/common/References';

export default function Chapter7() {
  return (
    <div className="space-y-8">
      {/* 1. Transfer Learning 소개 */}
      <section>
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          Transfer Learning & Fine-tuning
        </h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          Transfer Learning은 한 작업에서 학습한 지식을 다른 작업에 활용하는 기법입니다.
          대규모 데이터로 사전학습된 모델을 활용하면 적은 데이터로도 높은 성능을 달성할 수 있습니다.
        </p>

        <div className="bg-gradient-to-br from-green-50 to-teal-50 dark:from-green-900/20 dark:to-teal-900/20 rounded-2xl p-6 border border-green-200 dark:border-green-700 mb-6">
          <h3 className="text-lg font-semibold mb-3 text-green-900 dark:text-green-300">
            💡 Transfer Learning의 장점
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li><strong>데이터 효율성</strong>: 적은 데이터로 높은 성능</li>
            <li><strong>학습 시간 단축</strong>: 사전학습된 가중치로 시작</li>
            <li><strong>일반화 성능</strong>: 풍부한 표현력 학습</li>
            <li><strong>실전 활용도</strong>: 대부분의 실무 프로젝트에서 사용</li>
          </ul>
        </div>
      </section>

      {/* 2. Feature Extraction vs Fine-tuning */}
      <section>
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          Feature Extraction vs Fine-tuning
        </h2>

        <div className="grid md:grid-cols-2 gap-4 mb-6">
          {/* Feature Extraction */}
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6 border border-blue-200 dark:border-blue-800">
            <h4 className="font-semibold mb-3 text-blue-900 dark:text-blue-300 text-lg">🔒 Feature Extraction</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              사전학습 모델을 <strong>고정(freeze)</strong>하고 마지막 레이어만 학습
            </p>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• 모든 레이어 고정 (requires_grad = False)</li>
              <li>• 새로운 분류기(classifier)만 추가</li>
              <li>• 빠른 학습, 낮은 메모리 사용</li>
              <li>• 데이터가 매우 적을 때 (&lt; 1,000)</li>
              <li>• 도메인이 유사할 때</li>
            </ul>
          </div>

          {/* Fine-tuning */}
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-xl p-6 border border-purple-200 dark:border-purple-800">
            <h4 className="font-semibold mb-3 text-purple-900 dark:text-purple-300 text-lg">🔓 Fine-tuning</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              사전학습 모델의 <strong>일부 또는 전체</strong>를 재학습
            </p>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• 하위 레이어 고정, 상위 레이어만 학습</li>
              <li>• 또는 전체 레이어 미세 조정</li>
              <li>• 느린 학습, 높은 메모리 사용</li>
              <li>• 데이터가 충분할 때 (&gt; 10,000)</li>
              <li>• 도메인이 다를 때</li>
            </ul>
          </div>
        </div>

        {/* Layer Freezing 전략 */}
        <div className="bg-gradient-to-br from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-xl p-6 border border-orange-200 dark:border-orange-800">
          <h3 className="text-lg font-semibold mb-3 text-orange-900 dark:text-orange-300">
            🧊 Layer Freezing 전략
          </h3>
          <div className="grid md:grid-cols-3 gap-4 text-sm">
            <div>
              <strong className="text-gray-900 dark:text-gray-100">초반 레이어 (Low-level Features)</strong>
              <p className="text-gray-700 dark:text-gray-300 mt-1">
                Edge, Texture 등 범용적인 특징<br/>
                → 거의 항상 고정
              </p>
            </div>
            <div>
              <strong className="text-gray-900 dark:text-gray-100">중간 레이어 (Mid-level Features)</strong>
              <p className="text-gray-700 dark:text-gray-300 mt-1">
                Shapes, Parts 등 중간 수준 특징<br/>
                → 경우에 따라 조정
              </p>
            </div>
            <div>
              <strong className="text-gray-900 dark:text-gray-100">상위 레이어 (High-level Features)</strong>
              <p className="text-gray-700 dark:text-gray-300 mt-1">
                Task-specific한 특징<br/>
                → 거의 항상 재학습
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 3. 사전학습 모델 활용 */}
      <section>
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          주요 사전학습 모델
        </h2>

        <div className="grid md:grid-cols-2 gap-4">
          {/* Computer Vision */}
          <div className="bg-pink-50 dark:bg-pink-900/20 rounded-xl p-6 border border-pink-200 dark:border-pink-800">
            <h4 className="font-semibold mb-3 text-pink-900 dark:text-pink-300 text-lg">📸 Computer Vision</h4>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
              <li>• <strong>ResNet</strong>: 50, 101, 152 layers (ImageNet)</li>
              <li>• <strong>EfficientNet</strong>: 효율적인 scaling (B0-B7)</li>
              <li>• <strong>Vision Transformer (ViT)</strong>: Attention 기반</li>
              <li>• <strong>CLIP</strong>: 텍스트-이미지 멀티모달</li>
              <li>• <strong>DINO</strong>: Self-supervised learning</li>
            </ul>
          </div>

          {/* NLP */}
          <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-xl p-6 border border-indigo-200 dark:border-indigo-800">
            <h4 className="font-semibold mb-3 text-indigo-900 dark:text-indigo-300 text-lg">💬 Natural Language Processing</h4>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
              <li>• <strong>BERT</strong>: Bidirectional encoder (Google)</li>
              <li>• <strong>GPT</strong>: Auto-regressive decoder (OpenAI)</li>
              <li>• <strong>T5</strong>: Text-to-Text framework</li>
              <li>• <strong>RoBERTa</strong>: Robustly optimized BERT</li>
              <li>• <strong>ELECTRA</strong>: 효율적인 사전학습</li>
            </ul>
          </div>

          {/* Audio */}
          <div className="bg-teal-50 dark:bg-teal-900/20 rounded-xl p-6 border border-teal-200 dark:border-teal-800">
            <h4 className="font-semibold mb-3 text-teal-900 dark:text-teal-300 text-lg">🎵 Audio & Speech</h4>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
              <li>• <strong>Wav2Vec 2.0</strong>: Speech representation</li>
              <li>• <strong>HuBERT</strong>: Self-supervised audio</li>
              <li>• <strong>Whisper</strong>: Robust speech recognition</li>
              <li>• <strong>AudioMAE</strong>: Masked autoencoding</li>
            </ul>
          </div>

          {/* Multimodal */}
          <div className="bg-violet-50 dark:bg-violet-900/20 rounded-xl p-6 border border-violet-200 dark:border-violet-800">
            <h4 className="font-semibold mb-3 text-violet-900 dark:text-violet-300 text-lg">🌐 Multimodal</h4>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
              <li>• <strong>CLIP</strong>: 이미지-텍스트 (OpenAI)</li>
              <li>• <strong>DALL-E</strong>: 텍스트→이미지 생성</li>
              <li>• <strong>Flamingo</strong>: Few-shot learning</li>
              <li>• <strong>BEiT</strong>: Vision-Language pre-training</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 4. Fine-tuning 실전 가이드 */}
      <section>
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          Fine-tuning 실전 가이드
        </h2>

        <div className="bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-2xl p-8 border border-blue-200 dark:border-blue-800">
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-semibold text-lg mb-3 text-blue-900 dark:text-blue-300">1️⃣ 학습률 설정</h3>
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li>• 사전학습보다 <strong>10-100배 낮은 학습률</strong> 사용</li>
                <li>• 레이어별 차등 학습률 (Discriminative learning rate)</li>
                <li>• 하위 레이어: 1e-5, 상위 레이어: 1e-3</li>
                <li>• Warm-up + Cosine decay 추천</li>
              </ul>
            </div>

            <div>
              <h3 className="font-semibold text-lg mb-3 text-blue-900 dark:text-blue-300">2️⃣ 데이터 준비</h3>
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li>• 사전학습 모델의 <strong>전처리 방식</strong> 그대로 사용</li>
                <li>• 이미지 크기, 정규화 값 일치</li>
                <li>• Data Augmentation 적극 활용</li>
                <li>• Class imbalance 해결 (oversampling, weighted loss)</li>
              </ul>
            </div>

            <div>
              <h3 className="font-semibold text-lg mb-3 text-blue-900 dark:text-blue-300">3️⃣ 정규화</h3>
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li>• Dropout 비율 증가 (0.5 → 0.7)</li>
                <li>• Weight Decay 적절히 조정</li>
                <li>• Early Stopping 필수</li>
                <li>• Label Smoothing 고려</li>
              </ul>
            </div>

            <div>
              <h3 className="font-semibold text-lg mb-3 text-blue-900 dark:text-blue-300">4️⃣ 평가</h3>
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li>• Validation set으로 하이퍼파라미터 튜닝</li>
                <li>• Test set은 최종 평가에만 사용</li>
                <li>• Cross-validation 추천 (데이터 적을 때)</li>
                <li>• Confusion matrix, F1-score 확인</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* 5. Domain Adaptation */}
      <section>
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          Domain Adaptation & Few-shot Learning
        </h2>

        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-xl p-6 border border-yellow-200 dark:border-yellow-800">
            <h4 className="font-semibold mb-3 text-yellow-900 dark:text-yellow-300 text-lg">🔄 Domain Adaptation</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              Source domain에서 Target domain으로 지식 전이
            </p>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• <strong>Unsupervised DA</strong>: Target label 없음</li>
              <li>• <strong>Semi-supervised DA</strong>: 일부만 label</li>
              <li>• <strong>Adversarial DA</strong>: Domain classifier로 혼란</li>
              <li>• 예: 합성 데이터 → 실제 데이터</li>
            </ul>
          </div>

          <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-xl p-6 border border-emerald-200 dark:border-emerald-800">
            <h4 className="font-semibold mb-3 text-emerald-900 dark:text-emerald-300 text-lg">🎯 Few-shot Learning</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              매우 적은 샘플 (5-50개)로 새로운 클래스 학습
            </p>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• <strong>Meta-learning</strong>: Learning to learn</li>
              <li>• <strong>Prototypical Networks</strong>: 클래스 prototype 학습</li>
              <li>• <strong>MAML</strong>: Model-Agnostic Meta-Learning</li>
              <li>• 예: 새로운 제품 카테고리 분류</li>
            </ul>
          </div>
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '🔴 핵심 논문',
            icon: 'paper' as const,
            color: 'border-red-500',
            items: [
              {
                title: 'ImageNet Pre-training for Transfer Learning',
                authors: 'Huh, M., et al.',
                year: '2016',
                description: 'Transfer learning의 효과 분석',
                link: 'https://arxiv.org/abs/1608.08614'
              },
              {
                title: 'How transferable are features in deep neural networks?',
                authors: 'Yosinski, J., et al.',
                year: '2014',
                description: '레이어별 전이 가능성 분석',
                link: 'https://arxiv.org/abs/1411.1792'
              }
            ]
          },
          {
            title: '🛠️ 실전 라이브러리',
            icon: 'github' as const,
            color: 'border-green-500',
            items: [
              {
                title: 'Hugging Face Transformers',
                authors: 'Hugging Face',
                year: '2023',
                description: '사전학습 모델 라이브러리 (NLP)',
                link: 'https://github.com/huggingface/transformers'
              },
              {
                title: 'timm (PyTorch Image Models)',
                authors: 'Ross Wightman',
                year: '2023',
                description: '사전학습 모델 라이브러리 (Vision)',
                link: 'https://github.com/huggingface/pytorch-image-models'
              }
            ]
          }
        ]}
      />
    </div>
  );
}
