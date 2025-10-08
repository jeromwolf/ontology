'use client';

import References from '@/components/common/References';

export default function Chapter2() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4">CNN: 합성곱 신경망</h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          Convolutional Neural Network (CNN)은 이미지 처리에 특화된 신경망 구조입니다.
          Convolution 연산을 통해 이미지의 공간적 특징을 효과적으로 추출합니다.
        </p>

        <div className="bg-violet-50 dark:bg-violet-900/20 border border-violet-200 dark:border-violet-800 rounded-lg p-6 mb-6">
          <h3 className="font-semibold text-violet-900 dark:text-violet-100 mb-3">CNN의 핵심 구성 요소</h3>
          <ul className="space-y-2 text-violet-800 dark:text-violet-200">
            <li>• <strong>Convolution Layer</strong>: 필터를 이용한 특징 추출</li>
            <li>• <strong>Pooling Layer</strong>: 공간 크기 축소 및 주요 특징 선택</li>
            <li>• <strong>Fully Connected Layer</strong>: 최종 분류 또는 회귀</li>
            <li>• <strong>Activation Function</strong>: 비선형성 부여 (주로 ReLU)</li>
          </ul>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">주요 CNN 아키텍처</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6">
            <h3 className="font-semibold text-lg mb-3 text-blue-600 dark:text-blue-400">LeNet-5 (1998)</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              Yann LeCun - 손글씨 숫자 인식용 최초의 실용적 CNN
            </p>
            <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
              <li>✓ 7층 구조</li>
              <li>✓ MNIST 데이터셋에서 99.2% 정확도</li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6">
            <h3 className="font-semibold text-lg mb-3 text-green-600 dark:text-green-400">AlexNet (2012)</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              ImageNet 우승 - 딥러닝의 부활을 알린 모델
            </p>
            <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
              <li>✓ ReLU 활성화 함수 사용</li>
              <li>✓ Dropout으로 과적합 방지</li>
              <li>✓ GPU 병렬 학습</li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6">
            <h3 className="font-semibold text-lg mb-3 text-purple-600 dark:text-purple-400">VGGNet (2014)</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              단순하고 깊은 구조 - 3×3 필터만 사용
            </p>
            <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
              <li>✓ 16-19층의 깊은 구조</li>
              <li>✓ 균일한 아키텍처</li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6">
            <h3 className="font-semibold text-lg mb-3 text-red-600 dark:text-red-400">ResNet (2015)</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              Skip Connection으로 152층 이상 학습 가능
            </p>
            <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
              <li>✓ Residual Learning</li>
              <li>✓ Gradient 소실 문제 해결</li>
            </ul>
          </div>
        </div>
      </section>

      <References
        sections={[
          {
            title: 'CNN Architectures',
            icon: 'paper' as const,
            color: 'border-violet-500',
            items: [
              {
                title: 'ImageNet Classification with Deep CNNs (AlexNet)',
                authors: 'Alex Krizhevsky, et al.',
                year: '2012',
                description: '딥러닝 부활의 신호탄',
                link: 'https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf'
              },
              {
                title: 'Very Deep CNNs (VGGNet)',
                authors: 'Karen Simonyan, Andrew Zisserman',
                year: '2014',
                description: '단순하고 깊은 구조',
                link: 'https://arxiv.org/abs/1409.1556'
              },
              {
                title: 'Deep Residual Learning (ResNet)',
                authors: 'Kaiming He, et al.',
                year: '2015',
                description: 'Skip Connection의 혁신',
                link: 'https://arxiv.org/abs/1512.03385'
              }
            ]
          }
        ]}
      />
    </div>
  );
}
