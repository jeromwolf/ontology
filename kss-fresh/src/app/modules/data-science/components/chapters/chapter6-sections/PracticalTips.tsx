'use client'

import { CheckCircle, AlertCircle } from 'lucide-react'
import References from '@/components/common/References'

interface PracticalTipsProps {
  onComplete?: () => void
}

export default function PracticalTips({ onComplete }: PracticalTipsProps) {
  return (
    <>
      <section>
        <h2 className="text-3xl font-bold mb-6">6. 딥러닝 실전 팁</h2>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 p-6 rounded-xl">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <CheckCircle className="text-green-600" />
              모범 사례
            </h3>
            <ul className="space-y-2 text-sm">
              <li>✓ 데이터 정규화/표준화</li>
              <li>✓ 적절한 가중치 초기화</li>
              <li>✓ 배치 정규화 사용</li>
              <li>✓ 조기 종료 (Early Stopping)</li>
              <li>✓ 학습률 스케줄링</li>
              <li>✓ 정규화 (L1/L2, Dropout)</li>
              <li>✓ 데이터 증강</li>
              <li>✓ 앙상블 방법</li>
            </ul>
          </div>
          
          <div className="bg-gradient-to-br from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 p-6 rounded-xl">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <AlertCircle className="text-red-600" />
              일반적인 문제와 해결법
            </h3>
            <ul className="space-y-2 text-sm">
              <li><strong>과적합:</strong> Dropout, 데이터 증강, 정규화</li>
              <li><strong>과소적합:</strong> 모델 크기 증가, 학습 시간 연장</li>
              <li><strong>Vanishing Gradient:</strong> ReLU, BatchNorm, ResNet</li>
              <li><strong>Exploding Gradient:</strong> Gradient Clipping, 작은 학습률</li>
              <li><strong>느린 학습:</strong> 학습률 조정, 옵티마이저 변경</li>
              <li><strong>메모리 부족:</strong> 배치 크기 감소, 모델 경량화</li>
            </ul>
          </div>
        </div>

        {/* 추가 팁 */}
        <div className="mt-6 bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 p-6 rounded-xl">
          <h3 className="text-lg font-semibold mb-4">디버깅 체크리스트</h3>
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <h4 className="font-semibold mb-2">데이터 확인</h4>
              <ul className="text-sm space-y-1">
                <li>□ 입력 데이터 shape 확인</li>
                <li>□ 레이블 분포 확인</li>
                <li>□ 데이터 스케일 확인</li>
                <li>□ 누락값/이상치 처리</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-2">모델 확인</h4>
              <ul className="text-sm space-y-1">
                <li>□ 파라미터 수 확인</li>
                <li>□ 활성화 함수 체크</li>
                <li>□ 손실 함수 적절성</li>
                <li>□ 과적합 모니터링</li>
              </ul>
            </div>
          </div>
        </div>

        {/* 유용한 코드 스니펫 */}
        <div className="mt-6 bg-gray-900 rounded-xl p-6">
          <h3 className="text-white font-semibold mb-4">유용한 코드 스니펫</h3>
          <pre className="bg-gray-800 p-4 rounded-lg overflow-x-auto">
            <code className="text-sm text-gray-300">{`# 모델 요약 정보 출력
def model_summary(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

# 학습 곡선 시각화
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Validation')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# 조기 종료 콜백
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0`}</code>
          </pre>
        </div>
      </section>

      {/* 프로젝트 */}
      <section className="mt-12">
        <div className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white p-8 rounded-xl">
          <h2 className="text-2xl font-bold mb-4">🧠 실전 프로젝트: MNIST 손글씨 인식</h2>
          <p className="mb-6">
            딥러닝의 "Hello World"인 MNIST 데이터셋으로 시작해보세요.
            CNN을 사용해 99% 이상의 정확도를 달성하는 것이 목표입니다.
          </p>
          
          <div className="mb-6">
            <h3 className="text-lg font-semibold mb-2">프로젝트 단계:</h3>
            <ol className="list-decimal list-inside space-y-1 text-sm">
              <li>MNIST 데이터셋 로드 및 전처리</li>
              <li>간단한 MLP로 베이스라인 설정</li>
              <li>CNN 아키텍처 설계 (Conv → Pool → FC)</li>
              <li>데이터 증강 적용 (회전, 이동, 확대)</li>
              <li>하이퍼파라미터 튜닝</li>
              <li>앙상블로 성능 개선</li>
            </ol>
          </div>
          
          <div className="flex gap-4 flex-wrap">
            {onComplete && (
              <button 
                onClick={onComplete}
                className="bg-white text-indigo-600 px-6 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-colors"
              >
                챕터 완료하기
              </button>
            )}
            <button className="bg-indigo-500 text-white px-6 py-3 rounded-lg font-semibold hover:bg-indigo-400 transition-colors">
              Colab 노트북 열기
            </button>
          </div>
        </div>
      </section>

      <References
        sections={[
          {
            title: 'Foundational Books & Courses',
            icon: 'paper',
            color: 'border-indigo-500',
            items: [
              {
                title: 'Deep Learning',
                authors: 'Ian Goodfellow, Yoshua Bengio, Aaron Courville',
                year: '2016',
                description: '딥러닝의 바이블 - MIT Press 무료 온라인 제공',
                link: 'https://www.deeplearningbook.org/'
              },
              {
                title: 'Neural Networks and Deep Learning',
                authors: 'Michael Nielsen',
                year: '2015',
                description: '직관적이고 이해하기 쉬운 딥러닝 입문서',
                link: 'http://neuralnetworksanddeeplearning.com/'
              },
              {
                title: 'CS231n: Convolutional Neural Networks for Visual Recognition',
                authors: 'Stanford University',
                year: '2024',
                description: 'Stanford의 유명한 CNN 강의 - 전체 강의 영상 및 노트 제공',
                link: 'http://cs231n.stanford.edu/'
              },
              {
                title: 'Fast.ai Practical Deep Learning',
                authors: 'Jeremy Howard',
                year: '2024',
                description: '코드 중심의 실용적인 딥러닝 강의',
                link: 'https://course.fast.ai/'
              }
            ]
          },
          {
            title: 'Neural Network Architectures',
            icon: 'paper',
            color: 'border-blue-500',
            items: [
              {
                title: 'Learning representations by back-propagating errors',
                authors: 'David Rumelhart, Geoffrey Hinton, Ronald Williams',
                year: '1986',
                description: '역전파 알고리즘의 기초를 확립한 논문 (Nature)',
                link: 'https://www.nature.com/articles/323533a0'
              },
              {
                title: 'Gradient-based learning applied to document recognition',
                authors: 'Yann LeCun, et al.',
                year: '1998',
                description: 'LeNet - 현대 CNN의 시초 (Proceedings of the IEEE)',
                link: 'http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf'
              },
              {
                title: 'Deep Residual Learning for Image Recognition',
                authors: 'Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun',
                year: '2015',
                description: 'ResNet - 잔차 연결로 초심층 네트워크 학습 가능 (CVPR 2016)',
                link: 'https://arxiv.org/abs/1512.03385'
              },
              {
                title: 'Long Short-Term Memory',
                authors: 'Sepp Hochreiter, Jürgen Schmidhuber',
                year: '1997',
                description: 'LSTM - 장기 의존성 문제 해결 (Neural Computation)',
                link: 'https://www.bioinf.jku.at/publications/older/2604.pdf'
              }
            ]
          },
          {
            title: 'Optimization & Training Techniques',
            icon: 'paper',
            color: 'border-green-500',
            items: [
              {
                title: 'Adam: A Method for Stochastic Optimization',
                authors: 'Diederik P. Kingma, Jimmy Ba',
                year: '2014',
                description: '가장 널리 사용되는 최적화 알고리즘 (ICLR 2015)',
                link: 'https://arxiv.org/abs/1412.6980'
              },
              {
                title: 'Batch Normalization: Accelerating Deep Network Training',
                authors: 'Sergey Ioffe, Christian Szegedy',
                year: '2015',
                description: '배치 정규화로 학습 안정화 및 가속 (ICML 2015)',
                link: 'https://arxiv.org/abs/1502.03167'
              },
              {
                title: 'Dropout: A Simple Way to Prevent Neural Networks from Overfitting',
                authors: 'Nitish Srivastava, et al.',
                year: '2014',
                description: '과적합 방지를 위한 Dropout 기법 (JMLR)',
                link: 'http://jmlr.org/papers/v15/srivastava14a.html'
              },
              {
                title: 'Understanding the difficulty of training deep feedforward neural networks',
                authors: 'Xavier Glorot, Yoshua Bengio',
                year: '2010',
                description: 'Xavier 초기화 - 가중치 초기화의 중요성 (AISTATS)',
                link: 'http://proceedings.mlr.press/v9/glorot10a.html'
              }
            ]
          },
          {
            title: 'Frameworks & Practical Resources',
            icon: 'web',
            color: 'border-purple-500',
            items: [
              {
                title: 'PyTorch Documentation',
                description: 'PyTorch 공식 문서 및 튜토리얼',
                link: 'https://pytorch.org/docs/stable/index.html'
              },
              {
                title: 'TensorFlow Tutorials',
                description: 'TensorFlow 2.x 공식 가이드',
                link: 'https://www.tensorflow.org/tutorials'
              },
              {
                title: 'Keras Documentation',
                description: '초보자 친화적인 고수준 딥러닝 API',
                link: 'https://keras.io/'
              },
              {
                title: 'Papers with Code',
                description: '최신 논문과 구현 코드 모음',
                link: 'https://paperswithcode.com/'
              },
              {
                title: 'Distill.pub',
                description: '시각화 중심의 머신러닝 설명',
                link: 'https://distill.pub/'
              }
            ]
          }
        ]}
      />
    </>
  )
}