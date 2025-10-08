import { Module, Chapter } from '@/types/module'

export const deepLearningModule: Module = {
  id: 'deep-learning',
  name: 'Deep Learning',
  nameKo: '딥러닝 완전정복',
  description: '신경망 기초부터 최신 딥러닝 아키텍처까지 체계적 학습',
  version: '1.0.0',
  difficulty: 'intermediate',
  estimatedHours: 25,
  icon: '🧠',
  color: '#8b5cf6',

  prerequisites: ['probability-statistics', 'data-science'],

  chapters: [
    {
      id: '01-neural-networks',
      title: '신경망의 기초',
      description: 'Perceptron부터 다층 신경망까지 - 딥러닝의 출발점',
      estimatedMinutes: 60,
      keywords: ['perceptron', 'MLP', 'activation-function', 'gradient-descent', 'backpropagation'],
      learningObjectives: [
        'Perceptron의 수학적 원리와 한계 이해',
        '다층 신경망(MLP)의 구조와 표현력',
        '활성화 함수의 종류와 역할 (Sigmoid, ReLU, Tanh)',
        '경사하강법과 최적화 알고리즘',
        'Backpropagation 완전 분해',
        '과적합 방지와 정규화 기법'
      ]
    },
    {
      id: '02-cnn',
      title: 'CNN: 합성곱 신경망',
      description: '이미지 처리의 혁명 - Convolution의 모든 것',
      estimatedMinutes: 70,
      keywords: ['convolution', 'pooling', 'feature-map', 'filters', 'receptive-field'],
      learningObjectives: [
        'Convolution 연산의 수학적 원리',
        'Pooling의 종류와 역할 (Max, Average, Global)',
        'CNN 아키텍처 발전사 (LeNet → AlexNet → VGGNet → ResNet)',
        'Batch Normalization과 Layer Normalization',
        'Skip Connection과 Residual Learning',
        '1×1 Convolution의 활용'
      ]
    },
    {
      id: '03-rnn-lstm',
      title: 'RNN & LSTM: 시계열 데이터',
      description: '순환 신경망으로 시간의 흐름 학습하기',
      estimatedMinutes: 65,
      keywords: ['RNN', 'LSTM', 'GRU', 'sequence', 'time-series', 'memory'],
      learningObjectives: [
        'RNN의 구조와 시퀀스 모델링',
        'Vanishing Gradient 문제와 해결책',
        'LSTM의 게이트 메커니즘 (Forget, Input, Output)',
        'GRU의 간소화된 구조',
        'Bidirectional RNN과 Encoder-Decoder',
        '시계열 예측 실전 기법'
      ]
    },
    {
      id: '04-transformer',
      title: 'Transformer: Attention의 힘',
      description: 'RNN을 넘어선 병렬 처리 아키텍처',
      estimatedMinutes: 80,
      keywords: ['attention', 'self-attention', 'multi-head', 'positional-encoding', 'transformer'],
      learningObjectives: [
        'Attention 메커니즘의 수학적 이해 (Q, K, V)',
        'Self-Attention과 Multi-Head Attention',
        'Positional Encoding의 필요성과 구현',
        'Transformer Encoder-Decoder 구조',
        'BERT, GPT 등 Transformer 변형',
        'Vision Transformer (ViT) 확장'
      ]
    },
    {
      id: '05-gan-generative',
      title: 'GAN & 생성 모델',
      description: '창의적인 AI - 데이터를 생성하는 신경망',
      estimatedMinutes: 70,
      keywords: ['GAN', 'VAE', 'diffusion', 'generator', 'discriminator', 'latent-space'],
      learningObjectives: [
        'GAN의 Generator-Discriminator 대립 구조',
        'Mode Collapse와 학습 안정화 기법',
        'VAE (Variational AutoEncoder) 원리',
        'Diffusion Models의 노이즈 제거 과정',
        'Conditional GAN과 제어 가능 생성',
        'StyleGAN, DALL-E, Stable Diffusion 구조'
      ]
    },
    {
      id: '06-optimization',
      title: '최적화 & 정규화',
      description: '학습을 빠르고 안정적으로 - 고급 최적화 기법',
      estimatedMinutes: 55,
      keywords: ['optimizer', 'adam', 'dropout', 'batch-norm', 'learning-rate', 'regularization'],
      learningObjectives: [
        'SGD, Momentum, Adam 등 Optimizer 비교',
        'Learning Rate Scheduling 전략',
        'Dropout과 DropConnect',
        'Batch Normalization의 내부 동작',
        'L1, L2 Regularization',
        'Early Stopping과 Gradient Clipping'
      ]
    },
    {
      id: '07-transfer-learning',
      title: 'Transfer Learning & Fine-tuning',
      description: '사전학습 모델 활용 - 적은 데이터로 높은 성능',
      estimatedMinutes: 60,
      keywords: ['transfer-learning', 'fine-tuning', 'pretrained', 'feature-extraction', 'domain-adaptation'],
      learningObjectives: [
        'Transfer Learning의 원리와 효과',
        'Feature Extraction vs Fine-tuning',
        'Layer Freezing 전략',
        'Domain Adaptation 기법',
        'Few-shot Learning',
        'ImageNet, COCO 등 사전학습 모델 활용'
      ]
    },
    {
      id: '08-advanced-practice',
      title: '실전 딥러닝 프로젝트',
      description: '이론을 실전으로 - 프로젝트 구축과 배포',
      estimatedMinutes: 90,
      keywords: ['pytorch', 'tensorflow', 'deployment', 'mlops', 'hyperparameter', 'model-serving'],
      learningObjectives: [
        'PyTorch vs TensorFlow 선택 가이드',
        'Dataset과 DataLoader 구성',
        'Hyperparameter Tuning 전략',
        '모델 저장과 불러오기',
        'ONNX, TensorRT로 최적화',
        'FastAPI, TorchServe로 배포',
        'MLOps 파이프라인 구축'
      ]
    }
  ],

  simulators: [
    {
      id: 'neural-network-playground',
      name: 'Neural Network Playground',
      description: '신경망 구조를 직접 설계하고 실시간으로 학습 과정 시각화',
      component: 'NeuralNetworkPlayground'
    },
    {
      id: 'cnn-visualizer',
      name: 'CNN Filter Visualizer',
      description: 'CNN 필터와 Feature Map을 레이어별로 시각화',
      component: 'CNNVisualizer'
    },
    {
      id: 'attention-visualizer',
      name: 'Attention 메커니즘 시각화',
      description: 'Self-Attention과 Multi-Head Attention의 실시간 동작 과정',
      component: 'AttentionVisualizer'
    },
    {
      id: 'gan-generator',
      name: 'GAN 생성 실험실',
      description: 'GAN으로 이미지, 텍스트, 음악 생성 체험',
      component: 'GANGenerator'
    },
    {
      id: 'training-dashboard',
      name: '학습 과정 대시보드',
      description: 'Loss, Accuracy, Gradient 등 학습 지표 실시간 모니터링',
      component: 'TrainingDashboard'
    },
    {
      id: 'optimizer-comparison',
      name: 'Optimizer 성능 비교',
      description: 'SGD, Adam, RMSprop 등 다양한 최적화 알고리즘 비교 실험',
      component: 'OptimizerComparison'
    }
  ],

  tools: [
    {
      id: 'neural-network-playground',
      name: 'Neural Network Playground',
      description: '인터랙티브 신경망 실습 도구',
      url: '/modules/deep-learning/tools/neural-network-playground'
    }
  ]
}

export const getChapter = (chapterId: string): Chapter | undefined => {
  return deepLearningModule.chapters.find(chapter => chapter.id === chapterId)
}

export const getNextChapter = (currentChapterId: string): Chapter | undefined => {
  const currentIndex = deepLearningModule.chapters.findIndex(ch => ch.id === currentChapterId)
  return currentIndex < deepLearningModule.chapters.length - 1 ? deepLearningModule.chapters[currentIndex + 1] : undefined
}

export const getPrevChapter = (currentChapterId: string): Chapter | undefined => {
  const currentIndex = deepLearningModule.chapters.findIndex(ch => ch.id === currentChapterId)
  return currentIndex > 0 ? deepLearningModule.chapters[currentIndex - 1] : undefined
}
