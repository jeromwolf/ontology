'use client'

import { useState } from 'react'
import Link from 'next/link'
import { 
  Scan,
  Brain,
  Activity,
  Eye,
  ChevronRight,
  ArrowLeft,
  ArrowRight,
  CheckCircle,
  AlertCircle,
  Layers,
  Cpu,
  BarChart3,
  Target
} from 'lucide-react'

export default function MedicalImagingPage() {
  const [activeSection, setActiveSection] = useState('overview')

  const sections = [
    { id: 'overview', title: '개요', icon: Scan },
    { id: 'modalities', title: '영상 기법', icon: Layers },
    { id: 'cnn', title: 'CNN 아키텍처', icon: Cpu },
    { id: 'preprocessing', title: '전처리', icon: BarChart3 },
    { id: 'applications', title: '응용 사례', icon: Target }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-white dark:from-gray-900 dark:to-gray-800">
      {/* Header */}
      <header className="sticky top-0 z-30 bg-white/80 dark:bg-gray-900/80 backdrop-blur-xl border-b border-gray-200 dark:border-gray-800">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link
                href="/medical-ai"
                className="flex items-center gap-2 text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white transition-colors"
              >
                <ArrowLeft className="w-5 h-5" />
                <span>목록으로</span>
              </Link>
              <div className="h-6 w-px bg-gray-300 dark:bg-gray-700"></div>
              <h1 className="text-xl font-semibold text-gray-900 dark:text-white">
                Chapter 2: 의료 영상 분석
              </h1>
            </div>
            <div className="flex items-center gap-3">
              <span className="px-3 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 rounded-full text-sm font-medium">
                중급
              </span>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="grid lg:grid-cols-4 gap-8">
          {/* Sidebar Navigation */}
          <aside className="lg:col-span-1">
            <div className="sticky top-24 space-y-2">
              {sections.map((section) => {
                const Icon = section.icon
                return (
                  <button
                    key={section.id}
                    onClick={() => setActiveSection(section.id)}
                    className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all ${
                      activeSection === section.id
                        ? 'bg-gradient-to-r from-blue-500 to-cyan-600 text-white shadow-lg'
                        : 'text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-800'
                    }`}
                  >
                    <Icon className="w-5 h-5" />
                    <span className="font-medium">{section.title}</span>
                  </button>
                )
              })}
            </div>
          </aside>

          {/* Main Content */}
          <main className="lg:col-span-3 space-y-8">
            {/* Overview Section */}
            {activeSection === 'overview' && (
              <div className="space-y-6">
                <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
                  <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
                    의료 영상 분석 AI
                  </h2>
                  
                  <div className="prose prose-lg dark:prose-invert max-w-none">
                    <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
                      의료 영상 분석은 AI가 가장 성공적으로 적용된 의료 분야 중 하나입니다.
                      딥러닝 기술, 특히 CNN(Convolutional Neural Network)을 활용하여
                      방사선 전문의 수준의 정확도로 질병을 진단할 수 있게 되었습니다.
                    </p>

                    <div className="grid md:grid-cols-2 gap-6 my-8">
                      <div className="bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-lg p-6">
                        <Scan className="w-10 h-10 text-blue-600 dark:text-blue-400 mb-4" />
                        <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                          주요 영상 기법
                        </h3>
                        <ul className="space-y-2 text-gray-600 dark:text-gray-300">
                          <li className="flex items-start gap-2">
                            <CheckCircle className="w-5 h-5 text-green-500 mt-0.5" />
                            <span>X-Ray (흉부, 골절)</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <CheckCircle className="w-5 h-5 text-green-500 mt-0.5" />
                            <span>CT (컴퓨터 단층촬영)</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <CheckCircle className="w-5 h-5 text-green-500 mt-0.5" />
                            <span>MRI (자기공명영상)</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <CheckCircle className="w-5 h-5 text-green-500 mt-0.5" />
                            <span>초음파 (Ultrasound)</span>
                          </li>
                        </ul>
                      </div>

                      <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg p-6">
                        <Brain className="w-10 h-10 text-purple-600 dark:text-purple-400 mb-4" />
                        <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                          AI 기술
                        </h3>
                        <ul className="space-y-2 text-gray-600 dark:text-gray-300">
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-purple-500 mt-0.5" />
                            <span>CNN (ResNet, DenseNet)</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-purple-500 mt-0.5" />
                            <span>U-Net (세그멘테이션)</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-purple-500 mt-0.5" />
                            <span>Vision Transformer</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-purple-500 mt-0.5" />
                            <span>Transfer Learning</span>
                          </li>
                        </ul>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Modalities Section */}
            {activeSection === 'modalities' && (
              <div className="space-y-6">
                <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
                  <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
                    의료 영상 기법별 특징
                  </h2>
                  
                  <div className="space-y-6">
                    <div className="border-l-4 border-blue-500 pl-6">
                      <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                        X-Ray (방사선 촬영)
                      </h3>
                      <p className="text-gray-600 dark:text-gray-300 mb-3">
                        가장 기본적이고 널리 사용되는 영상 기법으로, 뼈와 폐 질환 진단에 효과적
                      </p>
                      <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <p className="font-medium text-gray-900 dark:text-white mb-2">장점</p>
                            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                              <li>• 빠른 촬영 속도</li>
                              <li>• 저렴한 비용</li>
                              <li>• 뼈 구조 명확</li>
                            </ul>
                          </div>
                          <div>
                            <p className="font-medium text-gray-900 dark:text-white mb-2">AI 응용</p>
                            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                              <li>• 폐렴 검출</li>
                              <li>• 골절 진단</li>
                              <li>• 결핵 스크리닝</li>
                            </ul>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="border-l-4 border-green-500 pl-6">
                      <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                        CT (Computed Tomography)
                      </h3>
                      <p className="text-gray-600 dark:text-gray-300 mb-3">
                        3D 단면 영상을 제공하여 복잡한 구조를 상세히 관찰 가능
                      </p>
                      <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <p className="font-medium text-gray-900 dark:text-white mb-2">장점</p>
                            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                              <li>• 3D 재구성 가능</li>
                              <li>• 높은 해상도</li>
                              <li>• 빠른 스캔</li>
                            </ul>
                          </div>
                          <div>
                            <p className="font-medium text-gray-900 dark:text-white mb-2">AI 응용</p>
                            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                              <li>• 종양 검출</li>
                              <li>• 뇌출혈 진단</li>
                              <li>• 폐 결절 분석</li>
                            </ul>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="border-l-4 border-purple-500 pl-6">
                      <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                        MRI (Magnetic Resonance Imaging)
                      </h3>
                      <p className="text-gray-600 dark:text-gray-300 mb-3">
                        자기장을 이용한 고해상도 연조직 영상, 방사선 노출 없음
                      </p>
                      <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <p className="font-medium text-gray-900 dark:text-white mb-2">장점</p>
                            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                              <li>• 연조직 대비 우수</li>
                              <li>• 방사선 없음</li>
                              <li>• 다양한 시퀀스</li>
                            </ul>
                          </div>
                          <div>
                            <p className="font-medium text-gray-900 dark:text-white mb-2">AI 응용</p>
                            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                              <li>• 뇌종양 분류</li>
                              <li>• 알츠하이머 예측</li>
                              <li>• 관절 질환 진단</li>
                            </ul>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* CNN Architecture Section */}
            {activeSection === 'cnn' && (
              <div className="space-y-6">
                <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
                  <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
                    의료 영상용 CNN 아키텍처
                  </h2>
                  
                  <div className="space-y-8">
                    <div>
                      <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
                        주요 CNN 모델
                      </h3>
                      
                      <div className="grid md:grid-cols-2 gap-6">
                        <div className="bg-gradient-to-br from-indigo-50 to-blue-50 dark:from-indigo-900/20 dark:to-blue-900/20 rounded-lg p-6">
                          <h4 className="font-semibold text-gray-900 dark:text-white mb-3">
                            ResNet-50
                          </h4>
                          <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                            Skip connection으로 깊은 네트워크 학습 가능
                          </p>
                          <div className="bg-white dark:bg-gray-800 rounded p-3 font-mono text-xs">
                            <code className="text-blue-600 dark:text-blue-400">
                              정확도: 96.2%<br/>
                              파라미터: 25.6M<br/>
                              속도: 45ms/image
                            </code>
                          </div>
                        </div>

                        <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg p-6">
                          <h4 className="font-semibold text-gray-900 dark:text-white mb-3">
                            DenseNet-121
                          </h4>
                          <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                            Dense connection으로 특징 재사용 극대화
                          </p>
                          <div className="bg-white dark:bg-gray-800 rounded p-3 font-mono text-xs">
                            <code className="text-purple-600 dark:text-purple-400">
                              정확도: 97.1%<br/>
                              파라미터: 8.1M<br/>
                              속도: 32ms/image
                            </code>
                          </div>
                        </div>

                        <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg p-6">
                          <h4 className="font-semibold text-gray-900 dark:text-white mb-3">
                            U-Net
                          </h4>
                          <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                            의료 영상 세그멘테이션의 표준 모델
                          </p>
                          <div className="bg-white dark:bg-gray-800 rounded p-3 font-mono text-xs">
                            <code className="text-green-600 dark:text-green-400">
                              IoU: 0.89<br/>
                              파라미터: 31M<br/>
                              속도: 78ms/image
                            </code>
                          </div>
                        </div>

                        <div className="bg-gradient-to-br from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-lg p-6">
                          <h4 className="font-semibold text-gray-900 dark:text-white mb-3">
                            Vision Transformer
                          </h4>
                          <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                            Self-attention 기반 최신 아키텍처
                          </p>
                          <div className="bg-white dark:bg-gray-800 rounded p-3 font-mono text-xs">
                            <code className="text-orange-600 dark:text-orange-400">
                              정확도: 98.3%<br/>
                              파라미터: 86M<br/>
                              속도: 112ms/image
                            </code>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
                        Transfer Learning 전략
                      </h3>
                      <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
                        <pre className="text-sm overflow-x-auto">
                          <code className="language-python text-gray-700 dark:text-gray-300">{`# ImageNet 사전학습 모델 활용
from torchvision import models
import torch.nn as nn

# ResNet50 불러오기
model = models.resnet50(pretrained=True)

# 마지막 레이어 수정 (14개 질병 분류)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 14)

# Fine-tuning 설정
for param in model.parameters():
    param.requires_grad = False
    
# 마지막 레이어만 학습
for param in model.fc.parameters():
    param.requires_grad = True`}</code>
                        </pre>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Preprocessing Section */}
            {activeSection === 'preprocessing' && (
              <div className="space-y-6">
                <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
                  <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
                    의료 영상 전처리
                  </h2>
                  
                  <div className="space-y-6">
                    <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
                      <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
                        전처리 파이프라인
                      </h3>
                      <div className="space-y-4">
                        <div className="flex items-start gap-4">
                          <div className="w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold">
                            1
                          </div>
                          <div className="flex-1">
                            <h4 className="font-semibold text-gray-900 dark:text-white">DICOM 변환</h4>
                            <p className="text-sm text-gray-600 dark:text-gray-400">
                              의료 영상 표준 포맷을 배열로 변환
                            </p>
                          </div>
                        </div>
                        <div className="flex items-start gap-4">
                          <div className="w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold">
                            2
                          </div>
                          <div className="flex-1">
                            <h4 className="font-semibold text-gray-900 dark:text-white">정규화</h4>
                            <p className="text-sm text-gray-600 dark:text-gray-400">
                              픽셀 값을 0-1 또는 -1 to 1 범위로 조정
                            </p>
                          </div>
                        </div>
                        <div className="flex items-start gap-4">
                          <div className="w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold">
                            3
                          </div>
                          <div className="flex-1">
                            <h4 className="font-semibold text-gray-900 dark:text-white">리사이징</h4>
                            <p className="text-sm text-gray-600 dark:text-gray-400">
                              모델 입력 크기에 맞게 조정 (예: 224x224)
                            </p>
                          </div>
                        </div>
                        <div className="flex items-start gap-4">
                          <div className="w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold">
                            4
                          </div>
                          <div className="flex-1">
                            <h4 className="font-semibold text-gray-900 dark:text-white">데이터 증강</h4>
                            <p className="text-sm text-gray-600 dark:text-gray-400">
                              회전, 이동, 밝기 조정으로 데이터 확장
                            </p>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
                        전처리 코드 예제
                      </h3>
                      <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
                        <pre className="text-sm overflow-x-auto">
                          <code className="language-python text-gray-700 dark:text-gray-300">{`import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_xray(image_path):
    # 이미지 로드
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    
    # 리사이징
    img = cv2.resize(img, (224, 224))
    
    # 정규화
    img = img / 255.0
    
    # 차원 추가 (채널)
    img = np.expand_dims(img, axis=-1)
    
    return img`}</code>
                        </pre>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Applications Section */}
            {activeSection === 'applications' && (
              <div className="space-y-6">
                <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
                  <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
                    실제 응용 사례
                  </h2>
                  
                  <div className="grid md:grid-cols-2 gap-6">
                    <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-6">
                      <Eye className="w-10 h-10 text-green-600 dark:text-green-400 mb-4" />
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        Google DeepMind - 안과 질환
                      </h3>
                      <p className="text-gray-600 dark:text-gray-400 mb-4">
                        OCT 스캔을 통한 50가지 이상의 안과 질환 진단
                      </p>
                      <div className="bg-gray-50 dark:bg-gray-900 rounded p-3">
                        <p className="text-sm font-mono text-gray-700 dark:text-gray-300">
                          정확도: 94.5%<br/>
                          처리시간: 30초<br/>
                          FDA 승인: 2018년
                        </p>
                      </div>
                    </div>

                    <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-6">
                      <Activity className="w-10 h-10 text-red-600 dark:text-red-400 mb-4" />
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        Stanford CheXNet - 폐렴 진단
                      </h3>
                      <p className="text-gray-600 dark:text-gray-400 mb-4">
                        흉부 X-Ray에서 14가지 질병 동시 검출
                      </p>
                      <div className="bg-gray-50 dark:bg-gray-900 rounded p-3">
                        <p className="text-sm font-mono text-gray-700 dark:text-gray-300">
                          AUC: 0.94<br/>
                          데이터셋: 112,120장<br/>
                          모델: DenseNet-121
                        </p>
                      </div>
                    </div>

                    <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-6">
                      <Brain className="w-10 h-10 text-purple-600 dark:text-purple-400 mb-4" />
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        NYU - 유방암 스크리닝
                      </h3>
                      <p className="text-gray-600 dark:text-gray-400 mb-4">
                        맘모그래피 영상으로 5년 내 유방암 위험 예측
                      </p>
                      <div className="bg-gray-50 dark:bg-gray-900 rounded p-3">
                        <p className="text-sm font-mono text-gray-700 dark:text-gray-300">
                          민감도: 90.1%<br/>
                          특이도: 89.5%<br/>
                          조기발견: 20% 향상
                        </p>
                      </div>
                    </div>

                    <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-6">
                      <Target className="w-10 h-10 text-blue-600 dark:text-blue-400 mb-4" />
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        Zebra Medical - 다중질환
                      </h3>
                      <p className="text-gray-600 dark:text-gray-400 mb-4">
                        CT, X-Ray로 골다공증, 지방간 등 다양한 질환 검출
                      </p>
                      <div className="bg-gray-50 dark:bg-gray-900 rounded p-3">
                        <p className="text-sm font-mono text-gray-700 dark:text-gray-300">
                          질환 수: 40+<br/>
                          병원 수: 100+<br/>
                          CE 마크 획득
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="mt-8 p-6 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
                    <div className="flex items-start gap-3">
                      <AlertCircle className="w-5 h-5 text-yellow-600 dark:text-yellow-400 mt-0.5" />
                      <div>
                        <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                          의료 AI 도입 시 고려사항
                        </h4>
                        <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                          <li>• FDA/CE 등 규제 기관 승인 필요</li>
                          <li>• 의료진과의 협업 체계 구축</li>
                          <li>• 지속적인 모델 업데이트와 검증</li>
                          <li>• 개인정보 보호 및 데이터 보안</li>
                        </ul>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Navigation */}
            <div className="flex justify-between items-center pt-8">
              <Link
                href="/medical-ai/chapter/introduction"
                className="flex items-center gap-2 px-6 py-3 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
              >
                <ArrowLeft className="w-5 h-5" />
                이전 챕터
              </Link>
              <Link
                href="/medical-ai/chapter/diagnosis-assistant"
                className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-red-600 to-pink-600 text-white rounded-lg hover:shadow-lg transition-all"
              >
                다음 챕터
                <ArrowRight className="w-5 h-5" />
              </Link>
            </div>
          </main>
        </div>
      </div>
    </div>
  )
}