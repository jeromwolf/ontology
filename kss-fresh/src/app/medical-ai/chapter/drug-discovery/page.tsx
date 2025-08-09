'use client'

import { useState } from 'react'
import Link from 'next/link'
import { 
  Beaker,
  Dna,
  TrendingUp,
  Target,
  Brain,
  Lightbulb,
  ChevronRight,
  ArrowLeft,
  ArrowRight,
  CheckCircle,
  AlertCircle,
  Clock,
  DollarSign,
  Activity,
  Cpu,
  Microscope,
  FlaskConical
} from 'lucide-react'

export default function DrugDiscoveryPage() {
  const [activeSection, setActiveSection] = useState('overview')

  const sections = [
    { id: 'overview', title: '개요', icon: Beaker },
    { id: 'ai-techniques', title: 'AI 기술', icon: Brain },
    { id: 'pipeline', title: '개발 파이프라인', icon: TrendingUp },
    { id: 'challenges', title: '도전 과제', icon: AlertCircle },
    { id: 'case-studies', title: '성공 사례', icon: Target }
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
                Chapter 4: AI 기반 신약 개발
              </h1>
            </div>
            <div className="flex items-center gap-3">
              <span className="px-3 py-1 bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400 rounded-full text-sm font-medium">
                고급
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
                        ? 'bg-gradient-to-r from-purple-500 to-pink-600 text-white shadow-lg'
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
                    AI 기반 신약 개발
                  </h2>
                  
                  <div className="prose prose-lg dark:prose-invert max-w-none">
                    <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
                      AI 기반 신약 개발은 인공지능 기술을 활용하여 새로운 약물을 발견하고 개발하는 과정을 
                      혁신적으로 가속화하는 접근법입니다. 전통적으로 10-15년이 걸리던 신약 개발 과정을 
                      AI를 통해 3-5년으로 단축할 수 있습니다.
                    </p>

                    <div className="grid md:grid-cols-2 gap-6 my-8">
                      <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg p-6">
                        <FlaskConical className="w-10 h-10 text-purple-600 dark:text-purple-400 mb-4" />
                        <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                          전통적 방법 vs AI
                        </h3>
                        <div className="space-y-3">
                          <div>
                            <div className="flex justify-between text-sm">
                              <span>개발 기간</span>
                              <span className="font-semibold">10-15년 → 3-5년</span>
                            </div>
                          </div>
                          <div>
                            <div className="flex justify-between text-sm">
                              <span>개발 비용</span>
                              <span className="font-semibold">$2.8B → $1B</span>
                            </div>
                          </div>
                          <div>
                            <div className="flex justify-between text-sm">
                              <span>성공률</span>
                              <span className="font-semibold">12% → 35%</span>
                            </div>
                          </div>
                        </div>
                      </div>

                      <div className="bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-lg p-6">
                        <Cpu className="w-10 h-10 text-blue-600 dark:text-blue-400 mb-4" />
                        <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                          핵심 기술
                        </h3>
                        <ul className="space-y-2 text-gray-600 dark:text-gray-300">
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-blue-500 mt-0.5" />
                            <span>분자 생성 모델</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-blue-500 mt-0.5" />
                            <span>단백질 구조 예측</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-blue-500 mt-0.5" />
                            <span>약물-표적 상호작용</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-blue-500 mt-0.5" />
                            <span>ADMET 예측</span>
                          </li>
                        </ul>
                      </div>
                    </div>

                    <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 my-8">
                      <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
                        AI 신약 개발의 장점
                      </h3>
                      <div className="grid md:grid-cols-3 gap-4">
                        <div className="text-center p-4 bg-white dark:bg-gray-800 rounded-lg">
                          <Clock className="w-8 h-8 text-purple-600 dark:text-purple-400 mx-auto mb-2" />
                          <div className="font-semibold">시간 단축</div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">
                            타겟 발굴부터 임상까지 70% 단축
                          </div>
                        </div>
                        <div className="text-center p-4 bg-white dark:bg-gray-800 rounded-lg">
                          <DollarSign className="w-8 h-8 text-green-600 dark:text-green-400 mx-auto mb-2" />
                          <div className="font-semibold">비용 절감</div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">
                            개발 비용 60% 이상 절감
                          </div>
                        </div>
                        <div className="text-center p-4 bg-white dark:bg-gray-800 rounded-lg">
                          <Target className="w-8 h-8 text-blue-600 dark:text-blue-400 mx-auto mb-2" />
                          <div className="font-semibold">정확도 향상</div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">
                            후보 물질 성공률 3배 증가
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* AI Techniques Section */}
            {activeSection === 'ai-techniques' && (
              <div className="space-y-6">
                <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
                  <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
                    신약 개발 AI 기술
                  </h2>
                  
                  <div className="space-y-8">
                    {/* Deep Generative Models */}
                    <div className="border-l-4 border-purple-500 pl-6">
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        생성형 딥러닝 모델
                      </h3>
                      <p className="text-gray-600 dark:text-gray-300 mb-4">
                        새로운 분자 구조를 생성하고 최적화하는 AI 모델
                      </p>
                      <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                        <pre className="text-sm overflow-x-auto">
                          <code className="language-python">
{`# VAE 기반 분자 생성 예제
import torch
import torch.nn as nn

class MolecularVAE(nn.Module):
    def __init__(self, vocab_size, latent_dim=128):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Embedding(vocab_size, 256),
            nn.LSTM(256, 512, batch_first=True),
            nn.Linear(512, latent_dim * 2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LSTM(512, 256, batch_first=True),
            nn.Linear(256, vocab_size)
        )
        
    def forward(self, smiles):
        # Encode SMILES to latent space
        h = self.encoder(smiles)
        mu, logvar = h.chunk(2, dim=-1)
        
        # Reparameterization trick
        z = self.reparameterize(mu, logvar)
        
        # Decode to new molecule
        return self.decoder(z)`}
                          </code>
                        </pre>
                      </div>
                    </div>

                    {/* Protein Structure Prediction */}
                    <div className="border-l-4 border-blue-500 pl-6">
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        단백질 구조 예측
                      </h3>
                      <p className="text-gray-600 dark:text-gray-300 mb-4">
                        AlphaFold와 같은 AI로 단백질 3D 구조를 예측하여 약물 타겟 발굴
                      </p>
                      <div className="grid md:grid-cols-2 gap-4">
                        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                          <h4 className="font-semibold mb-2">입력 데이터</h4>
                          <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                            <li>• 아미노산 서열</li>
                            <li>• MSA (Multiple Sequence Alignment)</li>
                            <li>• 템플릿 구조</li>
                          </ul>
                        </div>
                        <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
                          <h4 className="font-semibold mb-2">출력 결과</h4>
                          <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                            <li>• 3D 좌표</li>
                            <li>• 신뢰도 점수 (pLDDT)</li>
                            <li>• 바인딩 사이트 예측</li>
                          </ul>
                        </div>
                      </div>
                    </div>

                    {/* Drug-Target Interaction */}
                    <div className="border-l-4 border-green-500 pl-6">
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        약물-표적 상호작용 예측
                      </h3>
                      <p className="text-gray-600 dark:text-gray-300 mb-4">
                        그래프 신경망을 활용한 약물과 단백질의 결합 친화도 예측
                      </p>
                      <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                        <pre className="text-sm overflow-x-auto">
                          <code className="language-python">
{`# GNN 기반 DTI 예측
import torch_geometric
from torch_geometric.nn import GCNConv, global_mean_pool

class DTI_GNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.drug_conv1 = GCNConv(node_features, 128)
        self.drug_conv2 = GCNConv(128, 64)
        
        self.protein_conv1 = GCNConv(node_features, 128)
        self.protein_conv2 = GCNConv(128, 64)
        
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, drug_graph, protein_graph):
        # Process drug molecule
        drug_x = self.drug_conv1(drug_graph.x, drug_graph.edge_index)
        drug_x = self.drug_conv2(drug_x, drug_graph.edge_index)
        drug_x = global_mean_pool(drug_x, drug_graph.batch)
        
        # Process protein
        protein_x = self.protein_conv1(protein_graph.x, protein_graph.edge_index)
        protein_x = self.protein_conv2(protein_x, protein_graph.edge_index)
        protein_x = global_mean_pool(protein_x, protein_graph.batch)
        
        # Combine and predict binding affinity
        combined = torch.cat([drug_x, protein_x], dim=1)
        return self.fc(combined)`}
                          </code>
                        </pre>
                      </div>
                    </div>

                    {/* ADMET Prediction */}
                    <div className="border-l-4 border-orange-500 pl-6">
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        ADMET 특성 예측
                      </h3>
                      <p className="text-gray-600 dark:text-gray-300 mb-4">
                        약물의 흡수, 분포, 대사, 배설, 독성을 AI로 예측
                      </p>
                      <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
                        <div className="text-center p-3 bg-orange-50 dark:bg-orange-900/20 rounded">
                          <div className="font-bold text-orange-600 dark:text-orange-400">A</div>
                          <div className="text-xs">흡수</div>
                        </div>
                        <div className="text-center p-3 bg-orange-50 dark:bg-orange-900/20 rounded">
                          <div className="font-bold text-orange-600 dark:text-orange-400">D</div>
                          <div className="text-xs">분포</div>
                        </div>
                        <div className="text-center p-3 bg-orange-50 dark:bg-orange-900/20 rounded">
                          <div className="font-bold text-orange-600 dark:text-orange-400">M</div>
                          <div className="text-xs">대사</div>
                        </div>
                        <div className="text-center p-3 bg-orange-50 dark:bg-orange-900/20 rounded">
                          <div className="font-bold text-orange-600 dark:text-orange-400">E</div>
                          <div className="text-xs">배설</div>
                        </div>
                        <div className="text-center p-3 bg-orange-50 dark:bg-orange-900/20 rounded">
                          <div className="font-bold text-orange-600 dark:text-orange-400">T</div>
                          <div className="text-xs">독성</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Pipeline Section */}
            {activeSection === 'pipeline' && (
              <div className="space-y-6">
                <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
                  <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
                    AI 신약 개발 파이프라인
                  </h2>
                  
                  <div className="space-y-8">
                    {/* Pipeline Steps */}
                    <div className="relative">
                      <div className="absolute left-8 top-0 bottom-0 w-0.5 bg-gradient-to-b from-purple-500 to-pink-600"></div>
                      
                      <div className="space-y-6">
                        {/* Step 1 */}
                        <div className="relative flex items-start gap-6">
                          <div className="w-16 h-16 bg-gradient-to-br from-purple-500 to-pink-600 rounded-full flex items-center justify-center text-white font-bold shadow-lg z-10">
                            1
                          </div>
                          <div className="flex-1">
                            <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                              타겟 발굴 (Target Discovery)
                            </h3>
                            <p className="text-gray-600 dark:text-gray-300 mb-3">
                              질병 관련 단백질 타겟을 AI로 식별하고 검증
                            </p>
                            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                                <li>• 유전체 데이터 분석</li>
                                <li>• 질병 경로 매핑</li>
                                <li>• 드러그어빌리티 평가</li>
                                <li>• 소요 시간: 3-6개월</li>
                              </ul>
                            </div>
                          </div>
                        </div>

                        {/* Step 2 */}
                        <div className="relative flex items-start gap-6">
                          <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-cyan-600 rounded-full flex items-center justify-center text-white font-bold shadow-lg z-10">
                            2
                          </div>
                          <div className="flex-1">
                            <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                              선도물질 발굴 (Lead Discovery)
                            </h3>
                            <p className="text-gray-600 dark:text-gray-300 mb-3">
                              가상 스크리닝으로 수백만 개 화합물 중 후보 선별
                            </p>
                            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                                <li>• 가상 화합물 라이브러리 스크리닝</li>
                                <li>• 분자 도킹 시뮬레이션</li>
                                <li>• 히트 화합물 선별</li>
                                <li>• 소요 시간: 2-4개월</li>
                              </ul>
                            </div>
                          </div>
                        </div>

                        {/* Step 3 */}
                        <div className="relative flex items-start gap-6">
                          <div className="w-16 h-16 bg-gradient-to-br from-green-500 to-emerald-600 rounded-full flex items-center justify-center text-white font-bold shadow-lg z-10">
                            3
                          </div>
                          <div className="flex-1">
                            <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                              선도물질 최적화 (Lead Optimization)
                            </h3>
                            <p className="text-gray-600 dark:text-gray-300 mb-3">
                              AI로 화합물 구조를 최적화하여 효능과 안전성 향상
                            </p>
                            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                                <li>• 구조-활성 관계 (SAR) 분석</li>
                                <li>• ADMET 특성 최적화</li>
                                <li>• 생성 모델로 신규 유도체 설계</li>
                                <li>• 소요 시간: 6-12개월</li>
                              </ul>
                            </div>
                          </div>
                        </div>

                        {/* Step 4 */}
                        <div className="relative flex items-start gap-6">
                          <div className="w-16 h-16 bg-gradient-to-br from-orange-500 to-red-600 rounded-full flex items-center justify-center text-white font-bold shadow-lg z-10">
                            4
                          </div>
                          <div className="flex-1">
                            <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                              전임상 시험 (Preclinical)
                            </h3>
                            <p className="text-gray-600 dark:text-gray-300 mb-3">
                              AI 예측 모델로 동물실험 최소화 및 효율화
                            </p>
                            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                                <li>• 독성 예측 모델</li>
                                <li>• PK/PD 시뮬레이션</li>
                                <li>• 용량 최적화</li>
                                <li>• 소요 시간: 1-2년</li>
                              </ul>
                            </div>
                          </div>
                        </div>

                        {/* Step 5 */}
                        <div className="relative flex items-start gap-6">
                          <div className="w-16 h-16 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-full flex items-center justify-center text-white font-bold shadow-lg z-10">
                            5
                          </div>
                          <div className="flex-1">
                            <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                              임상 시험 (Clinical Trials)
                            </h3>
                            <p className="text-gray-600 dark:text-gray-300 mb-3">
                              AI로 환자 선별 및 임상 설계 최적화
                            </p>
                            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                                <li>• 환자 코호트 선별</li>
                                <li>• 바이오마커 예측</li>
                                <li>• 부작용 모니터링</li>
                                <li>• 소요 시간: 3-7년</li>
                              </ul>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Timeline Summary */}
                    <div className="bg-gradient-to-r from-purple-100 to-pink-100 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg p-6">
                      <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
                        전체 개발 기간 비교
                      </h3>
                      <div className="space-y-3">
                        <div>
                          <div className="flex justify-between mb-1">
                            <span className="text-sm">전통적 방법</span>
                            <span className="text-sm font-semibold">10-15년</span>
                          </div>
                          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                            <div className="bg-red-500 h-2 rounded-full" style={{width: '100%'}}></div>
                          </div>
                        </div>
                        <div>
                          <div className="flex justify-between mb-1">
                            <span className="text-sm">AI 활용</span>
                            <span className="text-sm font-semibold">3-5년</span>
                          </div>
                          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                            <div className="bg-green-500 h-2 rounded-full" style={{width: '40%'}}></div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Challenges Section */}
            {activeSection === 'challenges' && (
              <div className="space-y-6">
                <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
                  <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
                    도전 과제와 한계
                  </h2>
                  
                  <div className="space-y-6">
                    <div className="border-l-4 border-red-500 pl-6">
                      <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                        데이터 품질과 가용성
                      </h3>
                      <p className="text-gray-600 dark:text-gray-300 mb-3">
                        고품질의 대규모 데이터셋 부족은 AI 모델 성능의 주요 제약 요인
                      </p>
                      <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                        <li>• 실패한 약물 데이터의 부족</li>
                        <li>• 데이터 표준화 문제</li>
                        <li>• 지적재산권으로 인한 데이터 공유 제한</li>
                      </ul>
                    </div>

                    <div className="border-l-4 border-blue-500 pl-6">
                      <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                        생물학적 복잡성
                      </h3>
                      <p className="text-gray-600 dark:text-gray-300 mb-3">
                        인체의 복잡한 생물학적 시스템을 완벽히 모델링하기 어려움
                      </p>
                      <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                        <li>• 다중 타겟 상호작용</li>
                        <li>• 오프타겟 효과 예측</li>
                        <li>• 개인별 유전적 변이</li>
                      </ul>
                    </div>

                    <div className="border-l-4 border-purple-500 pl-6">
                      <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                        검증과 신뢰성
                      </h3>
                      <p className="text-gray-600 dark:text-gray-300 mb-3">
                        AI 예측 결과의 실험적 검증 필요성과 비용
                      </p>
                      <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                        <li>• 인실리코 예측과 실제 결과의 차이</li>
                        <li>• 모델 해석 가능성 부족</li>
                        <li>• 규제 기관의 AI 모델 승인 기준</li>
                      </ul>
                    </div>

                    <div className="border-l-4 border-green-500 pl-6">
                      <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                        통합과 협업
                      </h3>
                      <p className="text-gray-600 dark:text-gray-300 mb-3">
                        전통적 제약 개발 프로세스와 AI의 통합 어려움
                      </p>
                      <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                        <li>• 조직 문화의 변화 필요</li>
                        <li>• 다학제간 전문가 협업</li>
                        <li>• 기존 인프라와의 호환성</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Case Studies Section */}
            {activeSection === 'case-studies' && (
              <div className="space-y-6">
                <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
                  <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
                    성공 사례
                  </h2>
                  
                  <div className="space-y-6">
                    {/* Case 1 */}
                    <div className="bg-gradient-to-r from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-lg p-6">
                      <div className="flex items-start gap-4">
                        <div className="w-12 h-12 bg-blue-600 dark:bg-blue-500 rounded-lg flex items-center justify-center text-white font-bold">
                          1
                        </div>
                        <div className="flex-1">
                          <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                            Exscientia - DSP-1181
                          </h3>
                          <p className="text-gray-600 dark:text-gray-300 mb-3">
                            강박장애 치료제를 AI로 개발하여 임상 1상 진입
                          </p>
                          <div className="grid md:grid-cols-3 gap-4 mt-4">
                            <div className="text-center">
                              <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">12개월</div>
                              <div className="text-sm text-gray-500">개발 기간</div>
                            </div>
                            <div className="text-center">
                              <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">350개</div>
                              <div className="text-sm text-gray-500">합성 화합물</div>
                            </div>
                            <div className="text-center">
                              <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">80%</div>
                              <div className="text-sm text-gray-500">시간 단축</div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Case 2 */}
                    <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg p-6">
                      <div className="flex items-start gap-4">
                        <div className="w-12 h-12 bg-purple-600 dark:bg-purple-500 rounded-lg flex items-center justify-center text-white font-bold">
                          2
                        </div>
                        <div className="flex-1">
                          <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                            Atomwise - COVID-19 치료제
                          </h3>
                          <p className="text-gray-600 dark:text-gray-300 mb-3">
                            AI 가상 스크리닝으로 COVID-19 치료 후보물질 발견
                          </p>
                          <div className="grid md:grid-cols-3 gap-4 mt-4">
                            <div className="text-center">
                              <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">4일</div>
                              <div className="text-sm text-gray-500">스크리닝 시간</div>
                            </div>
                            <div className="text-center">
                              <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">6.8B</div>
                              <div className="text-sm text-gray-500">검토 화합물</div>
                            </div>
                            <div className="text-center">
                              <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">100+</div>
                              <div className="text-sm text-gray-500">후보 물질</div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Case 3 */}
                    <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg p-6">
                      <div className="flex items-start gap-4">
                        <div className="w-12 h-12 bg-green-600 dark:bg-green-500 rounded-lg flex items-center justify-center text-white font-bold">
                          3
                        </div>
                        <div className="flex-1">
                          <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                            Insilico Medicine - 섬유증 치료제
                          </h3>
                          <p className="text-gray-600 dark:text-gray-300 mb-3">
                            특발성 폐섬유증 치료제를 AI로 설계하여 임상 진입
                          </p>
                          <div className="grid md:grid-cols-3 gap-4 mt-4">
                            <div className="text-center">
                              <div className="text-2xl font-bold text-green-600 dark:text-green-400">18개월</div>
                              <div className="text-sm text-gray-500">발견→전임상</div>
                            </div>
                            <div className="text-center">
                              <div className="text-2xl font-bold text-green-600 dark:text-green-400">$2.6M</div>
                              <div className="text-sm text-gray-500">개발 비용</div>
                            </div>
                            <div className="text-center">
                              <div className="text-2xl font-bold text-green-600 dark:text-green-400">90%</div>
                              <div className="text-sm text-gray-500">비용 절감</div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Future Outlook */}
                    <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
                      <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
                        시장 전망
                      </h3>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="text-center">
                          <div className="text-2xl font-bold text-red-600 dark:text-red-400">$13.8B</div>
                          <div className="text-sm text-gray-500">2028년 시장 규모</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">40.2%</div>
                          <div className="text-sm text-gray-500">연평균 성장률</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-green-600 dark:text-green-400">150+</div>
                          <div className="text-sm text-gray-500">AI 신약개발 기업</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">50+</div>
                          <div className="text-sm text-gray-500">임상단계 AI 약물</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Navigation */}
            <div className="flex justify-between items-center pt-8">
              <Link
                href="/medical-ai/chapter/diagnosis-assistant"
                className="flex items-center gap-2 px-6 py-3 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
              >
                <ArrowLeft className="w-5 h-5" />
                이전 챕터
              </Link>
              <Link
                href="/medical-ai/chapter/genomics"
                className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg hover:shadow-lg transition-all"
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