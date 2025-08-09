'use client'

import { useState } from 'react'
import Link from 'next/link'
import { 
  Dna,
  Brain,
  TrendingUp,
  Target,
  Database,
  Lightbulb,
  ChevronRight,
  ArrowLeft,
  ArrowRight,
  CheckCircle,
  AlertCircle,
  Activity,
  Cpu,
  Microscope,
  FileText,
  GitBranch,
  Zap
} from 'lucide-react'

export default function GenomicsPage() {
  const [activeSection, setActiveSection] = useState('overview')

  const sections = [
    { id: 'overview', title: '개요', icon: Dna },
    { id: 'sequencing', title: '시퀀싱 기술', icon: Microscope },
    { id: 'analysis', title: '유전체 분석', icon: Brain },
    { id: 'applications', title: '임상 응용', icon: Target },
    { id: 'precision-medicine', title: '정밀의료', icon: Activity }
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
                Chapter 5: 유전체 분석 AI
              </h1>
            </div>
            <div className="flex items-center gap-3">
              <span className="px-3 py-1 bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400 rounded-full text-sm font-medium">
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
                        ? 'bg-gradient-to-r from-green-500 to-emerald-600 text-white shadow-lg'
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
                    유전체 분석과 AI
                  </h2>
                  
                  <div className="prose prose-lg dark:prose-invert max-w-none">
                    <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
                      유전체 분석 AI는 인간 게놈의 30억 개 염기쌍을 해석하고, 
                      질병과 관련된 유전적 변이를 찾아내며, 개인 맞춤형 치료법을 
                      제안하는 혁신적인 기술입니다.
                    </p>

                    <div className="grid md:grid-cols-2 gap-6 my-8">
                      <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg p-6">
                        <Dna className="w-10 h-10 text-green-600 dark:text-green-400 mb-4" />
                        <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                          게놈 데이터 규모
                        </h3>
                        <ul className="space-y-2 text-gray-600 dark:text-gray-300">
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-green-500 mt-0.5" />
                            <span>30억 개 염기쌍</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-green-500 mt-0.5" />
                            <span>2만 개 유전자</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-green-500 mt-0.5" />
                            <span>300만 개 변이</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-green-500 mt-0.5" />
                            <span>100GB 원시 데이터</span>
                          </li>
                        </ul>
                      </div>

                      <div className="bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-lg p-6">
                        <Brain className="w-10 h-10 text-blue-600 dark:text-blue-400 mb-4" />
                        <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                          AI 분석 영역
                        </h3>
                        <ul className="space-y-2 text-gray-600 dark:text-gray-300">
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-blue-500 mt-0.5" />
                            <span>변이 검출</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-blue-500 mt-0.5" />
                            <span>기능 예측</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-blue-500 mt-0.5" />
                            <span>질병 연관성</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-blue-500 mt-0.5" />
                            <span>약물 반응 예측</span>
                          </li>
                        </ul>
                      </div>
                    </div>

                    <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 my-8">
                      <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
                        시퀀싱 비용 혁명
                      </h3>
                      <div className="grid grid-cols-3 gap-4">
                        <div className="text-center">
                          <div className="text-3xl font-bold text-green-600 dark:text-green-400">$100M</div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">2001년</div>
                        </div>
                        <div className="text-center">
                          <div className="text-3xl font-bold text-blue-600 dark:text-blue-400">$1,000</div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">2020년</div>
                        </div>
                        <div className="text-center">
                          <div className="text-3xl font-bold text-purple-600 dark:text-purple-400">$100</div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">2025년 예상</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Sequencing Section */}
            {activeSection === 'sequencing' && (
              <div className="space-y-6">
                <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
                  <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
                    차세대 시퀀싱과 AI
                  </h2>
                  
                  <div className="space-y-8">
                    {/* NGS Technologies */}
                    <div className="border-l-4 border-green-500 pl-6">
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        NGS 기술과 AI 통합
                      </h3>
                      <p className="text-gray-600 dark:text-gray-300 mb-4">
                        차세대 시퀀싱(NGS) 데이터를 AI로 실시간 분석
                      </p>
                      <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                        <pre className="text-sm overflow-x-auto">
                          <code className="language-python">
{`# Base Calling with Deep Learning
import torch
import torch.nn as nn

class BaseCaller(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(4, 256, kernel_size=11, padding=5),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )
        self.lstm = nn.LSTM(256, 512, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(1024, 5)  # A, C, G, T, N
        
    def forward(self, signal):
        # signal: raw electrical signal from nanopore
        x = self.conv1d(signal)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        bases = self.classifier(lstm_out)
        return torch.softmax(bases, dim=-1)

# Variant Calling
class VariantCaller(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8),
            num_layers=6
        )
        self.variant_head = nn.Linear(512, 4)  # SNP, Indel, SV, CNV
        
    def forward(self, reads, reference):
        # Align and detect variants
        encoded = self.encoder(reads)
        variants = self.variant_head(encoded)
        return variants`}
                          </code>
                        </pre>
                      </div>
                    </div>

                    {/* Sequencing Technologies Comparison */}
                    <div className="grid md:grid-cols-3 gap-4">
                      <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
                        <h4 className="font-semibold mb-2 text-gray-900 dark:text-white">
                          Illumina
                        </h4>
                        <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                          <li>• Short reads (150-300bp)</li>
                          <li>• 높은 정확도 (99.9%)</li>
                          <li>• 대용량 처리</li>
                          <li>• WGS, WES 적합</li>
                        </ul>
                      </div>
                      <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                        <h4 className="font-semibold mb-2 text-gray-900 dark:text-white">
                          Oxford Nanopore
                        </h4>
                        <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                          <li>• Long reads (&gt;10kb)</li>
                          <li>• 실시간 시퀀싱</li>
                          <li>• 휴대용 장비</li>
                          <li>• 구조 변이 검출</li>
                        </ul>
                      </div>
                      <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
                        <h4 className="font-semibold mb-2 text-gray-900 dark:text-white">
                          PacBio HiFi
                        </h4>
                        <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                          <li>• Long reads (10-25kb)</li>
                          <li>• 높은 정확도 (99.9%)</li>
                          <li>• Phasing 가능</li>
                          <li>• De novo assembly</li>
                        </ul>
                      </div>
                    </div>

                    {/* Quality Control */}
                    <div className="border-l-4 border-blue-500 pl-6">
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        AI 기반 품질 관리
                      </h3>
                      <p className="text-gray-600 dark:text-gray-300 mb-4">
                        시퀀싱 데이터의 품질을 실시간으로 평가하고 오류를 자동 보정
                      </p>
                      <div className="grid md:grid-cols-2 gap-4">
                        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                          <h4 className="font-semibold mb-2">품질 지표</h4>
                          <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                            <li>• Phred Quality Score</li>
                            <li>• Coverage Depth</li>
                            <li>• GC Content Bias</li>
                            <li>• Duplicate Rate</li>
                          </ul>
                        </div>
                        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                          <h4 className="font-semibold mb-2">AI 보정</h4>
                          <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                            <li>• Error Correction</li>
                            <li>• Adapter Trimming</li>
                            <li>• Contamination Detection</li>
                            <li>• Batch Effect Removal</li>
                          </ul>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Analysis Section */}
            {activeSection === 'analysis' && (
              <div className="space-y-6">
                <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
                  <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
                    유전체 데이터 분석
                  </h2>
                  
                  <div className="space-y-8">
                    {/* Variant Analysis */}
                    <div className="border-l-4 border-purple-500 pl-6">
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        변이 분석 파이프라인
                      </h3>
                      <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                        <pre className="text-sm overflow-x-auto">
                          <code className="language-python">
{`# Genomic Variant Analysis Pipeline
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class VariantAnalyzer:
    def __init__(self):
        self.pathogenicity_model = RandomForestClassifier()
        self.load_annotations()
        
    def annotate_variant(self, chrom, pos, ref, alt):
        """변이 주석 달기"""
        annotations = {
            'gene': self.get_gene(chrom, pos),
            'consequence': self.predict_consequence(ref, alt),
            'maf': self.get_population_frequency(chrom, pos, alt),
            'conservation': self.get_conservation_score(chrom, pos),
            'protein_impact': self.predict_protein_impact(ref, alt)
        }
        return annotations
    
    def predict_pathogenicity(self, variant_features):
        """병원성 예측"""
        # Features: SIFT, PolyPhen, CADD, etc.
        probability = self.pathogenicity_model.predict_proba(variant_features)
        classification = ['Benign', 'Likely Benign', 'VUS', 
                         'Likely Pathogenic', 'Pathogenic']
        return classification[np.argmax(probability)]
    
    def prioritize_variants(self, variants_df):
        """변이 우선순위 결정"""
        # Score based on multiple factors
        variants_df['priority_score'] = (
            variants_df['pathogenicity'] * 0.4 +
            variants_df['gene_importance'] * 0.3 +
            variants_df['phenotype_match'] * 0.3
        )
        return variants_df.sort_values('priority_score', ascending=False)`}
                          </code>
                        </pre>
                      </div>
                    </div>

                    {/* Structural Variants */}
                    <div className="border-l-4 border-green-500 pl-6">
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        구조 변이 검출
                      </h3>
                      <p className="text-gray-600 dark:text-gray-300 mb-4">
                        대규모 삽입, 결실, 역위, 전좌 등 구조적 변이를 AI로 검출
                      </p>
                      <div className="grid md:grid-cols-4 gap-3">
                        <div className="text-center p-3 bg-green-50 dark:bg-green-900/20 rounded">
                          <GitBranch className="w-8 h-8 text-green-600 dark:text-green-400 mx-auto mb-2" />
                          <div className="text-sm font-semibold">Deletion</div>
                          <div className="text-xs text-gray-500">결실</div>
                        </div>
                        <div className="text-center p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                          <GitBranch className="w-8 h-8 text-blue-600 dark:text-blue-400 mx-auto mb-2 rotate-180" />
                          <div className="text-sm font-semibold">Insertion</div>
                          <div className="text-xs text-gray-500">삽입</div>
                        </div>
                        <div className="text-center p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                          <GitBranch className="w-8 h-8 text-purple-600 dark:text-purple-400 mx-auto mb-2 rotate-90" />
                          <div className="text-sm font-semibold">Inversion</div>
                          <div className="text-xs text-gray-500">역위</div>
                        </div>
                        <div className="text-center p-3 bg-orange-50 dark:bg-orange-900/20 rounded">
                          <GitBranch className="w-8 h-8 text-orange-600 dark:text-orange-400 mx-auto mb-2" />
                          <div className="text-sm font-semibold">Translocation</div>
                          <div className="text-xs text-gray-500">전좌</div>
                        </div>
                      </div>
                    </div>

                    {/* Expression Analysis */}
                    <div className="border-l-4 border-orange-500 pl-6">
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        유전자 발현 분석
                      </h3>
                      <p className="text-gray-600 dark:text-gray-300 mb-4">
                        RNA-seq 데이터를 통한 유전자 발현 패턴 분석
                      </p>
                      <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                        <pre className="text-sm overflow-x-auto">
                          <code className="language-python">
{`# Differential Expression Analysis
class RNASeqAnalyzer:
    def differential_expression(self, counts, conditions):
        """차등 발현 유전자 분석"""
        # Normalize counts
        normalized = self.deseq2_normalization(counts)
        
        # Statistical testing
        de_genes = []
        for gene in normalized.index:
            control = normalized.loc[gene, conditions == 'control']
            treatment = normalized.loc[gene, conditions == 'treatment']
            
            # Calculate fold change and p-value
            fold_change = np.mean(treatment) / np.mean(control)
            p_value = stats.ttest_ind(control, treatment)[1]
            
            de_genes.append({
                'gene': gene,
                'log2FC': np.log2(fold_change),
                'p_value': p_value,
                'q_value': self.adjust_pvalue(p_value)
            })
        
        return pd.DataFrame(de_genes)`}
                          </code>
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
                    임상 응용 분야
                  </h2>
                  
                  <div className="grid md:grid-cols-2 gap-6">
                    {/* Cancer Genomics */}
                    <div className="bg-gradient-to-br from-red-50 to-pink-50 dark:from-red-900/20 dark:to-pink-900/20 rounded-lg p-6">
                      <Target className="w-10 h-10 text-red-600 dark:text-red-400 mb-4" />
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        암 유전체학
                      </h3>
                      <ul className="space-y-2 text-gray-600 dark:text-gray-300">
                        <li>• 종양 변이 부담 (TMB) 분석</li>
                        <li>• 드라이버 변이 식별</li>
                        <li>• 약물 저항성 예측</li>
                        <li>• 액체 생검 (ctDNA)</li>
                        <li>• 면역치료 반응 예측</li>
                      </ul>
                    </div>

                    {/* Rare Disease */}
                    <div className="bg-gradient-to-br from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 rounded-lg p-6">
                      <Microscope className="w-10 h-10 text-purple-600 dark:text-purple-400 mb-4" />
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        희귀질환 진단
                      </h3>
                      <ul className="space-y-2 text-gray-600 dark:text-gray-300">
                        <li>• 멘델 질환 원인 변이 발견</li>
                        <li>• 가족 유전체 분석 (Trio)</li>
                        <li>• 표현형-유전형 매칭</li>
                        <li>• 신규 질환 유전자 발견</li>
                        <li>• 진단 오디세이 단축</li>
                      </ul>
                    </div>

                    {/* Pharmacogenomics */}
                    <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg p-6">
                      <Activity className="w-10 h-10 text-green-600 dark:text-green-400 mb-4" />
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        약물유전체학
                      </h3>
                      <ul className="space-y-2 text-gray-600 dark:text-gray-300">
                        <li>• 약물 대사 효소 변이</li>
                        <li>• 개인별 약물 용량 조절</li>
                        <li>• 부작용 위험 예측</li>
                        <li>• 약물 상호작용 분석</li>
                        <li>• 치료 반응 예측</li>
                      </ul>
                    </div>

                    {/* Prenatal Testing */}
                    <div className="bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-lg p-6">
                      <FileText className="w-10 h-10 text-blue-600 dark:text-blue-400 mb-4" />
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        산전 진단
                      </h3>
                      <ul className="space-y-2 text-gray-600 dark:text-gray-300">
                        <li>• NIPT (비침습적 산전검사)</li>
                        <li>• 염색체 이상 검출</li>
                        <li>• 단일 유전자 질환 스크리닝</li>
                        <li>• 태아 성별 확인</li>
                        <li>• 보인자 검사</li>
                      </ul>
                    </div>
                  </div>

                  {/* Success Metrics */}
                  <div className="mt-8 bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
                    <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
                      임상 성과
                    </h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="text-center">
                        <div className="text-2xl font-bold text-green-600 dark:text-green-400">35%</div>
                        <div className="text-sm text-gray-500">희귀질환 진단률</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">60%</div>
                        <div className="text-sm text-gray-500">암 표적치료 매칭</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">85%</div>
                        <div className="text-sm text-gray-500">약물 부작용 예방</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">99%</div>
                        <div className="text-sm text-gray-500">NIPT 정확도</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Precision Medicine Section */}
            {activeSection === 'precision-medicine' && (
              <div className="space-y-6">
                <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
                  <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
                    정밀의료 실현
                  </h2>
                  
                  <div className="space-y-8">
                    {/* Personalized Treatment */}
                    <div className="bg-gradient-to-r from-purple-100 to-pink-100 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg p-6">
                      <h3 className="text-2xl font-semibold mb-4 text-gray-900 dark:text-white">
                        개인 맞춤형 치료
                      </h3>
                      <div className="grid md:grid-cols-3 gap-4">
                        <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                          <Zap className="w-8 h-8 text-purple-600 dark:text-purple-400 mb-2" />
                          <h4 className="font-semibold mb-2">유전체 프로파일링</h4>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            개인의 전체 유전체 정보를 분석하여 질병 위험도와 약물 반응성 예측
                          </p>
                        </div>
                        <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                          <Database className="w-8 h-8 text-blue-600 dark:text-blue-400 mb-2" />
                          <h4 className="font-semibold mb-2">빅데이터 통합</h4>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            유전체, 임상, 생활습관 데이터를 통합하여 종합적 건강 관리
                          </p>
                        </div>
                        <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                          <Target className="w-8 h-8 text-green-600 dark:text-green-400 mb-2" />
                          <h4 className="font-semibold mb-2">표적 치료</h4>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            특정 유전자 변이에 맞는 표적 치료제 선택과 용량 최적화
                          </p>
                        </div>
                      </div>
                    </div>

                    {/* Population Genomics */}
                    <div className="border-l-4 border-blue-500 pl-6">
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        인구 유전체학
                      </h3>
                      <p className="text-gray-600 dark:text-gray-300 mb-4">
                        대규모 인구 집단의 유전체 데이터를 분석하여 질병 예방과 공중보건 향상
                      </p>
                      <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                        <div className="grid md:grid-cols-2 gap-4">
                          <div>
                            <h4 className="font-semibold mb-2">주요 프로젝트</h4>
                            <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                              <li>• UK Biobank (50만명)</li>
                              <li>• All of Us (100만명)</li>
                              <li>• gnomAD (14만명)</li>
                              <li>• 한국인 유전체 프로젝트</li>
                            </ul>
                          </div>
                          <div>
                            <h4 className="font-semibold mb-2">활용 분야</h4>
                            <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                              <li>• 질병 위험 예측 모델</li>
                              <li>• 인종별 약물 반응 차이</li>
                              <li>• 희귀 변이 발견</li>
                              <li>• 진화 의학 연구</li>
                            </ul>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Future Vision */}
                    <div className="bg-gradient-to-r from-green-100 to-emerald-100 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg p-6">
                      <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
                        미래 전망
                      </h3>
                      <div className="space-y-4">
                        <div className="flex items-start gap-3">
                          <Lightbulb className="w-6 h-6 text-yellow-500 mt-1" />
                          <div>
                            <h4 className="font-semibold text-gray-900 dark:text-white">실시간 유전체 모니터링</h4>
                            <p className="text-sm text-gray-600 dark:text-gray-400">
                              웨어러블 기기와 연동된 실시간 유전자 발현 변화 추적
                            </p>
                          </div>
                        </div>
                        <div className="flex items-start gap-3">
                          <Lightbulb className="w-6 h-6 text-yellow-500 mt-1" />
                          <div>
                            <h4 className="font-semibold text-gray-900 dark:text-white">AI 기반 신약 개발</h4>
                            <p className="text-sm text-gray-600 dark:text-gray-400">
                              개인 유전체에 최적화된 맞춤형 약물 설계
                            </p>
                          </div>
                        </div>
                        <div className="flex items-start gap-3">
                          <Lightbulb className="w-6 h-6 text-yellow-500 mt-1" />
                          <div>
                            <h4 className="font-semibold text-gray-900 dark:text-white">예방 중심 의료</h4>
                            <p className="text-sm text-gray-600 dark:text-gray-400">
                              출생 전부터 시작되는 평생 건강 관리 시스템
                            </p>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Market Stats */}
                    <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
                      <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
                        시장 전망
                      </h3>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                        <div>
                          <div className="text-2xl font-bold text-green-600 dark:text-green-400">$54B</div>
                          <div className="text-sm text-gray-500">2028년 시장규모</div>
                        </div>
                        <div>
                          <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">18.7%</div>
                          <div className="text-sm text-gray-500">연평균 성장률</div>
                        </div>
                        <div>
                          <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">60M</div>
                          <div className="text-sm text-gray-500">2025년 시퀀싱 인구</div>
                        </div>
                        <div>
                          <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">$100</div>
                          <div className="text-sm text-gray-500">전장 유전체 비용</div>
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
                href="/medical-ai/chapter/drug-discovery"
                className="flex items-center gap-2 px-6 py-3 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
              >
                <ArrowLeft className="w-5 h-5" />
                이전 챕터
              </Link>
              <Link
                href="/medical-ai/chapter/patient-monitoring"
                className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-green-600 to-emerald-600 text-white rounded-lg hover:shadow-lg transition-all"
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