'use client';

import { useState } from 'react';
import { Copy, CheckCircle, Brain } from 'lucide-react';

export default function Chapter7() {
  const [copiedCode, setCopiedCode] = useState<string | null>(null)

  const copyCode = (code: string, id: string) => {
    navigator.clipboard.writeText(code)
    setCopiedCode(id)
    setTimeout(() => setCopiedCode(null), 2000)
  }

  const integrationCode = `# Multi-omics 데이터 통합 분석
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import networkx as nx

class MultiOmicsIntegrator:
    def __init__(self):
        self.genomics_data = None
        self.transcriptomics_data = None
        self.proteomics_data = None
        self.metabolomics_data = None
        
    def load_omics_data(self, data_dict):
        """각 오믹스 데이터 로드 및 전처리"""
        self.genomics_data = self.preprocess_data(data_dict['genomics'])
        self.transcriptomics_data = self.preprocess_data(data_dict['transcriptomics'])
        self.proteomics_data = self.preprocess_data(data_dict['proteomics'])
        self.metabolomics_data = self.preprocess_data(data_dict['metabolomics'])
    
    def preprocess_data(self, data):
        """데이터 정규화 및 결측치 처리"""
        # 결측치 처리
        data = data.fillna(data.mean())
        
        # 정규화
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        return pd.DataFrame(data_scaled, 
                          index=data.index, 
                          columns=data.columns)
    
    def integrate_by_similarity(self):
        """Similarity Network Fusion (SNF) 기반 통합"""
        networks = []
        
        # 각 오믹스 데이터에서 유사도 네트워크 생성
        for data in [self.genomics_data, self.transcriptomics_data, 
                    self.proteomics_data, self.metabolomics_data]:
            if data is not None:
                similarity = self.compute_similarity_network(data)
                networks.append(similarity)
        
        # 네트워크 융합
        fused_network = self.fuse_networks(networks)
        return fused_network
    
    def pathway_enrichment(self, gene_list, database='KEGG'):
        """Pathway enrichment 분석"""
        enriched_pathways = []
        
        # 실제로는 KEGG/Reactome API 호출
        # 여기서는 예시 결과
        pathways = {
            'MAPK signaling pathway': {'p_value': 0.001, 'genes': 15},
            'Cell cycle': {'p_value': 0.003, 'genes': 12},
            'p53 signaling pathway': {'p_value': 0.005, 'genes': 10}
        }
        
        for pathway, stats in pathways.items():
            if stats['p_value'] < 0.05:
                enriched_pathways.append({
                    'pathway': pathway,
                    'p_value': stats['p_value'],
                    'gene_count': stats['genes']
                })
        
        return enriched_pathways
    
    def identify_biomarkers(self, clinical_outcome):
        """멀티오믹스 바이오마커 발굴"""
        biomarkers = {}
        
        # 각 오믹스 레벨에서 특징 선택
        biomarkers['genomic'] = self.select_features(
            self.genomics_data, clinical_outcome
        )
        biomarkers['transcriptomic'] = self.select_features(
            self.transcriptomics_data, clinical_outcome
        )
        biomarkers['proteomic'] = self.select_features(
            self.proteomics_data, clinical_outcome
        )
        
        # 통합 바이오마커 스코어 계산
        integrated_score = self.calculate_integrated_score(biomarkers)
        
        return biomarkers, integrated_score`

  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          1. 멀티오믹스 통합의 중요성
        </h2>
        <p className="mb-4">
          단일 오믹스 데이터만으로는 생물학적 시스템의 복잡성을 완전히 이해할 수 없습니다.
          멀티오믹스 통합은 유전체, 전사체, 단백질체, 대사체를 종합적으로 분석합니다.
        </p>
        
        <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6 mb-6">
          <h3 className="font-bold mb-3">오믹스 계층 구조</h3>
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <div className="w-2 h-12 bg-gradient-to-b from-blue-500 to-blue-300"></div>
              <div>
                <strong>Genomics:</strong> DNA 변이, SNPs, CNVs
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-2 h-12 bg-gradient-to-b from-green-500 to-green-300"></div>
              <div>
                <strong>Transcriptomics:</strong> mRNA 발현, splicing variants
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-2 h-12 bg-gradient-to-b from-purple-500 to-purple-300"></div>
              <div>
                <strong>Proteomics:</strong> 단백질 발현, PTMs
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-2 h-12 bg-gradient-to-b from-orange-500 to-orange-300"></div>
              <div>
                <strong>Metabolomics:</strong> 대사산물, 대사 경로
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          2. 통합 분석 파이프라인
        </h2>
        
        <div className="bg-gray-100 dark:bg-gray-900 rounded-lg p-4 mb-4">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-mono text-gray-600 dark:text-gray-400">multiomics_integration.py</span>
            <button
              onClick={() => copyCode(integrationCode, 'integration')}
              className="p-2 hover:bg-gray-200 dark:hover:bg-gray-800 rounded"
            >
              {copiedCode === 'integration' ? <CheckCircle className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
            </button>
          </div>
          <pre className="text-sm overflow-x-auto">
            <code>{integrationCode}</code>
          </pre>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          3. Network Biology
        </h2>
        <p className="mb-4">
          생물학적 네트워크 분석을 통해 복잡한 상호작용을 이해합니다.
        </p>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-emerald-200 dark:border-emerald-800">
            <h4 className="font-bold mb-2">네트워크 유형</h4>
            <ul className="space-y-1 text-sm">
              <li>• Protein-Protein Interaction (PPI)</li>
              <li>• Gene Regulatory Network (GRN)</li>
              <li>• Metabolic Network</li>
              <li>• Disease Network</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-teal-200 dark:border-teal-800">
            <h4 className="font-bold mb-2">분석 방법</h4>
            <ul className="space-y-1 text-sm">
              <li>• Hub gene identification</li>
              <li>• Module detection</li>
              <li>• Pathway crosstalk</li>
              <li>• Network perturbation</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          4. 바이오마커 발굴
        </h2>
        
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
          <h3 className="font-bold mb-3 flex items-center gap-2">
            <Brain className="w-5 h-5 text-blue-600" />
            통합 바이오마커 전략
          </h3>
          <ul className="space-y-2">
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-1">1.</span>
              <span>다층적 특징 선택 (Multi-layer feature selection)</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-1">2.</span>
              <span>Cross-omics validation</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-1">3.</span>
              <span>임상 검증 (Clinical validation)</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-1">4.</span>
              <span>예측 모델 구축</span>
            </li>
          </ul>
        </div>
      </section>
    </div>
  )
}