'use client'

import { useState } from 'react'
import { Copy, CheckCircle } from 'lucide-react'

export default function Chapter9() {
  const [copiedCode, setCopiedCode] = useState<string | null>(null)

  const copyCode = (code: string, id: string) => {
    navigator.clipboard.writeText(code)
    setCopiedCode(id)
    setTimeout(() => setCopiedCode(null), 2000)
  }

  const scRNACode = `# Single-cell RNA-seq 분석 파이프라인
import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns

class SingleCellAnalysis:
    def __init__(self, h5_file):
        """10X Genomics 데이터 로드"""
        self.adata = sc.read_10x_h5(h5_file)
        self.adata.var_names_make_unique()
        
    def quality_control(self):
        """품질 관리 및 필터링"""
        # 미토콘드리아 유전자 비율 계산
        self.adata.var['mt'] = self.adata.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(
            self.adata, 
            qc_vars=['mt'], 
            percent_top=None, 
            log1p=False, 
            inplace=True
        )
        
        # QC 메트릭 시각화
        sc.pl.violin(self.adata, 
                    ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
                    jitter=0.4, multi_panel=True)
        
        # 필터링 기준
        sc.pp.filter_cells(self.adata, min_genes=200)
        sc.pp.filter_genes(self.adata, min_cells=3)
        
        # 이상치 제거
        self.adata = self.adata[self.adata.obs.n_genes_by_counts < 2500, :]
        self.adata = self.adata[self.adata.obs.pct_counts_mt < 5, :]
        
        return self.adata
    
    def normalize_and_scale(self):
        """정규화 및 스케일링"""
        # 총 카운트 정규화
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        
        # 로그 변환
        sc.pp.log1p(self.adata)
        
        # Highly variable genes 찾기
        sc.pp.highly_variable_genes(
            self.adata, 
            min_mean=0.0125, 
            max_mean=3, 
            min_disp=0.5
        )
        self.adata.raw = self.adata
        self.adata = self.adata[:, self.adata.var.highly_variable]
        
        # 스케일링
        sc.pp.scale(self.adata, max_value=10)
        
        return self.adata
    
    def dimensionality_reduction(self):
        """차원 축소 및 클러스터링"""
        # PCA
        sc.tl.pca(self.adata, svd_solver='arpack')
        sc.pl.pca_variance_ratio(self.adata, log=True, n_pcs=50)
        
        # Neighborhood graph
        sc.pp.neighbors(self.adata, n_neighbors=10, n_pcs=40)
        
        # UMAP
        sc.tl.umap(self.adata)
        
        # Leiden clustering
        sc.tl.leiden(self.adata, resolution=0.5)
        
        return self.adata
    
    def cell_type_annotation(self):
        """세포 유형 어노테이션"""
        # 마커 유전자 기반 자동 어노테이션
        marker_genes = {
            'T cells': ['CD3D', 'CD3E', 'CD8A'],
            'B cells': ['CD19', 'CD79A', 'MS4A1'],
            'NK cells': ['GNLY', 'NKG7', 'KLRB1'],
            'Monocytes': ['CD14', 'LYZ', 'S100A8'],
            'Dendritic': ['FCER1A', 'CST3', 'IL3RA']
        }
        
        # 각 클러스터의 마커 유전자 발현 확인
        sc.tl.rank_genes_groups(self.adata, 'leiden', method='wilcoxon')
        
        # 세포 유형 할당
        cell_types = self.assign_cell_types(marker_genes)
        self.adata.obs['cell_type'] = cell_types
        
        return self.adata
    
    def trajectory_inference(self):
        """Pseudotime 분석"""
        # Diffusion pseudotime
        sc.tl.diffmap(self.adata)
        self.adata.uns['iroot'] = np.flatnonzero(
            self.adata.obs['leiden'] == '0'
        )[0]
        
        sc.tl.dpt(self.adata)
        
        # PAGA (Partition-based graph abstraction)
        sc.tl.paga(self.adata, groups='leiden')
        sc.pl.paga(self.adata, plot=False)
        
        return self.adata`

  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          1. Single-cell RNA Sequencing
        </h2>
        <p className="mb-4">
          단일세포 시퀀싱은 개별 세포 수준에서 유전자 발현을 측정하여 
          세포 이질성과 희귀 세포 집단을 발견할 수 있습니다.
        </p>
        
        <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6 mb-6">
          <h3 className="font-bold mb-3">scRNA-seq 플랫폼</h3>
          <table className="w-full">
            <thead>
              <tr className="border-b">
                <th className="text-left p-2">플랫폼</th>
                <th className="text-left p-2">처리량</th>
                <th className="text-left p-2">특징</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="p-2">10X Genomics</td>
                <td className="p-2">1,000-10,000 cells</td>
                <td className="p-2">Droplet-based, 3' capture</td>
              </tr>
              <tr>
                <td className="p-2">Smart-seq2</td>
                <td className="p-2">96-384 cells</td>
                <td className="p-2">Full-length, plate-based</td>
              </tr>
              <tr>
                <td className="p-2">Drop-seq</td>
                <td className="p-2">10,000+ cells</td>
                <td className="p-2">Low cost, droplet</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          2. scRNA-seq 분석 파이프라인
        </h2>
        
        <div className="bg-gray-100 dark:bg-gray-900 rounded-lg p-4 mb-4">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-mono text-gray-600 dark:text-gray-400">single_cell_analysis.py</span>
            <button
              onClick={() => copyCode(scRNACode, 'scrna')}
              className="p-2 hover:bg-gray-200 dark:hover:bg-gray-800 rounded"
            >
              {copiedCode === 'scrna' ? <CheckCircle className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
            </button>
          </div>
          <pre className="text-sm overflow-x-auto">
            <code>{scRNACode}</code>
          </pre>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          3. 세포 유형 식별
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-emerald-200 dark:border-emerald-800">
            <h4 className="font-bold mb-2">클러스터링 방법</h4>
            <ul className="space-y-1 text-sm">
              <li>• Leiden algorithm</li>
              <li>• Louvain clustering</li>
              <li>• K-means</li>
              <li>• Hierarchical clustering</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-teal-200 dark:border-teal-800">
            <h4 className="font-bold mb-2">어노테이션 방법</h4>
            <ul className="space-y-1 text-sm">
              <li>• Marker gene expression</li>
              <li>• Reference-based (SingleR)</li>
              <li>• Machine learning (scmap)</li>
              <li>• Manual curation</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          4. Trajectory & Pseudotime
        </h2>
        
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
          <h3 className="font-bold mb-3">세포 분화 경로 분석</h3>
          <ul className="space-y-2">
            <li className="flex items-start gap-2">
              <span className="text-blue-500 mt-1">•</span>
              <div>
                <strong>Monocle:</strong> 비선형 차원 축소 기반
              </div>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-500 mt-1">•</span>
              <div>
                <strong>Slingshot:</strong> Cluster-based lineage inference
              </div>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-500 mt-1">•</span>
              <div>
                <strong>PAGA:</strong> Graph abstraction approach
              </div>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-500 mt-1">•</span>
              <div>
                <strong>RNA velocity:</strong> 미래 상태 예측
              </div>
            </li>
          </ul>
        </div>
      </section>
    </div>
  )
}