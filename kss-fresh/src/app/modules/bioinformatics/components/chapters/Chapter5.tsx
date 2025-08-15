'use client';

import { useState } from 'react';
import { Copy, CheckCircle, FlaskConical } from 'lucide-react';

export default function Chapter5() {
  const [copiedCode, setCopiedCode] = useState<string | null>(null)

  const copyCode = (code: string, id: string) => {
    navigator.clipboard.writeText(code)
    setCopiedCode(id)
    setTimeout(() => setCopiedCode(null), 2000)
  }

  const alphafoldCode = `# AlphaFold 예측 결과 분석
import py3Dmol
import requests
import json

def fetch_alphafold_structure(uniprot_id):
    """AlphaFold DB에서 구조 정보 가져오기"""
    base_url = "https://alphafold.ebi.ac.uk/api"
    
    # PDB 파일 다운로드
    pdb_url = f"{base_url}/prediction/{uniprot_id}"
    response = requests.get(pdb_url)
    
    if response.status_code == 200:
        data = response.json()[0]
        
        # 신뢰도 점수 분석
        confidence_data = {
            'mean_plddt': data['meanPlddt'],
            'confidence_version': data['confidenceVersion'],
            'model_url': data['pdbUrl']
        }
        
        # PDB 파일 다운로드
        pdb_response = requests.get(data['pdbUrl'])
        pdb_content = pdb_response.text
        
        return pdb_content, confidence_data
    
    return None, None

def visualize_structure(pdb_content):
    """3D 구조 시각화"""
    view = py3Dmol.view(width=800, height=600)
    view.addModel(pdb_content, 'pdb')
    
    # pLDDT 점수에 따른 색상 매핑
    view.setStyle({'cartoon': {
        'colorscheme': {
            'prop': 'b',
            'gradient': 'roygb',
            'min': 50,
            'max': 90
        }
    }})
    
    view.zoomTo()
    return view

# 사용 예시
uniprot_id = "P00533"  # EGFR
pdb_content, confidence = fetch_alphafold_structure(uniprot_id)

if pdb_content:
    print(f"평균 pLDDT 점수: {confidence['mean_plddt']:.2f}")
    view = visualize_structure(pdb_content)
    view.show()`

  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          1. 단백질 구조 예측의 혁명: AlphaFold
        </h2>
        <p className="mb-4">
          AlphaFold2는 50년간의 단백질 접힘 문제를 해결한 AI 시스템으로, 
          아미노산 서열만으로 3D 구조를 원자 수준의 정확도로 예측합니다.
        </p>
        
        <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6 mb-6">
          <h3 className="font-bold mb-3">AlphaFold 신뢰도 점수 (pLDDT)</h3>
          <div className="space-y-2">
            <div className="flex items-center gap-3">
              <div className="w-4 h-4 bg-blue-600 rounded"></div>
              <span><strong>매우 높음 (&gt;90):</strong> 매우 정확한 예측</span>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-4 h-4 bg-cyan-500 rounded"></div>
              <span><strong>신뢰할 만함 (70-90):</strong> 전체적으로 정확</span>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-4 h-4 bg-yellow-500 rounded"></div>
              <span><strong>낮음 (50-70):</strong> 주의 필요</span>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-4 h-4 bg-orange-500 rounded"></div>
              <span><strong>매우 낮음 (&lt;50):</strong> 무질서 영역 가능</span>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          2. AlphaFold API 활용
        </h2>
        
        <div className="bg-gray-100 dark:bg-gray-900 rounded-lg p-4 mb-4">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-mono text-gray-600 dark:text-gray-400">alphafold_analysis.py</span>
            <button
              onClick={() => copyCode(alphafoldCode, 'alphafold')}
              className="p-2 hover:bg-gray-200 dark:hover:bg-gray-800 rounded"
            >
              {copiedCode === 'alphafold' ? <CheckCircle className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
            </button>
          </div>
          <pre className="text-sm overflow-x-auto">
            <code>{alphafoldCode}</code>
          </pre>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          3. 단백질-단백질 상호작용
        </h2>
        <p className="mb-4">
          단백질 간 상호작용 네트워크는 세포 기능을 이해하는 핵심입니다.
        </p>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-emerald-200 dark:border-emerald-800">
            <h4 className="font-bold mb-2">실험적 방법</h4>
            <ul className="space-y-1 text-sm">
              <li>• Yeast Two-Hybrid (Y2H)</li>
              <li>• Co-immunoprecipitation</li>
              <li>• Mass Spectrometry</li>
              <li>• FRET/BRET</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-teal-200 dark:border-teal-800">
            <h4 className="font-bold mb-2">계산적 예측</h4>
            <ul className="space-y-1 text-sm">
              <li>• Sequence-based</li>
              <li>• Structure-based docking</li>
              <li>• Machine learning</li>
              <li>• Co-evolution analysis</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          4. 단백질 기능 예측
        </h2>
        
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
          <h3 className="font-bold mb-3 flex items-center gap-2">
            <FlaskConical className="w-5 h-5 text-blue-600" />
            기능 어노테이션 도구
          </h3>
          <ul className="space-y-2">
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-1">•</span>
              <div>
                <strong>InterPro:</strong> 단백질 도메인과 기능 부위 예측
              </div>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-1">•</span>
              <div>
                <strong>Gene Ontology:</strong> 분자 기능, 생물학적 과정, 세포 위치
              </div>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-1">•</span>
              <div>
                <strong>KEGG:</strong> 대사 경로 매핑
              </div>
            </li>
          </ul>
        </div>
      </section>
    </div>
  )
}