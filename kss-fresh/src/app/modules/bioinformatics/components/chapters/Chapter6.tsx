'use client';

import { useState } from 'react';
import { Copy, CheckCircle } from 'lucide-react';

export default function Chapter6() {
  const [copiedCode, setCopiedCode] = useState<string | null>(null)

  const copyCode = (code: string, id: string) => {
    navigator.clipboard.writeText(code)
    setCopiedCode(id)
    setTimeout(() => setCopiedCode(null), 2000)
  }

  const dockingCode = `# AutoDock Vina를 사용한 분자 도킹
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import numpy as np

class DrugDiscoveryPipeline:
    def __init__(self, target_protein, ligand_library):
        self.target = target_protein
        self.ligands = ligand_library
        
    def calculate_drug_properties(self, smiles):
        """약물 유사성 특성 계산 (Lipinski's Rule of Five)"""
        mol = Chem.MolFromSmiles(smiles)
        
        properties = {
            'molecular_weight': Descriptors.MolWt(mol),
            'logP': Descriptors.MolLogP(mol),
            'h_donors': Descriptors.NumHDonors(mol),
            'h_acceptors': Descriptors.NumHAcceptors(mol),
            'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'tpsa': Descriptors.TPSA(mol)
        }
        
        # Lipinski's Rule of Five 체크
        lipinski_violations = 0
        if properties['molecular_weight'] > 500:
            lipinski_violations += 1
        if properties['logP'] > 5:
            lipinski_violations += 1
        if properties['h_donors'] > 5:
            lipinski_violations += 1
        if properties['h_acceptors'] > 10:
            lipinski_violations += 1
            
        properties['lipinski_violations'] = lipinski_violations
        properties['drug_like'] = lipinski_violations <= 1
        
        return properties
    
    def virtual_screening(self, threshold=-7.0):
        """Virtual screening으로 후보 물질 선별"""
        candidates = []
        
        for ligand in self.ligands:
            # 도킹 시뮬레이션 실행 (simplified)
            binding_affinity = self.run_docking(ligand)
            
            if binding_affinity < threshold:
                drug_props = self.calculate_drug_properties(ligand['smiles'])
                
                if drug_props['drug_like']:
                    candidates.append({
                        'name': ligand['name'],
                        'smiles': ligand['smiles'],
                        'binding_affinity': binding_affinity,
                        'properties': drug_props
                    })
        
        # 결합 친화도로 정렬
        candidates.sort(key=lambda x: x['binding_affinity'])
        return candidates
    
    def run_docking(self, ligand):
        """분자 도킹 시뮬레이션 (simplified)"""
        # 실제로는 AutoDock Vina 호출
        # 여기서는 예시 값 반환
        return np.random.uniform(-10, -5)`

  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          1. Computer-Aided Drug Design (CADD)
        </h2>
        <p className="mb-4">
          컴퓨터를 활용한 약물 설계는 신약 개발의 시간과 비용을 획기적으로 줄이는 핵심 기술입니다.
        </p>
        
        <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6 mb-6">
          <h3 className="font-bold mb-3">신약 개발 파이프라인</h3>
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-emerald-600 text-white rounded-full flex items-center justify-center text-sm font-bold">1</div>
              <span><strong>Target Identification:</strong> 질병 관련 타겟 단백질 발굴</span>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-emerald-600 text-white rounded-full flex items-center justify-center text-sm font-bold">2</div>
              <span><strong>Lead Discovery:</strong> Virtual Screening으로 후보 물질 발굴</span>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-emerald-600 text-white rounded-full flex items-center justify-center text-sm font-bold">3</div>
              <span><strong>Lead Optimization:</strong> ADMET 특성 개선</span>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-emerald-600 text-white rounded-full flex items-center justify-center text-sm font-bold">4</div>
              <span><strong>Preclinical Testing:</strong> 동물 실험 전 안전성 예측</span>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          2. 분자 도킹과 Virtual Screening
        </h2>
        
        <div className="bg-gray-100 dark:bg-gray-900 rounded-lg p-4 mb-4">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-mono text-gray-600 dark:text-gray-400">drug_discovery.py</span>
            <button
              onClick={() => copyCode(dockingCode, 'docking')}
              className="p-2 hover:bg-gray-200 dark:hover:bg-gray-800 rounded"
            >
              {copiedCode === 'docking' ? <CheckCircle className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
            </button>
          </div>
          <pre className="text-sm overflow-x-auto">
            <code>{dockingCode}</code>
          </pre>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          3. Lipinski's Rule of Five
        </h2>
        <p className="mb-4">
          경구 투여 약물의 약물 유사성을 평가하는 기준입니다.
        </p>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-emerald-200 dark:border-emerald-800">
          <table className="w-full">
            <thead>
              <tr className="border-b">
                <th className="text-left p-2">특성</th>
                <th className="text-left p-2">기준값</th>
                <th className="text-left p-2">중요성</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="p-2">분자량</td>
                <td className="p-2">≤ 500 Da</td>
                <td className="p-2">흡수율</td>
              </tr>
              <tr>
                <td className="p-2">LogP</td>
                <td className="p-2">≤ 5</td>
                <td className="p-2">지용성</td>
              </tr>
              <tr>
                <td className="p-2">수소 결합 공여체</td>
                <td className="p-2">≤ 5</td>
                <td className="p-2">투과성</td>
              </tr>
              <tr>
                <td className="p-2">수소 결합 수용체</td>
                <td className="p-2">≤ 10</td>
                <td className="p-2">용해도</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          4. AI 기반 약물 설계
        </h2>
        
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
          <h3 className="font-bold mb-3">최신 AI 기술</h3>
          <ul className="space-y-2">
            <li className="flex items-start gap-2">
              <span className="text-blue-500 mt-1">•</span>
              <div>
                <strong>Graph Neural Networks:</strong> 분자 구조 표현 학습
              </div>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-500 mt-1">•</span>
              <div>
                <strong>Generative Models:</strong> 새로운 분자 구조 생성
              </div>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-500 mt-1">•</span>
              <div>
                <strong>Reinforcement Learning:</strong> 특성 최적화
              </div>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-500 mt-1">•</span>
              <div>
                <strong>Transfer Learning:</strong> 적은 데이터로 예측
              </div>
            </li>
          </ul>
        </div>
      </section>
    </div>
  )
}