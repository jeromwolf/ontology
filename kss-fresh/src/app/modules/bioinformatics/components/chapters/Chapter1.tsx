'use client'

import { useState } from 'react'
import { Copy, CheckCircle } from 'lucide-react'

export default function Chapter1() {
  const [copiedCode, setCopiedCode] = useState<string | null>(null)

  const copyCode = (code: string, id: string) => {
    navigator.clipboard.writeText(code)
    setCopiedCode(id)
    setTimeout(() => setCopiedCode(null), 2000)
  }

  const dnaCode = `# DNA 이중나선 구조 시각화
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def visualize_dna_helix():
    """DNA 이중나선 구조 3D 시각화"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 나선 매개변수
    t = np.linspace(0, 8*np.pi, 200)
    radius = 1
    pitch = 0.8  # 나선 피치
    
    # 첫 번째 가닥 (forward strand)
    x1 = radius * np.cos(t)
    y1 = radius * np.sin(t)
    z1 = pitch * t
    
    # 두 번째 가닥 (reverse strand, 180도 회전)
    x2 = radius * np.cos(t + np.pi)
    y2 = radius * np.sin(t + np.pi)
    z2 = pitch * t
    
    # DNA 가닥 그리기
    ax.plot(x1, y1, z1, 'b-', linewidth=3, label='Strand 1 (5\' to 3\')')
    ax.plot(x2, y2, z2, 'r-', linewidth=3, label='Strand 2 (3\' to 5\')')
    
    # 염기쌍 연결선 (수소결합)
    for i in range(0, len(t), 10):
        ax.plot([x1[i], x2[i]], [y1[i], y2[i]], [z1[i], z2[i]], 
                'k--', alpha=0.3, linewidth=1)
    
    # 염기쌍 표시
    bases = ['A-T', 'G-C', 'C-G', 'T-A'] * 10
    for i, base in enumerate(bases[:20]):
        if i * 10 < len(t):
            mid_x = (x1[i*10] + x2[i*10]) / 2
            mid_y = (y1[i*10] + y2[i*10]) / 2
            mid_z = (z1[i*10] + z2[i*10]) / 2
            ax.text(mid_x, mid_y, mid_z, base, fontsize=8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (3\' to 5\' direction)')
    ax.set_title('DNA Double Helix Structure')
    ax.legend()
    
    # 회전 애니메이션을 위한 설정
    ax.view_init(elev=20, azim=45)
    plt.tight_layout()
    plt.show()

def analyze_gc_content(sequence):
    """DNA 서열의 GC 함량 분석"""
    gc_count = sequence.count('G') + sequence.count('C')
    total_bases = len(sequence)
    gc_percentage = (gc_count / total_bases) * 100
    
    print(f"Total bases: {total_bases}")
    print(f"GC content: {gc_percentage:.2f}%")
    
    # GC content가 생물학적 특성에 미치는 영향
    if gc_percentage < 40:
        stability = "Low"
        melting_temp = "Lower melting temperature"
    elif gc_percentage > 60:
        stability = "High"
        melting_temp = "Higher melting temperature"
    else:
        stability = "Moderate"
        melting_temp = "Moderate melting temperature"
    
    print(f"DNA stability: {stability}")
    print(f"Thermal stability: {melting_temp}")
    
    return gc_percentage

# 사용 예시
sample_dna = "ATGCGCTAGCTAGCGCGCATATATGCGCGCTAGCTAGCGC"
visualize_dna_helix()
gc_content = analyze_gc_content(sample_dna)`

  const proteinCode = `# 단백질 구조와 폴딩 분석
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class ProteinStructureAnalyzer:
    def __init__(self):
        # 20개 아미노산의 특성
        self.amino_acids = {
            'A': {'name': 'Alanine', 'hydrophobic': 1.8, 'charge': 0, 'size': 89},
            'R': {'name': 'Arginine', 'hydrophobic': -4.5, 'charge': 1, 'size': 174},
            'N': {'name': 'Asparagine', 'hydrophobic': -3.5, 'charge': 0, 'size': 132},
            'D': {'name': 'Aspartic acid', 'hydrophobic': -3.5, 'charge': -1, 'size': 133},
            'C': {'name': 'Cysteine', 'hydrophobic': 2.5, 'charge': 0, 'size': 121},
            'Q': {'name': 'Glutamine', 'hydrophobic': -3.5, 'charge': 0, 'size': 146},
            'E': {'name': 'Glutamic acid', 'hydrophobic': -3.5, 'charge': -1, 'size': 147},
            'G': {'name': 'Glycine', 'hydrophobic': -0.4, 'charge': 0, 'size': 75},
            'H': {'name': 'Histidine', 'hydrophobic': -3.2, 'charge': 0.1, 'size': 155},
            'I': {'name': 'Isoleucine', 'hydrophobic': 4.5, 'charge': 0, 'size': 131},
            'L': {'name': 'Leucine', 'hydrophobic': 3.8, 'charge': 0, 'size': 131},
            'K': {'name': 'Lysine', 'hydrophobic': -3.9, 'charge': 1, 'size': 146},
            'M': {'name': 'Methionine', 'hydrophobic': 1.9, 'charge': 0, 'size': 149},
            'F': {'name': 'Phenylalanine', 'hydrophobic': 2.8, 'charge': 0, 'size': 165},
            'P': {'name': 'Proline', 'hydrophobic': -1.6, 'charge': 0, 'size': 115},
            'S': {'name': 'Serine', 'hydrophobic': -0.8, 'charge': 0, 'size': 105},
            'T': {'name': 'Threonine', 'hydrophobic': -0.7, 'charge': 0, 'size': 119},
            'W': {'name': 'Tryptophan', 'hydrophobic': -0.9, 'charge': 0, 'size': 204},
            'Y': {'name': 'Tyrosine', 'hydrophobic': -1.3, 'charge': 0, 'size': 181},
            'V': {'name': 'Valine', 'hydrophobic': 4.2, 'charge': 0, 'size': 117}
        }
    
    def predict_secondary_structure(self, sequence):
        """간단한 이차구조 예측 (Chou-Fasman 기반)"""
        # 각 아미노산의 이차구조 성향
        helix_propensity = {
            'A': 1.42, 'R': 0.98, 'N': 0.67, 'D': 1.01, 'C': 0.70,
            'Q': 1.11, 'E': 1.51, 'G': 0.57, 'H': 1.00, 'I': 1.08,
            'L': 1.21, 'K': 1.16, 'M': 1.45, 'F': 1.13, 'P': 0.57,
            'S': 0.77, 'T': 0.83, 'W': 1.08, 'Y': 0.69, 'V': 1.06
        }
        
        sheet_propensity = {
            'A': 0.83, 'R': 0.93, 'N': 0.89, 'D': 0.54, 'C': 1.19,
            'Q': 1.10, 'E': 0.37, 'G': 0.75, 'H': 0.87, 'I': 1.60,
            'L': 1.30, 'K': 0.74, 'M': 1.05, 'F': 1.38, 'P': 0.55,
            'S': 0.75, 'T': 1.19, 'W': 1.37, 'Y': 1.47, 'V': 1.70
        }
        
        structure = []
        for aa in sequence:
            h_prop = helix_propensity.get(aa, 1.0)
            s_prop = sheet_propensity.get(aa, 1.0)
            
            if h_prop > s_prop and h_prop > 1.0:
                structure.append('H')  # Helix
            elif s_prop > h_prop and s_prop > 1.0:
                structure.append('E')  # Extended (sheet)
            else:
                structure.append('C')  # Coil
        
        return ''.join(structure)
    
    def analyze_hydrophobicity(self, sequence):
        """소수성 프로파일 분석"""
        window_size = 7
        hydrophobicity = []
        
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i + window_size]
            avg_hydrophobic = np.mean([
                self.amino_acids.get(aa, {}).get('hydrophobic', 0) 
                for aa in window
            ])
            hydrophobicity.append(avg_hydrophobic)
        
        # 시각화
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(hydrophobicity)), hydrophobicity, 'b-', linewidth=2)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Position')
        plt.ylabel('Hydrophobicity')
        plt.title('Protein Hydrophobicity Profile')
        plt.grid(True, alpha=0.3)
        
        # 막 관통 영역 예측 (소수성 > 1.5)
        transmembrane_regions = []
        for i, hydro in enumerate(hydrophobicity):
            if hydro > 1.5:
                plt.axvspan(i, i+1, alpha=0.3, color='yellow')
                transmembrane_regions.append(i)
        
        plt.show()
        return hydrophobicity, transmembrane_regions
    
    def predict_disulfide_bonds(self, sequence):
        """시스테인 잔기 간 이황결합 예측"""
        cysteine_positions = [i for i, aa in enumerate(sequence) if aa == 'C']
        
        if len(cysteine_positions) < 2:
            return []
        
        # 거리 기반 간단한 예측
        potential_bonds = []
        for i in range(len(cysteine_positions)):
            for j in range(i+1, len(cysteine_positions)):
                pos1, pos2 = cysteine_positions[i], cysteine_positions[j]
                distance = abs(pos2 - pos1)
                
                # 적절한 거리의 시스테인들을 이황결합 후보로 선정
                if 10 <= distance <= 200:
                    potential_bonds.append((pos1, pos2, distance))
        
        return sorted(potential_bonds, key=lambda x: x[2])

# 사용 예시
analyzer = ProteinStructureAnalyzer()
sample_protein = "MKLLVVLLTICSLPASEDVVKGNGDEQFSNYKKIFVSGNKDQDPHLLSVCGSAQLKWDVD"

# 이차구조 예측
secondary = analyzer.predict_secondary_structure(sample_protein)
print(f"Secondary structure: {secondary}")

# 소수성 분석
hydrophobicity, tm_regions = analyzer.analyze_hydrophobicity(sample_protein)
print(f"Potential transmembrane regions: {tm_regions}")

# 이황결합 예측
bonds = analyzer.predict_disulfide_bonds(sample_protein)
print(f"Potential disulfide bonds: {bonds}")`

  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          1. DNA 이중나선 구조
        </h2>
        <p className="mb-4">
          DNA는 두 개의 폴리뉴클레오타이드 가닥이 이중나선으로 꼬인 구조입니다. 
          각 가닥은 당-인산 백본과 네 종류의 염기(A, T, G, C)로 구성되어 있습니다.
        </p>
        
        <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6 mb-6">
          <h3 className="font-bold mb-3">DNA 구조의 핵심 특성</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h4 className="font-semibold mb-2">물리적 특성</h4>
              <ul className="space-y-1 text-sm">
                <li>• 나선 지름: 2 nm</li>
                <li>• 나선 피치: 3.4 nm (10염기쌍)</li>
                <li>• 염기쌍 간격: 0.34 nm</li>
                <li>• 나선 회전: 36° per base pair</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-2">화학적 특성</h4>
              <ul className="space-y-1 text-sm">
                <li>• A-T: 2개 수소결합</li>
                <li>• G-C: 3개 수소결합</li>
                <li>• 반평행 구조 (antiparallel)</li>
                <li>• Major groove와 Minor groove</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-gray-100 dark:bg-gray-900 rounded-lg p-4 mb-4">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-mono text-gray-600 dark:text-gray-400">dna_structure.py</span>
            <button
              onClick={() => copyCode(dnaCode, 'dna')}
              className="p-2 hover:bg-gray-200 dark:hover:bg-gray-800 rounded"
            >
              {copiedCode === 'dna' ? <CheckCircle className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
            </button>
          </div>
          <pre className="text-sm overflow-x-auto">
            <code>{dnaCode}</code>
          </pre>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          2. RNA의 종류와 기능
        </h2>
        <p className="mb-4">
          RNA는 DNA와 달리 단일가닥이며 우라실(U)을 포함합니다. 
          다양한 유형의 RNA가 각각 고유한 생물학적 기능을 수행합니다.
        </p>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-emerald-200 dark:border-emerald-800">
            <h4 className="font-bold mb-2">주요 RNA 유형</h4>
            <ul className="space-y-2 text-sm">
              <li><strong>mRNA:</strong> 유전정보 전달체</li>
              <li><strong>tRNA:</strong> 아미노산 운반체</li>
              <li><strong>rRNA:</strong> 리보솜 구성요소</li>
              <li><strong>miRNA:</strong> 유전자 발현 조절</li>
              <li><strong>lncRNA:</strong> 긴 비코딩 RNA</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-teal-200 dark:border-teal-800">
            <h4 className="font-bold mb-2">RNA 특이적 기능</h4>
            <ul className="space-y-2 text-sm">
              <li><strong>Splicing:</strong> 인트론 제거, 엑손 연결</li>
              <li><strong>Editing:</strong> 염기 치환 (A→I, C→U)</li>
              <li><strong>Modification:</strong> 메틸화, 슈도우리딘화</li>
              <li><strong>Localization:</strong> 세포 내 특정 위치로 이동</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          3. 단백질 구조와 폴딩
        </h2>
        
        <div className="bg-gray-100 dark:bg-gray-900 rounded-lg p-4 mb-4">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-mono text-gray-600 dark:text-gray-400">protein_analysis.py</span>
            <button
              onClick={() => copyCode(proteinCode, 'protein')}
              className="p-2 hover:bg-gray-200 dark:hover:bg-gray-800 rounded"
            >
              {copiedCode === 'protein' ? <CheckCircle className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
            </button>
          </div>
          <pre className="text-sm overflow-x-auto">
            <code>{proteinCode}</code>
          </pre>
        </div>

        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
          <h3 className="font-bold mb-3">단백질 구조 계층</h3>
          <ol className="space-y-2">
            <li className="flex items-start gap-2">
              <span className="text-blue-600 font-bold">1차 구조:</span>
              <span>아미노산 서열 (Primary sequence)</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-600 font-bold">2차 구조:</span>
              <span>α-helix, β-sheet, turn/loop</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-600 font-bold">3차 구조:</span>
              <span>전체적인 3D 폴딩</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-600 font-bold">4차 구조:</span>
              <span>여러 서브유닛의 조합</span>
            </li>
          </ol>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          4. 전사와 번역
        </h2>
        <p className="mb-4">
          유전 정보가 DNA → RNA → 단백질로 전달되는 중심 원리(Central Dogma)입니다.
        </p>
        
        <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6">
          <h3 className="font-bold mb-3">전사 과정</h3>
          <div className="space-y-2">
            <div className="flex items-center gap-3">
              <div className="w-6 h-6 bg-emerald-600 text-white rounded-full flex items-center justify-center text-xs font-bold">1</div>
              <span><strong>개시:</strong> RNA 폴리머라제가 프로모터에 결합</span>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-6 h-6 bg-emerald-600 text-white rounded-full flex items-center justify-center text-xs font-bold">2</div>
              <span><strong>연장:</strong> DNA 주형 가닥을 따라 RNA 합성</span>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-6 h-6 bg-emerald-600 text-white rounded-full flex items-center justify-center text-xs font-bold">3</div>
              <span><strong>종료:</strong> 종료 신호에서 전사 완료</span>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-6 h-6 bg-emerald-600 text-white rounded-full flex items-center justify-center text-xs font-bold">4</div>
              <span><strong>가공:</strong> 5' cap, 3' poly-A tail, splicing</span>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}