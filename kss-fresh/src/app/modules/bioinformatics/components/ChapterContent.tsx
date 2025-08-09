'use client'

import { useState } from 'react'
import { Dna, FlaskConical, Brain, Activity, Download, Copy, CheckCircle } from 'lucide-react'

interface ChapterContentProps {
  chapterId: string
}

export default function ChapterContent({ chapterId }: ChapterContentProps) {
  const [copiedCode, setCopiedCode] = useState<string | null>(null)

  const copyCode = (code: string, id: string) => {
    navigator.clipboard.writeText(code)
    setCopiedCode(id)
    setTimeout(() => setCopiedCode(null), 2000)
  }

  const renderContent = () => {
    switch (chapterId) {
      case 'biology-fundamentals':
        return <BiologyFundamentalsContent copyCode={copyCode} copiedCode={copiedCode} />
      case 'cell-genetics':
        return <CellGeneticsContent copyCode={copyCode} copiedCode={copiedCode} />
      case 'genomics-sequencing':
        return <GenomicsSequencingContent copyCode={copyCode} copiedCode={copiedCode} />
      case 'sequence-alignment':
        return <SequenceAlignmentContent copyCode={copyCode} copiedCode={copiedCode} />
      case 'proteomics-structure':
        return <ProteomicsStructureContent copyCode={copyCode} copiedCode={copiedCode} />
      case 'drug-discovery':
        return <DrugDiscoveryContent copyCode={copyCode} copiedCode={copiedCode} />
      case 'omics-integration':
        return <OmicsIntegrationContent copyCode={copyCode} copiedCode={copiedCode} />
      case 'ml-genomics':
        return <MLGenomicsContent copyCode={copyCode} copiedCode={copiedCode} />
      case 'single-cell':
        return <SingleCellContent copyCode={copyCode} copiedCode={copiedCode} />
      case 'clinical-applications':
        return <ClinicalApplicationsContent copyCode={copyCode} copiedCode={copiedCode} />
      default:
        return <div>챕터 콘텐츠를 불러올 수 없습니다.</div>
    }
  }

  return <div className="prose prose-lg dark:prose-invert max-w-none">{renderContent()}</div>
}

// Chapter 1: Biology Fundamentals
function BiologyFundamentalsContent({ copyCode, copiedCode }: any) {
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

// Chapter 2: Cell Genetics
function CellGeneticsContent({ copyCode, copiedCode }: any) {
  const epigeneticsCode = `# 후성유전학 분석 및 시각화
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class EpigeneticsAnalyzer:
    def __init__(self):
        self.histone_marks = {
            'H3K4me3': {'type': 'active', 'location': 'promoter', 'function': 'transcription initiation'},
            'H3K36me3': {'type': 'active', 'location': 'gene body', 'function': 'transcription elongation'},
            'H3K27me3': {'type': 'repressive', 'location': 'promoter', 'function': 'gene silencing'},
            'H3K9me3': {'type': 'repressive', 'location': 'heterochromatin', 'function': 'heterochromatin formation'},
            'H3K27ac': {'type': 'active', 'location': 'enhancer', 'function': 'enhancer activity'},
            'H3K4me1': {'type': 'neutral', 'location': 'enhancer', 'function': 'enhancer marking'}
        }
    
    def analyze_dna_methylation(self, cpg_sites, methylation_levels):
        """CpG 사이트 메틸화 분석"""
        # CpG islands 식별
        cpg_density = []
        gc_content = []
        
        window_size = 500
        for i in range(0, len(cpg_sites) - window_size, 100):
            window_sites = cpg_sites[i:i + window_size]
            
            # CpG 밀도 계산
            cpg_count = len([site for site in window_sites if 'CG' in site])
            density = cpg_count / window_size
            cpg_density.append(density)
            
            # GC 함량 계산
            total_gc = sum(site.count('G') + site.count('C') for site in window_sites)
            gc_percent = total_gc / (window_size * len(window_sites[0]))
            gc_content.append(gc_percent)
        
        # CpG islands 정의: CpG 밀도 > 0.6, GC 함량 > 0.5, 길이 > 500bp
        cpg_islands = []
        for i, (density, gc) in enumerate(zip(cpg_density, gc_content)):
            if density > 0.6 and gc > 0.5:
                cpg_islands.append(i * 100)
        
        # 메틸화 수준 시각화
        plt.figure(figsize=(15, 10))
        
        # 서브플롯 1: 메틸화 수준
        plt.subplot(3, 1, 1)
        plt.plot(methylation_levels, 'b-', alpha=0.7, linewidth=1)
        plt.ylabel('Methylation Level')
        plt.title('DNA Methylation Profile')
        plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='Hypermethylated')
        plt.axhline(y=0.2, color='g', linestyle='--', alpha=0.5, label='Hypomethylated')
        plt.legend()
        
        # 서브플롯 2: CpG 밀도
        plt.subplot(3, 1, 2)
        plt.plot(cpg_density, 'g-', linewidth=2)
        plt.ylabel('CpG Density')
        plt.title('CpG Density Profile')
        
        # CpG islands 표시
        for island in cpg_islands:
            plt.axvspan(island, island + 5, alpha=0.3, color='yellow')
        
        # 서브플롯 3: GC 함량
        plt.subplot(3, 1, 3)
        plt.plot(gc_content, 'purple', linewidth=2)
        plt.ylabel('GC Content')
        plt.xlabel('Genomic Position (100bp windows)')
        plt.title('GC Content Profile')
        
        plt.tight_layout()
        plt.show()
        
        return cpg_islands, np.mean(methylation_levels)
    
    def analyze_chromatin_state(self, histone_data):
        """크로마틴 상태 분석"""
        # 히스톤 수식의 조합으로 크로마틴 상태 예측
        states = []
        
        for i in range(len(histone_data['H3K4me3'])):
            h3k4me3 = histone_data['H3K4me3'][i]
            h3k27me3 = histone_data['H3K27me3'][i]
            h3k36me3 = histone_data['H3K36me3'][i]
            h3k27ac = histone_data['H3K27ac'][i]
            h3k9me3 = histone_data['H3K9me3'][i]
            
            # 크로마틴 상태 결정 논리
            if h3k4me3 > 2 and h3k27ac > 2:
                state = 'Active Promoter'
            elif h3k36me3 > 2:
                state = 'Transcribed'
            elif h3k27ac > 2 and h3k4me3 < 1:
                state = 'Active Enhancer'
            elif h3k27me3 > 2:
                if h3k4me3 > 1:
                    state = 'Bivalent'  # 양가성 크로마틴
                else:
                    state = 'Repressed'
            elif h3k9me3 > 2:
                state = 'Heterochromatin'
            else:
                state = 'Inactive'
            
            states.append(state)
        
        # 크로마틴 상태 시각화
        state_colors = {
            'Active Promoter': 'red',
            'Transcribed': 'green',
            'Active Enhancer': 'orange',
            'Bivalent': 'purple',
            'Repressed': 'blue',
            'Heterochromatin': 'black',
            'Inactive': 'gray'
        }
        
        plt.figure(figsize=(15, 8))
        
        # 히스톤 수식 히트맵
        plt.subplot(2, 1, 1)
        histone_matrix = np.array([histone_data[mark] for mark in self.histone_marks.keys()])
        sns.heatmap(histone_matrix, 
                   yticklabels=list(self.histone_marks.keys()),
                   cmap='YlOrRd', 
                   cbar_kws={'label': 'Signal Intensity'})
        plt.title('Histone Modification Landscape')
        
        # 크로마틴 상태
        plt.subplot(2, 1, 2)
        state_numeric = [list(state_colors.keys()).index(state) for state in states]
        plt.plot(state_numeric, linewidth=3)
        plt.ylabel('Chromatin State')
        plt.xlabel('Genomic Position')
        plt.title('Predicted Chromatin States')
        
        # y축 라벨 설정
        plt.yticks(range(len(state_colors)), list(state_colors.keys()))
        
        plt.tight_layout()
        plt.show()
        
        return states
    
    def gene_expression_regulation(self, promoter_methylation, enhancer_activity):
        """전사 조절 예측"""
        predicted_expression = []
        
        for meth, enh in zip(promoter_methylation, enhancer_activity):
            # 간단한 전사 조절 모델
            base_expression = 1.0
            
            # 프로모터 메틸화 효과 (억제)
            methylation_effect = max(0, 1 - meth * 2)
            
            # 인핸서 활성 효과 (증진)
            enhancer_effect = 1 + enh
            
            expression = base_expression * methylation_effect * enhancer_effect
            predicted_expression.append(expression)
        
        return predicted_expression

# 사용 예시
analyzer = EpigeneticsAnalyzer()

# 샘플 데이터 생성
np.random.seed(42)
n_regions = 1000

# DNA 메틸화 데이터
cpg_sites = [f"CG{''.join(np.random.choice(['A','T','G','C'], 10))}" for _ in range(n_regions)]
methylation_levels = np.random.beta(2, 2, n_regions)

# 히스톤 수식 데이터
histone_data = {
    'H3K4me3': np.random.exponential(1, n_regions),
    'H3K36me3': np.random.exponential(1, n_regions),
    'H3K27me3': np.random.exponential(1, n_regions),
    'H3K9me3': np.random.exponential(1, n_regions),
    'H3K27ac': np.random.exponential(1, n_regions),
    'H3K4me1': np.random.exponential(1, n_regions)
}

# 분석 실행
cpg_islands, avg_methylation = analyzer.analyze_dna_methylation(cpg_sites, methylation_levels)
chromatin_states = analyzer.analyze_chromatin_state(histone_data)

print(f"Found {len(cpg_islands)} CpG islands")
print(f"Average methylation level: {avg_methylation:.3f}")
print(f"Chromatin state distribution: {dict(zip(*np.unique(chromatin_states, return_counts=True)))}")`

  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          1. 염색체 구조와 조직
        </h2>
        <p className="mb-4">
          진핵세포의 염색체는 DNA가 히스톤 단백질과 함께 복잡하게 압축된 구조입니다.
          이러한 구조는 유전자 발현 조절에 중요한 역할을 합니다.
        </p>
        
        <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6 mb-6">
          <h3 className="font-bold mb-3">염색체 조직 계층</h3>
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center text-xs font-bold">1</div>
              <div>
                <strong>DNA 이중나선:</strong> 2nm 지름
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center text-xs font-bold">2</div>
              <div>
                <strong>뉴클레오솜:</strong> DNA + 히스톤 옥타머 (11nm)
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center text-xs font-bold">3</div>
              <div>
                <strong>30nm 크로마틴 섬유:</strong> 뉴클레오솜 응축
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center text-xs font-bold">4</div>
              <div>
                <strong>루프 도메인:</strong> 300nm 크로마틴 루프
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center text-xs font-bold">5</div>
              <div>
                <strong>응축 염색체:</strong> 700nm (분열기)
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          2. 유전자 조절 네트워크
        </h2>
        <p className="mb-4">
          유전자 발현은 전사인자, 프로모터, 인핸서, 사일런서 등의 복잡한 네트워크에 의해 조절됩니다.
        </p>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-emerald-200 dark:border-emerald-800">
            <h4 className="font-bold mb-2">조절 요소</h4>
            <ul className="space-y-1 text-sm">
              <li>• <strong>프로모터:</strong> 전사 시작점 근처</li>
              <li>• <strong>인핸서:</strong> 원거리 전사 증진</li>
              <li>• <strong>사일런서:</strong> 전사 억제</li>
              <li>• <strong>절연체:</strong> 도메인 경계</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-teal-200 dark:border-teal-800">
            <h4 className="font-bold mb-2">전사인자</h4>
            <ul className="space-y-1 text-sm">
              <li>• <strong>일반 전사인자:</strong> TFIIA, TFIIB, TFIID</li>
              <li>• <strong>특이 전사인자:</strong> p53, NF-κB, AP-1</li>
              <li>• <strong>코액티베이터:</strong> p300, CBP</li>
              <li>• <strong>코리프레서:</strong> HDAC, Sin3A</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          3. 후성유전학
        </h2>
        
        <div className="bg-gray-100 dark:bg-gray-900 rounded-lg p-4 mb-4">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-mono text-gray-600 dark:text-gray-400">epigenetics_analysis.py</span>
            <button
              onClick={() => copyCode(epigeneticsCode, 'epigenetics')}
              className="p-2 hover:bg-gray-200 dark:hover:bg-gray-800 rounded"
            >
              {copiedCode === 'epigenetics' ? <CheckCircle className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
            </button>
          </div>
          <pre className="text-sm overflow-x-auto">
            <code>{epigeneticsCode}</code>
          </pre>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          4. 히스톤 변형과 크로마틴 상태
        </h2>
        
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
          <h3 className="font-bold mb-3">주요 히스톤 변형</h3>
          <table className="w-full">
            <thead>
              <tr className="border-b">
                <th className="text-left p-2">변형</th>
                <th className="text-left p-2">위치</th>
                <th className="text-left p-2">기능</th>
                <th className="text-left p-2">효과</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="p-2">H3K4me3</td>
                <td className="p-2">프로모터</td>
                <td className="p-2">전사 개시</td>
                <td className="p-2 text-green-600">활성화</td>
              </tr>
              <tr>
                <td className="p-2">H3K27me3</td>
                <td className="p-2">프로모터</td>
                <td className="p-2">유전자 침묵</td>
                <td className="p-2 text-red-600">억제</td>
              </tr>
              <tr>
                <td className="p-2">H3K36me3</td>
                <td className="p-2">유전자 몸체</td>
                <td className="p-2">전사 연장</td>
                <td className="p-2 text-green-600">활성화</td>
              </tr>
              <tr>
                <td className="p-2">H3K27ac</td>
                <td className="p-2">인핸서</td>
                <td className="p-2">인핸서 활성</td>
                <td className="p-2 text-green-600">활성화</td>
              </tr>
              <tr>
                <td className="p-2">H3K9me3</td>
                <td className="p-2">헤테로크로마틴</td>
                <td className="p-2">헤테로크로마틴 형성</td>
                <td className="p-2 text-red-600">억제</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          5. DNA 메틸화
        </h2>
        <p className="mb-4">
          CpG 다이뉴클레오타이드의 시토신 메틸화는 유전자 발현을 조절하는 주요 후성유전학적 기전입니다.
        </p>
        
        <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6">
          <h3 className="font-bold mb-3">메틸화의 생물학적 의미</h3>
          <ul className="space-y-2">
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-1">•</span>
              <div>
                <strong>CpG Islands:</strong> 프로모터 지역의 CpG 밀집 구역
              </div>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-1">•</span>
              <div>
                <strong>Gene Silencing:</strong> 프로모터 메틸화로 전사 억제
              </div>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-1">•</span>
              <div>
                <strong>Genomic Imprinting:</strong> 부모 기원별 발현 조절
              </div>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-1">•</span>
              <div>
                <strong>X-inactivation:</strong> 여성의 X 염색체 불활성화
              </div>
            </li>
          </ul>
        </div>
      </section>
    </div>
  )
}

// Chapter 3: Genomics and Sequencing
function GenomicsSequencingContent({ copyCode, copiedCode }: any) {
  const fastqExample = `@SEQ_ID_001
GATTTGGGGTTCAAAGCAGTATCGATCAAATAGTAAATCCATTTGTTCAACTCACAGTTT
+
!''*((((***+))%%%++)(%%%%).1***-+*''))**55CCF>>>>>>CCCCCCC65`;

  const pythonCode = `# Biopython을 사용한 FASTQ 파일 품질 분석
from Bio import SeqIO
import matplotlib.pyplot as plt
import numpy as np

def analyze_fastq_quality(filename):
    """FASTQ 파일의 품질 점수 분석"""
    qualities = []
    
    for record in SeqIO.parse(filename, "fastq"):
        qualities.append(record.letter_annotations["phred_quality"])
    
    # 위치별 평균 품질 점수 계산
    max_length = max(len(q) for q in qualities)
    position_qualities = []
    
    for pos in range(max_length):
        pos_scores = [q[pos] for q in qualities if len(q) > pos]
        position_qualities.append(np.mean(pos_scores))
    
    # 품질 점수 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(position_qualities, linewidth=2)
    plt.axhline(y=30, color='r', linestyle='--', label='Q30 threshold')
    plt.xlabel('Position in read (bp)')
    plt.ylabel('Average Quality Score')
    plt.title('Per-base Sequence Quality')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    return position_qualities

# 사용 예시
quality_scores = analyze_fastq_quality('sample.fastq')`;

  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          1. DNA/RNA Sequencing Principles
        </h2>
        <p className="mb-4">
          Next-Generation Sequencing (NGS)은 대량의 DNA/RNA 서열을 병렬로 읽는 혁신적인 기술입니다.
          Illumina, PacBio, Oxford Nanopore 등 다양한 플랫폼이 있으며, 각각 고유한 장단점을 가지고 있습니다.
        </p>
        
        <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6 mb-6">
          <h3 className="font-bold mb-3">주요 시퀀싱 플랫폼 비교</h3>
          <table className="w-full">
            <thead>
              <tr className="border-b">
                <th className="text-left p-2">플랫폼</th>
                <th className="text-left p-2">읽기 길이</th>
                <th className="text-left p-2">정확도</th>
                <th className="text-left p-2">처리량</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="p-2">Illumina</td>
                <td className="p-2">150-300bp</td>
                <td className="p-2">99.9%</td>
                <td className="p-2">매우 높음</td>
              </tr>
              <tr>
                <td className="p-2">PacBio</td>
                <td className="p-2">10-25kb</td>
                <td className="p-2">99.0%</td>
                <td className="p-2">중간</td>
              </tr>
              <tr>
                <td className="p-2">Nanopore</td>
                <td className="p-2">10-100kb+</td>
                <td className="p-2">95-98%</td>
                <td className="p-2">중간</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          2. FASTQ 파일 형식과 품질 관리
        </h2>
        <p className="mb-4">
          FASTQ는 시퀀싱 데이터의 표준 형식으로, 서열 정보와 품질 점수를 함께 저장합니다.
        </p>
        
        <div className="bg-gray-100 dark:bg-gray-900 rounded-lg p-4 mb-4">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-mono text-gray-600 dark:text-gray-400">FASTQ 예시</span>
            <button
              onClick={() => copyCode(fastqExample, 'fastq')}
              className="p-2 hover:bg-gray-200 dark:hover:bg-gray-800 rounded"
            >
              {copiedCode === 'fastq' ? <CheckCircle className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
            </button>
          </div>
          <pre className="text-sm overflow-x-auto">
            <code>{fastqExample}</code>
          </pre>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          3. 품질 관리 파이프라인
        </h2>
        
        <div className="bg-gray-100 dark:bg-gray-900 rounded-lg p-4 mb-4">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-mono text-gray-600 dark:text-gray-400">quality_analysis.py</span>
            <button
              onClick={() => copyCode(pythonCode, 'python')}
              className="p-2 hover:bg-gray-200 dark:hover:bg-gray-800 rounded"
            >
              {copiedCode === 'python' ? <CheckCircle className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
            </button>
          </div>
          <pre className="text-sm overflow-x-auto">
            <code>{pythonCode}</code>
          </pre>
        </div>

        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 mt-6">
          <h3 className="font-bold mb-3 flex items-center gap-2">
            <Activity className="w-5 h-5 text-blue-600" />
            QC 체크리스트
          </h3>
          <ul className="space-y-2">
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-1">✓</span>
              <span>Per-base quality score &gt; Q30</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-1">✓</span>
              <span>Adapter contamination &lt; 1%</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-1">✓</span>
              <span>Duplicate rate &lt; 20%</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-1">✓</span>
              <span>GC content distribution 정상</span>
            </li>
          </ul>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          4. Genome Assembly
        </h2>
        <p className="mb-4">
          De novo assembly는 reference genome 없이 short reads로부터 전체 genome을 재구성하는 과정입니다.
        </p>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-emerald-200 dark:border-emerald-800">
            <h4 className="font-bold mb-2">Short-read Assembly</h4>
            <ul className="space-y-1 text-sm">
              <li>• SPAdes</li>
              <li>• Velvet</li>
              <li>• SOAPdenovo</li>
              <li>• ABySS</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-teal-200 dark:border-teal-800">
            <h4 className="font-bold mb-2">Long-read Assembly</h4>
            <ul className="space-y-1 text-sm">
              <li>• Canu</li>
              <li>• Flye</li>
              <li>• Wtdbg2</li>
              <li>• Shasta</li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  )
}

// Chapter 2: Sequence Alignment
function SequenceAlignmentContent({ copyCode, copiedCode }: any) {
  const blastCode = `# BLAST를 사용한 서열 유사성 검색
from Bio.Blast import NCBIWWW, NCBIXML
from Bio import SeqIO

def run_blast_search(sequence, database="nr", program="blastp"):
    """NCBI BLAST를 사용한 서열 검색"""
    print(f"Running {program} against {database}...")
    
    # BLAST 실행
    result_handle = NCBIWWW.qblast(
        program=program,
        database=database,
        sequence=sequence,
        expect=0.001,
        hitlist_size=10
    )
    
    # 결과 파싱
    blast_records = NCBIXML.parse(result_handle)
    
    results = []
    for blast_record in blast_records:
        for alignment in blast_record.alignments:
            for hsp in alignment.hsps:
                if hsp.expect < 0.001:
                    results.append({
                        'title': alignment.title,
                        'length': alignment.length,
                        'e_value': hsp.expect,
                        'score': hsp.score,
                        'identities': hsp.identities,
                        'query': hsp.query,
                        'match': hsp.match,
                        'subject': hsp.sbjct
                    })
    
    return results

# Multiple Sequence Alignment with Clustal
from Bio.Align.Applications import ClustalwCommandline

def multiple_alignment(input_file, output_file):
    """Clustal Omega를 사용한 다중 서열 정렬"""
    clustalw_cline = ClustalwCommandline(
        "clustalo",
        infile=input_file,
        outfile=output_file,
        verbose=True,
        auto=True
    )
    
    stdout, stderr = clustalw_cline()
    return output_file`

  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          1. 서열 정렬의 기초
        </h2>
        <p className="mb-4">
          서열 정렬은 DNA, RNA, 또는 단백질 서열 간의 유사성을 찾아 진화적 관계를 추론하는 핵심 기술입니다.
        </p>
        
        <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6 mb-6">
          <h3 className="font-bold mb-3">정렬 알고리즘 비교</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h4 className="font-semibold mb-2">Global Alignment (Needleman-Wunsch)</h4>
              <ul className="space-y-1 text-sm">
                <li>• 전체 서열 비교</li>
                <li>• 유사한 길이의 서열에 적합</li>
                <li>• O(mn) 시간 복잡도</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-2">Local Alignment (Smith-Waterman)</h4>
              <ul className="space-y-1 text-sm">
                <li>• 부분 서열 비교</li>
                <li>• 도메인 검색에 유용</li>
                <li>• 더 민감한 검색</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          2. BLAST와 유사성 검색
        </h2>
        
        <div className="bg-gray-100 dark:bg-gray-900 rounded-lg p-4 mb-4">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-mono text-gray-600 dark:text-gray-400">blast_search.py</span>
            <button
              onClick={() => copyCode(blastCode, 'blast')}
              className="p-2 hover:bg-gray-200 dark:hover:bg-gray-800 rounded"
            >
              {copiedCode === 'blast' ? <CheckCircle className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
            </button>
          </div>
          <pre className="text-sm overflow-x-auto">
            <code>{blastCode}</code>
          </pre>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-emerald-200 dark:border-emerald-800">
            <h4 className="font-bold mb-2">BLASTN</h4>
            <p className="text-sm">뉴클레오타이드 vs 뉴클레오타이드</p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-emerald-200 dark:border-emerald-800">
            <h4 className="font-bold mb-2">BLASTP</h4>
            <p className="text-sm">단백질 vs 단백질</p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-emerald-200 dark:border-emerald-800">
            <h4 className="font-bold mb-2">BLASTX</h4>
            <p className="text-sm">번역된 뉴클레오타이드 vs 단백질</p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          3. 계통 발생학적 분석
        </h2>
        <p className="mb-4">
          다중 서열 정렬 결과를 바탕으로 진화적 관계를 나타내는 계통수를 구축합니다.
        </p>
        
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
          <h3 className="font-bold mb-3">계통수 구축 방법</h3>
          <ul className="space-y-2">
            <li className="flex items-start gap-2">
              <span className="text-blue-500 mt-1">1.</span>
              <div>
                <strong>Distance-based:</strong> UPGMA, Neighbor-Joining
              </div>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-500 mt-1">2.</span>
              <div>
                <strong>Character-based:</strong> Maximum Parsimony
              </div>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-500 mt-1">3.</span>
              <div>
                <strong>Probabilistic:</strong> Maximum Likelihood, Bayesian
              </div>
            </li>
          </ul>
        </div>
      </section>
    </div>
  )
}

// Chapter 3: Proteomics and Structure
function ProteomicsStructureContent({ copyCode, copiedCode }: any) {
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

// Chapter 4: Drug Discovery
function DrugDiscoveryContent({ copyCode, copiedCode }: any) {
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

// Chapter 5: Multi-omics Integration
function OmicsIntegrationContent({ copyCode, copiedCode }: any) {
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

// Chapter 6: Machine Learning in Genomics
function MLGenomicsContent({ copyCode, copiedCode }: any) {
  const mlCode = `# 딥러닝을 사용한 유전체 변이 예측
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class GenomicCNN(nn.Module):
    """DNA 서열에서 변이 효과 예측 CNN 모델"""
    
    def __init__(self, seq_length=1000, num_filters=128):
        super(GenomicCNN, self).__init__()
        
        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(4, num_filters, kernel_size=12, padding=5)
        self.conv2 = nn.Conv1d(num_filters, num_filters*2, kernel_size=8, padding=3)
        self.conv3 = nn.Conv1d(num_filters*2, num_filters*4, kernel_size=4, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.bn2 = nn.BatchNorm1d(num_filters*2)
        self.bn3 = nn.BatchNorm1d(num_filters*4)
        
        # Pooling
        self.pool = nn.MaxPool1d(2)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=num_filters*4, 
            num_heads=8
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(num_filters*4 * (seq_length//8), 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        
        # Dropout
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # Convolutional blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # Flatten for FC layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Output (pathogenicity score)
        x = torch.sigmoid(self.fc3(x))
        
        return x

class VariantEffectPredictor:
    """변이 효과 예측 파이프라인"""
    
    def __init__(self, model_path=None):
        self.model = GenomicCNN()
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def encode_sequence(self, sequence):
        """DNA 서열을 one-hot encoding"""
        encoding = {'A': [1,0,0,0], 
                   'C': [0,1,0,0], 
                   'G': [0,0,1,0], 
                   'T': [0,0,0,1],
                   'N': [0,0,0,0]}
        
        encoded = []
        for base in sequence.upper():
            encoded.append(encoding.get(base, [0,0,0,0]))
        
        return torch.tensor(encoded).transpose(0, 1).unsqueeze(0)
    
    def predict_variant_effect(self, ref_seq, alt_seq):
        """Reference와 Alternative 서열의 효과 비교"""
        ref_encoded = self.encode_sequence(ref_seq)
        alt_encoded = self.encode_sequence(alt_seq)
        
        with torch.no_grad():
            ref_score = self.model(ref_encoded)
            alt_score = self.model(alt_encoded)
        
        effect = {
            'reference_score': ref_score.item(),
            'alternative_score': alt_score.item(),
            'delta_score': alt_score.item() - ref_score.item(),
            'predicted_effect': 'Pathogenic' if alt_score.item() > 0.5 else 'Benign'
        }
        
        return effect`

  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          1. 딥러닝과 유전체학의 만남
        </h2>
        <p className="mb-4">
          딥러닝은 유전체 데이터의 복잡한 패턴을 학습하여 변이 효과 예측, 
          유전자 발현 예측, 질병 위험도 평가 등에 혁신을 가져왔습니다.
        </p>
        
        <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6 mb-6">
          <h3 className="font-bold mb-3">주요 응용 분야</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h4 className="font-semibold mb-2">서열 기반 예측</h4>
              <ul className="space-y-1 text-sm">
                <li>• Variant effect prediction</li>
                <li>• Splice site prediction</li>
                <li>• Promoter/enhancer identification</li>
                <li>• TFBS prediction</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-2">임상 응용</h4>
              <ul className="space-y-1 text-sm">
                <li>• Disease risk prediction</li>
                <li>• Drug response prediction</li>
                <li>• Survival analysis</li>
                <li>• Treatment recommendation</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          2. CNN for Genomics
        </h2>
        
        <div className="bg-gray-100 dark:bg-gray-900 rounded-lg p-4 mb-4">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-mono text-gray-600 dark:text-gray-400">genomic_cnn.py</span>
            <button
              onClick={() => copyCode(mlCode, 'ml')}
              className="p-2 hover:bg-gray-200 dark:hover:bg-gray-800 rounded"
            >
              {copiedCode === 'ml' ? <CheckCircle className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
            </button>
          </div>
          <pre className="text-sm overflow-x-auto">
            <code>{mlCode}</code>
          </pre>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          3. Polygenic Risk Score (PRS)
        </h2>
        <p className="mb-4">
          수천~수백만 개의 유전 변이를 종합하여 질병 위험도를 계산합니다.
        </p>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-emerald-200 dark:border-emerald-800">
          <h3 className="font-bold mb-3">PRS 계산 과정</h3>
          <ol className="space-y-2">
            <li className="flex items-start gap-2">
              <span className="text-emerald-600 font-bold">1.</span>
              <span>GWAS summary statistics 수집</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-600 font-bold">2.</span>
              <span>SNP effect size 추정</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-600 font-bold">3.</span>
              <span>LD clumping & pruning</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-600 font-bold">4.</span>
              <span>가중 합계 계산: PRS = Σ(βi × dosagei)</span>
            </li>
          </ol>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          4. 정밀의학 응용
        </h2>
        
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
          <h3 className="font-bold mb-3 flex items-center gap-2">
            <Activity className="w-5 h-5 text-blue-600" />
            개인 맞춤 의학
          </h3>
          <ul className="space-y-2">
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-1">•</span>
              <div>
                <strong>약물유전체학:</strong> 개인별 약물 반응 예측
              </div>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-1">•</span>
              <div>
                <strong>암 유전체학:</strong> 종양 특이적 치료법 선택
              </div>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-1">•</span>
              <div>
                <strong>희귀질환:</strong> 유전자 진단 및 치료
              </div>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-1">•</span>
              <div>
                <strong>예방의학:</strong> 질병 위험도 기반 건강관리
              </div>
            </li>
          </ul>
        </div>
      </section>
    </div>
  )
}

// Chapter 7: Single-cell Genomics
function SingleCellContent({ copyCode, copiedCode }: any) {
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

// Chapter 8: Clinical Applications
function ClinicalApplicationsContent({ copyCode, copiedCode }: any) {
  const clinicalCode = `# 임상 유전체 분석 파이프라인
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from lifelines import KaplanMeierFitter, CoxPHFitter

class ClinicalGenomicsAnalyzer:
    def __init__(self):
        self.variant_db = self.load_variant_database()
        self.drug_db = self.load_pharmacogenomics_db()
        
    def analyze_tumor_mutations(self, vcf_file, bam_file):
        """종양 변이 분석 및 치료 옵션 제시"""
        
        # Somatic mutation calling
        mutations = self.call_somatic_mutations(vcf_file, bam_file)
        
        # Driver mutation identification
        driver_mutations = []
        for mutation in mutations:
            if self.is_driver_mutation(mutation):
                driver_mutations.append({
                    'gene': mutation['gene'],
                    'variant': mutation['variant'],
                    'vaf': mutation['vaf'],  # Variant Allele Frequency
                    'cosmic_id': mutation['cosmic_id'],
                    'clinical_significance': mutation['significance']
                })
        
        # Tumor Mutational Burden (TMB) 계산
        tmb = len(mutations) / 30  # mutations per Mb
        
        # Microsatellite Instability (MSI) 검사
        msi_status = self.check_msi_status(bam_file)
        
        # 치료 옵션 추천
        treatment_options = self.recommend_treatments(
            driver_mutations, tmb, msi_status
        )
        
        return {
            'driver_mutations': driver_mutations,
            'tmb': tmb,
            'msi_status': msi_status,
            'treatment_options': treatment_options
        }
    
    def liquid_biopsy_analysis(self, cfDNA_data):
        """Circulating tumor DNA (ctDNA) 분석"""
        
        results = {
            'ctDNA_fraction': None,
            'mutations_detected': [],
            'copy_number_alterations': [],
            'monitoring_recommendation': None
        }
        
        # ctDNA fraction 계산
        results['ctDNA_fraction'] = self.calculate_ctDNA_fraction(cfDNA_data)
        
        # Ultra-deep sequencing으로 low-frequency mutations 검출
        for region in cfDNA_data['target_regions']:
            mutations = self.detect_low_freq_mutations(
                region, 
                min_vaf=0.001  # 0.1% detection limit
            )
            results['mutations_detected'].extend(mutations)
        
        # Copy number alterations
        results['copy_number_alterations'] = self.detect_cna_from_cfDNA(cfDNA_data)
        
        # Monitoring recommendation
        if results['ctDNA_fraction'] > 0.01:
            results['monitoring_recommendation'] = 'High risk - monthly monitoring'
        elif results['ctDNA_fraction'] > 0.001:
            results['monitoring_recommendation'] = 'Medium risk - quarterly monitoring'
        else:
            results['monitoring_recommendation'] = 'Low risk - biannual monitoring'
        
        return results
    
    def pharmacogenomics_analysis(self, patient_genotype):
        """약물유전체 분석으로 개인 맞춤 처방"""
        
        recommendations = []
        
        # Key pharmacogenes 검사
        pharmacogenes = ['CYP2D6', 'CYP2C19', 'CYP2C9', 'VKORC1', 
                        'TPMT', 'NUDT15', 'DPYD', 'HLA-B']
        
        for gene in pharmacogenes:
            genotype = patient_genotype.get(gene)
            if genotype:
                # 약물 대사 표현형 예측
                phenotype = self.predict_metabolizer_status(gene, genotype)
                
                # 영향받는 약물 목록
                affected_drugs = self.drug_db[gene]
                
                for drug in affected_drugs:
                    recommendation = {
                        'drug': drug['name'],
                        'gene': gene,
                        'genotype': genotype,
                        'phenotype': phenotype,
                        'dosing_guideline': self.get_dosing_guideline(
                            drug, phenotype
                        ),
                        'alternative_drugs': drug.get('alternatives', [])
                    }
                    recommendations.append(recommendation)
        
        return recommendations
    
    def survival_analysis(self, genomic_data, clinical_data):
        """생존 분석 및 예후 예측"""
        
        # Cox proportional hazards model
        cph = CoxPHFitter()
        
        # Genomic and clinical features 결합
        combined_data = pd.concat([genomic_data, clinical_data], axis=1)
        
        # Model fitting
        cph.fit(combined_data, 
               duration_col='survival_time', 
               event_col='event_occurred')
        
        # Hazard ratios 계산
        hazard_ratios = cph.summary
        
        # Risk score 계산
        risk_scores = cph.predict_partial_hazard(combined_data)
        
        # Risk groups 분류
        risk_groups = pd.qcut(risk_scores, q=3, 
                              labels=['Low', 'Medium', 'High'])
        
        # Kaplan-Meier curves for each risk group
        kmf = KaplanMeierFitter()
        survival_curves = {}
        
        for group in ['Low', 'Medium', 'High']:
            group_data = combined_data[risk_groups == group]
            kmf.fit(group_data['survival_time'], 
                   group_data['event_occurred'])
            survival_curves[group] = kmf.survival_function_
        
        return {
            'hazard_ratios': hazard_ratios,
            'risk_scores': risk_scores,
            'risk_groups': risk_groups,
            'survival_curves': survival_curves
        }`

  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          1. Cancer Genomics
        </h2>
        <p className="mb-4">
          암 유전체학은 종양의 유전적 변화를 분석하여 정밀 치료를 가능하게 합니다.
        </p>
        
        <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6 mb-6">
          <h3 className="font-bold mb-3">종양 유전체 분석 항목</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h4 className="font-semibold mb-2">변이 분석</h4>
              <ul className="space-y-1 text-sm">
                <li>• Somatic mutations</li>
                <li>• Copy number alterations</li>
                <li>• Structural variants</li>
                <li>• Gene fusions</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-2">바이오마커</h4>
              <ul className="space-y-1 text-sm">
                <li>• TMB (Tumor Mutational Burden)</li>
                <li>• MSI (Microsatellite Instability)</li>
                <li>• HRD (Homologous Recombination Deficiency)</li>
                <li>• PD-L1 expression</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          2. 임상 유전체 분석 파이프라인
        </h2>
        
        <div className="bg-gray-100 dark:bg-gray-900 rounded-lg p-4 mb-4">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-mono text-gray-600 dark:text-gray-400">clinical_genomics.py</span>
            <button
              onClick={() => copyCode(clinicalCode, 'clinical')}
              className="p-2 hover:bg-gray-200 dark:hover:bg-gray-800 rounded"
            >
              {copiedCode === 'clinical' ? <CheckCircle className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
            </button>
          </div>
          <pre className="text-sm overflow-x-auto">
            <code>{clinicalCode}</code>
          </pre>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          3. Liquid Biopsy
        </h2>
        <p className="mb-4">
          혈액에서 순환 종양 DNA (ctDNA)를 분석하여 비침습적으로 암을 진단하고 모니터링합니다.
        </p>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-emerald-200 dark:border-emerald-800">
          <h3 className="font-bold mb-3">Liquid Biopsy 응용</h3>
          <ul className="space-y-2">
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-1">✓</span>
              <span><strong>조기 진단:</strong> 암 스크리닝 및 조기 발견</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-1">✓</span>
              <span><strong>치료 반응:</strong> 실시간 치료 효과 모니터링</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-1">✓</span>
              <span><strong>내성 감지:</strong> 약물 내성 변이 조기 발견</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-1">✓</span>
              <span><strong>재발 모니터링:</strong> MRD (Minimal Residual Disease) 추적</span>
            </li>
          </ul>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          4. Pharmacogenomics
        </h2>
        
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
          <h3 className="font-bold mb-3">주요 약물유전자</h3>
          <table className="w-full">
            <thead>
              <tr className="border-b">
                <th className="text-left p-2">유전자</th>
                <th className="text-left p-2">영향 약물</th>
                <th className="text-left p-2">임상 영향</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="p-2">CYP2D6</td>
                <td className="p-2">Codeine, Tamoxifen</td>
                <td className="p-2">진통/항암 효과</td>
              </tr>
              <tr>
                <td className="p-2">VKORC1</td>
                <td className="p-2">Warfarin</td>
                <td className="p-2">항응고제 용량</td>
              </tr>
              <tr>
                <td className="p-2">TPMT</td>
                <td className="p-2">6-MP, Azathioprine</td>
                <td className="p-2">골수 독성</td>
              </tr>
              <tr>
                <td className="p-2">HLA-B*5701</td>
                <td className="p-2">Abacavir</td>
                <td className="p-2">과민반응</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </div>
  )
}