'use client'

import { useState } from 'react'
import { Copy, CheckCircle } from 'lucide-react'

export default function Chapter2() {
  const [copiedCode, setCopiedCode] = useState<string | null>(null)

  const copyCode = (code: string, id: string) => {
    navigator.clipboard.writeText(code)
    setCopiedCode(id)
    setTimeout(() => setCopiedCode(null), 2000)
  }

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