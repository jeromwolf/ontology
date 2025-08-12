'use client'

import { useState } from 'react'
import { Copy, CheckCircle, Activity } from 'lucide-react'

export default function Chapter3() {
  const [copiedCode, setCopiedCode] = useState<string | null>(null)

  const copyCode = (code: string, id: string) => {
    navigator.clipboard.writeText(code)
    setCopiedCode(id)
    setTimeout(() => setCopiedCode(null), 2000)
  }

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