'use client'

import { useState } from 'react'
import { Copy, CheckCircle } from 'lucide-react'

export default function Chapter4() {
  const [copiedCode, setCopiedCode] = useState<string | null>(null)

  const copyCode = (code: string, id: string) => {
    navigator.clipboard.writeText(code)
    setCopiedCode(id)
    setTimeout(() => setCopiedCode(null), 2000)
  }

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