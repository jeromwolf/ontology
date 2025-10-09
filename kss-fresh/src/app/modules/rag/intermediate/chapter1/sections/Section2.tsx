'use client'

import { BarChart3 } from 'lucide-react'

export default function Section2() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-blue-100 dark:bg-blue-900/20 flex items-center justify-center">
          <BarChart3 className="text-blue-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">1.2 성능 벤치마크</h2>
          <p className="text-gray-600 dark:text-gray-400">실제 워크로드 기반 성능 비교</p>
        </div>
      </div>

      <div className="space-y-6">
        <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl border border-blue-200 dark:border-blue-700">
          <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-4">테스트 환경</h3>
          <ul className="space-y-2 text-sm text-blue-700 dark:text-blue-300">
            <li>• <strong>데이터셋:</strong> 100만개 벡터 (768차원)</li>
            <li>• <strong>하드웨어:</strong> 16 vCPU, 64GB RAM, NVMe SSD</li>
            <li>• <strong>인덱스 타입:</strong> HNSW (Hierarchical Navigable Small World)</li>
            <li>• <strong>측정 지표:</strong> QPS, Latency, Recall@10</li>
          </ul>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-gray-50 dark:bg-gray-700/50 p-6 rounded-xl">
            <h4 className="font-bold text-gray-900 dark:text-white mb-4">쿼리 성능 (QPS)</h4>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Qdrant</span>
                <div className="flex items-center gap-2">
                  <div className="w-32 bg-gray-200 rounded-full h-2">
                    <div className="bg-green-500 h-2 rounded-full" style={{width: '95%'}}></div>
                  </div>
                  <span className="text-xs text-gray-600 dark:text-gray-400">12,000</span>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Pinecone</span>
                <div className="flex items-center gap-2">
                  <div className="w-32 bg-gray-200 rounded-full h-2">
                    <div className="bg-green-500 h-2 rounded-full" style={{width: '88%'}}></div>
                  </div>
                  <span className="text-xs text-gray-600 dark:text-gray-400">11,000</span>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Milvus</span>
                <div className="flex items-center gap-2">
                  <div className="w-32 bg-gray-200 rounded-full h-2">
                    <div className="bg-green-500 h-2 rounded-full" style={{width: '80%'}}></div>
                  </div>
                  <span className="text-xs text-gray-600 dark:text-gray-400">10,000</span>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Weaviate</span>
                <div className="flex items-center gap-2">
                  <div className="w-32 bg-gray-200 rounded-full h-2">
                    <div className="bg-yellow-500 h-2 rounded-full" style={{width: '64%'}}></div>
                  </div>
                  <span className="text-xs text-gray-600 dark:text-gray-400">8,000</span>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Chroma</span>
                <div className="flex items-center gap-2">
                  <div className="w-32 bg-gray-200 rounded-full h-2">
                    <div className="bg-orange-500 h-2 rounded-full" style={{width: '40%'}}></div>
                  </div>
                  <span className="text-xs text-gray-600 dark:text-gray-400">5,000</span>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-gray-50 dark:bg-gray-700/50 p-6 rounded-xl">
            <h4 className="font-bold text-gray-900 dark:text-white mb-4">레이턴시 (P99, ms)</h4>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Qdrant</span>
                <div className="flex items-center gap-2">
                  <div className="w-32 bg-gray-200 rounded-full h-2">
                    <div className="bg-green-500 h-2 rounded-full" style={{width: '20%'}}></div>
                  </div>
                  <span className="text-xs text-gray-600 dark:text-gray-400">2ms</span>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Pinecone</span>
                <div className="flex items-center gap-2">
                  <div className="w-32 bg-gray-200 rounded-full h-2">
                    <div className="bg-green-500 h-2 rounded-full" style={{width: '25%'}}></div>
                  </div>
                  <span className="text-xs text-gray-600 dark:text-gray-400">2.5ms</span>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Milvus</span>
                <div className="flex items-center gap-2">
                  <div className="w-32 bg-gray-200 rounded-full h-2">
                    <div className="bg-yellow-500 h-2 rounded-full" style={{width: '35%'}}></div>
                  </div>
                  <span className="text-xs text-gray-600 dark:text-gray-400">3.5ms</span>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Weaviate</span>
                <div className="flex items-center gap-2">
                  <div className="w-32 bg-gray-200 rounded-full h-2">
                    <div className="bg-yellow-500 h-2 rounded-full" style={{width: '40%'}}></div>
                  </div>
                  <span className="text-xs text-gray-600 dark:text-gray-400">4ms</span>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Chroma</span>
                <div className="flex items-center gap-2">
                  <div className="w-32 bg-gray-200 rounded-full h-2">
                    <div className="bg-orange-500 h-2 rounded-full" style={{width: '60%'}}></div>
                  </div>
                  <span className="text-xs text-gray-600 dark:text-gray-400">6ms</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
