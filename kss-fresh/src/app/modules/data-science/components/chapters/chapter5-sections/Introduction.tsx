'use client'

import { Target, CheckCircle, Layers } from 'lucide-react'

export default function Introduction() {
  return (
    <>
      {/* 챕터 헤더 */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-4">비지도학습 - 클러스터링과 차원축소</h1>
        <p className="text-xl text-gray-600 dark:text-gray-400">
          K-means, DBSCAN, PCA, t-SNE, UMAP으로 데이터의 숨겨진 구조 발견하기
        </p>
      </div>

      {/* 학습 목표 */}
      <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 p-6 rounded-xl">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Target className="text-purple-600" />
          학습 목표
        </h2>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="flex items-start gap-3">
            <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
            <div>
              <p className="font-semibold">비지도학습의 개념과 활용</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">레이블 없는 데이터에서 패턴 발견</p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
            <div>
              <p className="font-semibold">클러스터링 알고리즘 마스터</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">K-means, DBSCAN, 계층적 클러스터링</p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
            <div>
              <p className="font-semibold">차원 축소 기법</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">PCA, t-SNE, UMAP, Autoencoder</p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
            <div>
              <p className="font-semibold">결과 해석과 검증</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">클러스터 품질 평가와 시각화</p>
            </div>
          </div>
        </div>
      </div>

      {/* 1. 비지도학습 개요 */}
      <section className="mt-8">
        <h2 className="text-3xl font-bold mb-6">1. 비지도학습이란?</h2>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700 mb-6">
          <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Layers className="text-purple-500" />
            비지도학습의 핵심 개념
          </h3>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            <strong>비지도학습(Unsupervised Learning)</strong>은 레이블이 없는 데이터에서 
            숨겨진 구조, 패턴, 관계를 발견하는 머신러닝 방법입니다. 
            데이터 자체의 특성을 이용해 유사한 것끼리 그룹화하거나 중요한 특성을 추출합니다.
          </p>
          
          <div className="grid md:grid-cols-3 gap-4 mt-6">
            <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 p-4 rounded-lg">
              <h4 className="font-semibold text-blue-700 dark:text-blue-400 mb-2">🎯 클러스터링</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                유사한 데이터포인트를 그룹으로 묶기
              </p>
              <ul className="mt-2 space-y-1 text-xs">
                <li>• 고객 세분화</li>
                <li>• 이상치 탐지</li>
                <li>• 이미지 분할</li>
              </ul>
            </div>
            
            <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 p-4 rounded-lg">
              <h4 className="font-semibold text-green-700 dark:text-green-400 mb-2">📐 차원 축소</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                고차원 데이터를 저차원으로 변환
              </p>
              <ul className="mt-2 space-y-1 text-xs">
                <li>• 시각화</li>
                <li>• 노이즈 제거</li>
                <li>• 계산 효율성</li>
              </ul>
            </div>
            
            <div className="bg-gradient-to-br from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 p-4 rounded-lg">
              <h4 className="font-semibold text-purple-700 dark:text-purple-400 mb-2">🔍 패턴 발견</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                데이터의 숨겨진 구조 탐색
              </p>
              <ul className="mt-2 space-y-1 text-xs">
                <li>• 연관 규칙</li>
                <li>• 밀도 추정</li>
                <li>• 표현 학습</li>
              </ul>
            </div>
          </div>
        </div>

        {/* 비지도학습 vs 지도학습 */}
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl">
            <h4 className="font-semibold text-blue-700 dark:text-blue-400 mb-3">지도학습 (Supervised)</h4>
            <ul className="space-y-2 text-sm">
              <li>✓ 레이블이 있는 데이터 사용</li>
              <li>✓ 예측이 목표 (분류/회귀)</li>
              <li>✓ 정답과 비교하여 학습</li>
              <li>✓ 성능 평가가 명확함</li>
            </ul>
          </div>
          
          <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-xl">
            <h4 className="font-semibold text-purple-700 dark:text-purple-400 mb-3">비지도학습 (Unsupervised)</h4>
            <ul className="space-y-2 text-sm">
              <li>✓ 레이블이 없는 데이터 사용</li>
              <li>✓ 구조 발견이 목표</li>
              <li>✓ 데이터 자체에서 패턴 학습</li>
              <li>✓ 평가가 주관적일 수 있음</li>
            </ul>
          </div>
        </div>
      </section>
    </>
  )
}