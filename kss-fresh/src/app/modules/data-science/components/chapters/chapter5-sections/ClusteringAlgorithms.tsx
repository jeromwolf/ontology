'use client'

import { useState } from 'react'

export default function ClusteringAlgorithms() {
  const [activeMethod, setActiveMethod] = useState('kmeans')

  return (
    <section>
      <h2 className="text-3xl font-bold mb-6">2. 주요 클러스터링 알고리즘</h2>
      
      <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700 mb-6">
        <div className="flex gap-2 mb-4 flex-wrap">
          {['kmeans', 'dbscan', 'hierarchical', 'gmm'].map((method) => (
            <button
              key={method}
              onClick={() => setActiveMethod(method)}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                activeMethod === method
                  ? 'bg-purple-500 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
              }`}
            >
              {method === 'kmeans' && 'K-Means'}
              {method === 'dbscan' && 'DBSCAN'}
              {method === 'hierarchical' && '계층적 클러스터링'}
              {method === 'gmm' && 'Gaussian Mixture'}
            </button>
          ))}
        </div>

        {activeMethod === 'kmeans' && <KMeansSection />}
        {activeMethod === 'dbscan' && <DBSCANSection />}
        {activeMethod === 'hierarchical' && <HierarchicalSection />}
        {activeMethod === 'gmm' && <GMMSection />}
      </div>
    </section>
  )
}

function KMeansSection() {
  return (
    <div>
      <h3 className="text-lg font-semibold mb-3 text-blue-600 dark:text-blue-400">K-Means 클러스터링</h3>
      
      <div className="grid md:grid-cols-2 gap-4 mb-4">
        <div>
          <h4 className="font-semibold mb-2">원리</h4>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            K개의 중심점(centroid)을 기준으로 가장 가까운 클러스터에 할당
          </p>
          <div className="mt-2">
            <p className="text-sm font-semibold">알고리즘:</p>
            <ol className="text-sm space-y-1 list-decimal list-inside">
              <li>K개의 초기 중심점 선택</li>
              <li>각 점을 가장 가까운 중심점에 할당</li>
              <li>클러스터별 새 중심점 계산</li>
              <li>수렴할 때까지 2-3 반복</li>
            </ol>
          </div>
        </div>
        <div>
          <h4 className="font-semibold mb-2">특징</h4>
          <ul className="space-y-1 text-sm">
            <li>✓ 빠르고 확장 가능</li>
            <li>✓ 구현이 간단</li>
            <li>✓ 구형 클러스터에 적합</li>
            <li>✗ K값 사전 지정 필요</li>
            <li>✗ 이상치에 민감</li>
            <li>✗ 비구형 클러스터 어려움</li>
          </ul>
        </div>
      </div>
      
      <KMeansCode />
    </div>
  )
}

function KMeansCode() {
  return (
    <div className="bg-gray-900 rounded-lg p-4">
      <pre className="text-sm text-gray-300 overflow-x-auto">
{`from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 데이터 스케일링 (중요!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 최적 K 찾기 - Elbow Method
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# 최적 K로 모델 학습
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10)
labels = kmeans.fit_predict(X_scaled)

# 클러스터 프로파일링
for i in range(optimal_k):
    cluster_data = X[labels == i]
    print(f"Cluster {i}: {len(cluster_data)} samples")`}</pre>
    </div>
  )
}

function DBSCANSection() {
  return (
    <div>
      <h3 className="text-lg font-semibold mb-3 text-purple-600 dark:text-purple-400">DBSCAN (Density-Based Spatial Clustering)</h3>
      
      <div className="grid md:grid-cols-2 gap-4 mb-4">
        <div>
          <h4 className="font-semibold mb-2">원리</h4>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            밀도 기반으로 클러스터를 형성. 임의 모양의 클러스터 발견 가능
          </p>
          <div className="mt-2">
            <p className="text-sm font-semibold">핵심 개념:</p>
            <ul className="text-sm space-y-1">
              <li>• <strong>ε (eps)</strong>: 이웃 반경</li>
              <li>• <strong>MinPts</strong>: 최소 점 개수</li>
              <li>• <strong>Core Point</strong>: ε 내 MinPts 이상</li>
              <li>• <strong>Border Point</strong>: Core의 이웃</li>
              <li>• <strong>Noise</strong>: 둘 다 아님</li>
            </ul>
          </div>
        </div>
        <div>
          <h4 className="font-semibold mb-2">특징</h4>
          <ul className="space-y-1 text-sm">
            <li>✓ K 지정 불필요</li>
            <li>✓ 임의 모양 클러스터</li>
            <li>✓ 이상치 자동 탐지</li>
            <li>✓ 밀도 차이 처리</li>
            <li>✗ 파라미터 민감</li>
            <li>✗ 고차원에서 어려움</li>
          </ul>
        </div>
      </div>
      
      <DBSCANCode />
    </div>
  )
}

function DBSCANCode() {
  return (
    <div className="bg-gray-900 rounded-lg p-4">
      <pre className="text-sm text-gray-300 overflow-x-auto">
{`from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# 최적 eps 찾기 - k-distance graph
k = 4  # MinPts - 1
neigh = NearestNeighbors(n_neighbors=k)
neigh.fit(X_scaled)
distances, indices = neigh.kneighbors(X_scaled)

# DBSCAN 적용
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

# 결과 분석
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f'Number of clusters: {n_clusters}')
print(f'Number of noise points: {n_noise}')`}</pre>
    </div>
  )
}

function HierarchicalSection() {
  return (
    <div>
      <h3 className="text-lg font-semibold mb-3 text-green-600 dark:text-green-400">계층적 클러스터링</h3>
      
      <div className="grid md:grid-cols-2 gap-4 mb-4">
        <div>
          <h4 className="font-semibold mb-2">원리</h4>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            데이터를 계층적으로 그룹화하여 덴드로그램 생성
          </p>
          <div className="mt-2">
            <p className="text-sm font-semibold">두 가지 접근법:</p>
            <ul className="text-sm space-y-1">
              <li>• <strong>Agglomerative</strong>: Bottom-up</li>
              <li>• <strong>Divisive</strong>: Top-down</li>
            </ul>
          </div>
        </div>
        <div>
          <h4 className="font-semibold mb-2">연결 방법</h4>
          <ul className="space-y-1 text-sm">
            <li>• Single: 최소 거리</li>
            <li>• Complete: 최대 거리</li>
            <li>• Average: 평균 거리</li>
            <li>• Ward: 분산 최소화</li>
          </ul>
        </div>
      </div>
      
      <HierarchicalCode />
    </div>
  )
}

function HierarchicalCode() {
  return (
    <div className="bg-gray-900 rounded-lg p-4">
      <pre className="text-sm text-gray-300 overflow-x-auto">
{`from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# 계층적 클러스터링
agg_clustering = AgglomerativeClustering(
    n_clusters=4,
    linkage='ward'
)
labels = agg_clustering.fit_predict(X_scaled)

# 덴드로그램 생성
linkage_matrix = linkage(X_scaled, method='ward')
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.show()`}</pre>
    </div>
  )
}

function GMMSection() {
  return (
    <div>
      <h3 className="text-lg font-semibold mb-3 text-orange-600 dark:text-orange-400">Gaussian Mixture Model (GMM)</h3>
      
      <div className="grid md:grid-cols-2 gap-4 mb-4">
        <div>
          <h4 className="font-semibold mb-2">원리</h4>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            데이터가 여러 가우시안 분포의 혼합으로 생성되었다고 가정
          </p>
          <div className="mt-2">
            <p className="text-sm font-semibold">특징:</p>
            <ul className="text-sm space-y-1">
              <li>• 확률적 클러스터링</li>
              <li>• Soft assignment</li>
              <li>• EM 알고리즘 사용</li>
              <li>• 타원형 클러스터</li>
            </ul>
          </div>
        </div>
        <div>
          <h4 className="font-semibold mb-2">장단점</h4>
          <ul className="space-y-1 text-sm">
            <li>✓ 확률적 해석 가능</li>
            <li>✓ 유연한 클러스터 모양</li>
            <li>✓ 불확실성 표현</li>
            <li>✗ 계산 비용 높음</li>
            <li>✗ 초기값 민감</li>
          </ul>
        </div>
      </div>
      
      <GMMCode />
    </div>
  )
}

function GMMCode() {
  return (
    <div className="bg-gray-900 rounded-lg p-4">
      <pre className="text-sm text-gray-300 overflow-x-auto">
{`from sklearn.mixture import GaussianMixture

# GMM 적용
gmm = GaussianMixture(
    n_components=4,
    covariance_type='full',
    random_state=42
)
gmm.fit(X_scaled)

# 확률적 예측
labels = gmm.predict(X_scaled)
probabilities = gmm.predict_proba(X_scaled)

# BIC/AIC로 최적 컴포넌트 수 찾기
n_components_range = range(2, 10)
bic_scores = []
aic_scores = []

for n in n_components_range:
    gmm = GaussianMixture(n_components=n)
    gmm.fit(X_scaled)
    bic_scores.append(gmm.bic(X_scaled))
    aic_scores.append(gmm.aic(X_scaled))`}</pre>
    </div>
  )
}