'use client'

import { useState } from 'react'

export default function EvaluationMetrics() {
  const [showVisualization, setShowVisualization] = useState(false)

  return (
    <section>
      <h2 className="text-3xl font-bold mb-6">4. 클러스터링 평가 지표</h2>
      
      <div className="grid md:grid-cols-2 gap-6 mb-6">
        <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold mb-4 text-blue-600 dark:text-blue-400">내부 평가 지표</h3>
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
            레이블 없이 클러스터 품질 평가
          </p>
          
          <div className="space-y-3">
            <MetricItem
              title="실루엣 계수"
              description="클러스터 내 응집도와 분리도 (-1 ~ 1)"
            />
            <MetricItem
              title="Calinski-Harabasz"
              description="클러스터 간/내 분산 비율 (높을수록 좋음)"
            />
            <MetricItem
              title="Davies-Bouldin"
              description="클러스터 간 유사도 (낮을수록 좋음)"
            />
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold mb-4 text-green-600 dark:text-green-400">외부 평가 지표</h3>
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
            실제 레이블과 비교 평가
          </p>
          
          <div className="space-y-3">
            <MetricItem
              title="Adjusted Rand Index"
              description="클러스터 일치도 (-1 ~ 1)"
            />
            <MetricItem
              title="Normalized Mutual Info"
              description="정보 이론 기반 (0 ~ 1)"
            />
            <MetricItem
              title="Homogeneity/Completeness"
              description="순도와 완전성 측정"
            />
          </div>
        </div>
      </div>

      {/* 평가 코드 */}
      <div className="bg-gray-900 rounded-xl p-6">
        <h3 className="text-white font-semibold mb-4">종합 클러스터 평가</h3>
        <button
          onClick={() => setShowVisualization(!showVisualization)}
          className="mb-4 px-4 py-2 bg-purple-500 text-white rounded hover:bg-purple-600"
        >
          {showVisualization ? '숨기기' : '평가 코드 보기'}
        </button>
        
        {showVisualization && <EvaluationCode />}
      </div>
    </section>
  )
}

function MetricItem({ title, description }: { title: string; description: string }) {
  return (
    <div>
      <h4 className="font-semibold text-sm">{title}</h4>
      <p className="text-xs text-gray-600 dark:text-gray-400">
        {description}
      </p>
    </div>
  )
}

function EvaluationCode() {
  return (
    <pre className="bg-gray-800 p-4 rounded-lg overflow-x-auto">
      <code className="text-sm text-gray-300">{`from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def evaluate_clustering(X, labels, true_labels=None):
    """클러스터링 결과 종합 평가"""
    
    results = {}
    
    # 내부 평가 지표
    if len(set(labels)) > 1:  # 클러스터가 2개 이상일 때만
        results['silhouette'] = silhouette_score(X, labels)
        results['calinski_harabasz'] = calinski_harabasz_score(X, labels)
        results['davies_bouldin'] = davies_bouldin_score(X, labels)
        
        # 실루엣 분석 시각화
        silhouette_vals = silhouette_samples(X, labels)
        
        plt.figure(figsize=(10, 8))
        y_lower = 10
        
        for i in range(len(set(labels))):
            cluster_silhouette_vals = silhouette_vals[labels == i]
            cluster_silhouette_vals.sort()
            
            size_cluster_i = cluster_silhouette_vals.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = plt.cm.viridis(float(i) / len(set(labels)))
            plt.fill_betweenx(np.arange(y_lower, y_upper),
                            0, cluster_silhouette_vals,
                            facecolor=color, edgecolor=color, alpha=0.7)
            
            plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
        
        plt.axvline(x=results['silhouette'], color="red", linestyle="--")
        plt.xlabel("Silhouette Coefficient Values")
        plt.ylabel("Cluster")
        plt.title("Silhouette Analysis")
        plt.show()
    
    # 외부 평가 지표 (실제 레이블이 있는 경우)
    if true_labels is not None:
        results['adjusted_rand'] = adjusted_rand_score(true_labels, labels)
        results['nmi'] = normalized_mutual_info_score(true_labels, labels)
        
        # Confusion matrix 스타일 시각화
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(true_labels, labels)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Cluster')
        plt.ylabel('True Label')
        plt.title('Cluster Assignment Matrix')
        plt.show()
    
    return results

# 여러 알고리즘 비교
algorithms = {
    'K-Means': KMeans(n_clusters=4, random_state=42),
    'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
    'GMM': GaussianMixture(n_components=4, random_state=42),
    'Agglomerative': AgglomerativeClustering(n_clusters=4)
}

comparison_results = []

for name, algorithm in algorithms.items():
    labels = algorithm.fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    if n_clusters > 1:
        scores = evaluate_clustering(X_scaled, labels)
        scores['algorithm'] = name
        scores['n_clusters'] = n_clusters
        comparison_results.append(scores)

# 결과 정리
results_df = pd.DataFrame(comparison_results)
print(results_df)

# 레이더 차트로 비교
categories = ['silhouette', 'calinski_harabasz_norm', 'davies_bouldin_inv']
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='polar')

for idx, row in results_df.iterrows():
    values = [
        row['silhouette'],
        row['calinski_harabasz'] / results_df['calinski_harabasz'].max(),
        1 / (1 + row['davies_bouldin'])  # 역수로 변환
    ]
    values += values[:1]  # 원 닫기
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    
    ax.plot(angles, values, 'o-', linewidth=2, label=row['algorithm'])
    ax.fill(angles, values, alpha=0.25)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.title('Clustering Algorithm Comparison')
plt.show()`}</code>
    </pre>
  )
}