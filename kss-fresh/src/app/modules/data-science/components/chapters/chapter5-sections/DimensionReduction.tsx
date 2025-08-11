'use client'

export default function DimensionReduction() {
  return (
    <section>
      <h2 className="text-3xl font-bold mb-6">3. 차원 축소 기법</h2>
      
      <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 p-6 rounded-xl mb-6">
        <h3 className="text-xl font-semibold mb-4">주요 차원 축소 방법</h3>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          <DimensionCard
            title="PCA"
            color="indigo"
            description="선형, 분산 최대화, 해석 가능"
          />
          <DimensionCard
            title="t-SNE"
            color="purple"
            description="비선형, 지역 구조 보존, 시각화용"
          />
          <DimensionCard
            title="UMAP"
            color="pink"
            description="비선형, 전역 구조 보존, 빠름"
          />
          <DimensionCard
            title="LDA"
            color="blue"
            description="선형, 클래스 분리 최대화"
          />
          <DimensionCard
            title="Autoencoder"
            color="green"
            description="비선형, 딥러닝 기반"
          />
          <DimensionCard
            title="ICA"
            color="orange"
            description="독립 성분 분석, 신호 분리"
          />
        </div>
      </div>

      {/* PCA 상세 */}
      <div className="bg-gray-900 rounded-xl p-6 mb-6">
        <h3 className="text-white font-semibold mb-4">PCA (Principal Component Analysis)</h3>
        <PCACode />
      </div>

      {/* t-SNE와 UMAP */}
      <div className="bg-gray-900 rounded-xl p-6">
        <h3 className="text-white font-semibold mb-4">t-SNE와 UMAP 비교</h3>
        <TSNEUMAPCode />
      </div>
    </section>
  )
}

function DimensionCard({ title, color, description }: { title: string; color: string; description: string }) {
  const colorClasses = {
    indigo: 'text-indigo-600 dark:text-indigo-400',
    purple: 'text-purple-600 dark:text-purple-400',
    pink: 'text-pink-600 dark:text-pink-400',
    blue: 'text-blue-600 dark:text-blue-400',
    green: 'text-green-600 dark:text-green-400',
    orange: 'text-orange-600 dark:text-orange-400'
  }

  return (
    <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
      <h4 className={`font-semibold mb-2 ${colorClasses[color as keyof typeof colorClasses]}`}>
        {title}
      </h4>
      <p className="text-sm text-gray-600 dark:text-gray-400">
        {description}
      </p>
    </div>
  )
}

function PCACode() {
  return (
    <pre className="bg-gray-800 p-4 rounded-lg overflow-x-auto">
      <code className="text-sm text-gray-300">{`from sklearn.decomposition import PCA
import numpy as np

# 1. PCA 수행
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# 2. 설명된 분산 비율
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Scree plot
ax1.bar(range(1, len(explained_variance_ratio)+1), 
        explained_variance_ratio)
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Explained Variance Ratio')
ax1.set_title('Scree Plot')

# 누적 분산
ax2.plot(range(1, len(cumulative_variance_ratio)+1), 
         cumulative_variance_ratio, 'bo-')
ax2.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
ax2.set_xlabel('Number of Components')
ax2.set_ylabel('Cumulative Explained Variance')
ax2.set_title('Variance Explained')
ax2.legend()

# 3. 95% 분산을 설명하는 최소 컴포넌트 수
n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
print(f"Components for 95% variance: {n_components_95}")

# 4. 주성분 해석
pca_interpret = PCA(n_components=3)
X_pca_3d = pca_interpret.fit_transform(X_scaled)

# 각 주성분에 대한 원본 특성의 기여도
components_df = pd.DataFrame(
    pca_interpret.components_.T,
    columns=['PC1', 'PC2', 'PC3'],
    index=feature_names
)

# 5. Biplot
def biplot(pca, X_pca, features):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 데이터 포인트
    ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
    
    # 특성 벡터
    for i, feature in enumerate(features):
        ax.arrow(0, 0, 
                pca.components_[0, i]*3, 
                pca.components_[1, i]*3,
                head_width=0.1, head_length=0.1, 
                fc='red', ec='red')
        ax.text(pca.components_[0, i]*3.2, 
               pca.components_[1, i]*3.2,
               feature, fontsize=12, ha='center')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    ax.set_title('PCA Biplot')
    ax.grid(True, alpha=0.3)`}</code>
    </pre>
  )
}

function TSNEUMAPCode() {
  return (
    <pre className="bg-gray-800 p-4 rounded-lg overflow-x-auto">
      <code className="text-sm text-gray-300">{`from sklearn.manifold import TSNE
import umap

# t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=30,        # 5-50, 지역 이웃 크기
    learning_rate=200,
    n_iter=1000,
    random_state=42
)
X_tsne = tsne.fit_transform(X_scaled)

# UMAP
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,       # 지역 이웃 크기
    min_dist=0.1,         # 점들 간 최소 거리
    metric='euclidean',
    random_state=42
)
X_umap = reducer.fit_transform(X_scaled)

# 시각화 비교
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# PCA
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, 
                cmap='viridis', alpha=0.6)
axes[0].set_title('PCA')

# t-SNE
axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, 
                cmap='viridis', alpha=0.6)
axes[1].set_title('t-SNE')

# UMAP
axes[2].scatter(X_umap[:, 0], X_umap[:, 1], c=labels, 
                cmap='viridis', alpha=0.6)
axes[2].set_title('UMAP')

plt.tight_layout()
plt.show()

# 파라미터 영향 분석
perplexities = [5, 30, 50, 100]
for i, perp in enumerate(perplexities):
    tsne = TSNE(n_components=2, perplexity=perp)
    X_tsne_temp = tsne.fit_transform(X_scaled)
    # 시각화 코드...`}</code>
    </pre>
  )
}