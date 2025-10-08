'use client'

import { ChevronRight } from 'lucide-react'
import References from '@/components/common/References'

interface PracticalGuideProps {
  onComplete?: () => void
}

export default function PracticalGuide({ onComplete }: PracticalGuideProps) {
  return (
    <>
      {/* 5. 실전 팁 */}
      <section>
        <h2 className="text-3xl font-bold mb-6">5. 비지도학습 실전 가이드</h2>
        
        <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 p-6 rounded-xl">
          <h3 className="text-xl font-semibold mb-4">알고리즘 선택 가이드</h3>
          
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-semibold mb-2">클러스터링 선택 기준</h4>
              <ul className="space-y-2 text-sm">
                <GuideItem
                  title="구형 클러스터 + K 알고 있음:"
                  recommendation="K-Means"
                />
                <GuideItem
                  title="임의 모양 + 밀도 기반:"
                  recommendation="DBSCAN"
                />
                <GuideItem
                  title="계층 구조 필요:"
                  recommendation="Hierarchical"
                />
                <GuideItem
                  title="확률적 멤버십:"
                  recommendation="GMM"
                />
              </ul>
            </div>
            
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-semibold mb-2">차원 축소 선택 기준</h4>
              <ul className="space-y-2 text-sm">
                <GuideItem
                  title="선형 관계 + 해석 필요:"
                  recommendation="PCA"
                />
                <GuideItem
                  title="시각화 목적:"
                  recommendation="t-SNE 또는 UMAP"
                />
                <GuideItem
                  title="클래스 분리:"
                  recommendation="LDA"
                />
                <GuideItem
                  title="비선형 + 재구성:"
                  recommendation="Autoencoder"
                />
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-semibold mb-2">전처리 체크리스트</h4>
              <ul className="space-y-2 text-sm">
                <GuideItem
                  title="스케일링:"
                  recommendation="StandardScaler 또는 MinMaxScaler 필수"
                />
                <GuideItem
                  title="이상치 처리:"
                  recommendation="IQR 또는 Isolation Forest 활용"
                />
                <GuideItem
                  title="결측치 처리:"
                  recommendation="SimpleImputer 또는 KNNImputer"
                />
                <GuideItem
                  title="특성 선택:"
                  recommendation="상관관계 높은 특성 제거"
                />
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-semibold mb-2">성능 최적화 팁</h4>
              <ul className="space-y-2 text-sm">
                <GuideItem
                  title="대용량 데이터:"
                  recommendation="MiniBatchKMeans, BallTree 사용"
                />
                <GuideItem
                  title="메모리 효율:"
                  recommendation="Sparse matrix, 증분 학습"
                />
                <GuideItem
                  title="속도 개선:"
                  recommendation="n_init 줄이기, 병렬 처리"
                />
                <GuideItem
                  title="안정성:"
                  recommendation="여러 random_state로 검증"
                />
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* 프로젝트 */}
      <section className="mt-12">
        <div className="bg-gradient-to-r from-purple-600 to-pink-600 text-white p-8 rounded-xl">
          <h2 className="text-2xl font-bold mb-4">🎨 실전 프로젝트: 고객 세분화</h2>
          <p className="mb-6">
            이커머스 고객 데이터를 사용해 의미 있는 고객 그룹을 발견하고,
            각 그룹의 특성을 분석해 마케팅 전략을 수립해보세요.
          </p>

          <div className="mb-6">
            <h3 className="text-lg font-semibold mb-2">프로젝트 단계:</h3>
            <ol className="list-decimal list-inside space-y-1 text-sm">
              <li>RFM (Recency, Frequency, Monetary) 특성 추출</li>
              <li>데이터 스케일링 및 전처리</li>
              <li>최적 클러스터 수 결정 (Elbow, Silhouette)</li>
              <li>K-Means와 DBSCAN 비교 분석</li>
              <li>고객 세그먼트 프로파일링</li>
              <li>세그먼트별 마케팅 전략 제안</li>
            </ol>
          </div>

          <div className="flex gap-4 flex-wrap">
            {onComplete && (
              <button
                onClick={onComplete}
                className="bg-white text-purple-600 px-6 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-colors"
              >
                챕터 완료하기
              </button>
            )}
            <button className="bg-purple-500 text-white px-6 py-3 rounded-lg font-semibold hover:bg-purple-400 transition-colors">
              프로젝트 템플릿 다운로드
            </button>
          </div>
        </div>
      </section>

      <References
        sections={[
          {
            title: 'Clustering Algorithms',
            icon: 'paper',
            color: 'border-purple-500',
            items: [
              {
                title: 'Algorithm AS 136: A K-Means Clustering Algorithm',
                authors: 'J. A. Hartigan, M. A. Wong',
                year: '1979',
                description: 'K-Means 알고리즘의 고전적 구현 (Applied Statistics)',
                link: 'https://www.jstor.org/stable/2346830'
              },
              {
                title: 'A Density-Based Algorithm for Discovering Clusters',
                authors: 'Martin Ester, Hans-Peter Kriegel, Jörg Sander, Xiaowei Xu',
                year: '1996',
                description: 'DBSCAN - 밀도 기반 클러스터링의 시작 (KDD 1996)',
                link: 'https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf'
              },
              {
                title: 'Clustering by Passing Messages Between Data Points',
                authors: 'Brendan J. Frey, Delbert Dueck',
                year: '2007',
                description: 'Affinity Propagation - 메시지 전달 기반 클러스터링 (Science)',
                link: 'https://www.science.org/doi/10.1126/science.1136800'
              },
              {
                title: 'Gaussian Mixture Models',
                authors: 'Douglas Reynolds',
                year: '2009',
                description: 'GMM 완벽 가이드 - 확률적 클러스터링 (Encyclopedia of Biometrics)',
                link: 'https://link.springer.com/referenceworkentry/10.1007/978-0-387-73003-5_196'
              }
            ]
          },
          {
            title: 'Dimensionality Reduction',
            icon: 'paper',
            color: 'border-blue-500',
            items: [
              {
                title: 'Principal Component Analysis',
                authors: 'Ian T. Jolliffe',
                year: '2002',
                description: 'PCA의 모든 것 - 이론과 응용 (Springer)',
                link: 'https://link.springer.com/book/10.1007/b98835'
              },
              {
                title: 'Visualizing Data using t-SNE',
                authors: 'Laurens van der Maaten, Geoffrey Hinton',
                year: '2008',
                description: 't-SNE - 고차원 시각화의 혁명 (JMLR)',
                link: 'https://www.jmlr.org/papers/v9/vandermaaten08a.html'
              },
              {
                title: 'UMAP: Uniform Manifold Approximation and Projection',
                authors: 'Leland McInnes, John Healy, James Melville',
                year: '2018',
                description: 'UMAP - t-SNE를 뛰어넘는 차원축소 기법 (ArXiv)',
                link: 'https://arxiv.org/abs/1802.03426'
              },
              {
                title: 'Nonlinear Dimensionality Reduction by Locally Linear Embedding',
                authors: 'Sam T. Roweis, Lawrence K. Saul',
                year: '2000',
                description: 'LLE - 국소 선형 임베딩 (Science)',
                link: 'https://www.science.org/doi/10.1126/science.290.5500.2323'
              }
            ]
          },
          {
            title: 'Evaluation & Validation',
            icon: 'paper',
            color: 'border-green-500',
            items: [
              {
                title: 'Silhouettes: A Graphical Aid to the Interpretation',
                authors: 'Peter J. Rousseeuw',
                year: '1987',
                description: 'Silhouette 점수 - 클러스터링 평가의 표준 (Computational and Applied Mathematics)',
                link: 'https://www.sciencedirect.com/science/article/pii/0377042787901257'
              },
              {
                title: 'An Efficient k-Means Clustering Algorithm',
                authors: 'Greg Hamerly, Charles Elkan',
                year: '2002',
                description: 'Elbow Method와 최적 k 선택 (Technical Report)',
                link: 'https://www.aaai.org/Papers/ICML/2003/ICML03-022.pdf'
              }
            ]
          },
          {
            title: 'Libraries & Tools',
            icon: 'web',
            color: 'border-orange-500',
            items: [
              {
                title: 'Scikit-learn Clustering',
                description: 'K-Means, DBSCAN, Hierarchical 등 모든 클러스터링 알고리즘',
                link: 'https://scikit-learn.org/stable/modules/clustering.html'
              },
              {
                title: 'UMAP Documentation',
                description: 'Python UMAP 구현 - 고속 차원축소',
                link: 'https://umap-learn.readthedocs.io/'
              },
              {
                title: 'Prince: Python Dimensionality Reduction',
                description: 'PCA, MCA, FAMD 등 다양한 차원축소 기법',
                link: 'https://github.com/MaxHalford/prince'
              },
              {
                title: 'HDBScan',
                description: '계층적 DBSCAN - 자동 클러스터 수 결정',
                link: 'https://hdbscan.readthedocs.io/'
              }
            ]
          }
        ]}
      />
    </>
  )
}

function GuideItem({ title, recommendation }: { title: string; recommendation: string }) {
  return (
    <li className="flex items-start gap-2">
      <ChevronRight className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
      <span>
        <strong>{title}</strong> {recommendation}
      </span>
    </li>
  )
}