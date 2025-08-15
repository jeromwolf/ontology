'use client';

import { useState } from 'react';
import { 
  BarChart3, LineChart, PieChart, ScatterChart, Activity,
  Eye, Search, Filter, Layers, Palette,
  CheckCircle, AlertCircle, Info, Target,
  ChevronRight, Play, Download, Lightbulb
} from 'lucide-react';

interface ChapterProps {
  onComplete?: () => void
}

export default function Chapter3({ onComplete }: ChapterProps) {
  const [activeViz, setActiveViz] = useState('distribution')
  const [selectedLibrary, setSelectedLibrary] = useState('matplotlib')

  return (
    <div className="space-y-8">
      {/* 챕터 헤더 */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-4">EDA와 데이터 시각화</h1>
        <p className="text-xl text-gray-600 dark:text-gray-400">
          Matplotlib, Seaborn, Plotly로 하는 효과적인 데이터 탐색과 시각화
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
              <p className="font-semibold">체계적인 EDA 프로세스</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">데이터 이해부터 인사이트 도출까지</p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
            <div>
              <p className="font-semibold">적절한 시각화 선택</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">데이터 타입별 최적 차트</p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
            <div>
              <p className="font-semibold">Python 시각화 라이브러리</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">Matplotlib, Seaborn, Plotly 마스터</p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
            <div>
              <p className="font-semibold">효과적인 스토리텔링</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">시각화로 인사이트 전달</p>
            </div>
          </div>
        </div>
      </div>

      {/* 1. EDA 개요 */}
      <section>
        <h2 className="text-3xl font-bold mb-6">1. 탐색적 데이터 분석(EDA) 프로세스</h2>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700 mb-6">
          <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Search className="text-purple-500" />
            EDA의 체계적 접근법
          </h3>
          
          <div className="space-y-4">
            {[
              {
                stage: "1단계: 데이터 개요 파악",
                tasks: ["shape, dtypes 확인", "메모리 사용량", "첫/마지막 행 확인"],
                color: "blue"
              },
              {
                stage: "2단계: 데이터 품질 검사",
                tasks: ["결측치 패턴 분석", "중복 데이터", "이상치 탐지"],
                color: "orange"
              },
              {
                stage: "3단계: 단변량 분석",
                tasks: ["분포 확인", "기술통계", "범주형 빈도"],
                color: "green"
              },
              {
                stage: "4단계: 다변량 분석",
                tasks: ["상관관계", "교차분석", "그룹별 비교"],
                color: "purple"
              },
              {
                stage: "5단계: 시각화 & 인사이트",
                tasks: ["패턴 발견", "가설 생성", "다음 단계 제안"],
                color: "pink"
              }
            ].map((phase, idx) => (
              <div key={idx} className={`bg-gradient-to-r from-${phase.color}-50 to-${phase.color}-100 dark:from-${phase.color}-900/20 dark:to-${phase.color}-800/20 p-4 rounded-lg`}>
                <h4 className="font-semibold mb-2">{phase.stage}</h4>
                <div className="flex flex-wrap gap-2">
                  {phase.tasks.map((task, tidx) => (
                    <span key={tidx} className="text-sm bg-white dark:bg-gray-700 px-3 py-1 rounded-full">
                      {task}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* EDA 체크리스트 */}
        <div className="bg-gradient-to-r from-indigo-50 to-blue-50 dark:from-indigo-900/20 dark:to-blue-900/20 p-6 rounded-xl">
          <h3 className="text-xl font-semibold mb-4">EDA 체크리스트</h3>
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <h4 className="font-semibold mb-2 text-indigo-700 dark:text-indigo-400">데이터 이해</h4>
              <ul className="space-y-1 text-sm">
                <li className="flex items-center gap-2">
                  <input type="checkbox" className="rounded" />
                  <span>각 변수의 의미 파악</span>
                </li>
                <li className="flex items-center gap-2">
                  <input type="checkbox" className="rounded" />
                  <span>데이터 수집 방법 이해</span>
                </li>
                <li className="flex items-center gap-2">
                  <input type="checkbox" className="rounded" />
                  <span>타겟 변수 정의 확인</span>
                </li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-2 text-blue-700 dark:text-blue-400">품질 검사</h4>
              <ul className="space-y-1 text-sm">
                <li className="flex items-center gap-2">
                  <input type="checkbox" className="rounded" />
                  <span>결측치 처리 전략</span>
                </li>
                <li className="flex items-center gap-2">
                  <input type="checkbox" className="rounded" />
                  <span>이상치 식별 및 처리</span>
                </li>
                <li className="flex items-center gap-2">
                  <input type="checkbox" className="rounded" />
                  <span>데이터 일관성 검증</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* 2. 시각화 기초 */}
      <section>
        <h2 className="text-3xl font-bold mb-6">2. 데이터 타입별 시각화 가이드</h2>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mb-6">
          {/* 수치형 데이터 */}
          <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <BarChart3 className="text-blue-500" />
              수치형 데이터
            </h3>
            <ul className="space-y-2 text-sm">
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-blue-400" />
                <span><strong>히스토그램:</strong> 분포 확인</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-blue-400" />
                <span><strong>박스플롯:</strong> 사분위수, 이상치</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-blue-400" />
                <span><strong>밀도 플롯:</strong> 연속적 분포</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-blue-400" />
                <span><strong>바이올린 플롯:</strong> 분포+박스플롯</span>
              </li>
            </ul>
          </div>

          {/* 범주형 데이터 */}
          <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <PieChart className="text-green-500" />
              범주형 데이터
            </h3>
            <ul className="space-y-2 text-sm">
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-green-400" />
                <span><strong>막대 그래프:</strong> 빈도 비교</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-green-400" />
                <span><strong>파이 차트:</strong> 비율 (5개 이하)</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-green-400" />
                <span><strong>도넛 차트:</strong> 개선된 파이</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-green-400" />
                <span><strong>트리맵:</strong> 계층적 데이터</span>
              </li>
            </ul>
          </div>

          {/* 시계열 데이터 */}
          <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <LineChart className="text-purple-500" />
              시계열 데이터
            </h3>
            <ul className="space-y-2 text-sm">
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-purple-400" />
                <span><strong>선 그래프:</strong> 트렌드</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-purple-400" />
                <span><strong>영역 차트:</strong> 누적 변화</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-purple-400" />
                <span><strong>캔들스틱:</strong> 금융 데이터</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-purple-400" />
                <span><strong>계절성 플롯:</strong> 주기 패턴</span>
              </li>
            </ul>
          </div>

          {/* 관계형 데이터 */}
          <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <ScatterChart className="text-orange-500" />
              관계형 데이터
            </h3>
            <ul className="space-y-2 text-sm">
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-orange-400" />
                <span><strong>산점도:</strong> 두 변수 관계</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-orange-400" />
                <span><strong>버블 차트:</strong> 3차원 정보</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-orange-400" />
                <span><strong>히트맵:</strong> 상관관계</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-orange-400" />
                <span><strong>페어플롯:</strong> 다중 관계</span>
              </li>
            </ul>
          </div>

          {/* 분포 비교 */}
          <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Layers className="text-red-500" />
              분포 비교
            </h3>
            <ul className="space-y-2 text-sm">
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-red-400" />
                <span><strong>박스플롯:</strong> 그룹별 비교</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-red-400" />
                <span><strong>릿지플롯:</strong> 여러 분포</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-red-400" />
                <span><strong>스트립플롯:</strong> 개별 점</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-red-400" />
                <span><strong>스웜플롯:</strong> 밀도 표현</span>
              </li>
            </ul>
          </div>

          {/* 지리 데이터 */}
          <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Activity className="text-teal-500" />
              지리/공간 데이터
            </h3>
            <ul className="space-y-2 text-sm">
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-teal-400" />
                <span><strong>코로플레스:</strong> 지역별 값</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-teal-400" />
                <span><strong>산점도 지도:</strong> 위치 표시</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-teal-400" />
                <span><strong>히트맵 지도:</strong> 밀도</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-teal-400" />
                <span><strong>플로우맵:</strong> 이동 경로</span>
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* 3. Python 시각화 라이브러리 */}
      <section>
        <h2 className="text-3xl font-bold mb-6">3. Python 시각화 라이브러리 마스터</h2>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700 mb-6">
          <div className="flex gap-2 mb-4">
            {['matplotlib', 'seaborn', 'plotly'].map((lib) => (
              <button
                key={lib}
                onClick={() => setSelectedLibrary(lib)}
                className={`px-4 py-2 rounded-lg font-medium transition-colors capitalize ${
                  selectedLibrary === lib
                    ? 'bg-purple-500 text-white'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                }`}
              >
                {lib}
              </button>
            ))}
          </div>

          {selectedLibrary === 'matplotlib' && (
            <div>
              <h3 className="text-lg font-semibold mb-3 text-blue-600 dark:text-blue-400">Matplotlib</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                가장 기본적이고 강력한 시각화 라이브러리. 세밀한 커스터마이징 가능.
              </p>
              <div className="bg-gray-900 rounded-lg p-4">
                <pre className="text-sm text-gray-300 overflow-x-auto">
{`import matplotlib.pyplot as plt
import numpy as np

# 기본 설정
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12

# 서브플롯 생성
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 히스토그램
data = np.random.normal(100, 15, 1000)
axes[0, 0].hist(data, bins=30, color='skyblue', 
                edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Histogram')
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Frequency')

# 2. 산점도
x = np.random.randn(100)
y = 2 * x + np.random.randn(100) * 0.5
axes[0, 1].scatter(x, y, alpha=0.6, c=x, cmap='viridis')
axes[0, 1].set_title('Scatter Plot')
axes[0, 1].set_xlabel('X')
axes[0, 1].set_ylabel('Y')

# 3. 선 그래프
x = np.linspace(0, 10, 100)
axes[1, 0].plot(x, np.sin(x), label='sin(x)', linewidth=2)
axes[1, 0].plot(x, np.cos(x), label='cos(x)', linewidth=2)
axes[1, 0].set_title('Line Plot')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. 막대 그래프
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]
axes[1, 1].bar(categories, values, color='coral')
axes[1, 1].set_title('Bar Chart')
axes[1, 1].set_ylabel('Values')

plt.tight_layout()
plt.show()`}</pre>
              </div>
            </div>
          )}

          {selectedLibrary === 'seaborn' && (
            <div>
              <h3 className="text-lg font-semibold mb-3 text-green-600 dark:text-green-400">Seaborn</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                통계적 시각화에 특화. 아름다운 기본 스타일과 간편한 API.
              </p>
              <div className="bg-gray-900 rounded-lg p-4">
                <pre className="text-sm text-gray-300 overflow-x-auto">
{`import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# 팁 데이터셋 로드
tips = sns.load_dataset('tips')

# Figure 설정
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 바이올린 플롯
sns.violinplot(data=tips, x='day', y='total_bill', 
               hue='sex', split=True, ax=axes[0, 0])
axes[0, 0].set_title('Violin Plot: Bill by Day and Gender')

# 2. 페어플롯용 상관관계 히트맵
numeric_cols = tips.select_dtypes(include=['float64', 'int64']).columns
corr_matrix = tips[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', 
            center=0, ax=axes[0, 1])
axes[0, 1].set_title('Correlation Heatmap')

# 3. 박스플롯 with 스웜플롯
sns.boxplot(data=tips, x='day', y='total_bill', 
            ax=axes[1, 0], palette='Set3')
sns.swarmplot(data=tips, x='day', y='total_bill', 
              color='black', alpha=0.5, ax=axes[1, 0])
axes[1, 0].set_title('Box Plot with Swarm')

# 4. 회귀선이 있는 산점도
sns.regplot(data=tips, x='total_bill', y='tip', 
            ax=axes[1, 1], color='darkblue',
            scatter_kws={'alpha': 0.5})
axes[1, 1].set_title('Regression Plot: Bill vs Tip')

plt.tight_layout()
plt.show()

# 추가: 페어플롯
g = sns.pairplot(tips, hue='time', diag_kind='kde', 
                 plot_kws={'alpha': 0.6})
g.fig.suptitle('Pairplot of Tips Dataset', y=1.02)
plt.show()`}</pre>
              </div>
            </div>
          )}

          {selectedLibrary === 'plotly' && (
            <div>
              <h3 className="text-lg font-semibold mb-3 text-purple-600 dark:text-purple-400">Plotly</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                인터랙티브 시각화의 강자. 웹 대시보드와 3D 시각화에 최적.
              </p>
              <div className="bg-gray-900 rounded-lg p-4">
                <pre className="text-sm text-gray-300 overflow-x-auto">
{`import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# 데이터 준비
df = px.data.iris()

# 1. 인터랙티브 산점도
fig1 = px.scatter(df, x='sepal_width', y='sepal_length', 
                  color='species', size='petal_length',
                  hover_data=['petal_width'],
                  title='Iris Dataset: Interactive Scatter Plot')
fig1.update_layout(height=500)
fig1.show()

# 2. 3D 산점도
fig2 = px.scatter_3d(df, x='sepal_length', y='sepal_width', 
                     z='petal_length', color='species',
                     title='3D Scatter Plot of Iris Dataset')
fig2.update_layout(height=600)
fig2.show()

# 3. 서브플롯
fig3 = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Histogram', 'Box Plot', 
                    'Violin Plot', 'Sunburst'),
    specs=[[{'type': 'histogram'}, {'type': 'box'}],
           [{'type': 'violin'}, {'type': 'sunburst'}]]
)

# 히스토그램
fig3.add_trace(
    go.Histogram(x=df['sepal_length'], name='Sepal Length'),
    row=1, col=1
)

# 박스플롯
for species in df['species'].unique():
    fig3.add_trace(
        go.Box(y=df[df['species']==species]['petal_length'], 
               name=species),
        row=1, col=2
    )

# 바이올린 플롯
fig3.add_trace(
    go.Violin(y=df['petal_width'], box_visible=True,
              meanline_visible=True, name='Petal Width'),
    row=2, col=1
)

# 선버스트 차트 (계층적 데이터용)
# 예제 데이터
sunburst_data = pd.DataFrame({
    'labels': ['Iris', 'Setosa', 'Versicolor', 'Virginica',
               'Small', 'Medium', 'Large'],
    'parents': ['', 'Iris', 'Iris', 'Iris',
                'Setosa', 'Versicolor', 'Virginica'],
    'values': [100, 33, 33, 34, 33, 33, 34]
})

fig3.add_trace(
    go.Sunburst(
        labels=sunburst_data['labels'],
        parents=sunburst_data['parents'],
        values=sunburst_data['values']
    ),
    row=2, col=2
)

fig3.update_layout(height=800, showlegend=False,
                   title_text="Plotly Subplots Example")
fig3.show()

# 4. 애니메이션
gapminder = px.data.gapminder()
fig4 = px.scatter(gapminder, x="gdpPercap", y="lifeExp", 
                  animation_frame="year", animation_group="country",
                  size="pop", color="continent", hover_name="country",
                  log_x=True, size_max=55, range_x=[100,100000], 
                  range_y=[25,90])
fig4.update_layout(title="GDP vs Life Expectancy Over Time")
fig4.show()`}</pre>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* 4. 효과적인 시각화 원칙 */}
      <section>
        <h2 className="text-3xl font-bold mb-6">4. 효과적인 데이터 시각화 원칙</h2>
        
        <div className="grid md:grid-cols-2 gap-6 mb-6">
          <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 p-6 rounded-xl">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <CheckCircle className="text-green-600" />
              좋은 시각화의 특징
            </h3>
            <ul className="space-y-2 text-sm">
              <li>✓ <strong>명확성:</strong> 메시지가 분명하게 전달</li>
              <li>✓ <strong>정확성:</strong> 데이터를 왜곡하지 않음</li>
              <li>✓ <strong>효율성:</strong> 불필요한 요소 제거</li>
              <li>✓ <strong>심미성:</strong> 보기 좋은 디자인</li>
              <li>✓ <strong>일관성:</strong> 통일된 스타일과 색상</li>
              <li>✓ <strong>접근성:</strong> 색맹 고려, 레이블 명확</li>
            </ul>
          </div>
          
          <div className="bg-gradient-to-br from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 p-6 rounded-xl">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <AlertCircle className="text-red-600" />
              피해야 할 실수
            </h3>
            <ul className="space-y-2 text-sm">
              <li>❌ <strong>3D 효과:</strong> 불필요한 입체 효과</li>
              <li>❌ <strong>과도한 색상:</strong> 무지개색 남용</li>
              <li>❌ <strong>잘린 축:</strong> Y축이 0에서 시작하지 않음</li>
              <li>❌ <strong>정보 과부하:</strong> 한 차트에 너무 많은 정보</li>
              <li>❌ <strong>부적절한 차트:</strong> 데이터에 맞지 않는 유형</li>
              <li>❌ <strong>레이블 부족:</strong> 축, 단위, 제목 누락</li>
            </ul>
          </div>
        </div>

        {/* 색상 팔레트 가이드 */}
        <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700">
          <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Palette className="text-purple-500" />
            색상 사용 가이드
          </h3>
          
          <div className="grid md:grid-cols-3 gap-4">
            <div>
              <h4 className="font-semibold mb-2">정성적 팔레트</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                범주형 데이터에 사용. 구분이 명확한 색상.
              </p>
              <div className="flex gap-1">
                <div className="w-8 h-8 bg-blue-500"></div>
                <div className="w-8 h-8 bg-orange-500"></div>
                <div className="w-8 h-8 bg-green-500"></div>
                <div className="w-8 h-8 bg-red-500"></div>
                <div className="w-8 h-8 bg-purple-500"></div>
              </div>
            </div>
            
            <div>
              <h4 className="font-semibold mb-2">순차적 팔레트</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                순서가 있는 데이터. 낮음→높음.
              </p>
              <div className="flex gap-1">
                <div className="w-8 h-8 bg-blue-100"></div>
                <div className="w-8 h-8 bg-blue-300"></div>
                <div className="w-8 h-8 bg-blue-500"></div>
                <div className="w-8 h-8 bg-blue-700"></div>
                <div className="w-8 h-8 bg-blue-900"></div>
              </div>
            </div>
            
            <div>
              <h4 className="font-semibold mb-2">발산형 팔레트</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                중심점이 있는 데이터. 음수↔양수.
              </p>
              <div className="flex gap-1">
                <div className="w-8 h-8 bg-red-600"></div>
                <div className="w-8 h-8 bg-red-300"></div>
                <div className="w-8 h-8 bg-gray-200"></div>
                <div className="w-8 h-8 bg-blue-300"></div>
                <div className="w-8 h-8 bg-blue-600"></div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 5. 실전 EDA 예제 */}
      <section>
        <h2 className="text-3xl font-bold mb-6">5. 실전 EDA 워크플로우</h2>
        
        <div className="bg-gray-900 rounded-xl p-6">
          <h3 className="text-white font-semibold mb-4">타이타닉 데이터셋 완전 분석</h3>
          <pre className="bg-gray-800 p-4 rounded-lg overflow-x-auto">
            <code className="text-sm text-gray-300">{`import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 데이터 로드
df = pd.read_csv('titanic.csv')

# 1. 데이터 개요
print("=== 데이터 정보 ===")
print(f"Shape: {df.shape}")
print(f"\\nColumns: {df.columns.tolist()}")
print(f"\\nData types:\\n{df.dtypes}")
print(f"\\nMemory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# 2. 결측치 분석
missing_df = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df)) * 100
}).sort_values('Missing_Percentage', ascending=False)

# 결측치 시각화
plt.figure(figsize=(10, 6))
sns.barplot(data=missing_df[missing_df['Missing_Percentage'] > 0], 
            x='Missing_Percentage', y='Column', palette='Reds_r')
plt.title('Missing Values by Column')
plt.xlabel('Percentage (%)')
plt.show()

# 3. 생존율 분석
survival_rate = df['Survived'].value_counts(normalize=True)
print(f"\\n전체 생존율: {survival_rate[1]:.2%}")

# 4. 다각도 생존율 분석
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 성별
sns.barplot(data=df, x='Sex', y='Survived', ax=axes[0, 0])
axes[0, 0].set_title('Survival Rate by Gender')
axes[0, 0].set_ylim(0, 1)

# 클래스
sns.barplot(data=df, x='Pclass', y='Survived', ax=axes[0, 1])
axes[0, 1].set_title('Survival Rate by Class')
axes[0, 1].set_ylim(0, 1)

# 나이 분포
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                         labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
sns.barplot(data=df, x='Age_Group', y='Survived', ax=axes[0, 2])
axes[0, 2].set_title('Survival Rate by Age Group')
axes[0, 2].set_ylim(0, 1)

# 승선 항구
sns.barplot(data=df, x='Embarked', y='Survived', ax=axes[1, 0])
axes[1, 0].set_title('Survival Rate by Embarkation')
axes[1, 0].set_ylim(0, 1)

# 가족 크기
df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
df['Family_Type'] = pd.cut(df['Family_Size'], bins=[0, 1, 4, 11], 
                           labels=['Alone', 'Small', 'Large'])
sns.barplot(data=df, x='Family_Type', y='Survived', ax=axes[1, 1])
axes[1, 1].set_title('Survival Rate by Family Size')
axes[1, 1].set_ylim(0, 1)

# 요금 분포
df['Fare_Group'] = pd.qcut(df['Fare'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
sns.barplot(data=df, x='Fare_Group', y='Survived', ax=axes[1, 2])
axes[1, 2].set_title('Survival Rate by Fare Quartile')
axes[1, 2].set_ylim(0, 1)

plt.tight_layout()
plt.show()

# 5. 상관관계 분석
# 수치형 변수만 선택
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
            center=0, square=True, linewidths=1, 
            cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix')
plt.show()

# 6. 인터랙티브 분석 (Plotly)
import plotly.express as px

# 나이-요금 관계 with 생존 여부
fig = px.scatter(df, x='Age', y='Fare', color='Survived', 
                 size='Family_Size', hover_data=['Name', 'Pclass'],
                 title='Age vs Fare colored by Survival',
                 color_continuous_scale='RdYlBu')
fig.update_layout(height=600)
fig.show()

# 7. 주요 발견사항 정리
insights = """
=== 주요 인사이트 ===
1. 여성의 생존율(74%)이 남성(19%)보다 훨씬 높음
2. 1등석 승객의 생존율(63%)이 3등석(24%)보다 높음
3. 어린이의 생존율이 성인보다 높음
4. 혼자 탑승한 승객보다 가족과 함께한 승객의 생존율이 높음
5. 높은 요금을 지불한 승객의 생존율이 높음
6. Cherbourg에서 탑승한 승객의 생존율이 가장 높음
"""
print(insights)`}</code>
          </pre>
        </div>
      </section>

      {/* 대시보드 만들기 */}
      <section className="mt-12">
        <div className="bg-gradient-to-r from-purple-600 to-pink-600 text-white p-8 rounded-xl">
          <h2 className="text-2xl font-bold mb-4">📊 인터랙티브 대시보드 만들기</h2>
          <p className="mb-6">
            배운 내용을 활용해 Plotly Dash나 Streamlit으로 인터랙티브 대시보드를 만들어보세요.
            사용자가 직접 데이터를 탐색할 수 있는 도구를 제공하는 것이 목표입니다.
          </p>
          <div className="flex gap-4">
            <button 
              onClick={onComplete}
              className="bg-white text-purple-600 px-6 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-colors"
            >
              챕터 완료하기
            </button>
            <button className="bg-purple-500 text-white px-6 py-3 rounded-lg font-semibold hover:bg-purple-400 transition-colors">
              대시보드 템플릿 보기
            </button>
          </div>
        </div>
      </section>
    </div>
  )
}