'use client';

import { useState } from 'react';
import { 
  BarChart, LineChart, PieChart, ScatterChart,
  TrendingUp, AlertCircle, CheckCircle, Info,
  FileSpreadsheet, Search, Filter, Eye,
  Activity, Database, Zap, Brain,
  ChevronRight, Play, Upload, Download, Target
} from 'lucide-react';

export default function Chapter2() {
  const [selectedDataset, setSelectedDataset] = useState('sales')
  const [activeTab, setActiveTab] = useState('overview')

  return (
    <div className="space-y-8">
      {/* 챕터 헤더 */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-4">탐색적 데이터 분석 (EDA) 완벽 가이드</h1>
        <p className="text-xl text-gray-600 dark:text-gray-400">
          데이터를 이해하고 인사이트를 발견하는 체계적인 접근법
        </p>
      </div>

      {/* 학습 목표 */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 p-6 rounded-xl">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Target className="text-blue-600" />
          학습 목표
        </h2>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="flex items-start gap-3">
            <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
            <div>
              <p className="font-semibold">EDA의 핵심 개념과 프로세스 이해</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">데이터 탐색의 체계적 접근법</p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
            <div>
              <p className="font-semibold">Pandas와 Polars를 활용한 데이터 분석</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">현대적인 데이터 처리 도구 마스터</p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
            <div>
              <p className="font-semibold">효과적인 데이터 시각화 기법</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">Matplotlib, Seaborn, Plotly 활용</p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
            <div>
              <p className="font-semibold">이상치 탐지와 데이터 품질 평가</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">신뢰할 수 있는 데이터 확보</p>
            </div>
          </div>
        </div>
      </div>

      {/* 1. EDA 개요 */}
      <section>
        <h2 className="text-3xl font-bold mb-6">1. EDA란 무엇인가?</h2>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700 mb-6">
          <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Brain className="text-purple-500" />
            탐색적 데이터 분석의 정의
          </h3>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            <strong>탐색적 데이터 분석(Exploratory Data Analysis, EDA)</strong>은 데이터를 이해하고 
            주요 특성을 파악하기 위한 접근법입니다. 통계 그래픽, 플롯, 정보 테이블 등을 활용하여 
            데이터의 패턴, 이상치, 관계를 발견합니다.
          </p>
          
          <div className="bg-gray-50 dark:bg-gray-900/50 p-4 rounded-lg">
            <p className="text-sm font-medium mb-2">John Tukey (1977)의 정의:</p>
            <blockquote className="italic text-gray-600 dark:text-gray-400 border-l-4 border-purple-500 pl-4">
              "EDA는 데이터가 우리에게 무엇을 말하고 있는지 듣는 과정이다"
            </blockquote>
          </div>
        </div>

        {/* EDA의 목적 */}
        <div className="grid md:grid-cols-2 gap-6 mb-6">
          <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 p-6 rounded-xl">
            <h4 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Search className="text-green-600" />
              EDA의 주요 목적
            </h4>
            <ul className="space-y-2">
              <li className="flex items-start gap-2">
                <span className="text-green-600 mt-1">•</span>
                <span>데이터의 주요 특성 파악</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-600 mt-1">•</span>
                <span>패턴과 트렌드 발견</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-600 mt-1">•</span>
                <span>이상치와 오류 탐지</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-600 mt-1">•</span>
                <span>가설 검증 및 생성</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-600 mt-1">•</span>
                <span>모델링 전략 수립</span>
              </li>
            </ul>
          </div>

          <div className="bg-gradient-to-br from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 p-6 rounded-xl">
            <h4 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <AlertCircle className="text-orange-600" />
              EDA vs CDA
            </h4>
            <div className="space-y-3">
              <div>
                <p className="font-medium text-orange-800 dark:text-orange-400">EDA (탐색적 분석)</p>
                <p className="text-sm text-gray-700 dark:text-gray-300">• 가설 생성 중심</p>
                <p className="text-sm text-gray-700 dark:text-gray-300">• 유연한 접근</p>
                <p className="text-sm text-gray-700 dark:text-gray-300">• 시각화 중심</p>
              </div>
              <div>
                <p className="font-medium text-red-800 dark:text-red-400">CDA (확증적 분석)</p>
                <p className="text-sm text-gray-700 dark:text-gray-300">• 가설 검증 중심</p>
                <p className="text-sm text-gray-700 dark:text-gray-300">• 엄격한 통계 검정</p>
                <p className="text-sm text-gray-700 dark:text-gray-300">• p-value 중심</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 2. EDA 프로세스 */}
      <section>
        <h2 className="text-3xl font-bold mb-6">2. EDA 프로세스</h2>
        
        <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 p-6 rounded-xl mb-6">
          <h3 className="text-xl font-semibold mb-4">체계적인 EDA 단계</h3>
          
          <div className="space-y-4">
            {[
              {
                step: "1. 데이터 수집 및 로드",
                tasks: ["데이터 소스 확인", "데이터 형식 파악", "메모리 효율적 로드"],
                icon: <Database className="text-indigo-600" />
              },
              {
                step: "2. 데이터 구조 파악",
                tasks: ["shape, dtypes 확인", "메타데이터 검토", "컬럼 의미 이해"],
                icon: <FileSpreadsheet className="text-purple-600" />
              },
              {
                step: "3. 데이터 품질 평가",
                tasks: ["결측치 확인", "중복 데이터 검사", "데이터 타입 검증"],
                icon: <AlertCircle className="text-orange-600" />
              },
              {
                step: "4. 단변량 분석",
                tasks: ["기술통계", "분포 확인", "히스토그램/박스플롯"],
                icon: <BarChart className="text-green-600" />
              },
              {
                step: "5. 다변량 분석",
                tasks: ["상관관계 분석", "산점도 행렬", "특성 간 관계"],
                icon: <ScatterChart className="text-blue-600" />
              },
              {
                step: "6. 시각화 및 인사이트",
                tasks: ["대시보드 생성", "핵심 발견사항 정리", "다음 단계 제안"],
                icon: <Eye className="text-pink-600" />
              }
            ].map((phase, idx) => (
              <div key={idx} className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
                <div className="flex items-start gap-3">
                  <div className="mt-1">{phase.icon}</div>
                  <div className="flex-1">
                    <h4 className="font-semibold text-lg mb-2">{phase.step}</h4>
                    <div className="flex flex-wrap gap-2">
                      {phase.tasks.map((task, tidx) => (
                        <span key={tidx} className="text-sm bg-gray-100 dark:bg-gray-700 px-3 py-1 rounded-full">
                          {task}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* 3. 실전 코드 예제 */}
      <section>
        <h2 className="text-3xl font-bold mb-6">3. 실전 EDA with Pandas</h2>
        
        {/* 코드 예제 탭 */}
        <div className="bg-gray-900 rounded-xl overflow-hidden mb-6">
          <div className="flex border-b border-gray-700">
            {['기본 탐색', '시각화', '고급 분석'].map((tab, idx) => (
              <button
                key={idx}
                onClick={() => setActiveTab(['overview', 'visualization', 'advanced'][idx])}
                className={`px-6 py-3 font-medium transition-colors ${
                  activeTab === ['overview', 'visualization', 'advanced'][idx]
                    ? 'bg-gray-800 text-white border-b-2 border-blue-500'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                {tab}
              </button>
            ))}
          </div>

          <div className="p-6">
            {activeTab === 'overview' && (
              <div>
                <h4 className="text-white font-semibold mb-3">1. 데이터 로드 및 기본 탐색</h4>
                <pre className="bg-gray-800 p-4 rounded-lg overflow-x-auto">
                  <code className="text-sm text-gray-300">{`import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
df = pd.read_csv('sales_data.csv')

# 기본 정보 확인
print(f"데이터 크기: {df.shape}")
print(f"메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# 데이터 타입 확인
print("\n=== 데이터 타입 ===")
print(df.dtypes.value_counts())

# 기본 통계
print("\n=== 수치형 변수 요약 ===")
print(df.describe())

# 결측치 확인
print("\n=== 결측치 현황 ===")
missing_df = pd.DataFrame({
    'columns': df.columns,
    'missing_count': df.isnull().sum(),
    'missing_percent': (df.isnull().sum() / len(df)) * 100
})
print(missing_df[missing_df['missing_count'] > 0])`}</code>
                </pre>

                <h4 className="text-white font-semibold mt-6 mb-3">2. 데이터 품질 체크</h4>
                <pre className="bg-gray-800 p-4 rounded-lg overflow-x-auto">
                  <code className="text-sm text-gray-300">{`# 중복 데이터 확인
duplicates = df.duplicated().sum()
print(f"중복 행 개수: {duplicates}")

# 유니크 값 확인
for col in df.select_dtypes(include=['object']).columns:
    unique_count = df[col].nunique()
    print(f"{col}: {unique_count} unique values")
    if unique_count < 10:
        print(f"  Values: {df[col].unique()}")

# 이상치 탐지 (IQR 방법)
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

# 수치형 컬럼에 대해 이상치 확인
for col in df.select_dtypes(include=[np.number]).columns:
    outliers = detect_outliers_iqr(df, col)
    if len(outliers) > 0:
        print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")`}</code>
                </pre>
              </div>
            )}

            {activeTab === 'visualization' && (
              <div>
                <h4 className="text-white font-semibold mb-3">시각화를 통한 데이터 탐색</h4>
                <pre className="bg-gray-800 p-4 rounded-lg overflow-x-auto">
                  <code className="text-sm text-gray-300">{`# 시각화 설정
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 1. 수치형 변수 분포 확인
numeric_cols = df.select_dtypes(include=[np.number]).columns
fig, axes = plt.subplots(len(numeric_cols), 2, figsize=(12, 4*len(numeric_cols)))

for idx, col in enumerate(numeric_cols):
    # 히스토그램
    axes[idx, 0].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
    axes[idx, 0].set_title(f'{col} - Histogram')
    axes[idx, 0].set_xlabel(col)
    axes[idx, 0].set_ylabel('Frequency')
    
    # 박스플롯
    axes[idx, 1].boxplot(df[col].dropna())
    axes[idx, 1].set_title(f'{col} - Boxplot')
    axes[idx, 1].set_ylabel(col)

plt.tight_layout()
plt.show()

# 2. 상관관계 히트맵
correlation_matrix = df[numeric_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap')
plt.show()

# 3. 페어플롯 (변수 간 관계)
if len(numeric_cols) <= 5:  # 변수가 너무 많으면 시간이 오래 걸림
    sns.pairplot(df[numeric_cols], diag_kind='kde', plot_kws={'alpha': 0.5})
    plt.suptitle('Pairwise Relationships', y=1.02)
    plt.show()`}</code>
                </pre>
              </div>
            )}

            {activeTab === 'advanced' && (
              <div>
                <h4 className="text-white font-semibold mb-3">고급 EDA 기법</h4>
                <pre className="bg-gray-800 p-4 rounded-lg overflow-x-auto">
                  <code className="text-sm text-gray-300">{`# 1. 시계열 분석 (날짜 컬럼이 있는 경우)
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # 시간에 따른 트렌드
    daily_sales = df.groupby('date')['sales'].sum()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 일별 매출
    ax1.plot(daily_sales.index, daily_sales.values, linewidth=1)
    ax1.set_title('Daily Sales Trend')
    ax1.set_ylabel('Sales')
    
    # 이동평균
    daily_sales.rolling(window=7).mean().plot(ax=ax2, label='7-day MA')
    daily_sales.rolling(window=30).mean().plot(ax=ax2, label='30-day MA')
    ax2.set_title('Moving Averages')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# 2. 범주형 변수 분석
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols[:3]:  # 상위 3개만
    plt.figure(figsize=(10, 5))
    
    # 빈도 분석
    value_counts = df[col].value_counts()
    
    plt.subplot(1, 2, 1)
    value_counts.head(10).plot(kind='bar')
    plt.title(f'{col} - Top 10 Values')
    plt.xticks(rotation=45, ha='right')
    
    # 파이 차트 (카테고리가 적은 경우)
    if len(value_counts) <= 8:
        plt.subplot(1, 2, 2)
        value_counts.plot(kind='pie', autopct='%1.1f%%')
        plt.title(f'{col} - Distribution')
        plt.ylabel('')
    
    plt.tight_layout()
    plt.show()

# 3. 이변량 분석 (타겟 변수가 있는 경우)
if 'target' in df.columns:
    # 각 특성과 타겟의 관계
    for col in numeric_cols:
        if col != 'target':
            plt.figure(figsize=(10, 5))
            
            # 산점도
            plt.subplot(1, 2, 1)
            plt.scatter(df[col], df['target'], alpha=0.5)
            plt.xlabel(col)
            plt.ylabel('Target')
            plt.title(f'{col} vs Target')
            
            # 상관계수
            corr = df[col].corr(df['target'])
            plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                    transform=plt.gca().transAxes, 
                    bbox=dict(boxstyle='round', facecolor='wheat'))
            
            plt.tight_layout()
            plt.show()`}</code>
                </pre>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* 4. Polars로 하는 현대적 EDA */}
      <section>
        <h2 className="text-3xl font-bold mb-6">4. Polars로 하는 현대적 EDA</h2>
        
        <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 p-6 rounded-xl mb-6">
          <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Zap className="text-purple-600" />
            왜 Polars인가?
          </h3>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-semibold text-purple-600 dark:text-purple-400 mb-2">⚡ 초고속 성능</h4>
              <p className="text-sm">Rust 기반으로 Pandas보다 10-100배 빠른 처리</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-semibold text-purple-600 dark:text-purple-400 mb-2">🧮 지연 실행</h4>
              <p className="text-sm">쿼리 최적화로 메모리 효율적인 처리</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-semibold text-purple-600 dark:text-purple-400 mb-2">🔄 병렬 처리</h4>
              <p className="text-sm">멀티코어 활용으로 대용량 데이터 처리</p>
            </div>
          </div>
        </div>

        <div className="bg-gray-900 rounded-xl p-6 mb-6">
          <h4 className="text-white font-semibold mb-3">Polars를 활용한 효율적인 EDA</h4>
          <pre className="bg-gray-800 p-4 rounded-lg overflow-x-auto">
            <code className="text-sm text-gray-300">{`import polars as pl
import polars.selectors as cs

# 1. 데이터 로드 (지연 실행 모드)
df = pl.scan_csv("large_dataset.csv")

# 2. 효율적인 데이터 프로파일링
profile = df.select([
    pl.count().alias("total_rows"),
    cs.numeric().null_count().name.prefix("null_"),
    cs.numeric().mean().name.prefix("mean_"),
    cs.numeric().std().name.prefix("std_"),
    cs.numeric().min().name.prefix("min_"),
    cs.numeric().max().name.prefix("max_"),
]).collect()

print(profile)

# 3. 조건부 집계 (매우 빠름!)
summary = df.lazy().group_by("category").agg([
    pl.col("sales").sum().alias("total_sales"),
    pl.col("sales").mean().alias("avg_sales"),
    pl.col("sales").std().alias("std_sales"),
    pl.col("customer_id").n_unique().alias("unique_customers"),
    (pl.col("sales") > pl.col("sales").mean()).sum().alias("above_avg_count")
]).sort("total_sales", descending=True).collect()

# 4. 윈도우 함수를 활용한 시계열 분석
time_analysis = df.lazy().with_columns([
    pl.col("date").str.to_date(),
]).sort("date").with_columns([
    pl.col("sales").rolling_mean(window_size=7).alias("sales_7d_ma"),
    pl.col("sales").rolling_std(window_size=7).alias("sales_7d_std"),
    pl.col("sales").pct_change().alias("sales_pct_change"),
    pl.col("sales").rank().over("category").alias("sales_rank_by_category")
]).collect()

# 5. 복잡한 필터링과 변환
outliers = df.lazy().filter(
    (pl.col("sales") > pl.col("sales").quantile(0.99)) |
    (pl.col("sales") < pl.col("sales").quantile(0.01))
).with_columns([
    pl.when(pl.col("sales") > pl.col("sales").quantile(0.99))
      .then(pl.lit("upper_outlier"))
      .otherwise(pl.lit("lower_outlier"))
      .alias("outlier_type")
]).collect()

print(f"Found {len(outliers)} outliers")`}</code>
          </pre>
        </div>
      </section>

      {/* 5. 실전 EDA 체크리스트 */}
      <section>
        <h2 className="text-3xl font-bold mb-6">5. 실전 EDA 체크리스트</h2>
        
        <div className="bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-900/20 dark:to-blue-900/20 p-6 rounded-xl">
          <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <CheckCircle className="text-green-600" />
            완벽한 EDA를 위한 체크리스트
          </h3>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold mb-3">📊 데이터 이해</h4>
              <div className="space-y-2">
                {[
                  "데이터 출처와 수집 방법 확인",
                  "각 변수의 의미와 단위 파악",
                  "타겟 변수 정의 명확히 이해",
                  "데이터 생성 시점과 주기 확인",
                  "샘플링 방법과 대표성 검토"
                ].map((item, idx) => (
                  <label key={idx} className="flex items-center gap-2 text-sm">
                    <input type="checkbox" className="rounded" />
                    <span>{item}</span>
                  </label>
                ))}
              </div>
            </div>

            <div>
              <h4 className="font-semibold mb-3">🔍 데이터 품질</h4>
              <div className="space-y-2">
                {[
                  "결측치 패턴과 원인 분석",
                  "이상치 탐지 및 처리 방안",
                  "중복 데이터 확인 및 제거",
                  "데이터 일관성 검증",
                  "타입 오류 및 형식 확인"
                ].map((item, idx) => (
                  <label key={idx} className="flex items-center gap-2 text-sm">
                    <input type="checkbox" className="rounded" />
                    <span>{item}</span>
                  </label>
                ))}
              </div>
            </div>

            <div>
              <h4 className="font-semibold mb-3">📈 통계 분석</h4>
              <div className="space-y-2">
                {[
                  "기술통계량 계산 (평균, 중앙값, 분산)",
                  "분포 형태 확인 (정규성, 왜도, 첨도)",
                  "상관관계 분석",
                  "그룹별 비교 분석",
                  "시계열 패턴 분석 (추세, 계절성)"
                ].map((item, idx) => (
                  <label key={idx} className="flex items-center gap-2 text-sm">
                    <input type="checkbox" className="rounded" />
                    <span>{item}</span>
                  </label>
                ))}
              </div>
            </div>

            <div>
              <h4 className="font-semibold mb-3">🎨 시각화</h4>
              <div className="space-y-2">
                {[
                  "단변량 분포 시각화",
                  "이변량 관계 시각화",
                  "다변량 시각화 (페어플롯, 히트맵)",
                  "시계열 시각화",
                  "대시보드 생성"
                ].map((item, idx) => (
                  <label key={idx} className="flex items-center gap-2 text-sm">
                    <input type="checkbox" className="rounded" />
                    <span>{item}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 실습 프로젝트 */}
      <section className="mt-12">
        <div className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white p-8 rounded-xl">
          <h2 className="text-2xl font-bold mb-4">🚀 실습 프로젝트</h2>
          <p className="mb-6">
            실제 데이터셋을 사용해 완전한 EDA를 수행해보세요. 
            Kaggle의 "Titanic" 또는 "House Prices" 데이터셋으로 시작하는 것을 추천합니다.
          </p>
          <div className="flex gap-4">
            <button className="bg-white text-indigo-600 px-6 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-colors">
              프로젝트 시작하기
            </button>
            <button className="bg-indigo-500 text-white px-6 py-3 rounded-lg font-semibold hover:bg-indigo-400 transition-colors">
              예제 노트북 보기
            </button>
          </div>
        </div>
      </section>
    </div>
  )
}