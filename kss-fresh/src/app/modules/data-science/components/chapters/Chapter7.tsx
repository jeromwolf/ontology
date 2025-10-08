'use client';

import React from 'react';
import { BookOpen, TrendingUp, Clock, BarChart2, Lightbulb } from 'lucide-react';
import References from '@/components/common/References';

interface Chapter7Props {
  onComplete?: () => void
}

export default function Chapter7({ onComplete }: Chapter7Props) {
  return (
    <div className="max-w-4xl mx-auto p-6 space-y-8">
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold text-primary">Chapter 7: 시계열 분석과 예측</h1>
        <p className="text-xl text-muted-foreground">
          시간에 따른 데이터 패턴을 분석하고 미래를 예측하는 방법을 학습합니다
        </p>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 px-6 py-4">
          <h2 className="text-xl font-semibold flex items-center gap-2">
            <Clock className="w-6 h-6 text-blue-600 dark:text-blue-400" />
            시계열 데이터란?
          </h2>
        </div>
        <div className="p-6 space-y-4">
          <div className="space-y-2">
            <h3 className="text-lg font-semibold">정의</h3>
            <p className="text-gray-600 dark:text-gray-400">
              시계열 데이터는 시간 순서대로 정렬된 데이터 포인트들의 집합입니다. 
              각 데이터 포인트는 특정 시점의 관측값을 나타냅니다.
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4">
              <h4 className="font-semibold mb-2">시계열 데이터의 특징</h4>
              <ul className="space-y-2 list-disc list-inside text-sm">
                <li>시간적 의존성 (Temporal Dependency)</li>
                <li>계절성 (Seasonality)</li>
                <li>추세 (Trend)</li>
                <li>노이즈 (Noise)</li>
              </ul>
            </div>
            
            <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4">
              <h4 className="font-semibold mb-2">응용 분야</h4>
              <ul className="space-y-2 list-disc list-inside text-sm">
                <li>주가 예측</li>
                <li>날씨 예보</li>
                <li>수요 예측</li>
                <li>트래픽 분석</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
        <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 px-6 py-4">
          <h2 className="text-xl font-semibold flex items-center gap-2">
            <TrendingUp className="w-6 h-6 text-purple-600 dark:text-purple-400" />
            시계열 구성 요소
          </h2>
        </div>
        <div className="p-6 space-y-6">
          <div className="space-y-4">
            <div className="border-l-4 border-blue-500 pl-4">
              <h4 className="font-semibold">1. 추세 (Trend)</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                데이터의 장기적인 방향성을 나타냅니다. 상승, 하락, 또는 수평 추세가 있을 수 있습니다.
              </p>
            </div>
            
            <div className="border-l-4 border-purple-500 pl-4">
              <h4 className="font-semibold">2. 계절성 (Seasonality)</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                고정된 기간에 반복되는 패턴입니다. 예: 여름철 아이스크림 판매량 증가
              </p>
            </div>
            
            <div className="border-l-4 border-pink-500 pl-4">
              <h4 className="font-semibold">3. 주기성 (Cyclical)</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                계절성과 달리 고정된 주기가 없는 반복 패턴입니다. 예: 경제 순환
              </p>
            </div>
            
            <div className="border-l-4 border-gray-500 pl-4">
              <h4 className="font-semibold">4. 불규칙 요소 (Irregular)</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                예측할 수 없는 랜덤한 변동입니다. 예: 자연재해, 파업 등
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
        <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 px-6 py-4">
          <h2 className="text-xl font-semibold flex items-center gap-2">
            <BarChart2 className="w-6 h-6 text-green-600 dark:text-green-400" />
            주요 시계열 분석 기법
          </h2>
        </div>
        <div className="p-6 space-y-6">
          <div className="space-y-4">
            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <h4 className="font-semibold mb-2">이동평균 (Moving Average)</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                과거 n개 기간의 평균을 계산하여 단기 변동을 smoothing합니다.
              </p>
              <pre className="bg-gray-100 dark:bg-gray-900 p-2 rounded text-xs overflow-x-auto">
                MA(t) = (x(t) + x(t-1) + ... + x(t-n+1)) / n
              </pre>
            </div>
            
            <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
              <h4 className="font-semibold mb-2">지수평활법 (Exponential Smoothing)</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                최근 데이터에 더 큰 가중치를 부여하는 예측 방법입니다.
              </p>
              <pre className="bg-gray-100 dark:bg-gray-900 p-2 rounded text-xs overflow-x-auto">
                S(t) = α × x(t) + (1-α) × S(t-1)
              </pre>
            </div>
            
            <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
              <h4 className="font-semibold mb-2">ARIMA 모델</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                자기회귀(AR), 차분(I), 이동평균(MA)을 결합한 강력한 예측 모델입니다.
              </p>
              <ul className="text-sm space-y-1 mt-2 list-disc list-inside">
                <li>AR(p): 과거 p개 시점의 값으로 현재 예측</li>
                <li>I(d): d번 차분하여 정상성 확보</li>
                <li>MA(q): 과거 q개 시점의 오차 항 사용</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
        <div className="bg-gradient-to-r from-indigo-50 to-blue-50 dark:from-indigo-900/20 dark:to-blue-900/20 px-6 py-4">
          <h2 className="text-xl font-semibold flex items-center gap-2">
            <BookOpen className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />
            Python으로 시계열 분석하기
          </h2>
        </div>
        <div className="p-6 space-y-4">
          <div>
            <h4 className="font-semibold mb-2">1. 데이터 준비</h4>
            <pre className="bg-gray-100 dark:bg-gray-900 p-4 rounded-lg text-sm overflow-x-auto">
{`import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# 시계열 데이터 생성
dates = pd.date_range('2020-01-01', periods=365, freq='D')
trend = np.linspace(100, 200, 365)
seasonal = 10 * np.sin(2 * np.pi * np.arange(365) / 30)
noise = np.random.normal(0, 5, 365)
ts = pd.Series(trend + seasonal + noise, index=dates)`}</pre>
          </div>
          
          <div>
            <h4 className="font-semibold mb-2">2. 시계열 분해</h4>
            <pre className="bg-gray-100 dark:bg-gray-900 p-4 rounded-lg text-sm overflow-x-auto">
{`# 시계열 분해
decomposition = seasonal_decompose(ts, model='additive', period=30)

# 결과 시각화
fig, axes = plt.subplots(4, 1, figsize=(12, 8))
ts.plot(ax=axes[0], title='원본 시계열')
decomposition.trend.plot(ax=axes[1], title='추세')
decomposition.seasonal.plot(ax=axes[2], title='계절성')
decomposition.resid.plot(ax=axes[3], title='잔차')
plt.tight_layout()`}</pre>
          </div>
          
          <div>
            <h4 className="font-semibold mb-2">3. ARIMA 모델 적용</h4>
            <pre className="bg-gray-100 dark:bg-gray-900 p-4 rounded-lg text-sm overflow-x-auto">
{`from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ACF, PACF 확인
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
plot_acf(ts, lags=40, ax=ax1)
plot_pacf(ts, lags=40, ax=ax2)

# ARIMA 모델 학습
model = ARIMA(ts, order=(2, 1, 2))
results = model.fit()

# 예측
forecast = results.forecast(steps=30)
print(forecast)`}</pre>
          </div>
        </div>
      </div>

      <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4 flex gap-3">
        <Lightbulb className="w-5 h-5 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
        <div>
          <strong>실전 팁:</strong> 시계열 예측의 정확도를 높이려면:
          <ul className="mt-2 space-y-1 list-disc list-inside">
            <li>데이터의 정상성(Stationarity)을 확인하고 필요시 변환하세요</li>
            <li>여러 모델을 비교하고 앙상블 기법을 활용하세요</li>
            <li>외부 변수(External Regressors)를 포함시켜 예측력을 향상시키세요</li>
            <li>예측 구간(Prediction Interval)을 함께 제공하여 불확실성을 표현하세요</li>
          </ul>
        </div>
      </div>

      <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/10 dark:to-indigo-900/10 rounded-xl p-6 border-2 border-blue-200 dark:border-blue-800">
        <h3 className="text-xl font-semibold mb-2">실습 프로젝트: 주식 가격 예측</h3>
        <p className="text-gray-600 dark:text-gray-400 mb-4">
          실제 주식 데이터를 사용하여 가격을 예측하는 모델을 만들어봅시다
        </p>
        <div className="space-y-4">
          <div className="space-y-3">
            <h4 className="font-semibold">프로젝트 단계:</h4>
            <ol className="space-y-2 list-decimal list-inside">
              <li>yfinance를 사용하여 주식 데이터 수집</li>
              <li>기술적 지표 추가 (이동평균, RSI, MACD)</li>
              <li>Prophet 또는 LSTM 모델로 예측</li>
              <li>백테스팅으로 성능 평가</li>
              <li>대시보드로 결과 시각화</li>
            </ol>
          </div>
          
          <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
            <p className="text-sm">
              주식 예측은 매우 어려운 문제입니다. 실제 투자에는 사용하지 마시고, 
              학습 목적으로만 활용하세요!
            </p>
          </div>
        </div>
      </div>

      <References
        sections={[
          {
            title: 'Time Series Analysis Foundations',
            icon: 'paper',
            color: 'border-blue-500',
            items: [
              {
                title: 'Time Series Analysis: Forecasting and Control',
                authors: 'George E. P. Box, Gwilym M. Jenkins, Gregory C. Reinsel',
                year: '2015',
                description: '시계열 분석의 바이블 - ARIMA 모델의 고전 (Wiley, 5th Edition)',
                link: 'https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+5th+Edition-p-9781118675021'
              },
              {
                title: 'Forecasting: Principles and Practice',
                authors: 'Rob J Hyndman, George Athanasopoulos',
                year: '2021',
                description: '현대적 예측 기법 완벽 가이드 - 무료 온라인북 (3rd Edition)',
                link: 'https://otexts.com/fpp3/'
              },
              {
                title: 'Autoregressive Conditional Heteroscedasticity',
                authors: 'Robert F. Engle',
                year: '1982',
                description: 'ARCH 모델 - 변동성 모델링의 시작 (노벨경제학상, Econometrica)',
                link: 'https://www.jstor.org/stable/1912773'
              }
            ]
          },
          {
            title: 'Modern Forecasting Methods',
            icon: 'paper',
            color: 'border-purple-500',
            items: [
              {
                title: 'Forecasting at Scale',
                authors: 'Sean J. Taylor, Benjamin Letham',
                year: '2018',
                description: 'Facebook Prophet - 자동화된 시계열 예측 (The American Statistician)',
                link: 'https://peerj.com/preprints/3190/'
              },
              {
                title: 'N-BEATS: Neural basis expansion analysis for time series',
                authors: 'Boris N. Oreshkin, et al.',
                year: '2020',
                description: '딥러닝 기반 시계열 예측 - SOTA 성능 (ICLR 2020)',
                link: 'https://arxiv.org/abs/1905.10437'
              },
              {
                title: 'Informer: Beyond Efficient Transformer',
                authors: 'Haoyi Zhou, et al.',
                year: '2021',
                description: 'Transformer 기반 장기 시계열 예측 (AAAI 2021)',
                link: 'https://arxiv.org/abs/2012.07436'
              }
            ]
          },
          {
            title: 'Specialized Techniques',
            icon: 'paper',
            color: 'border-green-500',
            items: [
              {
                title: 'Seasonal and Trend decomposition using Loess',
                authors: 'R. B. Cleveland, et al.',
                year: '1990',
                description: 'STL 분해 - 계절성과 트렌드 분리 (Journal of Official Statistics)',
                link: 'https://www.scb.se/contentassets/ca21efb41fee47d293bbee5bf7be7fb3/stl-a-seasonal-trend-decomposition-procedure-based-on-loess.pdf'
              },
              {
                title: 'Dynamic Time Warping Algorithm Review',
                authors: 'Toni Giorgino',
                year: '2009',
                description: 'DTW - 시계열 유사도 측정 (R Package)',
                link: 'https://www.jstatsoft.org/article/view/v031i07'
              }
            ]
          },
          {
            title: 'Libraries & Tools',
            icon: 'web',
            color: 'border-orange-500',
            items: [
              {
                title: 'Prophet Documentation',
                description: 'Facebook의 자동 시계열 예측 라이브러리',
                link: 'https://facebook.github.io/prophet/'
              },
              {
                title: 'statsmodels Time Series',
                description: 'Python ARIMA, SARIMAX, VAR 등 고전 모델',
                link: 'https://www.statsmodels.org/stable/tsa.html'
              },
              {
                title: 'pmdarima',
                description: 'Auto ARIMA - 자동 파라미터 선택',
                link: 'http://alkaline-ml.com/pmdarima/'
              },
              {
                title: 'NeuralProphet',
                description: 'Prophet + 딥러닝 - 차세대 예측 도구',
                link: 'https://neuralprophet.com/'
              },
              {
                title: 'Darts',
                description: '통합 시계열 라이브러리 - 고전부터 딥러닝까지',
                link: 'https://unit8co.github.io/darts/'
              }
            ]
          }
        ]}
      />

      <div className="flex justify-between items-center pt-8">
        <p className="text-sm text-gray-600 dark:text-gray-400">
          시계열 분석은 데이터 사이언스의 핵심 기술입니다
        </p>
        {onComplete && (
          <button
            onClick={onComplete}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            다음 챕터로
          </button>
        )}
      </div>
    </div>
  )
}