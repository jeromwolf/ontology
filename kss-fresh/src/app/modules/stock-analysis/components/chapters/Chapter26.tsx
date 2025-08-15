'use client';

import { useState } from 'react';

export default function Chapter26() {
  const [activeDataType, setActiveDataType] = useState('price');

  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6">금융 빅데이터 분석</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6">
          방대한 금융 데이터를 수집, 정제, 분석하여 투자 인사이트를 도출하는 방법을 배워봅시다.
          Python과 최신 데이터 분석 도구를 활용한 실전 빅데이터 분석 기법을 다룹니다.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🌊 금융 데이터의 바다</h2>
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-4">
            금융 데이터의 특성과 규모
          </h3>
          
          <div className="grid md:grid-cols-3 gap-4 mb-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">Volume (규모)</h4>
              <ul className="text-sm space-y-1">
                <li>• NYSE: 일일 40억 건 거래</li>
                <li>• 틱 데이터: 초당 수백만 건</li>
                <li>• 뉴스: 일일 수십만 기사</li>
                <li>• SNS: 실시간 수억 건</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">Velocity (속도)</h4>
              <ul className="text-sm space-y-1">
                <li>• 실시간 스트리밍</li>
                <li>• 마이크로초 단위 갱신</li>
                <li>• 24/7 글로벌 시장</li>
                <li>• 지연시간 민감도</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">Variety (다양성)</h4>
              <ul className="text-sm space-y-1">
                <li>• 정형: 가격, 거래량</li>
                <li>• 반정형: 재무제표</li>
                <li>• 비정형: 뉴스, SNS</li>
                <li>• 대안: 위성, IoT</li>
              </ul>
            </div>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
            <p className="text-sm font-semibold text-yellow-800 dark:text-yellow-200">
              💡 핵심 과제
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              이러한 방대한 데이터에서 의미 있는 신호를 찾고, 노이즈를 제거하며,
              실시간으로 처리하는 것이 금융 빅데이터 분석의 핵심입니다.
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📊 데이터 소스별 분석</h2>
        <div className="mb-4">
          <div className="flex gap-2 flex-wrap">
            <button
              onClick={() => setActiveDataType('price')}
              className={`px-4 py-2 rounded-lg font-medium ${
                activeDataType === 'price'
                  ? 'bg-purple-500 text-white'
                  : 'bg-gray-200 dark:bg-gray-700'
              }`}
            >
              가격 데이터
            </button>
            <button
              onClick={() => setActiveDataType('fundamental')}
              className={`px-4 py-2 rounded-lg font-medium ${
                activeDataType === 'fundamental'
                  ? 'bg-purple-500 text-white'
                  : 'bg-gray-200 dark:bg-gray-700'
              }`}
            >
              펀더멘털
            </button>
            <button
              onClick={() => setActiveDataType('alternative')}
              className={`px-4 py-2 rounded-lg font-medium ${
                activeDataType === 'alternative'
                  ? 'bg-purple-500 text-white'
                  : 'bg-gray-200 dark:bg-gray-700'
              }`}
            >
              대안 데이터
            </button>
            <button
              onClick={() => setActiveDataType('sentiment')}
              className={`px-4 py-2 rounded-lg font-medium ${
                activeDataType === 'sentiment'
                  ? 'bg-purple-500 text-white'
                  : 'bg-gray-200 dark:bg-gray-700'
              }`}
            >
              감성 분석
            </button>
          </div>
        </div>

        <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
          {activeDataType === 'price' && (
            <div>
              <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-3">
                가격 데이터 분석
              </h3>
              <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
                <pre className="text-sm">
{`import pandas as pd
import numpy as np
from datetime import datetime
import pyarrow.parquet as pq

class PriceDataAnalyzer:
    def __init__(self, data_path):
        self.data_path = data_path
        
    def load_tick_data(self, date, symbol):
        """틱 데이터 로드 (Parquet 형식)"""
        file_path = f"{self.data_path}/{date}/{symbol}.parquet"
        df = pd.read_parquet(file_path)
        
        # 타임스탬프 인덱스 설정
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def calculate_microstructure(self, df):
        """시장 미시구조 지표 계산"""
        # 유효 스프레드
        df['spread'] = df['ask'] - df['bid']
        df['mid_price'] = (df['ask'] + df['bid']) / 2
        
        # 주문 불균형
        df['order_imbalance'] = (df['bid_size'] - df['ask_size']) / \
                               (df['bid_size'] + df['ask_size'])
        
        # 가격 영향력
        df['price_impact'] = df['mid_price'].diff() / df['volume']
        
        # VWAP (Volume Weighted Average Price)
        df['vwap'] = (df['price'] * df['volume']).cumsum() / \
                     df['volume'].cumsum()
        
        return df`}</pre>
              </div>
              <div className="mt-4 space-y-2">
                <h4 className="font-semibold">주요 분석 포인트:</h4>
                <ul className="text-sm space-y-1">
                  <li>• 틱 데이터: 모든 거래와 호가 기록</li>
                  <li>• 분봉/일봉: 집계된 OHLCV 데이터</li>
                  <li>• 시장 미시구조: 스프레드, 깊이, 영향력</li>
                  <li>• 고빈도 패턴: 일중 계절성, 변동성 클러스터링</li>
                </ul>
              </div>
            </div>
          )}

          {activeDataType === 'fundamental' && (
            <div>
              <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-3">
                펀더멘털 데이터 분석
              </h3>
              <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
                <pre className="text-sm">
{`class FundamentalAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        
    def get_financial_statements(self, ticker, period='annual'):
        """재무제표 데이터 수집"""
        # API를 통한 데이터 수집 (예: SimFin, Quandl)
        statements = {
            'income': self.fetch_income_statement(ticker, period),
            'balance': self.fetch_balance_sheet(ticker, period),
            'cashflow': self.fetch_cashflow(ticker, period)
        }
        return statements
    
    def calculate_ratios(self, statements):
        """주요 재무 비율 계산"""
        ratios = {}
        
        # 수익성 지표
        ratios['ROE'] = statements['income']['net_income'] / \
                        statements['balance']['equity']
        ratios['ROA'] = statements['income']['net_income'] / \
                        statements['balance']['total_assets']
        ratios['profit_margin'] = statements['income']['net_income'] / \
                                  statements['income']['revenue']
        
        # 성장성 지표
        ratios['revenue_growth'] = statements['income']['revenue'].pct_change()
        ratios['earnings_growth'] = statements['income']['net_income'].pct_change()
        
        # 안정성 지표
        ratios['debt_ratio'] = statements['balance']['total_debt'] / \
                              statements['balance']['total_assets']
        ratios['current_ratio'] = statements['balance']['current_assets'] / \
                                 statements['balance']['current_liabilities']
        
        return ratios
    
    def industry_comparison(self, ticker, industry_peers):
        """산업 내 비교 분석"""
        peer_data = {}
        for peer in industry_peers:
            peer_data[peer] = self.calculate_ratios(
                self.get_financial_statements(peer)
            )
        
        # Z-score 정규화
        comparison = pd.DataFrame(peer_data).T
        z_scores = (comparison - comparison.mean()) / comparison.std()
        
        return z_scores`}</pre>
              </div>
              <div className="mt-4 space-y-2">
                <h4 className="font-semibold">분석 차원:</h4>
                <ul className="text-sm space-y-1">
                  <li>• 시계열 분석: 기업의 성장 추세</li>
                  <li>• 횡단면 분석: 동종업계 비교</li>
                  <li>• 품질 점수: Piotroski F-Score, M-Score</li>
                  <li>• 예측 모델: 실적 예측, 부도 확률</li>
                </ul>
              </div>
            </div>
          )}

          {activeDataType === 'alternative' && (
            <div>
              <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-3">
                대안 데이터 분석
              </h3>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
                <h4 className="font-semibold mb-2">위성 이미지 분석</h4>
                <div className="bg-gray-900 text-gray-100 rounded-lg p-3 overflow-x-auto">
                  <pre className="text-sm">
{`# 주차장 차량 수 계산으로 매출 예측
import cv2
import tensorflow as tf

def analyze_parking_lot(satellite_image):
    # 객체 탐지 모델 (YOLO, R-CNN)
    model = tf.keras.models.load_model('car_detection_model.h5')
    
    # 차량 탐지
    cars = model.predict(satellite_image)
    car_count = len(cars)
    
    # 시계열 분석
    historical_counts = load_historical_data()
    trend = calculate_trend(historical_counts, car_count)
    
    return {
        'current_count': car_count,
        'yoy_change': trend['yoy'],
        'sales_estimate': estimate_sales(car_count)
    }`}</pre>
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
                <h4 className="font-semibold mb-2">웹 스크래핑 & 크롤링</h4>
                <div className="bg-gray-900 text-gray-100 rounded-lg p-3 overflow-x-auto">
                  <pre className="text-sm">
{`# 이커머스 가격 모니터링
from scrapy import Spider
import pandas as pd

class PriceMonitorSpider(Spider):
    name = 'price_monitor'
    
    def parse(self, response):
        products = []
        for item in response.css('.product-item'):
            product = {
                'name': item.css('.title::text').get(),
                'price': float(item.css('.price::text').re_first(r'[\d.]+')),
                'stock': item.css('.stock::text').get(),
                'reviews': int(item.css('.review-count::text').re_first(r'\d+')),
                'timestamp': datetime.now()
            }
            products.append(product)
        
        # 가격 변동 및 재고 분석
        df = pd.DataFrame(products)
        insights = self.analyze_pricing_strategy(df)
        
        return insights`}</pre>
                </div>
              </div>

              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                  <h5 className="font-medium mb-2">신용카드 데이터</h5>
                  <ul className="text-sm space-y-1">
                    <li>• 소비 트렌드 분석</li>
                    <li>• 업종별 매출 추정</li>
                    <li>• 지역별 경제 활동</li>
                  </ul>
                </div>
                <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                  <h5 className="font-medium mb-2">앱 사용 데이터</h5>
                  <ul className="text-sm space-y-1">
                    <li>• MAU/DAU 트렌드</li>
                    <li>• 사용 시간 패턴</li>
                    <li>• 유저 이탈률</li>
                  </ul>
                </div>
              </div>
            </div>
          )}

          {activeDataType === 'sentiment' && (
            <div>
              <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-3">
                감성 분석 & 텍스트 마이닝
              </h3>
              <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto mb-4">
                <pre className="text-sm">
{`from transformers import pipeline
import pandas as pd
from textblob import TextBlob
import nltk

class FinancialSentimentAnalyzer:
    def __init__(self):
        # FinBERT: 금융 특화 언어 모델
        self.finbert = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert"
        )
        
    def analyze_news_sentiment(self, news_df):
        """뉴스 감성 분석"""
        sentiments = []
        
        for idx, row in news_df.iterrows():
            # 제목과 본문 결합
            text = f"{row['title']} {row['content'][:500]}"
            
            # FinBERT 감성 분석
            result = self.finbert(text)[0]
            
            sentiment = {
                'timestamp': row['timestamp'],
                'ticker': row['ticker'],
                'sentiment': result['label'],
                'confidence': result['score'],
                'source': row['source']
            }
            sentiments.append(sentiment)
        
        return pd.DataFrame(sentiments)
    
    def social_media_analysis(self, tweets):
        """소셜 미디어 감성 분석"""
        # 전처리
        tweets['clean_text'] = tweets['text'].apply(self.preprocess_text)
        
        # 감성 점수 계산
        tweets['polarity'] = tweets['clean_text'].apply(
            lambda x: TextBlob(x).sentiment.polarity
        )
        
        # 주요 키워드 추출
        keywords = self.extract_keywords(tweets['clean_text'])
        
        # 시간대별 감성 집계
        hourly_sentiment = tweets.set_index('timestamp').resample('H').agg({
            'polarity': ['mean', 'std', 'count'],
            'retweets': 'sum',
            'likes': 'sum'
        })
        
        return {
            'sentiment_trend': hourly_sentiment,
            'keywords': keywords,
            'influencer_posts': self.identify_influencers(tweets)
        }`}</pre>
              </div>
              
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-2">텍스트 소스</h4>
                  <ul className="text-sm space-y-1">
                    <li>📰 뉴스 기사 (Reuters, Bloomberg)</li>
                    <li>📊 애널리스트 리포트</li>
                    <li>📱 소셜 미디어 (Twitter, Reddit)</li>
                    <li>📝 기업 공시 및 컨퍼런스 콜</li>
                    <li>💬 투자자 포럼 및 커뮤니티</li>
                  </ul>
                </div>
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <h4 className="font-semibold mb-2">분석 기법</h4>
                  <ul className="text-sm space-y-1">
                    <li>• NLP: 자연어 처리</li>
                    <li>• Topic Modeling: LDA, BERT</li>
                    <li>• Named Entity Recognition</li>
                    <li>• Event Detection</li>
                    <li>• Aspect-based Sentiment</li>
                  </ul>
                </div>
              </div>
            </div>
          )}
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🔧 빅데이터 처리 인프라</h2>
        <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg p-6">
          <h3 className="font-semibold mb-4">확장 가능한 데이터 파이프라인</h3>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <h4 className="font-semibold mb-3">데이터 처리 아키텍처</h4>
            <div className="bg-gray-100 dark:bg-gray-700 rounded p-4 font-mono text-sm">
              <pre>
{`┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Data Sources│ ──► │ Data Pipeline│ ──► │ Data Storage│
└─────────────┘     └──────────────┘     └─────────────┘
      │                     │                     │
      ▼                     ▼                     ▼
- Market Data        - Apache Kafka         - Data Lake (S3)
- News Feeds        - Apache Spark         - Time Series DB
- Social Media      - Apache Airflow       - Graph Database
- Alt Data          - Stream Processing    - Feature Store`}</pre>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">실시간 처리</h4>
              <div className="bg-gray-900 text-gray-100 rounded-lg p-3 overflow-x-auto">
                <pre className="text-sm">
{`# Kafka + Spark Streaming
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("FinancialStream") \
    .getOrCreate()

# Kafka 스트림 읽기
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "market-data") \
    .load()

# 실시간 집계
aggregated = df \
    .groupBy(window(col("timestamp"), "1 minute")) \
    .agg(
        avg("price").alias("avg_price"),
        sum("volume").alias("total_volume"),
        stddev("price").alias("volatility")
    )`}</pre>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">배치 처리</h4>
              <div className="bg-gray-900 text-gray-100 rounded-lg p-3 overflow-x-auto">
                <pre className="text-sm">
{`# Airflow DAG 정의
from airflow import DAG
from airflow.operators.python import PythonOperator

def process_daily_data():
    # S3에서 데이터 로드
    df = spark.read.parquet("s3://bucket/daily/")
    
    # 일일 지표 계산
    daily_metrics = df.groupBy("symbol").agg(
        F.mean("return").alias("avg_return"),
        F.stddev("return").alias("volatility"),
        F.corr("volume", "return").alias("vol_ret_corr")
    )
    
    # 결과 저장
    daily_metrics.write.parquet("s3://bucket/metrics/")

dag = DAG('daily_analysis', schedule_interval='@daily')
task = PythonOperator(
    task_id='process_data',
    python_callable=process_daily_data,
    dag=dag
)`}</pre>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📈 고급 분석 기법</h2>
        <div className="space-y-4">
          <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-indigo-800 dark:text-indigo-200 mb-3">
              머신러닝 기반 예측 모델
            </h3>
            
            <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm">
{`import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import shap

class MarketPredictionModel:
    def __init__(self):
        self.models = {
            'xgboost': xgb.XGBRegressor(n_estimators=1000, max_depth=5),
            'random_forest': RandomForestRegressor(n_estimators=500),
            'lstm': self.build_lstm_model()
        }
        
    def engineer_features(self, df):
        """특징 공학"""
        features = pd.DataFrame()
        
        # 기술적 지표
        features['rsi'] = self.calculate_rsi(df['close'])
        features['macd'] = self.calculate_macd(df['close'])
        features['bb_position'] = self.bollinger_position(df['close'])
        
        # 시장 미시구조
        features['bid_ask_spread'] = df['ask'] - df['bid']
        features['order_flow'] = df['buy_volume'] - df['sell_volume']
        features['price_impact'] = df['close'].diff() / df['volume']
        
        # 시간 특징
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
        features['is_month_end'] = df.index.is_month_end
        
        # 감성 지표
        features['news_sentiment'] = self.get_news_sentiment(df.index)
        features['social_sentiment'] = self.get_social_sentiment(df.index)
        
        return features
    
    def train_ensemble(self, X, y):
        """앙상블 모델 학습"""
        predictions = {}
        
        for name, model in self.models.items():
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5)
            print(f"{name} CV Score: {cv_scores.mean():.4f}")
            
            # 학습
            model.fit(X, y)
            predictions[name] = model.predict(X)
        
        # 메타 모델 (스태킹)
        meta_features = pd.DataFrame(predictions)
        self.meta_model = LinearRegression()
        self.meta_model.fit(meta_features, y)
        
        # SHAP 값으로 특징 중요도 분석
        explainer = shap.Explainer(self.models['xgboost'])
        shap_values = explainer(X)
        
        return shap_values`}</pre>
            </div>
          </div>

          <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-orange-800 dark:text-orange-200 mb-3">
              네트워크 분석
            </h3>
            
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-semibold mb-2">상관관계 네트워크</h4>
                <ul className="text-sm space-y-1">
                  <li>• 종목 간 연결성 분석</li>
                  <li>• 섹터 클러스터링</li>
                  <li>• 전염 효과 모델링</li>
                  <li>• 시스템 리스크 측정</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-semibold mb-2">공급망 네트워크</h4>
                <ul className="text-sm space-y-1">
                  <li>• 기업 간 거래 관계</li>
                  <li>• 공급망 취약성 분석</li>
                  <li>• 파급 효과 시뮬레이션</li>
                  <li>• ESG 리스크 전파</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">⚡ 실시간 대시보드 구축</h2>
        <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6">
          <h3 className="font-semibold mb-4">모니터링 시스템</h3>
          
          <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
            <pre className="text-sm">
{`# Dash/Plotly 실시간 대시보드
import dash
from dash import dcc, html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import redis

app = dash.Dash(__name__)
redis_client = redis.Redis()

app.layout = html.Div([
    html.H1('금융 빅데이터 실시간 모니터링'),
    
    dcc.Graph(id='market-heatmap'),
    dcc.Graph(id='sentiment-gauge'),
    dcc.Graph(id='volume-profile'),
    
    dcc.Interval(
        id='interval-component',
        interval=1000  # 1초마다 업데이트
    )
])

@app.callback(
    Output('market-heatmap', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_heatmap(n):
    # Redis에서 실시간 데이터 조회
    market_data = redis_client.get('market_snapshot')
    
    # 히트맵 생성
    fig = go.Figure(data=go.Heatmap(
        z=market_data['returns'],
        x=market_data['symbols'],
        y=market_data['sectors'],
        colorscale='RdYlGn'
    ))
    
    return fig`}</pre>
          </div>

          <div className="mt-4 grid md:grid-cols-3 gap-4">
            <div className="bg-white dark:bg-gray-700 rounded-lg p-3">
              <h4 className="font-semibold mb-2">시장 지표</h4>
              <ul className="text-sm space-y-1">
                <li>📊 섹터별 히트맵</li>
                <li>📈 상관관계 매트릭스</li>
                <li>🎯 이상 거래 탐지</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-700 rounded-lg p-3">
              <h4 className="font-semibold mb-2">감성 지표</h4>
              <ul className="text-sm space-y-1">
                <li>😊 실시간 감성 점수</li>
                <li>📰 뉴스 플로우</li>
                <li>🔥 트렌딩 키워드</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-700 rounded-lg p-3">
              <h4 className="font-semibold mb-2">리스크 지표</h4>
              <ul className="text-sm space-y-1">
                <li>⚠️ VaR 모니터링</li>
                <li>📉 드로우다운 추적</li>
                <li>🔔 임계값 알림</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎯 실전 프로젝트</h2>
        <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-lg p-6">
          <h3 className="font-semibold mb-4">종합 분석 프로젝트 예시</h3>
          
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">프로젝트: ESG 투자 전략 개발</h4>
              <ol className="space-y-2 text-sm">
                <li>
                  <strong>1. 데이터 수집</strong>
                  <ul className="pl-4 mt-1">
                    <li>• ESG 평가 데이터 (MSCI, Sustainalytics)</li>
                    <li>• 위성 이미지 (탄소 배출 모니터링)</li>
                    <li>• 뉴스/SNS (ESG 관련 이슈)</li>
                  </ul>
                </li>
                <li>
                  <strong>2. 특징 공학</strong>
                  <ul className="pl-4 mt-1">
                    <li>• ESG 점수 변화율</li>
                    <li>• 논란 이슈 빈도</li>
                    <li>• 동종업계 대비 순위</li>
                  </ul>
                </li>
                <li>
                  <strong>3. 예측 모델</strong>
                  <ul className="pl-4 mt-1">
                    <li>• ESG 개선 기업 예측</li>
                    <li>• 리스크 이벤트 조기 경보</li>
                    <li>• 수익률 영향 분석</li>
                  </ul>
                </li>
                <li>
                  <strong>4. 포트폴리오 구성</strong>
                  <ul className="pl-4 mt-1">
                    <li>• ESG 통합 최적화</li>
                    <li>• 리스크 조정 수익률</li>
                    <li>• 임팩트 측정</li>
                  </ul>
                </li>
              </ol>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">💡 빅데이터 분석 팁</h2>
        <div className="bg-blue-100 dark:bg-blue-900/30 rounded-lg p-6">
          <h3 className="font-semibold mb-4">성공적인 금융 빅데이터 프로젝트를 위한 조언</h3>
          
          <div className="space-y-3">
            <div className="flex items-start gap-3">
              <span className="text-blue-600 dark:text-blue-400">▶</span>
              <div>
                <strong>데이터 품질이 핵심</strong>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  아무리 좋은 모델도 나쁜 데이터로는 무용지물. 전처리에 70% 시간 투자
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-blue-600 dark:text-blue-400">▶</span>
              <div>
                <strong>단순한 모델부터 시작</strong>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  복잡한 딥러닝보다 해석 가능한 선형 모델이 더 유용할 수 있음
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-blue-600 dark:text-blue-400">▶</span>
              <div>
                <strong>실시간과 배치의 균형</strong>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  모든 것을 실시간으로 처리할 필요 없음. 용도에 맞게 선택
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-blue-600 dark:text-blue-400">▶</span>
              <div>
                <strong>비용 대비 효과 고려</strong>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  클라우드 비용이 수익을 초과하지 않도록 아키텍처 최적화
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}