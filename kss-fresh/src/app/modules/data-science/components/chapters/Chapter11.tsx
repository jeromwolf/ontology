'use client'

import React, { useState } from 'react'
import { TrendingUp, DollarSign, Users, Target, BarChart3, PieChart, LineChart, Lightbulb } from 'lucide-react'

interface Chapter11Props {
  onComplete?: () => void
}

export default function Chapter11({ onComplete }: Chapter11Props) {
  const [activeTab, setActiveTab] = useState('overview')

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-8">
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold text-blue-600 dark:text-blue-400">Chapter 11: 비즈니스 분석 (Business Analytics)</h1>
        <p className="text-xl text-gray-600 dark:text-gray-400">
          데이터를 활용하여 비즈니스 인사이트를 도출하고 의사결정을 지원하는 방법을 학습합니다
        </p>
      </div>

      <div className="border border-blue-200 dark:border-blue-800 rounded-lg">
        <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-t-lg">
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <TrendingUp className="w-6 h-6" />
            비즈니스 분석이란?
          </h2>
        </div>
        <div className="p-6 space-y-4">
          <div className="space-y-2">
            <h3 className="text-lg font-semibold">정의</h3>
            <p className="text-gray-600 dark:text-gray-400">
              비즈니스 분석은 데이터와 통계적 방법을 사용하여 과거 성과를 이해하고, 
              현재 상황을 파악하며, 미래 전략을 수립하는 프로세스입니다. 
              기술적 스킬과 비즈니스 도메인 지식을 결합하여 실행 가능한 인사이트를 제공합니다.
            </p>
          </div>
          
          <div className="grid md:grid-cols-3 gap-4">
            <div className="border rounded-lg">
              <div className="p-4 pb-3 border-b">
                <h3 className="text-base font-bold">기술적 분석</h3>
              </div>
              <div className="p-4">
                <p className="text-sm text-gray-600 dark:text-gray-400">무엇이 일어났는가?</p>
                <ul className="mt-2 space-y-1 list-disc list-inside text-xs">
                  <li>과거 데이터 요약</li>
                  <li>KPI 대시보드</li>
                  <li>보고서 작성</li>
                </ul>
              </div>
            </div>
            
            <div className="border rounded-lg">
              <div className="p-4 pb-3 border-b">
                <h3 className="text-base font-bold">진단적 분석</h3>
              </div>
              <div className="p-4">
                <p className="text-sm text-gray-600 dark:text-gray-400">왜 일어났는가?</p>
                <ul className="mt-2 space-y-1 list-disc list-inside text-xs">
                  <li>원인 분석</li>
                  <li>상관관계 파악</li>
                  <li>드릴다운 분석</li>
                </ul>
              </div>
            </div>
            
            <div className="border rounded-lg">
              <div className="p-4 pb-3 border-b">
                <h3 className="text-base font-bold">예측적 분석</h3>
              </div>
              <div className="p-4">
                <p className="text-sm text-gray-600 dark:text-gray-400">무엇이 일어날 것인가?</p>
                <ul className="mt-2 space-y-1 list-disc list-inside text-xs">
                  <li>수요 예측</li>
                  <li>이탈 예측</li>
                  <li>트렌드 분석</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="w-full">
        <div className="grid w-full grid-cols-5 bg-muted p-1 rounded-lg">
          <button
            onClick={() => setActiveTab('overview')}
            className={`px-4 py-2 rounded-md transition-colors ${
              activeTab === 'overview'
                ? 'bg-background shadow-sm font-medium'
                : 'hover:bg-background/50'
            }`}
          >
            개요
          </button>
          <button
            onClick={() => setActiveTab('kpi')}
            className={`px-4 py-2 rounded-md transition-colors ${
              activeTab === 'kpi'
                ? 'bg-background shadow-sm font-medium'
                : 'hover:bg-background/50'
            }`}
          >
            KPI 분석
          </button>
          <button
            onClick={() => setActiveTab('customer')}
            className={`px-4 py-2 rounded-md transition-colors ${
              activeTab === 'customer'
                ? 'bg-background shadow-sm font-medium'
                : 'hover:bg-background/50'
            }`}
          >
            고객 분석
          </button>
          <button
            onClick={() => setActiveTab('financial')}
            className={`px-4 py-2 rounded-md transition-colors ${
              activeTab === 'financial'
                ? 'bg-background shadow-sm font-medium'
                : 'hover:bg-background/50'
            }`}
          >
            재무 분석
          </button>
          <button
            onClick={() => setActiveTab('marketing')}
            className={`px-4 py-2 rounded-md transition-colors ${
              activeTab === 'marketing'
                ? 'bg-background shadow-sm font-medium'
                : 'hover:bg-background/50'
            }`}
          >
            마케팅 분석
          </button>
        </div>

        {activeTab === 'overview' && (
          <div className="space-y-4 mt-6">
            <div className="border rounded-lg">
              <div className="bg-secondary/5 p-6 rounded-t-lg">
                <h3 className="text-xl font-bold">비즈니스 분석 프레임워크</h3>
              </div>
              <div className="p-6 space-y-6">
                <div className="space-y-4">
                  <div className="border-l-4 border-primary pl-4">
                    <h4 className="font-semibold">1. 비즈니스 이해</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                      조직의 목표, 프로세스, 도전과제 파악
                    </p>
                    <div className="mt-3 space-y-2">
                      <div className="bg-muted p-3 rounded text-sm">
                        <strong>핵심 질문:</strong>
                        <ul className="mt-1 space-y-1 list-disc list-inside">
                          <li>우리의 비즈니스 목표는 무엇인가?</li>
                          <li>현재 직면한 주요 과제는 무엇인가?</li>
                          <li>어떤 의사결정이 필요한가?</li>
                        </ul>
                      </div>
                    </div>
                  </div>
                  
                  <div className="border-l-4 border-secondary pl-4">
                    <h4 className="font-semibold">2. 데이터 준비</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                      필요한 데이터 식별, 수집, 정제
                    </p>
                    <pre className="bg-muted p-3 rounded text-xs mt-2 overflow-x-auto">
{`# 데이터 소스 통합 예시
import pandas as pd

# 다양한 소스에서 데이터 로드
sales_data = pd.read_csv('sales.csv')
customer_data = pd.read_csv('customers.csv')
product_data = pd.read_csv('products.csv')

# 데이터 병합
merged_data = sales_data.merge(customer_data, on='customer_id')
merged_data = merged_data.merge(product_data, on='product_id')

# 데이터 품질 확인
print(f"결측값: {merged_data.isnull().sum()}")
print(f"중복 레코드: {merged_data.duplicated().sum()}")`}</pre>
                  </div>
                  
                  <div className="border-l-4 border-accent pl-4">
                    <h4 className="font-semibold">3. 분석 수행</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                      적절한 분석 기법을 사용하여 인사이트 도출
                    </p>
                  </div>
                  
                  <div className="border-l-4 border-muted pl-4">
                    <h4 className="font-semibold">4. 인사이트 전달</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                      시각화와 스토리텔링을 통한 효과적인 커뮤니케이션
                    </p>
                  </div>
                </div>
                
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                    <h4 className="font-semibold mb-2">분석 도구</h4>
                    <ul className="space-y-1 text-sm">
                      <li>• <strong>SQL:</strong> 데이터 추출 및 집계</li>
                      <li>• <strong>Python/R:</strong> 고급 분석 및 모델링</li>
                      <li>• <strong>Tableau/Power BI:</strong> 시각화</li>
                      <li>• <strong>Excel:</strong> 빠른 분석 및 프로토타이핑</li>
                    </ul>
                  </div>
                  <div className="p-4 bg-secondary/5 rounded-lg">
                    <h4 className="font-semibold mb-2">핵심 역량</h4>
                    <ul className="space-y-1 text-sm">
                      <li>• 비즈니스 도메인 지식</li>
                      <li>• 데이터 분석 기술</li>
                      <li>• 커뮤니케이션 능력</li>
                      <li>• 문제 해결 능력</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'kpi' && (
          <div className="space-y-4 mt-6">
            <div className="border rounded-lg">
              <div className="bg-accent/5 p-6 rounded-t-lg">
                <h3 className="text-xl font-bold flex items-center gap-2">
                  <BarChart3 className="w-6 h-6" />
                  KPI 대시보드 구축
                </h3>
              </div>
              <div className="p-6 space-y-6">
                <div className="p-4 bg-muted rounded-lg">
                  <h4 className="font-semibold mb-3">주요 성과 지표 (KPI) 설계</h4>
                  <pre className="bg-muted-foreground/10 p-3 rounded text-sm overflow-x-auto">
{`import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class KPIDashboard:
    def __init__(self, data):
        self.data = data
        
    def calculate_revenue_metrics(self):
        """매출 관련 KPI 계산"""
        metrics = {
            'total_revenue': self.data['revenue'].sum(),
            'avg_order_value': self.data['revenue'].mean(),
            'revenue_growth': self.calculate_growth_rate('revenue'),
            'revenue_by_segment': self.data.groupby('segment')['revenue'].sum()
        }
        return metrics
    
    def calculate_customer_metrics(self):
        """고객 관련 KPI 계산"""
        metrics = {
            'total_customers': self.data['customer_id'].nunique(),
            'new_customers': self.data[self.data['first_purchase'] == True].shape[0],
            'retention_rate': self.calculate_retention_rate(),
            'customer_lifetime_value': self.calculate_clv()
        }
        return metrics
    
    def calculate_operational_metrics(self):
        """운영 관련 KPI 계산"""
        metrics = {
            'conversion_rate': self.data['converted'].mean() * 100,
            'avg_processing_time': self.data['processing_time'].mean(),
            'fulfillment_rate': (self.data['fulfilled'] == True).mean() * 100,
            'return_rate': (self.data['returned'] == True).mean() * 100
        }
        return metrics
    
    def create_dashboard(self):
        """대시보드 생성"""
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('매출 추이', '고객 세그먼트', '전환율',
                          '고객 획득', '제품 성과', '지역별 매출'),
            specs=[[{'type': 'scatter'}, {'type': 'pie'}, {'type': 'indicator'}],
                   [{'type': 'bar'}, {'type': 'scatter'}, {'type': 'geo'}]]
        )
        
        # 매출 추이
        fig.add_trace(
            go.Scatter(x=self.data['date'], y=self.data['revenue'],
                      mode='lines+markers', name='매출'),
            row=1, col=1
        )
        
        # 고객 세그먼트
        segment_data = self.data.groupby('segment').size()
        fig.add_trace(
            go.Pie(labels=segment_data.index, values=segment_data.values),
            row=1, col=2
        )
        
        # 전환율 지표
        conversion_rate = self.data['converted'].mean() * 100
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=conversion_rate,
                title={'text': "전환율 (%)"},
                delta={'reference': 3.5},
                gauge={'axis': {'range': [None, 10]},
                       'bar': {'color': "darkblue"},
                       'steps': [
                           {'range': [0, 2.5], 'color': "lightgray"},
                           {'range': [2.5, 5], 'color': "gray"}],
                       'threshold': {
                           'line': {'color': "red", 'width': 4},
                           'thickness': 0.75,
                           'value': 4.5}}
            ),
            row=1, col=3
        )
        
        return fig`}</pre>
                </div>

                <div className="grid md:grid-cols-3 gap-4">
                  <div className="border rounded-lg">
                    <div className="p-4 border-b">
                      <h4 className="text-sm font-bold flex items-center gap-2">
                        <DollarSign className="w-4 h-4" />
                        재무 KPI
                      </h4>
                    </div>
                    <div className="p-4 space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span>매출 성장률</span>
                        <span className="font-mono text-green-600">+23.5%</span>
                      </div>
                      <div className="flex justify-between">
                        <span>영업이익률</span>
                        <span className="font-mono">18.2%</span>
                      </div>
                      <div className="flex justify-between">
                        <span>ROI</span>
                        <span className="font-mono">156%</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="border rounded-lg">
                    <div className="p-4 border-b">
                      <h4 className="text-sm font-bold flex items-center gap-2">
                        <Users className="w-4 h-4" />
                        고객 KPI
                      </h4>
                    </div>
                    <div className="p-4 space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span>고객 만족도</span>
                        <span className="font-mono">4.6/5.0</span>
                      </div>
                      <div className="flex justify-between">
                        <span>이탈률</span>
                        <span className="font-mono text-red-600">2.3%</span>
                      </div>
                      <div className="flex justify-between">
                        <span>NPS</span>
                        <span className="font-mono">+42</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="border rounded-lg">
                    <div className="p-4 border-b">
                      <h4 className="text-sm font-bold flex items-center gap-2">
                        <Target className="w-4 h-4" />
                        운영 KPI
                      </h4>
                    </div>
                    <div className="p-4 space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span>처리 시간</span>
                        <span className="font-mono">2.4일</span>
                      </div>
                      <div className="flex justify-between">
                        <span>정시 배송률</span>
                        <span className="font-mono">96.8%</span>
                      </div>
                      <div className="flex justify-between">
                        <span>재고 회전율</span>
                        <span className="font-mono">8.2회</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'customer' && (
          <div className="space-y-4 mt-6">
            <div className="border rounded-lg">
              <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-t-lg">
                <h3 className="text-xl font-bold flex items-center gap-2">
                  <Users className="w-6 h-6" />
                  고객 분석
                </h3>
              </div>
              <div className="p-6 space-y-6">
                <div className="space-y-4">
                  <div>
                    <h4 className="font-semibold mb-2">1. RFM 분석 (Recency, Frequency, Monetary)</h4>
                    <pre className="bg-muted p-4 rounded-lg text-sm overflow-x-auto">
{`import pandas as pd
import numpy as np
from datetime import datetime

def rfm_analysis(df):
    """RFM 분석을 통한 고객 세그먼테이션"""
    
    # 현재 날짜
    today = pd.to_datetime('today')
    
    # RFM 메트릭 계산
    rfm = df.groupby('customer_id').agg({
        'order_date': lambda x: (today - x.max()).days,  # Recency
        'order_id': 'count',                              # Frequency
        'revenue': 'sum'                                  # Monetary
    })
    
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    
    # RFM 점수 계산 (1-5 척도)
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])
    
    # RFM 점수 결합
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + \
                       rfm['F_Score'].astype(str) + \
                       rfm['M_Score'].astype(str)
    
    # 고객 세그먼트 정의
    def segment_customers(row):
        if row['RFM_Score'] in ['555', '554', '544', '545', '454', '455', '445']:
            return '챔피언'
        elif row['RFM_Score'] in ['543', '444', '435', '355', '354', '345', '344', '335']:
            return '충성 고객'
        elif row['RFM_Score'] in ['553', '551', '552', '541', '542', '533', '532', '531', '452', '451']:
            return '잠재 충성고객'
        elif row['RFM_Score'] in ['512', '511', '422', '421', '412', '411', '311']:
            return '신규 고객'
        elif row['RFM_Score'] in ['525', '524', '523', '522', '521', '515', '514', '513', '425', '424', '413', '414', '415', '315', '314', '313']:
            return '유망 고객'
        elif row['RFM_Score'] in ['535', '534', '443', '434', '343', '334', '325', '324']:
            return '관심 필요'
        elif row['RFM_Score'] in ['331', '321', '312', '221', '213', '231', '241', '251']:
            return '잠든 고객'
        elif row['RFM_Score'] in ['155', '154', '144', '214', '215', '115', '114']:
            return '이탈 위험'
        elif row['RFM_Score'] in ['255', '254', '245', '244', '253', '252', '243', '242', '235', '234', '225', '224', '153', '152', '145']:
            return '이탈 고객'
        elif row['RFM_Score'] in ['332', '322', '231', '241', '251', '233', '232', '223', '222', '132', '123', '122', '212', '211']:
            return '동면 고객'
        else:
            return '기타'
    
    rfm['Segment'] = rfm.apply(segment_customers, axis=1)
    
    return rfm

# 세그먼트별 전략
segment_strategies = {
    '챔피언': '특별 보상 프로그램, VIP 혜택 제공',
    '충성 고객': '상위 티어 프로모션, 추천 프로그램',
    '잠재 충성고객': '멤버십 프로그램, 브랜드 참여 유도',
    '신규 고객': '온보딩 프로그램, 환영 혜택',
    '유망 고객': '카테고리 확장 제안, 크로스셀링',
    '관심 필요': '개인화된 오퍼, 재참여 캠페인',
    '이탈 위험': '긴급 리텐션 캠페인, 특별 할인',
    '이탈 고객': '윈백 캠페인, 피드백 수집'
}`}</pre>
                  </div>

                  <div>
                    <h4 className="font-semibold mb-2">2. 고객 생애 가치 (CLV) 예측</h4>
                    <pre className="bg-muted p-4 rounded-lg text-sm overflow-x-auto">
{`from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def predict_clv(customer_data):
    """머신러닝을 활용한 고객 생애 가치 예측"""
    
    # 특성 엔지니어링
    features = customer_data[[
        'days_since_first_purchase',
        'total_orders',
        'avg_order_value',
        'days_between_orders',
        'product_categories_purchased',
        'preferred_payment_method',
        'customer_service_contacts',
        'return_rate'
    ]]
    
    # 타겟: 향후 12개월 예상 수익
    target = customer_data['next_12_months_revenue']
    
    # 모델 학습
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 예측 및 평가
    predictions = model.predict(X_test)
    
    # 특성 중요도
    feature_importance = pd.DataFrame({
        'feature': features.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, feature_importance`}</pre>
                  </div>

                  <div className="grid md:grid-cols-2 gap-4">
                    <div className="p-4 bg-accent/5 rounded-lg">
                      <h4 className="font-semibold mb-2">코호트 분석</h4>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                        시간에 따른 고객 그룹의 행동 패턴 추적
                      </p>
                      <ul className="text-sm space-y-1">
                        <li>• 월별 신규 가입자 리텐션</li>
                        <li>• 첫 구매 후 재구매율</li>
                        <li>• 시즌별 고객 행동 변화</li>
                      </ul>
                    </div>
                    <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                      <h4 className="font-semibold mb-2">고객 여정 분석</h4>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                        터치포인트별 고객 경험 최적화
                      </p>
                      <ul className="text-sm space-y-1">
                        <li>• 채널별 전환 경로</li>
                        <li>• 이탈 포인트 식별</li>
                        <li>• 구매 결정 요인 분석</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'financial' && (
          <div className="space-y-4 mt-6">
            <div className="border rounded-lg">
              <div className="bg-secondary/5 p-6 rounded-t-lg">
                <h3 className="text-xl font-bold flex items-center gap-2">
                  <DollarSign className="w-6 h-6" />
                  재무 분석
                </h3>
              </div>
              <div className="p-6 space-y-6">
                <div className="space-y-4">
                  <div>
                    <h4 className="font-semibold mb-2">1. 수익성 분석</h4>
                    <pre className="bg-muted p-4 rounded-lg text-sm overflow-x-auto">
{`import pandas as pd
import numpy as np

class FinancialAnalyzer:
    def __init__(self, financial_data):
        self.data = financial_data
        
    def profitability_analysis(self):
        """수익성 지표 계산"""
        metrics = {}
        
        # 매출총이익률 (Gross Profit Margin)
        metrics['gross_margin'] = (
            (self.data['revenue'] - self.data['cogs']) / 
            self.data['revenue']
        ).mean() * 100
        
        # 영업이익률 (Operating Margin)
        metrics['operating_margin'] = (
            self.data['operating_income'] / 
            self.data['revenue']
        ).mean() * 100
        
        # 순이익률 (Net Profit Margin)
        metrics['net_margin'] = (
            self.data['net_income'] / 
            self.data['revenue']
        ).mean() * 100
        
        # EBITDA Margin
        metrics['ebitda_margin'] = (
            self.data['ebitda'] / 
            self.data['revenue']
        ).mean() * 100
        
        return metrics
    
    def break_even_analysis(self, fixed_costs, variable_cost_ratio, price):
        """손익분기점 분석"""
        # 손익분기점 수량
        break_even_units = fixed_costs / (price - (price * variable_cost_ratio))
        
        # 손익분기점 매출
        break_even_revenue = break_even_units * price
        
        # 안전마진
        current_revenue = self.data['revenue'].sum()
        safety_margin = ((current_revenue - break_even_revenue) / 
                        current_revenue) * 100
        
        return {
            'break_even_units': break_even_units,
            'break_even_revenue': break_even_revenue,
            'safety_margin': safety_margin
        }
    
    def variance_analysis(self, budget_data, actual_data):
        """예산 대비 실적 분석"""
        variance = pd.DataFrame({
            'Budget': budget_data,
            'Actual': actual_data,
            'Variance': actual_data - budget_data,
            'Variance%': ((actual_data - budget_data) / budget_data) * 100
        })
        
        # 주요 차이 항목 식별
        significant_variances = variance[
            abs(variance['Variance%']) > 10
        ].sort_values('Variance%', ascending=False)
        
        return variance, significant_variances`}</pre>
                  </div>

                  <div>
                    <h4 className="font-semibold mb-2">2. 현금흐름 예측</h4>
                    <pre className="bg-muted p-4 rounded-lg text-sm overflow-x-auto">
{`from statsmodels.tsa.holtwinters import ExponentialSmoothing

def forecast_cash_flow(historical_cash_flow, periods=12):
    """시계열 분석을 통한 현금흐름 예측"""
    
    # Holt-Winters 모델 적용
    model = ExponentialSmoothing(
        historical_cash_flow,
        seasonal_periods=12,
        trend='add',
        seasonal='add'
    )
    
    fitted_model = model.fit()
    
    # 예측
    forecast = fitted_model.forecast(periods)
    
    # 신뢰구간 계산
    residuals = fitted_model.resid
    std_error = np.std(residuals)
    confidence_interval = 1.96 * std_error
    
    forecast_df = pd.DataFrame({
        'forecast': forecast,
        'lower_bound': forecast - confidence_interval,
        'upper_bound': forecast + confidence_interval
    })
    
    # 현금 부족 위험 시점 식별
    cash_shortage_risk = forecast_df[forecast_df['lower_bound'] < 0]
    
    return forecast_df, cash_shortage_risk`}</pre>
                  </div>

                  <div className="grid md:grid-cols-3 gap-4">
                    <div className="border rounded-lg">
                      <div className="p-4 border-b">
                        <h4 className="text-sm font-bold">수익성 지표</h4>
                      </div>
                      <div className="p-4 space-y-2 text-xs">
                        <div className="flex justify-between">
                          <span>ROA</span>
                          <span className="font-mono">12.5%</span>
                        </div>
                        <div className="flex justify-between">
                          <span>ROE</span>
                          <span className="font-mono">18.3%</span>
                        </div>
                        <div className="flex justify-between">
                          <span>ROCE</span>
                          <span className="font-mono">15.7%</span>
                        </div>
                      </div>
                    </div>
                    
                    <div className="border rounded-lg">
                      <div className="p-4 border-b">
                        <h4 className="text-sm font-bold">유동성 지표</h4>
                      </div>
                      <div className="p-4 space-y-2 text-xs">
                        <div className="flex justify-between">
                          <span>유동비율</span>
                          <span className="font-mono">2.1</span>
                        </div>
                        <div className="flex justify-between">
                          <span>당좌비율</span>
                          <span className="font-mono">1.5</span>
                        </div>
                        <div className="flex justify-between">
                          <span>현금비율</span>
                          <span className="font-mono">0.8</span>
                        </div>
                      </div>
                    </div>
                    
                    <div className="border rounded-lg">
                      <div className="p-4 border-b">
                        <h4 className="text-sm font-bold">효율성 지표</h4>
                      </div>
                      <div className="p-4 space-y-2 text-xs">
                        <div className="flex justify-between">
                          <span>총자산회전율</span>
                          <span className="font-mono">1.8회</span>
                        </div>
                        <div className="flex justify-between">
                          <span>매출채권회전율</span>
                          <span className="font-mono">12.5회</span>
                        </div>
                        <div className="flex justify-between">
                          <span>재고자산회전율</span>
                          <span className="font-mono">8.2회</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'marketing' && (
          <div className="space-y-4 mt-6">
            <div className="border rounded-lg">
              <div className="bg-accent/5 p-6 rounded-t-lg">
                <h3 className="text-xl font-bold flex items-center gap-2">
                  <PieChart className="w-6 h-6" />
                  마케팅 분석
                </h3>
              </div>
              <div className="p-6 space-y-6">
                <div className="space-y-4">
                  <div>
                    <h4 className="font-semibold mb-2">1. 캠페인 성과 분석</h4>
                    <pre className="bg-muted p-4 rounded-lg text-sm overflow-x-auto">
{`class MarketingAnalytics:
    def __init__(self, campaign_data):
        self.data = campaign_data
        
    def campaign_roi_analysis(self):
        """캠페인 ROI 분석"""
        roi_metrics = self.data.groupby('campaign_name').agg({
            'cost': 'sum',
            'revenue': 'sum',
            'conversions': 'sum',
            'impressions': 'sum',
            'clicks': 'sum'
        })
        
        # ROI 계산
        roi_metrics['ROI'] = (
            (roi_metrics['revenue'] - roi_metrics['cost']) / 
            roi_metrics['cost']
        ) * 100
        
        # CPA (Cost Per Acquisition)
        roi_metrics['CPA'] = roi_metrics['cost'] / roi_metrics['conversions']
        
        # CTR (Click-Through Rate)
        roi_metrics['CTR'] = (roi_metrics['clicks'] / 
                             roi_metrics['impressions']) * 100
        
        # CVR (Conversion Rate)
        roi_metrics['CVR'] = (roi_metrics['conversions'] / 
                             roi_metrics['clicks']) * 100
        
        return roi_metrics.sort_values('ROI', ascending=False)
    
    def attribution_modeling(self):
        """멀티터치 어트리뷰션 분석"""
        # 고객 여정 데이터
        journey_data = self.data.groupby('customer_id')['touchpoint'].apply(list)
        
        # 어트리뷰션 모델
        attribution_models = {
            'first_touch': self.first_touch_attribution,
            'last_touch': self.last_touch_attribution,
            'linear': self.linear_attribution,
            'time_decay': self.time_decay_attribution,
            'data_driven': self.data_driven_attribution
        }
        
        results = {}
        for model_name, model_func in attribution_models.items():
            results[model_name] = model_func(journey_data)
            
        return results
    
    def market_basket_analysis(self, transaction_data):
        """장바구니 분석"""
        from mlxtend.frequent_patterns import apriori, association_rules
        
        # 트랜잭션 데이터를 원-핫 인코딩
        basket = transaction_data.groupby(['transaction_id', 'product'])['quantity'].sum().unstack().fillna(0)
        basket = basket.applymap(lambda x: 1 if x > 0 else 0)
        
        # 빈발 아이템셋 찾기
        frequent_items = apriori(basket, min_support=0.01, use_colnames=True)
        
        # 연관 규칙 생성
        rules = association_rules(frequent_items, metric='lift', min_threshold=1.5)
        
        # 주요 규칙 필터링
        significant_rules = rules[
            (rules['confidence'] > 0.5) & 
            (rules['lift'] > 2)
        ].sort_values('lift', ascending=False)
        
        return significant_rules`}</pre>
                  </div>

                  <div className="grid md:grid-cols-2 gap-4">
                    <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                      <h4 className="font-semibold mb-3">A/B 테스트 분석</h4>
                      <pre className="bg-muted p-3 rounded text-xs overflow-x-auto">
{`from scipy import stats

def ab_test_analysis(control, variant):
    """A/B 테스트 통계적 유의성 검정"""
    
    # 전환율 계산
    control_cvr = control['converted'].mean()
    variant_cvr = variant['converted'].mean()
    
    # 상대적 개선도
    lift = ((variant_cvr - control_cvr) / 
            control_cvr) * 100
    
    # 통계적 검정 (Chi-square test)
    contingency_table = pd.crosstab(
        index=[control['converted'], 
               variant['converted']], 
        columns=['control', 'variant']
    )
    
    chi2, p_value, dof, expected = stats.chi2_contingency(
        contingency_table
    )
    
    # 신뢰구간 계산
    se = np.sqrt(
        variant_cvr * (1 - variant_cvr) / len(variant) +
        control_cvr * (1 - control_cvr) / len(control)
    )
    
    confidence_interval = 1.96 * se
    
    return {
        'control_cvr': control_cvr,
        'variant_cvr': variant_cvr,
        'lift': lift,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'confidence_interval': confidence_interval
    }`}</pre>
                    </div>
                    
                    <div className="p-4 bg-secondary/5 rounded-lg">
                      <h4 className="font-semibold mb-3">고객 획득 비용 (CAC) 최적화</h4>
                      <div className="space-y-3">
                        <div className="text-sm">
                          <strong>채널별 CAC:</strong>
                          <ul className="mt-1 space-y-1 list-disc list-inside">
                            <li>검색 광고: $45</li>
                            <li>소셜 미디어: $32</li>
                            <li>이메일 마케팅: $12</li>
                            <li>추천 프로그램: $28</li>
                          </ul>
                        </div>
                        <div className="text-sm">
                          <strong>LTV/CAC 비율:</strong>
                          <div className="mt-1 text-2xl font-bold text-green-600">3.2:1</div>
                          <p className="text-xs text-gray-600 dark:text-gray-400">
                            (목표: 3:1 이상)
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="border border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <Lightbulb className="h-4 w-4 mt-1 flex-shrink-0" />
          <div className="space-y-1">
            <strong>비즈니스 분석 베스트 프랙티스:</strong>
            <ul className="mt-2 space-y-1 list-disc list-inside">
              <li>비즈니스 목표와 분석 목적을 명확히 정의</li>
              <li>데이터 품질과 정확성을 지속적으로 검증</li>
              <li>기술적 인사이트를 비즈니스 언어로 번역</li>
              <li>실행 가능한 권고사항 제시</li>
              <li>지속적인 모니터링과 피드백 루프 구축</li>
              <li>이해관계자와 적극적으로 소통</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="border-2 border-blue-200 dark:border-blue-800 rounded-lg">
        <div className="p-6 border-b">
          <h3 className="text-xl font-bold">실습 프로젝트: 종합 비즈니스 대시보드</h3>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
            실제 비즈니스 데이터를 활용한 종합 분석 대시보드를 구축해봅시다
          </p>
        </div>
        <div className="p-6 space-y-4">
          <div className="space-y-3">
            <h4 className="font-semibold">프로젝트 요구사항:</h4>
            <ol className="space-y-2 list-decimal list-inside">
              <li>매출, 고객, 제품 데이터 통합</li>
              <li>핵심 KPI 대시보드 구축</li>
              <li>고객 세그먼테이션 및 CLV 분석</li>
              <li>매출 예측 모델 개발</li>
              <li>마케팅 캠페인 ROI 분석</li>
              <li>경영진 보고서 자동화</li>
            </ol>
          </div>
          
          <div className="bg-muted p-4 rounded-lg">
            <h5 className="font-semibold text-sm mb-2">기대 효과:</h5>
            <ul className="text-sm space-y-1 list-disc list-inside">
              <li>데이터 기반 의사결정 문화 정착</li>
              <li>매출 예측 정확도 85% 이상</li>
              <li>마케팅 ROI 30% 개선</li>
              <li>고객 이탈률 20% 감소</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="flex justify-between items-center pt-8">
        <p className="text-sm text-gray-600 dark:text-gray-400">
          데이터는 새로운 석유입니다 - 비즈니스 가치로 전환하세요
        </p>
        {onComplete && (
          <button
            onClick={onComplete}
            className="px-4 py-2 bg-primary text-blue-600 dark:text-blue-400-foreground rounded-lg hover:bg-primary/90 transition-colors"
          >
            다음 챕터로
          </button>
        )}
      </div>
    </div>
  )
}