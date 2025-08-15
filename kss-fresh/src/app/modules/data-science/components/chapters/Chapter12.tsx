'use client';

import React, { useState } from 'react';
import { BookOpen, Building2, ShoppingCart, Heart, Smartphone, TrendingUp, Users, Lightbulb } from 'lucide-react';

interface Chapter12Props {
  onComplete?: () => void
}

export default function Chapter12({ onComplete }: Chapter12Props) {
  const [activeCase, setActiveCase] = useState('netflix')

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-8">
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold text-blue-600 dark:text-blue-400">Chapter 12: 실전 사례 연구</h1>
        <p className="text-xl text-gray-600 dark:text-gray-400">
          실제 기업들의 데이터 사이언스 성공 사례를 통해 실무 적용 방법을 학습합니다
        </p>
      </div>

      <div className="border border-primary/20 rounded-lg">
        <div className="bg-primary/5 p-6 rounded-t-lg">
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <BookOpen className="w-6 h-6" />
            사례 연구의 중요성
          </h2>
        </div>
        <div className="p-6 space-y-4">
          <p className="text-gray-600 dark:text-gray-400">
            사례 연구를 통해 이론과 실무의 격차를 줄이고, 실제 비즈니스 환경에서 
            데이터 사이언스를 어떻게 적용하는지 학습할 수 있습니다. 각 사례는 
            문제 정의부터 구현, 결과 평가까지 전체 프로세스를 다룹니다.
          </p>
          
          <div className="grid md:grid-cols-3 gap-4">
            <div className="border rounded-lg">
              <div className="p-4 pb-3 border-b">
                <h3 className="text-sm font-bold">학습 포인트</h3>
              </div>
              <div className="p-4">
                <ul className="space-y-1 list-disc list-inside text-xs">
                  <li>실제 데이터의 복잡성</li>
                  <li>비즈니스 제약 조건</li>
                  <li>팀 협업 방식</li>
                  <li>성과 측정 방법</li>
                </ul>
              </div>
            </div>
            
            <div className="border rounded-lg">
              <div className="p-4 pb-3 border-b">
                <h3 className="text-sm font-bold">산업별 적용</h3>
              </div>
              <div className="p-4">
                <ul className="space-y-1 list-disc list-inside text-xs">
                  <li>엔터테인먼트</li>
                  <li>이커머스</li>
                  <li>헬스케어</li>
                  <li>금융</li>
                </ul>
              </div>
            </div>
            
            <div className="border rounded-lg">
              <div className="p-4 pb-3 border-b">
                <h3 className="text-sm font-bold">핵심 기술</h3>
              </div>
              <div className="p-4">
                <ul className="space-y-1 list-disc list-inside text-xs">
                  <li>추천 시스템</li>
                  <li>예측 모델링</li>
                  <li>최적화</li>
                  <li>A/B 테스팅</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="w-full">
        <div className="flex gap-2 p-1 bg-muted rounded-lg overflow-x-auto">
          <button
            onClick={() => setActiveCase('netflix')}
            className={`px-4 py-2 rounded-md transition-colors whitespace-nowrap ${
              activeCase === 'netflix'
                ? 'bg-background shadow-sm font-medium'
                : 'hover:bg-background/50'
            }`}
          >
            Netflix 추천 시스템
          </button>
          <button
            onClick={() => setActiveCase('amazon')}
            className={`px-4 py-2 rounded-md transition-colors whitespace-nowrap ${
              activeCase === 'amazon'
                ? 'bg-background shadow-sm font-medium'
                : 'hover:bg-background/50'
            }`}
          >
            Amazon 수요 예측
          </button>
          <button
            onClick={() => setActiveCase('kakaobank')}
            className={`px-4 py-2 rounded-md transition-colors whitespace-nowrap ${
              activeCase === 'kakaobank'
                ? 'bg-background shadow-sm font-medium'
                : 'hover:bg-background/50'
            }`}
          >
            카카오뱅크 신용평가
          </button>
          <button
            onClick={() => setActiveCase('coupang')}
            className={`px-4 py-2 rounded-md transition-colors whitespace-nowrap ${
              activeCase === 'coupang'
                ? 'bg-background shadow-sm font-medium'
                : 'hover:bg-background/50'
            }`}
          >
            쿠팡 물류 최적화
          </button>
        </div>

        {activeCase === 'netflix' && (
          <div className="space-y-4 mt-6">
            <div className="border rounded-lg">
              <div className="bg-secondary/5 p-6 rounded-t-lg">
                <h3 className="text-xl font-bold flex items-center gap-2">
                  <Building2 className="w-6 h-6" />
                  Netflix 개인화 추천 시스템
                </h3>
              </div>
              <div className="p-6 space-y-6">
                <div className="space-y-4">
                  <div className="p-4 bg-destructive/5 rounded-lg">
                    <h4 className="font-semibold mb-2">비즈니스 과제</h4>
                    <ul className="space-y-2 text-sm">
                      <li>• 1억 5천만+ 사용자의 다양한 취향 충족</li>
                      <li>• 콘텐츠 시청 시간 증가 및 이탈 방지</li>
                      <li>• 신규 콘텐츠의 효과적인 노출</li>
                      <li>• 글로벌 다양성 고려한 추천</li>
                    </ul>
                  </div>

                  <div>
                    <h4 className="font-semibold mb-2">기술적 접근</h4>
                    <pre className="bg-muted p-4 rounded-lg text-sm overflow-x-auto">
{`# Netflix 추천 시스템 아키텍처 예시
class NetflixRecommender:
    def __init__(self):
        self.models = {
            'collaborative': CollaborativeFilteringModel(),
            'content_based': ContentBasedModel(),
            'deep_learning': DeepLearningModel(),
            'contextual': ContextualBanditModel()
        }
        
    def get_recommendations(self, user_id, context):
        """다중 모델 앙상블 추천"""
        
        # 1. 협업 필터링: 유사 사용자의 시청 패턴
        collab_recs = self.models['collaborative'].predict(
            user_id, 
            num_recommendations=100
        )
        
        # 2. 콘텐츠 기반: 장르, 배우, 감독 등 메타데이터
        user_profile = self.get_user_profile(user_id)
        content_recs = self.models['content_based'].predict(
            user_profile,
            available_content=self.get_available_content()
        )
        
        # 3. 딥러닝: 시청 시퀀스 및 임베딩
        watch_history = self.get_watch_history(user_id)
        dl_recs = self.models['deep_learning'].predict(
            watch_history,
            context_features=context
        )
        
        # 4. 맥락적 밴딧: 실시간 A/B 테스팅
        bandit_recs = self.models['contextual'].select_arms(
            user_features=user_profile,
            context=context
        )
        
        # 앙상블 및 다양성 보장
        final_recs = self.ensemble_and_diversify(
            [collab_recs, content_recs, dl_recs, bandit_recs],
            diversity_weight=0.3
        )
        
        return final_recs
    
    def personalize_homepage(self, user_id):
        """개인화된 홈페이지 구성"""
        rows = []
        
        # 각 행별로 다른 알고리즘/테마 적용
        rows.append({
            'title': 'Top Picks for You',
            'items': self.get_recommendations(user_id, {'type': 'top_picks'})
        })
        
        rows.append({
            'title': 'Because You Watched ' + self.get_last_watched(user_id),
            'items': self.get_similar_content(self.get_last_watched_id(user_id))
        })
        
        rows.append({
            'title': 'Trending Now',
            'items': self.get_trending_with_personalization(user_id)
        })
        
        return rows`}</pre>
                  </div>

                  <div className="grid md:grid-cols-2 gap-4">
                    <div className="p-4 bg-primary/5 rounded-lg">
                      <h4 className="font-semibold mb-2">핵심 메트릭</h4>
                      <ul className="space-y-2 text-sm">
                        <li className="flex justify-between">
                          <span>시청 완료율</span>
                          <span className="font-mono">+35%</span>
                        </li>
                        <li className="flex justify-between">
                          <span>평균 시청 시간</span>
                          <span className="font-mono">+23%</span>
                        </li>
                        <li className="flex justify-between">
                          <span>구독 유지율</span>
                          <span className="font-mono">93%</span>
                        </li>
                        <li className="flex justify-between">
                          <span>추천 클릭률</span>
                          <span className="font-mono">80%</span>
                        </li>
                      </ul>
                    </div>
                    <div className="p-4 bg-secondary/5 rounded-lg">
                      <h4 className="font-semibold mb-2">기술 스택</h4>
                      <ul className="space-y-1 text-sm">
                        <li>• Apache Spark (데이터 처리)</li>
                        <li>• TensorFlow (딥러닝)</li>
                        <li>• AWS (인프라)</li>
                        <li>• Cassandra (데이터베이스)</li>
                        <li>• A/B Testing Platform</li>
                      </ul>
                    </div>
                  </div>

                  <div className="border border-primary/20 bg-primary/5 rounded-lg p-4">
                    <div className="flex items-start gap-3">
                      <Lightbulb className="h-4 w-4 mt-1 flex-shrink-0" />
                      <div className="space-y-1">
                        <strong>핵심 인사이트:</strong>
                        <ul className="mt-2 space-y-1 list-disc list-inside text-sm">
                          <li>단일 알고리즘보다 다중 모델 앙상블이 효과적</li>
                          <li>실시간 컨텍스트(시간, 디바이스 등) 고려 중요</li>
                          <li>추천의 다양성과 정확도 사이 균형 필요</li>
                          <li>지속적인 A/B 테스팅으로 개선</li>
                        </ul>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeCase === 'amazon' && (
          <div className="space-y-4 mt-6">
            <div className="border rounded-lg">
              <div className="bg-accent/5 p-6 rounded-t-lg">
                <h3 className="text-xl font-bold flex items-center gap-2">
                  <ShoppingCart className="w-6 h-6" />
                  Amazon 수요 예측 시스템
                </h3>
              </div>
              <div className="p-6 space-y-6">
                <div className="space-y-4">
                  <div className="p-4 bg-destructive/5 rounded-lg">
                    <h4 className="font-semibold mb-2">비즈니스 과제</h4>
                    <ul className="space-y-2 text-sm">
                      <li>• 수백만 개 상품의 재고 최적화</li>
                      <li>• 계절성 및 트렌드 변화 대응</li>
                      <li>• 글로벌 공급망 효율화</li>
                      <li>• 배송 시간 단축 및 비용 절감</li>
                    </ul>
                  </div>

                  <div>
                    <h4 className="font-semibold mb-2">예측 모델 구현</h4>
                    <pre className="bg-muted p-4 rounded-lg text-sm overflow-x-auto">
{`import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor

class AmazonDemandForecaster:
    def __init__(self):
        self.models = {}
        self.feature_engineering = FeatureEngineering()
        
    def forecast_demand(self, product_id, horizon=30):
        """상품별 수요 예측"""
        
        # 1. 시계열 데이터 준비
        historical_data = self.get_product_history(product_id)
        
        # 2. 특성 엔지니어링
        features = self.feature_engineering.create_features(
            historical_data,
            include_seasonality=True,
            include_promotions=True,
            include_competitor_prices=True,
            include_weather=True
        )
        
        # 3. 모델별 예측
        predictions = {}
        
        # Prophet (시계열 예측)
        prophet_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            holidays=self.get_holidays()
        )
        prophet_model.fit(historical_data[['ds', 'y']])
        future = prophet_model.make_future_dataframe(periods=horizon)
        predictions['prophet'] = prophet_model.predict(future)
        
        # Random Forest (다변량 예측)
        rf_model = RandomForestRegressor(n_estimators=100)
        X_train, y_train = features[:-horizon], historical_data['y'][:-horizon]
        rf_model.fit(X_train, y_train)
        predictions['rf'] = rf_model.predict(features[-horizon:])
        
        # DeepAR (딥러닝 시계열)
        deepar_predictions = self.deepar_forecast(
            historical_data, 
            features,
            horizon
        )
        predictions['deepar'] = deepar_predictions
        
        # 4. 앙상블 및 불확실성 추정
        ensemble_forecast = self.weighted_ensemble(
            predictions,
            weights={'prophet': 0.3, 'rf': 0.3, 'deepar': 0.4}
        )
        
        # 5. 재고 최적화 권장사항
        recommendations = self.optimize_inventory(
            ensemble_forecast,
            service_level=0.95,
            holding_cost=self.get_holding_cost(product_id),
            stockout_cost=self.get_stockout_cost(product_id)
        )
        
        return {
            'forecast': ensemble_forecast,
            'confidence_interval': self.calculate_confidence_interval(predictions),
            'recommendations': recommendations
        }
    
    def optimize_inventory(self, forecast, service_level, holding_cost, stockout_cost):
        """안전재고 및 주문량 최적화"""
        
        # 수요 분포 추정
        demand_mean = np.mean(forecast)
        demand_std = np.std(forecast)
        
        # 안전재고 계산 (정규분포 가정)
        from scipy.stats import norm
        z_score = norm.ppf(service_level)
        safety_stock = z_score * demand_std * np.sqrt(self.lead_time)
        
        # 경제적 주문량 (EOQ) 계산
        annual_demand = demand_mean * 365
        ordering_cost = self.ordering_cost
        eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
        
        # 재주문점 계산
        reorder_point = demand_mean * self.lead_time + safety_stock
        
        return {
            'safety_stock': safety_stock,
            'economic_order_quantity': eoq,
            'reorder_point': reorder_point,
            'expected_stockouts': self.calculate_expected_stockouts(
                demand_mean, demand_std, safety_stock
            )
        }`}</pre>
                  </div>

                  <div className="grid md:grid-cols-3 gap-4">
                    <div className="p-4 bg-primary/5 rounded-lg">
                      <h4 className="font-semibold mb-2 text-sm">예측 정확도</h4>
                      <ul className="space-y-1 text-xs">
                        <li className="flex justify-between">
                          <span>MAPE</span>
                          <span className="font-mono">8.3%</span>
                        </li>
                        <li className="flex justify-between">
                          <span>RMSE</span>
                          <span className="font-mono">124.5</span>
                        </li>
                        <li className="flex justify-between">
                          <span>MAE</span>
                          <span className="font-mono">89.2</span>
                        </li>
                      </ul>
                    </div>
                    <div className="p-4 bg-secondary/5 rounded-lg">
                      <h4 className="font-semibold mb-2 text-sm">비즈니스 성과</h4>
                      <ul className="space-y-1 text-xs">
                        <li className="flex justify-between">
                          <span>재고 회전율</span>
                          <span className="font-mono">+45%</span>
                        </li>
                        <li className="flex justify-between">
                          <span>재고 비용</span>
                          <span className="font-mono">-23%</span>
                        </li>
                        <li className="flex justify-between">
                          <span>품절률</span>
                          <span className="font-mono">-67%</span>
                        </li>
                      </ul>
                    </div>
                    <div className="p-4 bg-accent/5 rounded-lg">
                      <h4 className="font-semibold mb-2 text-sm">기술 혁신</h4>
                      <ul className="space-y-1 text-xs">
                        <li>• 실시간 예측 업데이트</li>
                        <li>• 멀티 에셜론 최적화</li>
                        <li>• 자동화된 구매 결정</li>
                        <li>• 공급망 시뮬레이션</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeCase === 'kakaobank' && (
          <div className="space-y-4 mt-6">
            <div className="border rounded-lg">
              <div className="bg-primary/5 p-6 rounded-t-lg">
                <h3 className="text-xl font-bold flex items-center gap-2">
                  <Building2 className="w-6 h-6" />
                  카카오뱅크 AI 신용평가 시스템
                </h3>
              </div>
              <div className="p-6 space-y-6">
                <div className="space-y-4">
                  <div className="p-4 bg-destructive/5 rounded-lg">
                    <h4 className="font-semibold mb-2">비즈니스 과제</h4>
                    <ul className="space-y-2 text-sm">
                      <li>• 전통적 신용평가의 한계 극복</li>
                      <li>• 금융 이력이 부족한 씬파일러 평가</li>
                      <li>• 실시간 신용 리스크 평가</li>
                      <li>• 공정성과 설명가능성 확보</li>
                    </ul>
                  </div>

                  <div>
                    <h4 className="font-semibold mb-2">AI 신용평가 모델</h4>
                    <pre className="bg-muted p-4 rounded-lg text-sm overflow-x-auto">
{`class KakaoBankCreditScoring:
    def __init__(self):
        self.traditional_model = TraditionalCreditModel()
        self.alternative_model = AlternativeDataModel()
        self.ml_ensemble = MLEnsembleModel()
        
    def evaluate_credit_risk(self, customer_id):
        """통합 신용 위험 평가"""
        
        # 1. 전통적 신용 데이터
        traditional_features = {
            'credit_history': self.get_credit_history(customer_id),
            'income': self.get_income_data(customer_id),
            'debt_ratio': self.calculate_debt_ratio(customer_id),
            'employment': self.get_employment_status(customer_id)
        }
        
        # 2. 대안 데이터 수집
        alternative_features = {
            'transaction_patterns': self.analyze_transactions(customer_id),
            'payment_behavior': self.analyze_payment_behavior(customer_id),
            'digital_footprint': self.analyze_digital_behavior(customer_id),
            'social_commerce': self.get_commerce_data(customer_id)
        }
        
        # 3. 특성 엔지니어링
        engineered_features = self.feature_engineering(
            traditional_features,
            alternative_features
        )
        
        # 4. 모델 앙상블
        predictions = {
            'logistic': self.logistic_model.predict_proba(engineered_features),
            'xgboost': self.xgboost_model.predict_proba(engineered_features),
            'neural_net': self.neural_model.predict_proba(engineered_features),
            'catboost': self.catboost_model.predict_proba(engineered_features)
        }
        
        # 5. 가중 평균 및 캘리브레이션
        final_score = self.calibrated_ensemble(predictions)
        
        # 6. 설명가능한 AI (XAI)
        explanations = self.generate_explanations(
            customer_id,
            engineered_features,
            final_score
        )
        
        # 7. 신용 등급 및 한도 결정
        credit_decision = self.make_credit_decision(
            score=final_score,
            explanations=explanations,
            risk_tolerance=self.get_risk_tolerance()
        )
        
        return credit_decision
    
    def generate_explanations(self, customer_id, features, score):
        """SHAP을 활용한 모델 설명"""
        import shap
        
        # SHAP 값 계산
        explainer = shap.TreeExplainer(self.xgboost_model)
        shap_values = explainer.shap_values(features)
        
        # 주요 영향 요인 추출
        feature_importance = pd.DataFrame({
            'feature': features.columns,
            'impact': np.abs(shap_values).mean(axis=0)
        }).sort_values('impact', ascending=False)
        
        # 고객 맞춤 설명 생성
        explanations = {
            'positive_factors': self.get_positive_factors(shap_values, features),
            'negative_factors': self.get_negative_factors(shap_values, features),
            'improvement_suggestions': self.generate_suggestions(feature_importance)
        }
        
        return explanations
    
    def monitor_fairness(self, predictions, sensitive_attributes):
        """공정성 모니터링"""
        from fairlearn.metrics import demographic_parity_ratio
        
        fairness_metrics = {}
        
        for attribute in sensitive_attributes:
            # 인구통계학적 균형성 검증
            dp_ratio = demographic_parity_ratio(
                y_true=predictions['actual'],
                y_pred=predictions['predicted'],
                sensitive_features=predictions[attribute]
            )
            
            fairness_metrics[attribute] = {
                'demographic_parity': dp_ratio,
                'equalized_odds': self.calculate_equalized_odds(
                    predictions, attribute
                ),
                'disparate_impact': self.calculate_disparate_impact(
                    predictions, attribute
                )
            }
            
        return fairness_metrics`}</pre>
                  </div>

                  <div className="grid md:grid-cols-2 gap-4">
                    <div className="p-4 bg-primary/5 rounded-lg">
                      <h4 className="font-semibold mb-2">성과 지표</h4>
                      <ul className="space-y-2 text-sm">
                        <li className="flex justify-between">
                          <span>불량률 감소</span>
                          <span className="font-mono">-42%</span>
                        </li>
                        <li className="flex justify-between">
                          <span>승인율 증가</span>
                          <span className="font-mono">+28%</span>
                        </li>
                        <li className="flex justify-between">
                          <span>처리 시간</span>
                          <span className="font-mono">3초</span>
                        </li>
                        <li className="flex justify-between">
                          <span>고객 만족도</span>
                          <span className="font-mono">4.7/5</span>
                        </li>
                      </ul>
                    </div>
                    <div className="p-4 bg-secondary/5 rounded-lg">
                      <h4 className="font-semibold mb-2">혁신 포인트</h4>
                      <ul className="space-y-1 text-sm">
                        <li>• 실시간 리스크 평가</li>
                        <li>• 대안 데이터 활용</li>
                        <li>• 설명가능한 AI 적용</li>
                        <li>• 공정성 알고리즘 도입</li>
                        <li>• 지속적 모델 개선</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeCase === 'coupang' && (
          <div className="space-y-4 mt-6">
            <div className="border rounded-lg">
              <div className="bg-secondary/5 p-6 rounded-t-lg">
                <h3 className="text-xl font-bold flex items-center gap-2">
                  <TrendingUp className="w-6 h-6" />
                  쿠팡 로켓배송 물류 최적화
                </h3>
              </div>
              <div className="p-6 space-y-6">
                <div className="space-y-4">
                  <div className="p-4 bg-destructive/5 rounded-lg">
                    <h4 className="font-semibold mb-2">비즈니스 과제</h4>
                    <ul className="space-y-2 text-sm">
                      <li>• 새벽배송 시간 윈도우 최적화</li>
                      <li>• 배송 경로 및 차량 배치 최적화</li>
                      <li>• 실시간 수요 변동 대응</li>
                      <li>• 배송 비용 절감 및 효율성 향상</li>
                    </ul>
                  </div>

                  <div>
                    <h4 className="font-semibold mb-2">물류 최적화 시스템</h4>
                    <pre className="bg-muted p-4 rounded-lg text-sm overflow-x-auto">
{`class CoupangLogisticsOptimizer:
    def __init__(self):
        self.route_optimizer = RouteOptimizer()
        self.demand_predictor = DemandPredictor()
        self.fleet_manager = FleetManager()
        
    def optimize_daily_operations(self, date):
        """일일 물류 운영 최적화"""
        
        # 1. 수요 예측
        demand_forecast = self.predict_regional_demand(date)
        
        # 2. 창고별 재고 배치 최적화
        inventory_allocation = self.optimize_inventory_placement(
            demand_forecast,
            current_inventory=self.get_current_inventory(),
            transfer_costs=self.get_transfer_costs()
        )
        
        # 3. 배송 경로 최적화
        delivery_routes = self.optimize_delivery_routes(
            orders=self.get_orders(date),
            vehicles=self.get_available_vehicles(),
            time_windows=self.get_delivery_windows()
        )
        
        # 4. 동적 가격 최적화
        pricing_strategy = self.optimize_delivery_pricing(
            demand_forecast,
            capacity_utilization=self.calculate_capacity_utilization(),
            competitor_prices=self.get_competitor_prices()
        )
        
        return {
            'inventory_plan': inventory_allocation,
            'delivery_routes': delivery_routes,
            'pricing': pricing_strategy,
            'expected_performance': self.simulate_performance(
                inventory_allocation, delivery_routes
            )
        }
    
    def optimize_delivery_routes(self, orders, vehicles, time_windows):
        """차량 경로 문제(VRP) 최적화"""
        from ortools.constraint_solver import routing_enums_pb2
        from ortools.constraint_solver import pywrapcp
        
        # 거리 매트릭스 생성
        distance_matrix = self.create_distance_matrix(orders)
        
        # Create Routing Index Manager
        manager = pywrapcp.RoutingIndexManager(
            len(orders), len(vehicles), 0
        )
        
        # Create Routing Model
        routing = pywrapcp.RoutingModel(manager)
        
        # 비용 함수 정의
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # 시간 제약 추가
        time_dimension = routing.AddDimension(
            transit_callback_index,
            30,  # 대기 시간
            720,  # 최대 근무 시간 (12시간)
            False,
            'Time'
        )
        
        # 시간 윈도우 제약
        for location_idx, time_window in enumerate(time_windows):
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(
                time_window[0], time_window[1]
            )
            
        # 차량 용량 제약
        capacity_dimension = routing.AddDimensionWithVehicleCapacity(
            self.demand_callback,
            0,  # null capacity slack
            vehicle_capacities,
            True,
            'Capacity'
        )
        
        # 최적화 파라미터 설정
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.FromSeconds(30)
        
        # 솔루션 찾기
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            return self.extract_routes(manager, routing, solution)
        else:
            return self.fallback_routing(orders, vehicles)
    
    def real_time_adjustment(self, current_state, new_event):
        """실시간 경로 재조정"""
        
        if new_event['type'] == 'traffic_jam':
            # 교통 체증 대응
            affected_routes = self.identify_affected_routes(
                current_state['routes'],
                new_event['location']
            )
            
            for route in affected_routes:
                alternative_route = self.find_alternative_route(
                    route,
                    avoid_location=new_event['location']
                )
                self.update_driver_route(route['driver_id'], alternative_route)
                
        elif new_event['type'] == 'urgent_order':
            # 긴급 주문 처리
            nearest_vehicle = self.find_nearest_available_vehicle(
                new_event['pickup_location']
            )
            
            if nearest_vehicle:
                updated_route = self.insert_order_to_route(
                    nearest_vehicle['current_route'],
                    new_event['order']
                )
                self.update_driver_route(nearest_vehicle['id'], updated_route)
                
        return self.get_updated_state()`}</pre>
                  </div>

                  <div className="grid md:grid-cols-3 gap-4">
                    <div className="p-4 bg-primary/5 rounded-lg">
                      <h4 className="font-semibold mb-2 text-sm">운영 효율성</h4>
                      <ul className="space-y-1 text-xs">
                        <li className="flex justify-between">
                          <span>배송 시간</span>
                          <span className="font-mono">-35%</span>
                        </li>
                        <li className="flex justify-between">
                          <span>연료 비용</span>
                          <span className="font-mono">-28%</span>
                        </li>
                        <li className="flex justify-between">
                          <span>차량 가동률</span>
                          <span className="font-mono">92%</span>
                        </li>
                      </ul>
                    </div>
                    <div className="p-4 bg-secondary/5 rounded-lg">
                      <h4 className="font-semibold mb-2 text-sm">서비스 품질</h4>
                      <ul className="space-y-1 text-xs">
                        <li className="flex justify-between">
                          <span>정시 배송률</span>
                          <span className="font-mono">99.2%</span>
                        </li>
                        <li className="flex justify-between">
                          <span>고객 만족도</span>
                          <span className="font-mono">4.8/5</span>
                        </li>
                        <li className="flex justify-between">
                          <span>배송 커버리지</span>
                          <span className="font-mono">95%</span>
                        </li>
                      </ul>
                    </div>
                    <div className="p-4 bg-accent/5 rounded-lg">
                      <h4 className="font-semibold mb-2 text-sm">기술 혁신</h4>
                      <ul className="space-y-1 text-xs">
                        <li>• 실시간 경로 최적화</li>
                        <li>• AI 수요 예측</li>
                        <li>• 자동 재고 보충</li>
                        <li>• IoT 차량 추적</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="border border-primary/20 bg-primary/5 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <Lightbulb className="h-4 w-4 mt-1 flex-shrink-0" />
          <div className="space-y-1">
            <strong>사례 연구에서 배운 교훈:</strong>
            <ul className="mt-2 space-y-1 list-disc list-inside">
              <li>비즈니스 목표와 기술 솔루션의 정렬이 핵심</li>
              <li>데이터 품질과 인프라 투자가 성공의 기반</li>
              <li>지속적인 실험과 개선이 경쟁 우위 창출</li>
              <li>크로스펑셔널 팀 협업이 혁신을 가속화</li>
              <li>윤리적 고려사항과 사용자 신뢰가 장기적 성공 좌우</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="border-2 border-primary/20 rounded-lg">
        <div className="p-6 border-b">
          <h3 className="text-xl font-bold">실습 프로젝트: 나만의 데이터 사이언스 프로젝트</h3>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
            학습한 내용을 바탕으로 실제 비즈니스 문제를 해결하는 프로젝트를 진행해봅시다
          </p>
        </div>
        <div className="p-6 space-y-4">
          <div className="space-y-3">
            <h4 className="font-semibold">프로젝트 단계:</h4>
            <ol className="space-y-2 list-decimal list-inside">
              <li>비즈니스 문제 정의 및 목표 설정</li>
              <li>데이터 수집 및 탐색적 분석</li>
              <li>특성 엔지니어링 및 모델링</li>
              <li>모델 평가 및 최적화</li>
              <li>배포 계획 수립</li>
              <li>성과 측정 및 개선</li>
            </ol>
          </div>
          
          <div className="bg-muted p-4 rounded-lg">
            <h5 className="font-semibold text-sm mb-2">추천 프로젝트 주제:</h5>
            <ul className="text-sm space-y-1 list-disc list-inside">
              <li>고객 이탈 예측 및 방지 시스템</li>
              <li>개인화 추천 엔진 구축</li>
              <li>수요 예측 및 재고 최적화</li>
              <li>이상 거래 탐지 시스템</li>
              <li>텍스트 분석을 통한 고객 인사이트 도출</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="flex justify-between items-center pt-8">
        <p className="text-sm text-gray-600 dark:text-gray-400">
          축하합니다! 데이터 사이언스 기초 과정을 완료했습니다
        </p>
        {onComplete && (
          <button
            onClick={onComplete}
            className="px-4 py-2 bg-primary text-blue-600 dark:text-blue-400-foreground rounded-lg hover:bg-primary/90 transition-colors"
          >
            과정 완료
          </button>
        )}
      </div>
    </div>
  )
}