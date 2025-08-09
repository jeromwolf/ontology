// AI-based Stock Recommendation Engine
// Enterprise-grade recommendation system for B2B clients

import { 
  StockRecommendation, 
  TechnicalSignal, 
  AIAnalysis,
  MarketData 
} from '@/lib/services/market-data/types';

export interface RecommendationParams {
  riskProfile: 'conservative' | 'moderate' | 'aggressive';
  investmentHorizon: number; // days
  sectors?: string[];
  excludeTickers?: string[];
  minMarketCap?: number;
  maxPE?: number;
  preferredStrategy?: 'value' | 'growth' | 'momentum' | 'dividend';
}

export interface PricePrediction {
  ticker: string;
  currentPrice: number;
  predictions: {
    oneDay: { price: number; confidence: number };
    oneWeek: { price: number; confidence: number };
    oneMonth: { price: number; confidence: number };
    threeMonths: { price: number; confidence: number };
  };
  modelUsed: string;
  features: string[];
}

export interface RiskMetrics {
  ticker: string;
  overallRisk: number; // 0-10
  volatility: number;
  beta: number;
  sharpeRatio: number;
  sortinoRatio: number;
  maxDrawdown: number;
  VaR95: number; // Value at Risk at 95% confidence
  CVaR95: number; // Conditional VaR
  liquidityRisk: number;
  sectorRisk: number;
}

export interface OptimizationResult {
  originalPortfolio: PortfolioAllocation[];
  optimizedPortfolio: PortfolioAllocation[];
  improvements: {
    expectedReturn: number;
    riskReduction: number;
    sharpeRatioImprovement: number;
  };
  rebalancingActions: RebalancingAction[];
}

interface PortfolioAllocation {
  ticker: string;
  weight: number;
  shares: number;
  value: number;
}

interface RebalancingAction {
  ticker: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  shares: number;
  reason: string;
}

export class StockRecommendationEngine {
  private apiKey: string;
  private modelEndpoint: string;
  
  constructor(apiKey?: string) {
    this.apiKey = apiKey || process.env.NEXT_PUBLIC_AI_API_KEY || '';
    this.modelEndpoint = process.env.NEXT_PUBLIC_AI_ENDPOINT || 'http://localhost:8000/api/v1';
  }

  // Get AI-powered stock recommendations
  async getRecommendations(params: RecommendationParams): Promise<StockRecommendation[]> {
    try {
      // In production, this would call our ML model API
      // For now, we'll implement a sophisticated mock
      
      const candidates = await this.getCandidateStocks(params);
      const analyzed = await Promise.all(
        candidates.map(stock => this.analyzeStock(stock, params))
      );
      
      // Filter and rank based on AI scores
      const recommendations = analyzed
        .filter(stock => this.meetsRecommendationCriteria(stock, params))
        .sort((a, b) => b.confidence - a.confidence)
        .slice(0, 10); // Top 10 recommendations
      
      return recommendations;
    } catch (error) {
      console.error('Error generating recommendations:', error);
      throw error;
    }
  }

  // Predict future prices using AI models
  async predictPrice(ticker: string, days: number): Promise<PricePrediction> {
    try {
      // In production, this would call our LSTM/Transformer model
      const currentPrice = await this.getCurrentPrice(ticker);
      
      // Mock AI prediction with realistic variations
      const predictions = {
        oneDay: this.generatePrediction(currentPrice, 1),
        oneWeek: this.generatePrediction(currentPrice, 7),
        oneMonth: this.generatePrediction(currentPrice, 30),
        threeMonths: this.generatePrediction(currentPrice, 90)
      };
      
      return {
        ticker,
        currentPrice,
        predictions,
        modelUsed: 'LSTM-Attention-v2.3',
        features: [
          'price_history_90d',
          'volume_patterns',
          'technical_indicators',
          'news_sentiment',
          'sector_performance',
          'macro_indicators'
        ]
      };
    } catch (error) {
      console.error('Error predicting price:', error);
      throw error;
    }
  }

  // Calculate comprehensive risk metrics
  async calculateRiskScore(ticker: string): Promise<RiskMetrics> {
    try {
      // Fetch historical data for risk calculations
      const historicalData = await this.getHistoricalData(ticker);
      
      // Calculate various risk metrics
      const volatility = this.calculateVolatility(historicalData);
      const beta = await this.calculateBeta(ticker, historicalData);
      const { sharpe, sortino } = this.calculateRatios(historicalData);
      const maxDrawdown = this.calculateMaxDrawdown(historicalData);
      const { var95, cvar95 } = this.calculateVaR(historicalData);
      
      // AI-based risk scoring
      const overallRisk = this.calculateOverallRisk({
        volatility,
        beta,
        maxDrawdown,
        liquidityRisk: await this.assessLiquidityRisk(ticker),
        sectorRisk: await this.assessSectorRisk(ticker)
      });
      
      return {
        ticker,
        overallRisk,
        volatility,
        beta,
        sharpeRatio: sharpe,
        sortinoRatio: sortino,
        maxDrawdown,
        VaR95: var95,
        CVaR95: cvar95,
        liquidityRisk: await this.assessLiquidityRisk(ticker),
        sectorRisk: await this.assessSectorRisk(ticker)
      };
    } catch (error) {
      console.error('Error calculating risk score:', error);
      throw error;
    }
  }

  // Optimize portfolio allocation
  async optimizePortfolio(holdings: PortfolioAllocation[]): Promise<OptimizationResult> {
    try {
      // Calculate current portfolio metrics
      const currentMetrics = await this.calculatePortfolioMetrics(holdings);
      
      // Run optimization algorithm (Markowitz, Black-Litterman, etc.)
      const optimized = await this.runOptimization(holdings, currentMetrics);
      
      // Calculate improvement metrics
      const optimizedMetrics = await this.calculatePortfolioMetrics(optimized);
      
      // Generate rebalancing actions
      const rebalancingActions = this.generateRebalancingActions(holdings, optimized);
      
      return {
        originalPortfolio: holdings,
        optimizedPortfolio: optimized,
        improvements: {
          expectedReturn: optimizedMetrics.expectedReturn - currentMetrics.expectedReturn,
          riskReduction: currentMetrics.risk - optimizedMetrics.risk,
          sharpeRatioImprovement: optimizedMetrics.sharpeRatio - currentMetrics.sharpeRatio
        },
        rebalancingActions
      };
    } catch (error) {
      console.error('Error optimizing portfolio:', error);
      throw error;
    }
  }

  // Private helper methods

  private async getCandidateStocks(params: RecommendationParams): Promise<string[]> {
    // In production, this would query our stock universe database
    // Filter by market cap, PE, sectors, etc.
    
    const allStocks = [
      '005930', // Samsung Electronics
      '000660', // SK Hynix
      '035420', // NAVER
      '035720', // Kakao
      '005380', // Hyundai Motor
      '051910', // LG Chem
      '006400', // Samsung SDI
      '003550', // LG Corp
      '105560', // KB Financial
      '055550', // Shinhan Financial
      'AAPL',   // Apple
      'MSFT',   // Microsoft
      'GOOGL',  // Google
      'AMZN',   // Amazon
      'TSLA',   // Tesla
      'NVDA',   // NVIDIA
      'META',   // Meta
      'BRK.B',  // Berkshire Hathaway
      'JPM',    // JP Morgan
      'V'       // Visa
    ];
    
    // Filter based on params
    return allStocks.filter(ticker => {
      if (params.excludeTickers?.includes(ticker)) return false;
      // Add more filtering logic based on sectors, market cap, etc.
      return true;
    });
  }

  private async analyzeStock(
    ticker: string, 
    params: RecommendationParams
  ): Promise<StockRecommendation> {
    // Comprehensive stock analysis using multiple signals
    
    const [technical, fundamental, sentiment] = await Promise.all([
      this.analyzeTechnical(ticker),
      this.analyzeFundamental(ticker),
      this.analyzeSentiment(ticker)
    ]);
    
    // AI model would combine all signals
    const confidence = this.calculateConfidence(technical, fundamental, sentiment, params);
    const expectedReturn = await this.calculateExpectedReturn(ticker, params.investmentHorizon);
    const riskLevel = await this.assessRiskLevel(ticker);
    
    return {
      ticker,
      name: await this.getStockName(ticker),
      confidence,
      expectedReturn,
      riskLevel,
      timeHorizon: this.getTimeHorizon(params.investmentHorizon),
      reasons: this.generateReasons(technical, fundamental, sentiment),
      technicalSignals: technical,
      fundamentalScore: fundamental.score,
      newsScore: sentiment.score,
      price: await this.getCurrentPrice(ticker),
      targetPrice: await this.calculateTargetPrice(ticker, params.investmentHorizon),
      stopLoss: await this.calculateStopLoss(ticker, params.riskProfile)
    };
  }

  private async analyzeTechnical(ticker: string): Promise<TechnicalSignal[]> {
    // Technical analysis using multiple indicators
    const signals: TechnicalSignal[] = [];
    
    // Moving Averages
    signals.push({
      indicator: 'SMA_20_50',
      signal: Math.random() > 0.5 ? 'BUY' : 'SELL',
      strength: Math.random(),
      value: Math.random() * 100,
      description: '20일선이 50일선을 상향 돌파'
    });
    
    // RSI
    const rsi = Math.random() * 100;
    signals.push({
      indicator: 'RSI_14',
      signal: rsi < 30 ? 'BUY' : rsi > 70 ? 'SELL' : 'HOLD',
      strength: Math.abs(50 - rsi) / 50,
      value: rsi,
      description: `RSI ${rsi.toFixed(2)} - ${rsi < 30 ? '과매도' : rsi > 70 ? '과매수' : '중립'}`
    });
    
    // MACD
    signals.push({
      indicator: 'MACD',
      signal: Math.random() > 0.5 ? 'BUY' : 'HOLD',
      strength: Math.random(),
      value: (Math.random() - 0.5) * 10,
      description: 'MACD 히스토그램 상승세'
    });
    
    // Bollinger Bands
    signals.push({
      indicator: 'BollingerBands',
      signal: Math.random() > 0.7 ? 'BUY' : 'HOLD',
      strength: Math.random(),
      value: Math.random() * 2,
      description: '하단 밴드 근처에서 반등'
    });
    
    return signals;
  }

  private async analyzeFundamental(ticker: string): Promise<{ score: number; metrics: any }> {
    // Fundamental analysis
    return {
      score: Math.random() * 0.4 + 0.6, // 0.6-1.0 range
      metrics: {
        pe: Math.random() * 30 + 5,
        pbr: Math.random() * 3 + 0.5,
        roe: Math.random() * 0.3 + 0.1,
        debtRatio: Math.random() * 0.5,
        profitGrowth: (Math.random() - 0.3) * 0.5
      }
    };
  }

  private async analyzeSentiment(ticker: string): Promise<{ score: number; sources: any }> {
    // News and social sentiment analysis
    return {
      score: Math.random() * 0.4 + 0.6, // 0.6-1.0 range
      sources: {
        news: Math.random() * 2 - 1,
        social: Math.random() * 2 - 1,
        analyst: Math.random() * 2 - 1
      }
    };
  }

  private meetsRecommendationCriteria(
    stock: StockRecommendation, 
    params: RecommendationParams
  ): boolean {
    // Filter based on risk profile
    if (params.riskProfile === 'conservative' && stock.riskLevel > 5) return false;
    if (params.riskProfile === 'moderate' && stock.riskLevel > 7) return false;
    
    // Filter based on confidence
    if (stock.confidence < 0.6) return false;
    
    // Filter based on expected return
    if (stock.expectedReturn < 0.05) return false; // Min 5% expected return
    
    return true;
  }

  private generatePrediction(currentPrice: number, days: number): { price: number; confidence: number } {
    // Simulate realistic price prediction
    const dailyVolatility = 0.02; // 2% daily volatility
    const trend = 0.0003; // 0.03% daily trend (11% annually)
    
    const randomWalk = Math.sqrt(days) * dailyVolatility * (Math.random() - 0.5) * 2;
    const trendComponent = trend * days;
    
    const predictedPrice = currentPrice * (1 + trendComponent + randomWalk);
    const confidence = Math.max(0.5, 1 - (days / 365)); // Confidence decreases with time
    
    return { price: predictedPrice, confidence };
  }

  private async getCurrentPrice(ticker: string): Promise<number> {
    // In production, fetch from market data service
    const basePrice = ticker.length === 6 ? 50000 : 150; // KRW vs USD
    return basePrice + (Math.random() - 0.5) * basePrice * 0.1;
  }

  private getTimeHorizon(days: number): '단기' | '중기' | '장기' {
    if (days <= 30) return '단기';
    if (days <= 180) return '중기';
    return '장기';
  }

  private async getStockName(ticker: string): Promise<string> {
    // In production, fetch from database
    const names: { [key: string]: string } = {
      '005930': '삼성전자',
      '000660': 'SK하이닉스',
      '035420': 'NAVER',
      '035720': '카카오',
      'AAPL': 'Apple Inc.',
      'MSFT': 'Microsoft Corp.',
      'GOOGL': 'Alphabet Inc.',
      'NVDA': 'NVIDIA Corp.'
    };
    
    return names[ticker] || ticker;
  }

  private generateReasons(
    technical: TechnicalSignal[], 
    fundamental: any, 
    sentiment: any
  ): string[] {
    const reasons: string[] = [];
    
    // Technical reasons
    const buySignals = technical.filter(s => s.signal === 'BUY');
    if (buySignals.length >= 2) {
      reasons.push(`${buySignals.length}개의 기술적 매수 신호 감지`);
    }
    
    // Fundamental reasons
    if (fundamental.metrics.pe < 15) {
      reasons.push('저평가된 PER 수준');
    }
    if (fundamental.metrics.roe > 0.15) {
      reasons.push('높은 자기자본이익률 (ROE)');
    }
    
    // Sentiment reasons
    if (sentiment.score > 0.7) {
      reasons.push('긍정적인 뉴스 센티먼트');
    }
    
    // Growth reasons
    if (fundamental.metrics.profitGrowth > 0.1) {
      reasons.push('강한 이익 성장세');
    }
    
    return reasons;
  }

  private calculateConfidence(
    technical: TechnicalSignal[],
    fundamental: any,
    sentiment: any,
    params: RecommendationParams
  ): number {
    // Weighted average of different signals
    const technicalScore = technical.reduce((sum, signal) => 
      sum + (signal.signal === 'BUY' ? signal.strength : 0), 0
    ) / technical.length;
    
    const weights = {
      technical: params.preferredStrategy === 'momentum' ? 0.5 : 0.3,
      fundamental: params.preferredStrategy === 'value' ? 0.5 : 0.3,
      sentiment: 0.4
    };
    
    const totalWeight = weights.technical + weights.fundamental + weights.sentiment;
    
    return (
      (technicalScore * weights.technical +
       fundamental.score * weights.fundamental +
       sentiment.score * weights.sentiment) / totalWeight
    );
  }

  private async calculateExpectedReturn(ticker: string, horizon: number): Promise<number> {
    // Calculate expected return based on multiple factors
    const historicalReturn = 0.12; // 12% annual average
    const dailyReturn = historicalReturn / 252; // Trading days
    
    // Add some randomness for realism
    const randomFactor = (Math.random() - 0.5) * 0.1;
    
    return dailyReturn * horizon + randomFactor;
  }

  private async assessRiskLevel(ticker: string): Promise<number> {
    // Risk assessment on 0-10 scale
    const volatility = Math.random() * 0.3 + 0.1; // 10-40% volatility
    const beta = Math.random() * 0.5 + 0.8; // 0.8-1.3 beta
    
    return Math.min(10, volatility * 20 + beta * 2);
  }

  private async calculateTargetPrice(ticker: string, horizon: number): Promise<number> {
    const currentPrice = await this.getCurrentPrice(ticker);
    const expectedReturn = await this.calculateExpectedReturn(ticker, horizon);
    
    return currentPrice * (1 + expectedReturn);
  }

  private async calculateStopLoss(ticker: string, riskProfile: RecommendationParams['riskProfile']): Promise<number> {
    const currentPrice = await this.getCurrentPrice(ticker);
    
    const stopLossPercentage = {
      conservative: 0.05,  // 5%
      moderate: 0.08,     // 8%
      aggressive: 0.12    // 12%
    };
    
    return currentPrice * (1 - stopLossPercentage[riskProfile]);
  }

  private async getHistoricalData(ticker: string): Promise<number[]> {
    // Mock historical prices
    return Array.from({ length: 252 }, () => Math.random() * 100 + 50);
  }

  private calculateVolatility(prices: number[]): number {
    // Calculate annualized volatility
    const returns = prices.slice(1).map((price, i) => 
      Math.log(price / prices[i])
    );
    
    const mean = returns.reduce((a, b) => a + b) / returns.length;
    const variance = returns.reduce((sum, ret) => 
      sum + Math.pow(ret - mean, 2), 0
    ) / returns.length;
    
    return Math.sqrt(variance * 252); // Annualized
  }

  private async calculateBeta(ticker: string, prices: number[]): Promise<number> {
    // Calculate beta vs market
    // In production, compare with actual market index
    return Math.random() * 0.5 + 0.8; // 0.8-1.3 range
  }

  private calculateRatios(prices: number[]): { sharpe: number; sortino: number } {
    // Simplified Sharpe and Sortino ratio calculation
    const returns = prices.slice(1).map((price, i) => 
      (price - prices[i]) / prices[i]
    );
    
    const avgReturn = returns.reduce((a, b) => a + b) / returns.length;
    const riskFreeRate = 0.03 / 252; // 3% annual risk-free rate
    
    const stdDev = Math.sqrt(
      returns.reduce((sum, ret) => sum + Math.pow(ret - avgReturn, 2), 0) / returns.length
    );
    
    const downside = returns.filter(r => r < 0);
    const downsideStdDev = Math.sqrt(
      downside.reduce((sum, ret) => sum + Math.pow(ret, 2), 0) / downside.length
    );
    
    return {
      sharpe: (avgReturn - riskFreeRate) / stdDev * Math.sqrt(252),
      sortino: (avgReturn - riskFreeRate) / downsideStdDev * Math.sqrt(252)
    };
  }

  private calculateMaxDrawdown(prices: number[]): number {
    let maxDrawdown = 0;
    let peak = prices[0];
    
    for (const price of prices) {
      if (price > peak) {
        peak = price;
      }
      const drawdown = (peak - price) / peak;
      if (drawdown > maxDrawdown) {
        maxDrawdown = drawdown;
      }
    }
    
    return maxDrawdown;
  }

  private calculateVaR(prices: number[]): { var95: number; cvar95: number } {
    const returns = prices.slice(1).map((price, i) => 
      (price - prices[i]) / prices[i]
    );
    
    returns.sort((a, b) => a - b);
    const var95Index = Math.floor(returns.length * 0.05);
    const var95 = -returns[var95Index];
    
    const cvar95 = -returns.slice(0, var95Index).reduce((a, b) => a + b) / var95Index;
    
    return { var95, cvar95 };
  }

  private calculateOverallRisk(factors: {
    volatility: number;
    beta: number;
    maxDrawdown: number;
    liquidityRisk: number;
    sectorRisk: number;
  }): number {
    // Weighted risk score 0-10
    const weights = {
      volatility: 0.3,
      beta: 0.2,
      maxDrawdown: 0.2,
      liquidityRisk: 0.15,
      sectorRisk: 0.15
    };
    
    return Math.min(10,
      factors.volatility * 10 * weights.volatility +
      factors.beta * 5 * weights.beta +
      factors.maxDrawdown * 10 * weights.maxDrawdown +
      factors.liquidityRisk * weights.liquidityRisk +
      factors.sectorRisk * weights.sectorRisk
    );
  }

  private async assessLiquidityRisk(ticker: string): Promise<number> {
    // Assess liquidity risk based on volume, spread, etc.
    return Math.random() * 5; // 0-5 scale
  }

  private async assessSectorRisk(ticker: string): Promise<number> {
    // Assess sector-specific risks
    return Math.random() * 5; // 0-5 scale
  }

  private async calculatePortfolioMetrics(portfolio: PortfolioAllocation[]): Promise<any> {
    // Calculate portfolio-level metrics
    return {
      expectedReturn: 0.12,
      risk: 0.15,
      sharpeRatio: 0.8
    };
  }

  private async runOptimization(
    holdings: PortfolioAllocation[], 
    currentMetrics: any
  ): Promise<PortfolioAllocation[]> {
    // Run portfolio optimization algorithm
    // In production, this would use Markowitz optimization, Black-Litterman, etc.
    return holdings.map(holding => ({
      ...holding,
      weight: holding.weight + (Math.random() - 0.5) * 0.1
    }));
  }

  private generateRebalancingActions(
    original: PortfolioAllocation[],
    optimized: PortfolioAllocation[]
  ): RebalancingAction[] {
    const actions: RebalancingAction[] = [];
    
    optimized.forEach((opt, i) => {
      const orig = original[i];
      const diff = opt.weight - orig.weight;
      
      if (Math.abs(diff) > 0.02) { // 2% threshold
        actions.push({
          ticker: opt.ticker,
          action: diff > 0 ? 'BUY' : 'SELL',
          shares: Math.abs(Math.floor(diff * 10000)), // Mock calculation
          reason: diff > 0 ? '목표 비중 상향' : '목표 비중 하향'
        });
      }
    });
    
    return actions;
  }
}

// Export singleton instance
export const recommendationEngine = new StockRecommendationEngine();