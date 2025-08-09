// Advanced Backtesting Engine for Investment Strategies
// B2B-grade backtesting with institutional-level features

import { OHLCV, MarketData } from '@/lib/services/market-data/types';

export interface Strategy {
  id: string;
  name: string;
  description: string;
  parameters: { [key: string]: any };
  signals: StrategySignal[];
  entryConditions: ConditionSet;
  exitConditions: ConditionSet;
  stopLoss?: number; // percentage
  takeProfit?: number; // percentage
  positionSizing: PositionSizingMethod;
}

export interface StrategySignal {
  timestamp: Date;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number; // 0-1
  price: number;
  volume: number;
  metadata: { [key: string]: any };
}

export interface ConditionSet {
  technical: TechnicalCondition[];
  fundamental?: FundamentalCondition[];
  sentiment?: SentimentCondition[];
  macro?: MacroCondition[];
  logic: 'AND' | 'OR';
}

export interface TechnicalCondition {
  indicator: string;
  operator: '>' | '<' | '=' | '>=' | '<=' | 'crossover' | 'crossunder';
  value: number | string;
  lookback?: number;
}

export interface FundamentalCondition {
  metric: 'pe' | 'pbr' | 'roe' | 'debt_ratio' | 'earnings_growth';
  operator: '>' | '<' | '=' | '>=' | '<=';
  value: number;
}

export interface SentimentCondition {
  source: 'news' | 'social' | 'analyst';
  operator: '>' | '<' | '=' | '>=' | '<=';
  value: number; // -1 to 1
}

export interface MacroCondition {
  indicator: 'interest_rate' | 'inflation' | 'gdp_growth' | 'unemployment';
  operator: '>' | '<' | '=' | '>=' | '<=';
  value: number;
}

export type PositionSizingMethod = 
  | { type: 'fixed'; amount: number }
  | { type: 'percentage'; percentage: number }
  | { type: 'kelly'; confidence: number }
  | { type: 'volatility'; target: number };

export interface BacktestParams {
  strategy: Strategy;
  tickers: string[];
  startDate: Date;
  endDate: Date;
  initialCapital: number;
  commission: CommissionModel;
  slippage: SlippageModel;
  benchmark?: string; // ticker for benchmark comparison
  rebalanceFrequency?: 'daily' | 'weekly' | 'monthly' | 'quarterly';
}

export interface CommissionModel {
  type: 'fixed' | 'percentage' | 'tiered';
  value: number;
  minimum?: number;
  maximum?: number;
}

export interface SlippageModel {
  type: 'fixed' | 'linear' | 'square_root';
  value: number; // basis points
}

export interface BacktestResult {
  strategy: Strategy;
  params: BacktestParams;
  performance: PerformanceMetrics;
  trades: Trade[];
  equity: EquityCurve[];
  drawdowns: DrawdownPeriod[];
  monthlyReturns: MonthlyReturn[];
  riskMetrics: RiskMetrics;
  benchmark?: BenchmarkComparison;
  analysis: BacktestAnalysis;
}

export interface PerformanceMetrics {
  totalReturn: number;
  annualizedReturn: number;
  volatility: number;
  sharpeRatio: number;
  sortinoRatio: number;
  calmarRatio: number;
  maxDrawdown: number;
  maxDrawdownDuration: number; // days
  winRate: number;
  profitFactor: number;
  averageWin: number;
  averageLoss: number;
  totalTrades: number;
  averageHoldingPeriod: number; // days
}

export interface Trade {
  id: string;
  ticker: string;
  entryDate: Date;
  exitDate: Date;
  entryPrice: number;
  exitPrice: number;
  quantity: number;
  side: 'LONG' | 'SHORT';
  pnl: number;
  pnlPercent: number;
  commission: number;
  slippage: number;
  holdingPeriod: number; // days
  entrySignal: StrategySignal;
  exitSignal: StrategySignal;
  maxFavorableExcursion: number;
  maxAdverseExcursion: number;
}

export interface EquityCurve {
  date: Date;
  equity: number;
  cash: number;
  positions: number;
  totalValue: number;
  drawdown: number;
  returns: number;
}

export interface DrawdownPeriod {
  start: Date;
  end: Date;
  duration: number; // days
  maxDrawdown: number;
  recovery: Date;
}

export interface MonthlyReturn {
  year: number;
  month: number;
  return: number;
  benchmark?: number;
}

export interface RiskMetrics {
  var95: number; // Value at Risk 95%
  cvar95: number; // Conditional VaR 95%
  beta: number;
  alpha: number;
  trackingError: number;
  informationRatio: number;
  treynorRatio: number;
  ulcerIndex: number;
  painIndex: number;
}

export interface BenchmarkComparison {
  ticker: string;
  correlation: number;
  beta: number;
  alpha: number;
  outperformance: number;
  trackingError: number;
  informationRatio: number;
}

export interface BacktestAnalysis {
  strengths: string[];
  weaknesses: string[];
  recommendations: string[];
  riskWarnings: string[];
  suitability: {
    riskProfile: 'conservative' | 'moderate' | 'aggressive';
    timeHorizon: string;
    marketConditions: string[];
  };
}

export interface WalkForwardParams {
  strategy: Strategy;
  ticker: string;
  startDate: Date;
  endDate: Date;
  trainingPeriod: number; // days
  testingPeriod: number; // days
  step: number; // days to move forward
  optimizationTarget: 'sharpe' | 'return' | 'calmar' | 'sortino';
}

export interface WalkForwardResult {
  params: WalkForwardParams;
  results: {
    period: { start: Date; end: Date };
    inSample: BacktestResult;
    outOfSample: BacktestResult;
    degradation: number; // performance difference
  }[];
  summary: {
    averageInSample: PerformanceMetrics;
    averageOutOfSample: PerformanceMetrics;
    averageDegradation: number;
    consistency: number; // 0-1
    robustness: number; // 0-1
  };
}

export interface MonteCarloParams {
  strategy: Strategy;
  ticker: string;
  baseResult: BacktestResult;
  iterations: number;
  perturbation: {
    returns: number; // standard deviation
    volume: number; // standard deviation
    spread: number; // basis points
  };
}

export interface MonteCarloResult {
  params: MonteCarloParams;
  iterations: {
    id: number;
    totalReturn: number;
    maxDrawdown: number;
    sharpeRatio: number;
    finalEquity: number;
  }[];
  statistics: {
    totalReturn: {
      mean: number;
      std: number;
      percentiles: { p5: number; p25: number; p50: number; p75: number; p95: number };
    };
    maxDrawdown: {
      mean: number;
      std: number;
      percentiles: { p5: number; p25: number; p50: number; p75: number; p95: number };
    };
    sharpeRatio: {
      mean: number;
      std: number;
      percentiles: { p5: number; p25: number; p50: number; p75: number; p95: number };
    };
    probabilityOfProfit: number;
    probabilityOfRuin: number;
  };
}

export class BacktestingEngine {
  private commission: CommissionModel = { type: 'percentage', value: 0.1 }; // 0.1%
  private slippage: SlippageModel = { type: 'linear', value: 5 }; // 5 bps
  
  constructor() {}

  // Main backtesting function
  async runBacktest(params: BacktestParams): Promise<BacktestResult> {
    try {
      console.log('Starting backtest:', params.strategy.name);
      
      // Validate parameters
      this.validateParams(params);
      
      // Initialize portfolio
      const portfolio = this.initializePortfolio(params.initialCapital);
      
      // Get historical data for all tickers
      const historicalData = await this.getHistoricalData(
        params.tickers,
        params.startDate,
        params.endDate
      );
      
      // Run simulation
      const { trades, equity } = await this.runSimulation(params, historicalData);
      
      // Calculate performance metrics
      const performance = this.calculatePerformanceMetrics(equity, trades);
      
      // Calculate risk metrics
      const riskMetrics = this.calculateRiskMetrics(equity);
      
      // Analyze drawdowns
      const drawdowns = this.analyzeDrawdowns(equity);
      
      // Calculate monthly returns
      const monthlyReturns = this.calculateMonthlyReturns(equity);
      
      // Benchmark comparison (if specified)
      const benchmark = params.benchmark ? 
        await this.compareToBenchmark(equity, params.benchmark, params.startDate, params.endDate) :
        undefined;
      
      // Generate analysis
      const analysis = this.generateAnalysis(performance, riskMetrics, trades);
      
      return {
        strategy: params.strategy,
        params,
        performance,
        trades,
        equity,
        drawdowns,
        monthlyReturns,
        riskMetrics,
        benchmark,
        analysis
      };
    } catch (error) {
      console.error('Backtest failed:', error);
      throw error;
    }
  }

  // Walk-forward analysis
  async runWalkForwardAnalysis(params: WalkForwardParams): Promise<WalkForwardResult> {
    const results: WalkForwardResult['results'] = [];
    
    let currentStart = params.startDate;
    
    while (currentStart < params.endDate) {
      const trainingEnd = new Date(currentStart.getTime() + params.trainingPeriod * 24 * 60 * 60 * 1000);
      const testingEnd = new Date(trainingEnd.getTime() + params.testingPeriod * 24 * 60 * 60 * 1000);
      
      if (testingEnd > params.endDate) break;
      
      // In-sample backtest (training period)
      const inSampleParams: BacktestParams = {
        strategy: params.strategy,
        tickers: [params.ticker],
        startDate: currentStart,
        endDate: trainingEnd,
        initialCapital: 100000,
        commission: this.commission,
        slippage: this.slippage
      };
      
      const inSample = await this.runBacktest(inSampleParams);
      
      // Out-of-sample backtest (testing period)
      const outOfSampleParams: BacktestParams = {
        strategy: params.strategy,
        tickers: [params.ticker],
        startDate: trainingEnd,
        endDate: testingEnd,
        initialCapital: 100000,
        commission: this.commission,
        slippage: this.slippage
      };
      
      const outOfSample = await this.runBacktest(outOfSampleParams);
      
      // Calculate degradation
      const degradation = this.calculatePerformanceDegradation(
        inSample.performance,
        outOfSample.performance,
        params.optimizationTarget
      );
      
      results.push({
        period: { start: currentStart, end: testingEnd },
        inSample,
        outOfSample,
        degradation
      });
      
      // Move forward
      currentStart = new Date(currentStart.getTime() + params.step * 24 * 60 * 60 * 1000);
    }
    
    // Calculate summary statistics
    const summary = this.calculateWalkForwardSummary(results);
    
    return {
      params,
      results,
      summary
    };
  }

  // Monte Carlo simulation
  async runMonteCarloSimulation(params: MonteCarloParams): Promise<MonteCarloResult> {
    const iterations: MonteCarloResult['iterations'] = [];
    
    for (let i = 0; i < params.iterations; i++) {
      // Perturb the original data
      const perturbedData = this.perturbData(
        params.baseResult.equity,
        params.perturbation
      );
      
      // Run simulation with perturbed data
      const result = await this.simulateWithPerturbedData(
        params.strategy,
        perturbedData
      );
      
      iterations.push({
        id: i,
        totalReturn: result.performance.totalReturn,
        maxDrawdown: result.performance.maxDrawdown,
        sharpeRatio: result.performance.sharpeRatio,
        finalEquity: result.equity[result.equity.length - 1].totalValue
      });
    }
    
    // Calculate statistics
    const statistics = this.calculateMonteCarloStatistics(iterations);
    
    return {
      params,
      iterations,
      statistics
    };
  }

  // Private helper methods

  private validateParams(params: BacktestParams): void {
    if (params.startDate >= params.endDate) {
      throw new Error('Start date must be before end date');
    }
    
    if (params.initialCapital <= 0) {
      throw new Error('Initial capital must be positive');
    }
    
    if (params.tickers.length === 0) {
      throw new Error('At least one ticker must be specified');
    }
  }

  private initializePortfolio(initialCapital: number) {
    return {
      cash: initialCapital,
      positions: new Map(),
      totalValue: initialCapital
    };
  }

  private async getHistoricalData(
    tickers: string[],
    startDate: Date,
    endDate: Date
  ): Promise<Map<string, OHLCV[]>> {
    const data = new Map<string, OHLCV[]>();
    
    for (const ticker of tickers) {
      // In production, fetch real historical data
      // For now, generate mock data
      const mockData = this.generateMockData(startDate, endDate);
      data.set(ticker, mockData);
    }
    
    return data;
  }

  private generateMockData(startDate: Date, endDate: Date): OHLCV[] {
    const data: OHLCV[] = [];
    const days = Math.floor((endDate.getTime() - startDate.getTime()) / (24 * 60 * 60 * 1000));
    
    let price = 100; // Starting price
    
    for (let i = 0; i < days; i++) {
      const date = new Date(startDate.getTime() + i * 24 * 60 * 60 * 1000);
      
      // Random walk with slight upward bias
      const dailyReturn = (Math.random() - 0.48) * 0.03; // Slight positive bias
      price *= (1 + dailyReturn);
      
      const high = price * (1 + Math.random() * 0.02);
      const low = price * (1 - Math.random() * 0.02);
      const open = price * (1 + (Math.random() - 0.5) * 0.01);
      const volume = Math.floor(Math.random() * 1000000) + 100000;
      
      data.push({
        timestamp: date,
        open,
        high,
        low,
        close: price,
        volume
      });
    }
    
    return data;
  }

  private async runSimulation(
    params: BacktestParams,
    historicalData: Map<string, OHLCV[]>
  ): Promise<{ trades: Trade[]; equity: EquityCurve[] }> {
    const trades: Trade[] = [];
    const equity: EquityCurve[] = [];
    const portfolio = this.initializePortfolio(params.initialCapital);
    
    // Get all dates and sort
    const allDates = new Set<number>();
    historicalData.forEach(data => {
      data.forEach(bar => allDates.add(bar.timestamp.getTime()));
    });
    const sortedDates = Array.from(allDates).sort().map(d => new Date(d));
    
    for (const date of sortedDates) {
      // Get market data for this date
      const marketData = this.getMarketDataForDate(historicalData, date);
      
      // Generate signals
      const signals = await this.generateSignals(params.strategy, marketData, date);
      
      // Execute trades based on signals
      const dayTrades = this.executeTrades(signals, portfolio, marketData, params);
      trades.push(...dayTrades);
      
      // Update portfolio value
      const totalValue = this.calculatePortfolioValue(portfolio, marketData);
      
      // Record equity curve
      equity.push({
        date,
        equity: totalValue,
        cash: portfolio.cash,
        positions: this.calculatePositionsValue(portfolio, marketData),
        totalValue,
        drawdown: 0, // Will be calculated later
        returns: equity.length > 0 ? (totalValue / equity[equity.length - 1].totalValue) - 1 : 0
      });
    }
    
    // Calculate drawdowns for equity curve
    this.calculateDrawdowns(equity);
    
    return { trades, equity };
  }

  private getMarketDataForDate(
    historicalData: Map<string, OHLCV[]>,
    date: Date
  ): Map<string, OHLCV> {
    const marketData = new Map<string, OHLCV>();
    
    historicalData.forEach((data, ticker) => {
      const bar = data.find(d => d.timestamp.getTime() === date.getTime());
      if (bar) {
        marketData.set(ticker, bar);
      }
    });
    
    return marketData;
  }

  private async generateSignals(
    strategy: Strategy,
    marketData: Map<string, OHLCV>,
    date: Date
  ): Promise<StrategySignal[]> {
    const signals: StrategySignal[] = [];
    
    // Simple mock signal generation
    // In production, this would evaluate the strategy conditions
    marketData.forEach((data, ticker) => {
      // Random signal for demonstration
      const random = Math.random();
      let action: 'BUY' | 'SELL' | 'HOLD' = 'HOLD';
      
      if (random < 0.1) action = 'BUY';
      else if (random > 0.9) action = 'SELL';
      
      if (action !== 'HOLD') {
        signals.push({
          timestamp: date,
          action,
          confidence: Math.random(),
          price: data.close,
          volume: data.volume,
          metadata: { ticker }
        });
      }
    });
    
    return signals;
  }

  private executeTrades(
    signals: StrategySignal[],
    portfolio: any,
    marketData: Map<string, OHLCV>,
    params: BacktestParams
  ): Trade[] {
    const trades: Trade[] = [];
    
    // Simple trade execution logic
    // In production, this would be much more sophisticated
    signals.forEach(signal => {
      const ticker = signal.metadata.ticker;
      const data = marketData.get(ticker);
      
      if (data && signal.action === 'BUY' && portfolio.cash > data.close) {
        // Calculate position size
        const positionValue = this.calculatePositionSize(
          params.strategy.positionSizing,
          portfolio.cash,
          data.close
        );
        
        const shares = Math.floor(positionValue / data.close);
        const cost = shares * data.close;
        
        if (cost <= portfolio.cash && shares > 0) {
          portfolio.cash -= cost;
          
          const position = portfolio.positions.get(ticker) || { shares: 0, avgPrice: 0 };
          position.shares += shares;
          position.avgPrice = ((position.avgPrice * (position.shares - shares)) + cost) / position.shares;
          portfolio.positions.set(ticker, position);
          
          // Create trade record (will be completed on exit)
          // For now, we'll create a simple trade
        }
      }
    });
    
    return trades;
  }

  private calculatePositionSize(
    method: PositionSizingMethod,
    availableCash: number,
    price: number
  ): number {
    switch (method.type) {
      case 'fixed':
        return Math.min(method.amount, availableCash);
      case 'percentage':
        return availableCash * (method.percentage / 100);
      case 'kelly':
        // Simplified Kelly criterion
        return availableCash * method.confidence * 0.25; // Conservative
      case 'volatility':
        // Volatility-based sizing
        return availableCash * (method.target / 100);
      default:
        return availableCash * 0.1; // Default 10%
    }
  }

  private calculatePortfolioValue(
    portfolio: any,
    marketData: Map<string, OHLCV>
  ): number {
    let totalValue = portfolio.cash;
    
    portfolio.positions.forEach((position: any, ticker: string) => {
      const data = marketData.get(ticker);
      if (data) {
        totalValue += position.shares * data.close;
      }
    });
    
    return totalValue;
  }

  private calculatePositionsValue(
    portfolio: any,
    marketData: Map<string, OHLCV>
  ): number {
    let positionsValue = 0;
    
    portfolio.positions.forEach((position: any, ticker: string) => {
      const data = marketData.get(ticker);
      if (data) {
        positionsValue += position.shares * data.close;
      }
    });
    
    return positionsValue;
  }

  private calculateDrawdowns(equity: EquityCurve[]): void {
    let peak = equity[0]?.totalValue || 0;
    
    equity.forEach(point => {
      if (point.totalValue > peak) {
        peak = point.totalValue;
      }
      point.drawdown = peak > 0 ? (peak - point.totalValue) / peak : 0;
    });
  }

  private calculatePerformanceMetrics(
    equity: EquityCurve[],
    trades: Trade[]
  ): PerformanceMetrics {
    if (equity.length === 0) {
      throw new Error('No equity data available');
    }
    
    const startValue = equity[0].totalValue;
    const endValue = equity[equity.length - 1].totalValue;
    const totalReturn = (endValue - startValue) / startValue;
    
    // Calculate returns
    const returns = equity.slice(1).map((point, i) => 
      (point.totalValue - equity[i].totalValue) / equity[i].totalValue
    );
    
    const avgReturn = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const annualizedReturn = Math.pow(1 + totalReturn, 365 / equity.length) - 1;
    
    // Calculate volatility
    const variance = returns.reduce((sum, ret) => 
      sum + Math.pow(ret - avgReturn, 2), 0
    ) / returns.length;
    const volatility = Math.sqrt(variance * 365); // Annualized
    
    // Sharpe ratio (assuming 3% risk-free rate)
    const riskFreeRate = 0.03;
    const sharpeRatio = volatility > 0 ? (annualizedReturn - riskFreeRate) / volatility : 0;
    
    // Sortino ratio (downside deviation)
    const downsideReturns = returns.filter(ret => ret < 0);
    const downsideVariance = downsideReturns.reduce((sum, ret) => 
      sum + Math.pow(ret, 2), 0
    ) / downsideReturns.length;
    const downsideDeviation = Math.sqrt(downsideVariance * 365);
    const sortinoRatio = downsideDeviation > 0 ? 
      (annualizedReturn - riskFreeRate) / downsideDeviation : 0;
    
    // Max drawdown
    const maxDrawdown = Math.max(...equity.map(point => point.drawdown));
    
    // Calmar ratio
    const calmarRatio = maxDrawdown > 0 ? annualizedReturn / maxDrawdown : 0;
    
    // Trade statistics
    const winningTrades = trades.filter(trade => trade.pnl > 0);
    const losingTrades = trades.filter(trade => trade.pnl < 0);
    const winRate = trades.length > 0 ? winningTrades.length / trades.length : 0;
    
    const totalWins = winningTrades.reduce((sum, trade) => sum + trade.pnl, 0);
    const totalLosses = Math.abs(losingTrades.reduce((sum, trade) => sum + trade.pnl, 0));
    const profitFactor = totalLosses > 0 ? totalWins / totalLosses : 0;
    
    const averageWin = winningTrades.length > 0 ? 
      totalWins / winningTrades.length : 0;
    const averageLoss = losingTrades.length > 0 ? 
      totalLosses / losingTrades.length : 0;
    
    const averageHoldingPeriod = trades.length > 0 ?
      trades.reduce((sum, trade) => sum + trade.holdingPeriod, 0) / trades.length : 0;
    
    // Max drawdown duration
    let maxDrawdownDuration = 0;
    let currentDrawdownDuration = 0;
    let inDrawdown = false;
    
    equity.forEach(point => {
      if (point.drawdown > 0) {
        if (!inDrawdown) {
          inDrawdown = true;
          currentDrawdownDuration = 1;
        } else {
          currentDrawdownDuration++;
        }
      } else {
        if (inDrawdown) {
          maxDrawdownDuration = Math.max(maxDrawdownDuration, currentDrawdownDuration);
          inDrawdown = false;
        }
      }
    });
    
    return {
      totalReturn,
      annualizedReturn,
      volatility,
      sharpeRatio,
      sortinoRatio,
      calmarRatio,
      maxDrawdown,
      maxDrawdownDuration,
      winRate,
      profitFactor,
      averageWin,
      averageLoss,
      totalTrades: trades.length,
      averageHoldingPeriod
    };
  }

  private calculateRiskMetrics(equity: EquityCurve[]): RiskMetrics {
    const returns = equity.slice(1).map((point, i) => 
      (point.totalValue - equity[i].totalValue) / equity[i].totalValue
    );
    
    // Sort returns for VaR calculation
    const sortedReturns = [...returns].sort((a, b) => a - b);
    
    // VaR and CVaR at 95% confidence
    const var95Index = Math.floor(returns.length * 0.05);
    const var95 = -sortedReturns[var95Index] || 0;
    const cvar95 = var95Index > 0 ? 
      -sortedReturns.slice(0, var95Index).reduce((sum, ret) => sum + ret, 0) / var95Index : 0;
    
    // Mock other risk metrics
    // In production, these would be calculated with proper benchmark data
    const beta = 1.0 + (Math.random() - 0.5) * 0.4; // 0.8-1.2
    const alpha = (Math.random() - 0.5) * 0.1; // -5% to +5%
    const trackingError = Math.random() * 0.05; // 0-5%
    const informationRatio = alpha / (trackingError || 1);
    const treynorRatio = alpha / beta;
    
    // Ulcer Index (measure of downside volatility)
    const ulcerIndex = Math.sqrt(
      equity.reduce((sum, point) => sum + Math.pow(point.drawdown * 100, 2), 0) / equity.length
    );
    
    // Pain Index (average drawdown)
    const painIndex = equity.reduce((sum, point) => sum + point.drawdown, 0) / equity.length;
    
    return {
      var95,
      cvar95,
      beta,
      alpha,
      trackingError,
      informationRatio,
      treynorRatio,
      ulcerIndex,
      painIndex
    };
  }

  private analyzeDrawdowns(equity: EquityCurve[]): DrawdownPeriod[] {
    const drawdowns: DrawdownPeriod[] = [];
    let currentDrawdown: {
      start: Date;
      maxDrawdown: number;
      duration: number;
    } | null = null;
    let peak = equity[0]?.totalValue || 0;
    
    equity.forEach(point => {
      if (point.totalValue > peak) {
        // New peak - end current drawdown if exists
        if (currentDrawdown) {
          drawdowns.push({
            start: currentDrawdown.start,
            end: point.date,
            duration: currentDrawdown.duration,
            maxDrawdown: currentDrawdown.maxDrawdown,
            recovery: point.date
          });
          currentDrawdown = null;
        }
        peak = point.totalValue;
      } else if (point.totalValue < peak) {
        // In drawdown
        if (!currentDrawdown) {
          // Start new drawdown
          currentDrawdown = {
            start: point.date,
            maxDrawdown: point.drawdown,
            duration: 1
          };
        } else {
          // Continue drawdown
          currentDrawdown.maxDrawdown = Math.max(
            currentDrawdown.maxDrawdown || 0,
            point.drawdown
          );
          currentDrawdown.duration = (currentDrawdown.duration || 0) + 1;
        }
      }
    });
    
    // Close final drawdown if exists
    if (currentDrawdown) {
      drawdowns.push({
        start: currentDrawdown.start,
        end: equity[equity.length - 1].date,
        duration: currentDrawdown.duration,
        maxDrawdown: currentDrawdown.maxDrawdown,
        recovery: equity[equity.length - 1].date
      });
    }
    
    return drawdowns;
  }

  private calculateMonthlyReturns(equity: EquityCurve[]): MonthlyReturn[] {
    const monthlyReturns: MonthlyReturn[] = [];
    const monthlyData = new Map<string, { start: number; end: number }>();
    
    equity.forEach(point => {
      const key = `${point.date.getFullYear()}-${point.date.getMonth()}`;
      const existing = monthlyData.get(key);
      
      if (!existing) {
        monthlyData.set(key, { start: point.totalValue, end: point.totalValue });
      } else {
        existing.end = point.totalValue;
      }
    });
    
    monthlyData.forEach((data, key) => {
      const [year, month] = key.split('-').map(Number);
      const monthReturn = (data.end - data.start) / data.start;
      
      monthlyReturns.push({
        year,
        month,
        return: monthReturn
      });
    });
    
    return monthlyReturns.sort((a, b) => a.year - b.year || a.month - b.month);
  }

  private async compareToBenchmark(
    equity: EquityCurve[],
    benchmarkTicker: string,
    startDate: Date,
    endDate: Date
  ): Promise<BenchmarkComparison> {
    // Mock benchmark comparison
    // In production, fetch actual benchmark data
    const correlation = 0.7 + Math.random() * 0.3; // 0.7-1.0
    const beta = 0.8 + Math.random() * 0.4; // 0.8-1.2
    const alpha = (Math.random() - 0.5) * 0.1; // -5% to +5%
    const outperformance = alpha;
    const trackingError = Math.random() * 0.05; // 0-5%
    const informationRatio = alpha / (trackingError || 1);
    
    return {
      ticker: benchmarkTicker,
      correlation,
      beta,
      alpha,
      outperformance,
      trackingError,
      informationRatio
    };
  }

  private generateAnalysis(
    performance: PerformanceMetrics,
    riskMetrics: RiskMetrics,
    trades: Trade[]
  ): BacktestAnalysis {
    const strengths: string[] = [];
    const weaknesses: string[] = [];
    const recommendations: string[] = [];
    const riskWarnings: string[] = [];
    
    // Analyze performance
    if (performance.sharpeRatio > 1.5) {
      strengths.push('높은 위험조정수익률 (Sharpe Ratio > 1.5)');
    } else if (performance.sharpeRatio < 0.5) {
      weaknesses.push('낮은 위험조정수익률 (Sharpe Ratio < 0.5)');
    }
    
    if (performance.maxDrawdown < 0.1) {
      strengths.push('낮은 최대 낙폭 (< 10%)');
    } else if (performance.maxDrawdown > 0.3) {
      weaknesses.push('높은 최대 낙폭 (> 30%)');
      riskWarnings.push('큰 손실 위험 존재');
    }
    
    if (performance.winRate > 0.6) {
      strengths.push('높은 승률 (> 60%)');
    } else if (performance.winRate < 0.4) {
      weaknesses.push('낮은 승률 (< 40%)');
    }
    
    // Generate recommendations
    if (performance.volatility > 0.3) {
      recommendations.push('포지션 크기 축소를 통한 변동성 관리');
    }
    
    if (trades.length < 10) {
      recommendations.push('더 많은 거래 기회 확보 필요');
    }
    
    if (performance.averageHoldingPeriod > 100) {
      recommendations.push('보유 기간이 길어 시장 효율성 검토 필요');
    }
    
    // Determine suitability
    let riskProfile: 'conservative' | 'moderate' | 'aggressive' = 'moderate';
    if (performance.maxDrawdown < 0.15 && performance.volatility < 0.2) {
      riskProfile = 'conservative';
    } else if (performance.maxDrawdown > 0.25 || performance.volatility > 0.3) {
      riskProfile = 'aggressive';
    }
    
    return {
      strengths,
      weaknesses,
      recommendations,
      riskWarnings,
      suitability: {
        riskProfile,
        timeHorizon: performance.averageHoldingPeriod > 30 ? '장기' : '단기',
        marketConditions: ['상승장', '횡보장'] // Mock conditions
      }
    };
  }

  private calculatePerformanceDegradation(
    inSample: PerformanceMetrics,
    outOfSample: PerformanceMetrics,
    target: WalkForwardParams['optimizationTarget']
  ): number {
    const inSampleValue = this.getMetricValue(inSample, target);
    const outOfSampleValue = this.getMetricValue(outOfSample, target);
    
    return inSampleValue > 0 ? (inSampleValue - outOfSampleValue) / inSampleValue : 0;
  }

  private getMetricValue(
    performance: PerformanceMetrics,
    metric: WalkForwardParams['optimizationTarget']
  ): number {
    switch (metric) {
      case 'sharpe': return performance.sharpeRatio;
      case 'return': return performance.annualizedReturn;
      case 'calmar': return performance.calmarRatio;
      case 'sortino': return performance.sortinoRatio;
      default: return performance.sharpeRatio;
    }
  }

  private calculateWalkForwardSummary(
    results: WalkForwardResult['results']
  ): WalkForwardResult['summary'] {
    // Calculate averages
    const avgInSample = this.calculateAveragePerformance(
      results.map(r => r.inSample.performance)
    );
    const avgOutOfSample = this.calculateAveragePerformance(
      results.map(r => r.outOfSample.performance)
    );
    
    const avgDegradation = results.reduce((sum, r) => sum + r.degradation, 0) / results.length;
    
    // Calculate consistency (low standard deviation of returns)
    const outOfSampleReturns = results.map(r => r.outOfSample.performance.annualizedReturn);
    const avgReturn = outOfSampleReturns.reduce((sum, ret) => sum + ret, 0) / outOfSampleReturns.length;
    const variance = outOfSampleReturns.reduce((sum, ret) => 
      sum + Math.pow(ret - avgReturn, 2), 0
    ) / outOfSampleReturns.length;
    const consistency = 1 / (1 + Math.sqrt(variance)); // 0-1 scale
    
    // Calculate robustness (percentage of profitable periods)
    const profitablePeriods = results.filter(r => r.outOfSample.performance.totalReturn > 0).length;
    const robustness = profitablePeriods / results.length;
    
    return {
      averageInSample: avgInSample,
      averageOutOfSample: avgOutOfSample,
      averageDegradation: avgDegradation,
      consistency,
      robustness
    };
  }

  private calculateAveragePerformance(performances: PerformanceMetrics[]): PerformanceMetrics {
    const count = performances.length;
    
    return {
      totalReturn: performances.reduce((sum, p) => sum + p.totalReturn, 0) / count,
      annualizedReturn: performances.reduce((sum, p) => sum + p.annualizedReturn, 0) / count,
      volatility: performances.reduce((sum, p) => sum + p.volatility, 0) / count,
      sharpeRatio: performances.reduce((sum, p) => sum + p.sharpeRatio, 0) / count,
      sortinoRatio: performances.reduce((sum, p) => sum + p.sortinoRatio, 0) / count,
      calmarRatio: performances.reduce((sum, p) => sum + p.calmarRatio, 0) / count,
      maxDrawdown: performances.reduce((sum, p) => sum + p.maxDrawdown, 0) / count,
      maxDrawdownDuration: performances.reduce((sum, p) => sum + p.maxDrawdownDuration, 0) / count,
      winRate: performances.reduce((sum, p) => sum + p.winRate, 0) / count,
      profitFactor: performances.reduce((sum, p) => sum + p.profitFactor, 0) / count,
      averageWin: performances.reduce((sum, p) => sum + p.averageWin, 0) / count,
      averageLoss: performances.reduce((sum, p) => sum + p.averageLoss, 0) / count,
      totalTrades: performances.reduce((sum, p) => sum + p.totalTrades, 0) / count,
      averageHoldingPeriod: performances.reduce((sum, p) => sum + p.averageHoldingPeriod, 0) / count
    };
  }

  private perturbData(
    equity: EquityCurve[],
    perturbation: MonteCarloParams['perturbation']
  ): EquityCurve[] {
    // Create perturbed version of equity curve
    return equity.map(point => ({
      ...point,
      returns: point.returns + (Math.random() - 0.5) * perturbation.returns * 2
    }));
  }

  private async simulateWithPerturbedData(
    strategy: Strategy,
    perturbedData: EquityCurve[]
  ): Promise<BacktestResult> {
    // Mock simulation with perturbed data
    // In production, this would re-run the full backtest
    const mockResult: BacktestResult = {
      strategy,
      params: {} as BacktestParams,
      performance: {
        totalReturn: (Math.random() - 0.3) * 0.4, // -30% to +10%
        annualizedReturn: (Math.random() - 0.3) * 0.4,
        volatility: Math.random() * 0.3 + 0.1,
        sharpeRatio: (Math.random() - 0.5) * 2,
        sortinoRatio: (Math.random() - 0.5) * 2,
        calmarRatio: (Math.random() - 0.5) * 2,
        maxDrawdown: Math.random() * 0.5,
        maxDrawdownDuration: Math.floor(Math.random() * 100),
        winRate: Math.random(),
        profitFactor: Math.random() * 3,
        averageWin: Math.random() * 0.05,
        averageLoss: Math.random() * 0.05,
        totalTrades: Math.floor(Math.random() * 100),
        averageHoldingPeriod: Math.floor(Math.random() * 30)
      },
      trades: [],
      equity: perturbedData,
      drawdowns: [],
      monthlyReturns: [],
      riskMetrics: {} as RiskMetrics,
      analysis: {} as BacktestAnalysis
    };
    
    return mockResult;
  }

  private calculateMonteCarloStatistics(
    iterations: MonteCarloResult['iterations']
  ): MonteCarloResult['statistics'] {
    // Helper function to calculate percentiles
    const calculatePercentiles = (values: number[]) => {
      const sorted = [...values].sort((a, b) => a - b);
      return {
        p5: sorted[Math.floor(sorted.length * 0.05)],
        p25: sorted[Math.floor(sorted.length * 0.25)],
        p50: sorted[Math.floor(sorted.length * 0.5)],
        p75: sorted[Math.floor(sorted.length * 0.75)],
        p95: sorted[Math.floor(sorted.length * 0.95)]
      };
    };
    
    // Helper function to calculate mean and std
    const calculateStats = (values: number[]) => {
      const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
      const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
      return {
        mean,
        std: Math.sqrt(variance),
        percentiles: calculatePercentiles(values)
      };
    };
    
    const totalReturns = iterations.map(iter => iter.totalReturn);
    const maxDrawdowns = iterations.map(iter => iter.maxDrawdown);
    const sharpeRatios = iterations.map(iter => iter.sharpeRatio);
    
    const profitableIterations = iterations.filter(iter => iter.totalReturn > 0).length;
    const ruinIterations = iterations.filter(iter => iter.totalReturn < -0.5).length; // 50% loss
    
    return {
      totalReturn: calculateStats(totalReturns),
      maxDrawdown: calculateStats(maxDrawdowns),
      sharpeRatio: calculateStats(sharpeRatios),
      probabilityOfProfit: profitableIterations / iterations.length,
      probabilityOfRuin: ruinIterations / iterations.length
    };
  }
}

// Export singleton instance
export const backtestingEngine = new BacktestingEngine();