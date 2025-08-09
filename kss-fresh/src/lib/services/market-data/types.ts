// Market Data Types for Real-time Stock Analysis

export interface MarketData {
  ticker: string;
  exchange: 'KRX' | 'KOSDAQ' | 'NASDAQ' | 'NYSE';
  price: number;
  currency: 'KRW' | 'USD';
  timestamp: Date;
  volume: number;
  change: number;
  changePercent: number;
  high: number;
  low: number;
  open: number;
  previousClose: number;
  marketCap?: number;
  pe?: number;
  eps?: number;
}

export interface OrderBook {
  ticker: string;
  timestamp: Date;
  bids: OrderLevel[];
  asks: OrderLevel[];
  spread: number;
  midPrice: number;
}

export interface OrderLevel {
  price: number;
  volume: number;
  orders: number;
}

export interface PriceData {
  ticker: string;
  price: number;
  timestamp: Date;
  volume: number;
  bid: number;
  ask: number;
}

export interface VolumeMetrics {
  ticker: string;
  totalVolume: number;
  avgVolume10d: number;
  avgVolume30d: number;
  volumeRatio: number;
  buyVolume: number;
  sellVolume: number;
  institutionalVolume?: number;
  retailVolume?: number;
}

export interface MarketIndicators {
  kospi: {
    index: number;
    change: number;
    changePercent: number;
  };
  kosdaq: {
    index: number;
    change: number;
    changePercent: number;
  };
  sp500?: {
    index: number;
    change: number;
    changePercent: number;
  };
  nasdaq?: {
    index: number;
    change: number;
    changePercent: number;
  };
  vix?: number;
  dollarWon?: number;
  marketBreadth: {
    advances: number;
    declines: number;
    unchanged: number;
  };
}

export interface HistoricalData {
  ticker: string;
  period: '1d' | '1w' | '1m' | '3m' | '6m' | '1y' | '3y' | '5y';
  interval: '1m' | '5m' | '15m' | '30m' | '1h' | '1d' | '1w' | '1M';
  data: OHLCV[];
}

export interface OHLCV {
  timestamp: Date;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface StockRecommendation {
  ticker: string;
  name: string;
  confidence: number; // 0-1
  expectedReturn: number;
  riskLevel: number;
  timeHorizon: '단기' | '중기' | '장기';
  reasons: string[];
  technicalSignals: TechnicalSignal[];
  fundamentalScore: number;
  newsScore: number;
  price: number;
  targetPrice: number;
  stopLoss: number;
}

export interface TechnicalSignal {
  indicator: string;
  signal: 'BUY' | 'SELL' | 'HOLD';
  strength: number; // 0-1
  value: number;
  description: string;
}

export interface AIAnalysis {
  ticker: string;
  timestamp: Date;
  pricePrediction: {
    oneDay: number;
    oneWeek: number;
    oneMonth: number;
    confidence: number;
  };
  sentiment: {
    news: number; // -1 to 1
    social: number; // -1 to 1
    analyst: number; // -1 to 1
    overall: number; // -1 to 1
  };
  riskMetrics: {
    volatility: number;
    beta: number;
    sharpeRatio: number;
    maxDrawdown: number;
    VaR95: number;
  };
  recommendation: 'STRONG_BUY' | 'BUY' | 'HOLD' | 'SELL' | 'STRONG_SELL';
}

export interface WebSocketMessage {
  type: 'price' | 'orderbook' | 'trade' | 'news' | 'alert';
  data: any;
  timestamp: Date;
}

export interface MarketDataConfig {
  exchanges: ('KRX' | 'KOSDAQ' | 'NASDAQ' | 'NYSE')[];
  subscriptions: string[]; // ticker symbols
  updateInterval: number; // milliseconds
  includeOrderBook: boolean;
  includeNews: boolean;
  enableWebSocket?: boolean;
  apiKeys?: {
    krx?: string;
    alphaVantage?: string;
    iexCloud?: string;
    newsAPI?: string;
  };
}

export type DataProvider = 'KRX' | 'ALPHA_VANTAGE' | 'IEX_CLOUD' | 'YAHOO_FINANCE';