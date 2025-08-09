// Real-time Market Data Service
// B2B-grade service for financial institutions
// Now powered by UnifiedDataService with multiple providers

import { EventEmitter } from 'events';
import { unifiedDataService, UnifiedDataService } from './UnifiedDataService';
import { 
  MarketData, 
  OrderBook, 
  PriceData, 
  VolumeMetrics, 
  MarketIndicators,
  HistoricalData,
  WebSocketMessage,
  MarketDataConfig,
  OHLCV
} from './types';

export class RealTimeDataService extends EventEmitter {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 5000;
  private subscriptions = new Set<string>();
  private cache = new Map<string, MarketData>();
  private config: MarketDataConfig;
  private unifiedService: UnifiedDataService;
  
  constructor(config: MarketDataConfig) {
    super();
    this.config = config;
    this.unifiedService = unifiedDataService;
    this.setupUnifiedServiceListeners();
  }

  // Setup listeners for unified service events
  private setupUnifiedServiceListeners(): void {
    this.unifiedService.on('data', (event) => {
      this.cache.set(event.ticker, event.data);
      this.emit('price', event.data);
    });
    
    this.unifiedService.on('error', (error) => {
      this.emit('error', error);
    });
  }

  // Initialize connection to data providers
  async connect(): Promise<void> {
    try {
      // The unified service handles provider connections internally
      // We just need to set up our WebSocket if enabled
      if (this.config.enableWebSocket !== false) {
        await this.connectWebSocket();
      }
      await this.initializeDataFeeds();
      this.emit('connected');
    } catch (error) {
      this.emit('error', error);
      throw error;
    }
  }

  // Connect to WebSocket for real-time data
  private async connectWebSocket(): Promise<void> {
    return new Promise((resolve, reject) => {
      // In production, this would connect to actual market data providers
      // For now, we'll simulate with a mock WebSocket server
      const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'wss://api.kss-finance.com/v1/stream';
      
      this.ws = new WebSocket(wsUrl);
      
      this.ws.onopen = () => {
        console.log('WebSocket connected');
        this.reconnectAttempts = 0;
        this.authenticateWebSocket();
        resolve();
      };
      
      this.ws.onmessage = (event) => {
        this.handleWebSocketMessage(event.data);
      };
      
      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.emit('error', error);
      };
      
      this.ws.onclose = () => {
        console.log('WebSocket disconnected');
        this.handleReconnect();
      };
    });
  }

  // Authenticate WebSocket connection
  private authenticateWebSocket(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'auth',
        apiKey: this.config.apiKeys?.krx || process.env.NEXT_PUBLIC_API_KEY
      }));
    }
  }

  // Handle incoming WebSocket messages
  private handleWebSocketMessage(data: string): void {
    try {
      const message: WebSocketMessage = JSON.parse(data);
      
      switch (message.type) {
        case 'price':
          this.handlePriceUpdate(message.data);
          break;
        case 'orderbook':
          this.handleOrderBookUpdate(message.data);
          break;
        case 'trade':
          this.handleTradeUpdate(message.data);
          break;
        case 'news':
          this.handleNewsUpdate(message.data);
          break;
        case 'alert':
          this.handleAlert(message.data);
          break;
      }
    } catch (error) {
      console.error('Error parsing WebSocket message:', error);
    }
  }

  // Handle price updates
  private handlePriceUpdate(data: PriceData): void {
    const ticker = data.ticker;
    const cached = this.cache.get(ticker);
    
    if (cached) {
      // Update cache with new price data
      cached.price = data.price;
      cached.timestamp = data.timestamp;
      cached.volume = data.volume;
      
      // Calculate change
      if (cached.previousClose) {
        cached.change = data.price - cached.previousClose;
        cached.changePercent = (cached.change / cached.previousClose) * 100;
      }
    }
    
    this.emit('price', data);
  }

  // Handle order book updates
  private handleOrderBookUpdate(data: OrderBook): void {
    this.emit('orderbook', data);
  }

  // Handle trade updates
  private handleTradeUpdate(data: any): void {
    this.emit('trade', data);
  }

  // Handle news updates
  private handleNewsUpdate(data: any): void {
    this.emit('news', data);
  }

  // Handle alerts
  private handleAlert(data: any): void {
    this.emit('alert', data);
  }

  // Reconnect logic
  private handleReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`Reconnecting... Attempt ${this.reconnectAttempts}`);
      
      setTimeout(() => {
        this.connect().catch(console.error);
      }, this.reconnectDelay * this.reconnectAttempts);
    } else {
      this.emit('error', new Error('Max reconnection attempts reached'));
    }
  }

  // Initialize data feeds
  private async initializeDataFeeds(): Promise<void> {
    // Initialize connections to various data providers
    // This would include KRX, KOSCOM, Alpha Vantage, etc.
    
    // Subscribe to initial tickers
    for (const ticker of this.config.subscriptions) {
      await this.subscribe(ticker);
    }
  }

  // Subscribe to real-time data for specific tickers
  async subscribe(ticker: string): Promise<void> {
    if (this.subscriptions.has(ticker)) {
      return; // Already subscribed
    }
    
    this.subscriptions.add(ticker);
    
    // Subscribe through unified service
    await this.unifiedService.subscribe(ticker);
    
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'subscribe',
        tickers: [ticker],
        channels: ['price', 'orderbook', 'trade']
      }));
    }
    
    // Fetch initial data
    await this.fetchInitialData(ticker);
  }

  // Unsubscribe from ticker
  async unsubscribe(ticker: string): Promise<void> {
    this.subscriptions.delete(ticker);
    
    // Unsubscribe from unified service
    await this.unifiedService.unsubscribe(ticker);
    
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'unsubscribe',
        tickers: [ticker]
      }));
    }
  }

  // Fetch initial data for a ticker
  private async fetchInitialData(ticker: string): Promise<void> {
    try {
      // In production, this would fetch from actual APIs
      // For now, we'll simulate with mock data
      const marketData = await this.fetchMarketData(ticker);
      this.cache.set(ticker, marketData);
      this.emit('initial-data', marketData);
    } catch (error) {
      console.error(`Error fetching initial data for ${ticker}:`, error);
    }
  }

  // Fetch market data using unified service
  async fetchMarketData(ticker: string): Promise<MarketData> {
    try {
      return await this.unifiedService.getMarketData(ticker);
    } catch (error) {
      console.error(`Error fetching market data for ${ticker}:`, error);
      throw error;
    }
  }

  // Get order book using unified service
  async getOrderBook(ticker: string): Promise<OrderBook> {
    try {
      return await this.unifiedService.getOrderBook(ticker);
    } catch (error) {
      console.error(`Error fetching order book for ${ticker}:`, error);
      throw error;
    }
  }

  // Get volume analysis using unified service
  async getVolumeAnalysis(ticker: string): Promise<VolumeMetrics> {
    try {
      return await this.unifiedService.getVolumeAnalysis(ticker);
    } catch (error) {
      console.error(`Error fetching volume analysis for ${ticker}:`, error);
      throw error;
    }
  }

  // Get market indicators using unified service
  async getMarketIndicators(): Promise<MarketIndicators> {
    try {
      return await this.unifiedService.getMarketIndicators();
    } catch (error) {
      console.error('Error fetching market indicators:', error);
      throw error;
    }
  }

  // Get historical data using unified service
  async getHistoricalData(
    ticker: string, 
    period: HistoricalData['period'],
    interval: HistoricalData['interval']
  ): Promise<HistoricalData> {
    try {
      const endDate = new Date();
      const startDate = this.getStartDateFromPeriod(period);
      const ohlcvData = await this.unifiedService.getHistoricalData(ticker, startDate, endDate, interval);
      
      return {
        ticker,
        period,
        interval,
        data: ohlcvData
      };
    } catch (error) {
      console.error(`Error fetching historical data for ${ticker}:`, error);
      throw error;
    }
  }

  // Helper to get start date from period
  private getStartDateFromPeriod(period: HistoricalData['period']): Date {
    const now = new Date();
    const periodDays = {
      '1d': 1, '1w': 7, '1m': 30, '3m': 90, 
      '6m': 180, '1y': 365, '3y': 1095, '5y': 1825
    };
    
    const days = periodDays[period] || 30;
    return new Date(now.getTime() - days * 24 * 60 * 60 * 1000);
  }

  // Helper: Get exchange from ticker
  private getExchange(ticker: string): MarketData['exchange'] {
    if (ticker.startsWith('A')) return 'NYSE';
    if (ticker.length === 6 && /^\d+$/.test(ticker)) return 'KRX';
    if (ticker.includes('.KQ')) return 'KOSDAQ';
    return 'NASDAQ';
  }

  // Helper: Get data points for period
  private getDataPointsForPeriod(
    period: HistoricalData['period'], 
    interval: HistoricalData['interval']
  ): number {
    const periodDays = {
      '1d': 1, '1w': 7, '1m': 30, '3m': 90, 
      '6m': 180, '1y': 365, '3y': 1095, '5y': 1825
    };
    
    const intervalMinutes = {
      '1m': 1, '5m': 5, '15m': 15, '30m': 30,
      '1h': 60, '1d': 1440, '1w': 10080, '1M': 43200
    };
    
    const days = periodDays[period];
    const minutes = intervalMinutes[interval];
    
    return Math.floor((days * 24 * 60) / minutes);
  }

  // Helper: Get interval in milliseconds
  private getIntervalMillis(interval: HistoricalData['interval']): number {
    const map = {
      '1m': 60000, '5m': 300000, '15m': 900000, '30m': 1800000,
      '1h': 3600000, '1d': 86400000, '1w': 604800000, '1M': 2592000000
    };
    return map[interval];
  }

  // Get current market data
  getMarketData(ticker: string): MarketData | undefined {
    return this.cache.get(ticker);
  }

  // Get all subscribed tickers
  getSubscriptions(): string[] {
    return Array.from(this.subscriptions);
  }

  // Disconnect service
  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.subscriptions.clear();
    this.cache.clear();
    this.emit('disconnected');
  }
}

// Export singleton instance for easy use
export const marketDataService = new RealTimeDataService({
  exchanges: ['KRX', 'KOSDAQ', 'NYSE', 'NASDAQ'],
  subscriptions: [],
  updateInterval: 1000,
  includeOrderBook: true,
  includeNews: true
});