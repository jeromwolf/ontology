// Unified Data Service
// Integrates multiple market data providers with intelligent routing and caching

import { EventEmitter } from 'events';
import { krxProvider, KrxDataProvider } from './providers/KrxDataProvider';
import { alphaVantageProvider, AlphaVantageProvider } from './providers/AlphaVantageProvider';
import { MarketData, OHLCV, OrderBook, VolumeMetrics, MarketIndicators, DataProvider } from './types';

export interface DataServiceConfig {
  enableCaching: boolean;
  cacheExpiry: number; // milliseconds
  fallbackToMock: boolean;
  enableWebSocket: boolean;
  retryAttempts: number;
  retryDelay: number;
}

export interface CacheEntry<T> {
  data: T;
  timestamp: Date;
  expiresAt: Date;
}

export class UnifiedDataService extends EventEmitter {
  private config: DataServiceConfig;
  private cache = new Map<string, CacheEntry<any>>();
  private providers: Map<DataProvider, any> = new Map();
  private subscriptions = new Set<string>();
  private wsConnections = new Map<string, WebSocket>();
  
  constructor(config: DataServiceConfig) {
    super();
    this.config = config;
    this.initializeProviders();
  }

  // Initialize data providers
  private initializeProviders(): void {
    this.providers.set('KRX', krxProvider);
    this.providers.set('ALPHA_VANTAGE', alphaVantageProvider);
  }

  // Get market data with intelligent provider routing
  async getMarketData(ticker: string): Promise<MarketData> {
    const cacheKey = `market_data_${ticker}`;
    
    // Check cache first
    if (this.config.enableCaching) {
      const cached = this.getCachedData<MarketData>(cacheKey);
      if (cached) {
        return cached;
      }
    }

    try {
      let data: MarketData;
      const provider = this.determineProvider(ticker);
      
      switch (provider) {
        case 'KRX':
          data = await this.getKrxData(ticker);
          break;
        case 'ALPHA_VANTAGE':
          data = await this.getAlphaVantageData(ticker);
          break;
        default:
          throw new Error(`No provider available for ticker: ${ticker}`);
      }

      // Cache the result
      if (this.config.enableCaching) {
        this.setCachedData(cacheKey, data);
      }

      this.emit('data', { ticker, data });
      return data;
    } catch (error) {
      console.error(`Error fetching market data for ${ticker}:`, error);
      
      if (this.config.fallbackToMock) {
        return this.generateMockData(ticker);
      }
      
      throw error;
    }
  }

  // Get historical data with provider routing
  async getHistoricalData(
    ticker: string,
    startDate: Date,
    endDate: Date,
    interval: '1D' | '5min' | '15min' | '30min' | '1H' = '1D'
  ): Promise<OHLCV[]> {
    const cacheKey = `historical_${ticker}_${startDate.toISOString()}_${endDate.toISOString()}_${interval}`;
    
    // Check cache
    if (this.config.enableCaching) {
      const cached = this.getCachedData<OHLCV[]>(cacheKey);
      if (cached) {
        return cached;
      }
    }

    try {
      let data: OHLCV[];
      const provider = this.determineProvider(ticker);
      
      switch (provider) {
        case 'KRX':
          data = await krxProvider.getHistoricalData(ticker, startDate, endDate, interval as '1D' | '1W' | '1M');
          break;
        case 'ALPHA_VANTAGE':
          // Alpha Vantage has different intervals
          const avInterval = this.mapIntervalToAlphaVantage(interval);
          if (avInterval.includes('min')) {
            data = await alphaVantageProvider.getIntradayData(ticker, avInterval as any);
          } else {
            data = await alphaVantageProvider.getHistoricalData(ticker);
          }
          // Filter by date range
          data = data.filter(item => 
            item.timestamp >= startDate && item.timestamp <= endDate
          );
          break;
        default:
          data = this.generateMockHistoricalData(ticker, startDate, endDate);
      }

      // Cache the result
      if (this.config.enableCaching) {
        this.setCachedData(cacheKey, data, 5 * 60 * 1000); // 5 minutes for historical data
      }

      return data;
    } catch (error) {
      console.error(`Error fetching historical data for ${ticker}:`, error);
      
      if (this.config.fallbackToMock) {
        return this.generateMockHistoricalData(ticker, startDate, endDate);
      }
      
      throw error;
    }
  }

  // Get order book data
  async getOrderBook(ticker: string): Promise<OrderBook> {
    const cacheKey = `orderbook_${ticker}`;
    
    // Check cache (very short expiry for order book)
    if (this.config.enableCaching) {
      const cached = this.getCachedData<OrderBook>(cacheKey);
      if (cached) {
        return cached;
      }
    }

    try {
      let data: OrderBook;
      const provider = this.determineProvider(ticker);
      
      switch (provider) {
        case 'KRX':
          data = await krxProvider.getOrderBook(ticker);
          break;
        default:
          // Alpha Vantage doesn't provide order book data
          data = this.generateMockOrderBook(ticker);
      }

      // Cache with very short expiry (10 seconds)
      if (this.config.enableCaching) {
        this.setCachedData(cacheKey, data, 10 * 1000);
      }

      return data;
    } catch (error) {
      console.error(`Error fetching order book for ${ticker}:`, error);
      return this.generateMockOrderBook(ticker);
    }
  }

  // Get volume analysis
  async getVolumeAnalysis(ticker: string): Promise<VolumeMetrics> {
    const cacheKey = `volume_${ticker}`;
    
    if (this.config.enableCaching) {
      const cached = this.getCachedData<VolumeMetrics>(cacheKey);
      if (cached) {
        return cached;
      }
    }

    try {
      // Get recent historical data to calculate volume metrics
      const endDate = new Date();
      const startDate = new Date(endDate.getTime() - 30 * 24 * 60 * 60 * 1000); // 30 days
      const historicalData = await this.getHistoricalData(ticker, startDate, endDate);
      
      const volumeMetrics = this.calculateVolumeMetrics(ticker, historicalData);
      
      if (this.config.enableCaching) {
        this.setCachedData(cacheKey, volumeMetrics, 60 * 1000); // 1 minute
      }

      return volumeMetrics;
    } catch (error) {
      console.error(`Error calculating volume analysis for ${ticker}:`, error);
      return this.generateMockVolumeMetrics(ticker);
    }
  }

  // Get market indicators
  async getMarketIndicators(): Promise<MarketIndicators> {
    const cacheKey = 'market_indicators';
    
    if (this.config.enableCaching) {
      const cached = this.getCachedData<MarketIndicators>(cacheKey);
      if (cached) {
        return cached;
      }
    }

    try {
      // Fetch major index data
      const [kospiData, kosdaqData, sp500Data, nasdaqData] = await Promise.allSettled([
        this.getMarketData('KS11'), // KOSPI
        this.getMarketData('KQ11'), // KOSDAQ
        this.getMarketData('SPY'),  // S&P 500 ETF
        this.getMarketData('QQQ')   // NASDAQ ETF
      ]);

      const indicators: MarketIndicators = {
        kospi: kospiData.status === 'fulfilled' ? {
          index: kospiData.value.price,
          change: kospiData.value.change || 0,
          changePercent: kospiData.value.changePercent || 0
        } : { index: 2500, change: 0, changePercent: 0 },
        
        kosdaq: kosdaqData.status === 'fulfilled' ? {
          index: kosdaqData.value.price,
          change: kosdaqData.value.change || 0,
          changePercent: kosdaqData.value.changePercent || 0
        } : { index: 850, change: 0, changePercent: 0 },
        
        sp500: sp500Data.status === 'fulfilled' ? {
          index: sp500Data.value.price,
          change: sp500Data.value.change || 0,
          changePercent: sp500Data.value.changePercent || 0
        } : { index: 4500, change: 0, changePercent: 0 },
        
        nasdaq: nasdaqData.status === 'fulfilled' ? {
          index: nasdaqData.value.price,
          change: nasdaqData.value.change || 0,
          changePercent: nasdaqData.value.changePercent || 0
        } : { index: 15000, change: 0, changePercent: 0 },
        
        vix: 20, // Mock VIX
        dollarWon: 1320, // Mock USD/KRW
        marketBreadth: {
          advances: 1200,
          declines: 800,
          unchanged: 100
        }
      };

      if (this.config.enableCaching) {
        this.setCachedData(cacheKey, indicators, 30 * 1000); // 30 seconds
      }

      return indicators;
    } catch (error) {
      console.error('Error fetching market indicators:', error);
      return this.generateMockMarketIndicators();
    }
  }

  // Subscribe to real-time updates
  async subscribe(ticker: string): Promise<void> {
    if (this.subscriptions.has(ticker)) {
      return;
    }

    this.subscriptions.add(ticker);
    
    if (this.config.enableWebSocket) {
      await this.setupWebSocketSubscription(ticker);
    } else {
      // Fallback to polling
      this.setupPollingSubscription(ticker);
    }
  }

  // Unsubscribe from updates
  async unsubscribe(ticker: string): Promise<void> {
    this.subscriptions.delete(ticker);
    
    const ws = this.wsConnections.get(ticker);
    if (ws) {
      ws.close();
      this.wsConnections.delete(ticker);
    }
  }

  // Private helper methods

  private determineProvider(ticker: string): DataProvider {
    // Korean stocks (6-digit codes)
    if (/^\d{6}$/.test(ticker) || ticker.includes('.KS') || ticker.includes('.KQ')) {
      return 'KRX';
    }
    
    // US stocks
    if (/^[A-Z]{1,5}$/.test(ticker)) {
      return 'ALPHA_VANTAGE';
    }
    
    // Default to Alpha Vantage for other symbols
    return 'ALPHA_VANTAGE';
  }

  private async getKrxData(ticker: string): Promise<MarketData> {
    return await this.retryOperation(() => krxProvider.getRealTimeData(ticker));
  }

  private async getAlphaVantageData(ticker: string): Promise<MarketData> {
    return await this.retryOperation(() => alphaVantageProvider.getRealTimeData(ticker));
  }

  private async retryOperation<T>(operation: () => Promise<T>): Promise<T> {
    let lastError: Error;
    
    for (let attempt = 1; attempt <= this.config.retryAttempts; attempt++) {
      try {
        return await operation();
      } catch (error) {
        lastError = error as Error;
        
        if (attempt < this.config.retryAttempts) {
          await new Promise(resolve => setTimeout(resolve, this.config.retryDelay * attempt));
        }
      }
    }
    
    throw lastError!;
  }

  private getCachedData<T>(key: string): T | null {
    const entry = this.cache.get(key);
    if (!entry) return null;
    
    if (new Date() > entry.expiresAt) {
      this.cache.delete(key);
      return null;
    }
    
    return entry.data;
  }

  private setCachedData<T>(key: string, data: T, expiry?: number): void {
    const expiryTime = expiry || this.config.cacheExpiry;
    const entry: CacheEntry<T> = {
      data,
      timestamp: new Date(),
      expiresAt: new Date(Date.now() + expiryTime)
    };
    
    this.cache.set(key, entry);
  }

  private mapIntervalToAlphaVantage(interval: string): string {
    const mapping: { [key: string]: string } = {
      '5min': '5min',
      '15min': '15min',
      '30min': '30min',
      '1H': '60min',
      '1D': 'daily'
    };
    
    return mapping[interval] || 'daily';
  }

  private calculateVolumeMetrics(ticker: string, historicalData: OHLCV[]): VolumeMetrics {
    const recent = historicalData.slice(-30); // Last 30 days
    const totalVolume = recent[recent.length - 1]?.volume || 0;
    const avgVolume10d = recent.slice(-10).reduce((sum, item) => sum + item.volume, 0) / 10;
    const avgVolume30d = recent.reduce((sum, item) => sum + item.volume, 0) / recent.length;
    
    return {
      ticker,
      totalVolume,
      avgVolume10d,
      avgVolume30d,
      volumeRatio: avgVolume10d / avgVolume30d,
      buyVolume: totalVolume * 0.52, // Mock buy/sell split
      sellVolume: totalVolume * 0.48,
      institutionalVolume: totalVolume * 0.65,
      retailVolume: totalVolume * 0.35
    };
  }

  private async setupWebSocketSubscription(ticker: string): Promise<void> {
    // Implementation would depend on provider WebSocket APIs
    console.log(`Setting up WebSocket subscription for ${ticker}`);
  }

  private setupPollingSubscription(ticker: string): void {
    const interval = setInterval(async () => {
      try {
        const data = await this.getMarketData(ticker);
        this.emit('price', { ticker, data });
      } catch (error) {
        console.error(`Polling error for ${ticker}:`, error);
      }
    }, 5000); // Poll every 5 seconds
    
    // Store interval for cleanup
    (this as any).pollingIntervals = (this as any).pollingIntervals || new Map();
    (this as any).pollingIntervals.set(ticker, interval);
  }

  // Mock data generators
  private generateMockData(ticker: string): MarketData {
    const basePrice = ticker.length === 6 ? 50000 : 150;
    const price = basePrice + (Math.random() - 0.5) * basePrice * 0.1;
    const change = (Math.random() - 0.5) * basePrice * 0.05;
    
    return {
      ticker,
      exchange: ticker.length === 6 ? 'KRX' : 'NYSE',
      price,
      currency: ticker.length === 6 ? 'KRW' : 'USD',
      timestamp: new Date(),
      volume: Math.floor(Math.random() * 1000000),
      change,
      changePercent: (change / (price - change)) * 100,
      high: price * 1.02,
      low: price * 0.98,
      open: price * (0.99 + Math.random() * 0.02),
      previousClose: price - change
    };
  }

  private generateMockHistoricalData(ticker: string, startDate: Date, endDate: Date): OHLCV[] {
    const data: OHLCV[] = [];
    const basePrice = ticker.length === 6 ? 50000 : 150;
    let currentPrice = basePrice;
    
    const currentDate = new Date(startDate);
    while (currentDate <= endDate) {
      const volatility = 0.02;
      const change = (Math.random() - 0.5) * currentPrice * volatility;
      currentPrice += change;
      
      const open = currentPrice;
      const close = currentPrice + (Math.random() - 0.5) * currentPrice * volatility;
      const high = Math.max(open, close) * (1 + Math.random() * 0.01);
      const low = Math.min(open, close) * (1 - Math.random() * 0.01);
      
      data.push({
        timestamp: new Date(currentDate),
        open,
        high,
        low,
        close,
        volume: Math.floor(Math.random() * 1000000)
      });
      
      currentDate.setDate(currentDate.getDate() + 1);
    }
    
    return data;
  }

  private generateMockOrderBook(ticker: string): OrderBook {
    const basePrice = ticker.length === 6 ? 50000 : 150;
    const spread = basePrice * 0.001;
    
    const bids = Array.from({ length: 10 }, (_, i) => ({
      price: basePrice - (i + 1) * spread,
      volume: Math.floor(Math.random() * 10000),
      orders: Math.floor(Math.random() * 50) + 1
    }));
    
    const asks = Array.from({ length: 10 }, (_, i) => ({
      price: basePrice + (i + 1) * spread,
      volume: Math.floor(Math.random() * 10000),
      orders: Math.floor(Math.random() * 50) + 1
    }));
    
    return {
      ticker,
      timestamp: new Date(),
      bids,
      asks,
      spread: asks[0].price - bids[0].price,
      midPrice: (bids[0].price + asks[0].price) / 2
    };
  }

  private generateMockVolumeMetrics(ticker: string): VolumeMetrics {
    const totalVolume = Math.floor(Math.random() * 10000000);
    
    return {
      ticker,
      totalVolume,
      avgVolume10d: totalVolume * 0.9,
      avgVolume30d: totalVolume * 0.85,
      volumeRatio: 1.1,
      buyVolume: totalVolume * 0.55,
      sellVolume: totalVolume * 0.45,
      institutionalVolume: totalVolume * 0.7,
      retailVolume: totalVolume * 0.3
    };
  }

  private generateMockMarketIndicators(): MarketIndicators {
    return {
      kospi: {
        index: 2500 + Math.random() * 100 - 50,
        change: (Math.random() - 0.5) * 50,
        changePercent: (Math.random() - 0.5) * 2
      },
      kosdaq: {
        index: 850 + Math.random() * 50 - 25,
        change: (Math.random() - 0.5) * 20,
        changePercent: (Math.random() - 0.5) * 3
      },
      sp500: {
        index: 4500 + Math.random() * 100 - 50,
        change: (Math.random() - 0.5) * 50,
        changePercent: (Math.random() - 0.5) * 1.5
      },
      nasdaq: {
        index: 15000 + Math.random() * 500 - 250,
        change: (Math.random() - 0.5) * 200,
        changePercent: (Math.random() - 0.5) * 2.5
      },
      vix: 15 + Math.random() * 10,
      dollarWon: 1300 + Math.random() * 50,
      marketBreadth: {
        advances: Math.floor(Math.random() * 1000),
        declines: Math.floor(Math.random() * 1000),
        unchanged: Math.floor(Math.random() * 100)
      }
    };
  }

  // Cleanup
  disconnect(): void {
    this.subscriptions.clear();
    this.cache.clear();
    
    // Close WebSocket connections
    for (const [ticker, ws] of this.wsConnections) {
      ws.close();
    }
    this.wsConnections.clear();
    
    // Clear polling intervals
    if ((this as any).pollingIntervals) {
      for (const [ticker, interval] of (this as any).pollingIntervals) {
        clearInterval(interval);
      }
      (this as any).pollingIntervals.clear();
    }
    
    this.emit('disconnected');
  }

  // Get cache statistics
  getCacheStats(): { size: number; hitRate: number } {
    return {
      size: this.cache.size,
      hitRate: 0.85 // Mock hit rate
    };
  }
}

// Export singleton instance
export const unifiedDataService = new UnifiedDataService({
  enableCaching: true,
  cacheExpiry: 60 * 1000, // 1 minute default
  fallbackToMock: true,
  enableWebSocket: false, // Enable when WebSocket endpoints are available
  retryAttempts: 3,
  retryDelay: 1000
});