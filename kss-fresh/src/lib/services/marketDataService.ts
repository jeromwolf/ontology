// 실시간 시장 데이터 서비스
import { ChartData, OrderBookEntry } from '@/components/charts/ProChart/types';

export interface MarketDataConfig {
  symbol: string;
  interval: string;
  onData: (data: ChartData) => void;
  onOrderBook?: (bids: OrderBookEntry[], asks: OrderBookEntry[]) => void;
  onTrade?: (trade: TradeData) => void;
  onError?: (error: Error) => void;
}

export interface TradeData {
  price: number;
  quantity: number;
  type: 'buy' | 'sell';
  timestamp: Date;
}

export class MarketDataService {
  private ws: WebSocket | null = null;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private config: MarketDataConfig;
  private isConnected: boolean = false;
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number = 5;
  private reconnectDelay: number = 1000;

  constructor(config: MarketDataConfig) {
    this.config = config;
  }

  // WebSocket 연결
  connect() {
    try {
      // 실제 서비스에서는 실제 WebSocket 서버 URL 사용
      // 예: wss://api.exchange.com/ws
      this.ws = new WebSocket('wss://demo.piesocket.com/v3/channel_123?api_key=VCXCEuvhGcBDP7XhiJJUDvR1e1D3eiVjgZ9VRiaV');
      
      this.ws.onopen = () => {
        console.log('WebSocket connected');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        
        // 구독 메시지 전송
        this.subscribe();
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.handleMessage(data);
        } catch (error) {
          console.error('Failed to parse message:', error);
        }
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.config.onError?.(new Error('WebSocket connection error'));
      };

      this.ws.onclose = () => {
        console.log('WebSocket disconnected');
        this.isConnected = false;
        this.attemptReconnect();
      };
    } catch (error) {
      console.error('Failed to connect:', error);
      this.config.onError?.(error as Error);
    }
  }

  // 구독 메시지 전송
  private subscribe() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      // 실제 서비스에서는 실제 구독 메시지 형식 사용
      this.ws.send(JSON.stringify({
        type: 'subscribe',
        channels: [
          `ticker.${this.config.symbol}`,
          `orderbook.${this.config.symbol}`,
          `trades.${this.config.symbol}`
        ]
      }));
    }
  }

  // 메시지 처리
  private handleMessage(data: any) {
    switch (data.type) {
      case 'ticker':
        this.handleTickerData(data);
        break;
      case 'orderbook':
        this.handleOrderBookData(data);
        break;
      case 'trade':
        this.handleTradeData(data);
        break;
      default:
        // 데모 데이터 생성
        this.generateDemoData();
    }
  }

  // 틱 데이터 처리
  private handleTickerData(data: any) {
    const chartData: ChartData = {
      time: new Date().toISOString(),
      open: data.open || 0,
      high: data.high || 0,
      low: data.low || 0,
      close: data.close || 0,
      volume: data.volume || 0,
      ma5: data.ma5,
      ma20: data.ma20,
      ma60: data.ma60,
      rsi: data.rsi,
      macd: data.macd,
      signal: data.signal,
      histogram: data.histogram
    };
    
    this.config.onData(chartData);
  }

  // 호가 데이터 처리
  private handleOrderBookData(data: any) {
    if (this.config.onOrderBook) {
      this.config.onOrderBook(data.bids || [], data.asks || []);
    }
  }

  // 체결 데이터 처리
  private handleTradeData(data: any) {
    if (this.config.onTrade) {
      const trade: TradeData = {
        price: data.price || 0,
        quantity: data.quantity || 0,
        type: data.side === 'sell' ? 'sell' : 'buy',
        timestamp: new Date(data.timestamp || Date.now())
      };
      
      this.config.onTrade(trade);
    }
  }

  // 데모 데이터 생성 (실제 서비스에서는 제거)
  private generateDemoData() {
    setInterval(() => {
      if (!this.isConnected) return;

      const lastPrice = 69000 + Math.random() * 1000;
      const change = (Math.random() - 0.5) * 100;
      
      const chartData: ChartData = {
        time: new Date().toISOString(),
        open: lastPrice,
        high: lastPrice + Math.random() * 50,
        low: lastPrice - Math.random() * 50,
        close: lastPrice + change,
        volume: Math.floor(Math.random() * 1000000),
        ma5: lastPrice - 50,
        ma20: lastPrice - 100,
        ma60: lastPrice - 150,
        rsi: 50 + (Math.random() - 0.5) * 30,
        macd: (Math.random() - 0.5) * 100,
        signal: (Math.random() - 0.5) * 50,
        histogram: (Math.random() - 0.5) * 25
      };
      
      this.config.onData(chartData);
    }, 1000);
  }

  // 재연결 시도
  private attemptReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      this.config.onError?.(new Error('Unable to reconnect to market data'));
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
    
    console.log(`Attempting to reconnect in ${delay}ms... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
    
    this.reconnectTimer = setTimeout(() => {
      this.connect();
    }, delay);
  }

  // 연결 해제
  disconnect() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    
    this.isConnected = false;
  }

  // 연결 상태 확인
  isConnected() {
    return this.isConnected && this.ws?.readyState === WebSocket.OPEN;
  }

  // 심볼 변경
  changeSymbol(symbol: string) {
    this.config.symbol = symbol;
    
    if (this.isConnected && this.ws) {
      // 기존 구독 해제
      this.ws.send(JSON.stringify({
        type: 'unsubscribe',
        channels: [
          `ticker.${this.config.symbol}`,
          `orderbook.${this.config.symbol}`,
          `trades.${this.config.symbol}`
        ]
      }));
      
      // 새 심볼 구독
      this.config.symbol = symbol;
      this.subscribe();
    }
  }
}