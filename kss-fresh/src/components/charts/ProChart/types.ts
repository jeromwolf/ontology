// 재사용 가능한 차트 타입 정의
import { Time } from 'lightweight-charts';

export interface ChartData {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  // 이동평균선
  ma5?: number;
  ma20?: number;
  ma60?: number;
  ma120?: number;
  // 지수이동평균
  ema12?: number;
  ema26?: number;
  // MACD
  macd?: number;
  signal?: number;
  histogram?: number;
  // RSI
  rsi?: number;
  // 볼린저밴드
  bollinger_upper?: number;
  bollinger_middle?: number;
  bollinger_lower?: number;
  // 스토캐스틱
  stochastic_k?: number;
  stochastic_d?: number;
  // 추가 지표들
  adx?: number;
  cci?: number;
  williams_r?: number;
  obv?: number;
}

export interface Indicator {
  id: string;
  name: string;
  type: 'overlay' | 'oscillator' | 'volume';
  enabled: boolean;
  params: Record<string, any>;
  color: string;
  lineWidth?: number;
}

export interface DrawingTool {
  id: string;
  type: 'trendline' | 'horizontal' | 'vertical' | 'fibonacci' | 'channel' | 'rectangle' | 'ellipse' | 'text' | 'arrow' | 'pitchfork';
  points: { x: number; y: number; time?: string; price?: number }[];
  color: string;
  lineWidth: number;
  lineStyle?: 'solid' | 'dashed' | 'dotted';
  text?: string;
  fillColor?: string;
  locked?: boolean;
}

export interface OrderLevel {
  id: string;
  type: 'buy' | 'sell' | 'stop_loss' | 'take_profit' | 'limit';
  price: number;
  quantity: number;
  filled: number;
  timestamp?: Date;
  status?: 'pending' | 'partial' | 'filled' | 'cancelled';
}

export interface OrderBookEntry {
  price: number;
  quantity: number;
  total: number;
  percentage: number;
}

export interface MarketDepth {
  bids: OrderBookEntry[];
  asks: OrderBookEntry[];
  spread: number;
  midPrice: number;
}

export interface ChartTheme {
  background: string;
  text: string;
  grid: string;
  candleUp: string;
  candleDown: string;
  volumeUp: string;
  volumeDown: string;
  crosshair: string;
}