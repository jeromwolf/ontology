'use client';

import { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import { ArrowLeft, TrendingUp, Calculator, ChartLine, BarChart3, Shield, Activity, Settings, Download, Maximize2, RefreshCw, Info, ChevronRight, DollarSign, Percent, Clock, Target, AlertTriangle, Layers, LineChart } from 'lucide-react';

interface OptionLeg {
  id: string;
  type: 'call' | 'put';
  position: 'long' | 'short';
  strike: number;
  premium: number;
  quantity: number;
  expiry: string;
}

interface Strategy {
  name: string;
  legs: OptionLeg[];
  description: string;
}

interface Greeks {
  delta: number;
  gamma: number;
  theta: number;
  vega: number;
  rho: number;
}

interface PayoffPoint {
  price: number;
  profit: number;
}

interface VolatilitySurface {
  strike: number;
  expiry: number;
  iv: number;
}

export default function OptionsStrategyAnalyzerPage() {
  const [selectedStrategy, setSelectedStrategy] = useState('covered-call');
  const [spotPrice, setSpotPrice] = useState(150);
  const [volatility, setVolatility] = useState(25);
  const [riskFreeRate, setRiskFreeRate] = useState(5);
  const [daysToExpiry, setDaysToExpiry] = useState(30);
  const [legs, setLegs] = useState<OptionLeg[]>([]);
  const [greeks, setGreeks] = useState<Greeks>({ delta: 0, gamma: 0, theta: 0, vega: 0, rho: 0 });
  const [payoffData, setPayoffData] = useState<PayoffPoint[]>([]);
  const [probProfit, setProbProfit] = useState(0);
  const [maxProfit, setMaxProfit] = useState(0);
  const [maxLoss, setMaxLoss] = useState(0);
  const [breakeven, setBreakeven] = useState<number[]>([]);
  const [selectedTab, setSelectedTab] = useState('builder');
  
  const payoffCanvasRef = useRef<HTMLCanvasElement>(null);
  const volatilityCanvasRef = useRef<HTMLCanvasElement>(null);
  const timeDecayCanvasRef = useRef<HTMLCanvasElement>(null);
  const skewCanvasRef = useRef<HTMLCanvasElement>(null);

  // Predefined strategies
  const strategies: Record<string, Strategy> = {
    'covered-call': {
      name: 'Covered Call',
      description: 'Long stock + Short call. Income generation strategy.',
      legs: [
        { id: '1', type: 'call', position: 'short', strike: 155, premium: 2.5, quantity: -1, expiry: '30' }
      ]
    },
    'protective-put': {
      name: 'Protective Put',
      description: 'Long stock + Long put. Downside protection strategy.',
      legs: [
        { id: '1', type: 'put', position: 'long', strike: 145, premium: 2.0, quantity: 1, expiry: '30' }
      ]
    },
    'bull-call-spread': {
      name: 'Bull Call Spread',
      description: 'Long call + Short call at higher strike. Bullish strategy with limited risk.',
      legs: [
        { id: '1', type: 'call', position: 'long', strike: 150, premium: 3.5, quantity: 1, expiry: '30' },
        { id: '2', type: 'call', position: 'short', strike: 155, premium: 1.5, quantity: -1, expiry: '30' }
      ]
    },
    'bear-put-spread': {
      name: 'Bear Put Spread',
      description: 'Long put + Short put at lower strike. Bearish strategy with limited risk.',
      legs: [
        { id: '1', type: 'put', position: 'long', strike: 150, premium: 3.0, quantity: 1, expiry: '30' },
        { id: '2', type: 'put', position: 'short', strike: 145, premium: 1.5, quantity: -1, expiry: '30' }
      ]
    },
    'iron-condor': {
      name: 'Iron Condor',
      description: 'Bull put spread + Bear call spread. Neutral strategy for range-bound markets.',
      legs: [
        { id: '1', type: 'put', position: 'short', strike: 140, premium: 1.0, quantity: -1, expiry: '30' },
        { id: '2', type: 'put', position: 'long', strike: 135, premium: 0.5, quantity: 1, expiry: '30' },
        { id: '3', type: 'call', position: 'short', strike: 160, premium: 1.0, quantity: -1, expiry: '30' },
        { id: '4', type: 'call', position: 'long', strike: 165, premium: 0.5, quantity: 1, expiry: '30' }
      ]
    },
    'butterfly': {
      name: 'Long Butterfly',
      description: 'Long 1 ITM call, Short 2 ATM calls, Long 1 OTM call. Low volatility strategy.',
      legs: [
        { id: '1', type: 'call', position: 'long', strike: 145, premium: 6.0, quantity: 1, expiry: '30' },
        { id: '2', type: 'call', position: 'short', strike: 150, premium: 3.5, quantity: -2, expiry: '30' },
        { id: '3', type: 'call', position: 'long', strike: 155, premium: 1.5, quantity: 1, expiry: '30' }
      ]
    },
    'straddle': {
      name: 'Long Straddle',
      description: 'Long call + Long put at same strike. High volatility strategy.',
      legs: [
        { id: '1', type: 'call', position: 'long', strike: 150, premium: 3.5, quantity: 1, expiry: '30' },
        { id: '2', type: 'put', position: 'long', strike: 150, premium: 3.0, quantity: 1, expiry: '30' }
      ]
    },
    'strangle': {
      name: 'Long Strangle',
      description: 'Long OTM call + Long OTM put. High volatility strategy with lower cost.',
      legs: [
        { id: '1', type: 'call', position: 'long', strike: 155, premium: 1.5, quantity: 1, expiry: '30' },
        { id: '2', type: 'put', position: 'long', strike: 145, premium: 1.5, quantity: 1, expiry: '30' }
      ]
    },
    'calendar': {
      name: 'Calendar Spread',
      description: 'Short near-term option + Long far-term option at same strike.',
      legs: [
        { id: '1', type: 'call', position: 'short', strike: 150, premium: 3.5, quantity: -1, expiry: '30' },
        { id: '2', type: 'call', position: 'long', strike: 150, premium: 5.0, quantity: 1, expiry: '60' }
      ]
    },
    'diagonal': {
      name: 'Diagonal Spread',
      description: 'Calendar spread with different strikes. Time decay + directional play.',
      legs: [
        { id: '1', type: 'call', position: 'short', strike: 155, premium: 1.5, quantity: -1, expiry: '30' },
        { id: '2', type: 'call', position: 'long', strike: 150, premium: 5.0, quantity: 1, expiry: '60' }
      ]
    }
  };

  // Load strategy when selection changes
  useEffect(() => {
    const strategy = strategies[selectedStrategy];
    if (strategy) {
      setLegs(strategy.legs.map(leg => ({ ...leg, id: Math.random().toString() })));
    }
  }, [selectedStrategy]);

  // Calculate Black-Scholes option price
  const blackScholes = (S: number, K: number, T: number, r: number, sigma: number, type: 'call' | 'put'): number => {
    const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
    const d2 = d1 - sigma * Math.sqrt(T);
    
    const cumulativeNormal = (x: number): number => {
      return 0.5 * (1 + erf(x / Math.sqrt(2)));
    };
    
    const erf = (x: number): number => {
      const a1 =  0.254829592;
      const a2 = -0.284496736;
      const a3 =  1.421413741;
      const a4 = -1.453152027;
      const a5 =  1.061405429;
      const p  =  0.3275911;
      const sign = x >= 0 ? 1 : -1;
      x = Math.abs(x);
      const t = 1.0 / (1.0 + p * x);
      const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
      return sign * y;
    };
    
    if (type === 'call') {
      return S * cumulativeNormal(d1) - K * Math.exp(-r * T) * cumulativeNormal(d2);
    } else {
      return K * Math.exp(-r * T) * cumulativeNormal(-d2) - S * cumulativeNormal(-d1);
    }
  };

  // Calculate Greeks
  const calculateGreeks = () => {
    let totalDelta = 0;
    let totalGamma = 0;
    let totalTheta = 0;
    let totalVega = 0;
    let totalRho = 0;
    
    const T = daysToExpiry / 365;
    const r = riskFreeRate / 100;
    const sigma = volatility / 100;
    
    legs.forEach(leg => {
      const d1 = (Math.log(spotPrice / leg.strike) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
      const d2 = d1 - sigma * Math.sqrt(T);
      
      const normPDF = (x: number) => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
      const normCDF = (x: number) => 0.5 * (1 + erf(x / Math.sqrt(2)));
      const erf = (x: number): number => {
        const sign = x >= 0 ? 1 : -1;
        x = Math.abs(x);
        const a1 =  0.254829592;
        const a2 = -0.284496736;
        const a3 =  1.421413741;
        const a4 = -1.453152027;
        const a5 =  1.061405429;
        const p  =  0.3275911;
        const t = 1.0 / (1.0 + p * x);
        const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
        return sign * y;
      };
      
      const multiplier = leg.quantity;
      
      // Delta
      let delta = leg.type === 'call' ? normCDF(d1) : normCDF(d1) - 1;
      totalDelta += delta * multiplier * 100;
      
      // Gamma
      const gamma = normPDF(d1) / (spotPrice * sigma * Math.sqrt(T));
      totalGamma += gamma * multiplier * 100;
      
      // Theta
      let theta;
      if (leg.type === 'call') {
        theta = -(spotPrice * normPDF(d1) * sigma) / (2 * Math.sqrt(T)) - r * leg.strike * Math.exp(-r * T) * normCDF(d2);
      } else {
        theta = -(spotPrice * normPDF(d1) * sigma) / (2 * Math.sqrt(T)) + r * leg.strike * Math.exp(-r * T) * normCDF(-d2);
      }
      totalTheta += (theta / 365) * multiplier * 100;
      
      // Vega
      const vega = spotPrice * normPDF(d1) * Math.sqrt(T);
      totalVega += (vega / 100) * multiplier * 100;
      
      // Rho
      let rho;
      if (leg.type === 'call') {
        rho = leg.strike * T * Math.exp(-r * T) * normCDF(d2);
      } else {
        rho = -leg.strike * T * Math.exp(-r * T) * normCDF(-d2);
      }
      totalRho += (rho / 100) * multiplier * 100;
    });
    
    setGreeks({
      delta: totalDelta,
      gamma: totalGamma,
      theta: totalTheta,
      vega: totalVega,
      rho: totalRho
    });
  };

  // Calculate payoff data
  const calculatePayoff = () => {
    const priceRange = [];
    const step = spotPrice * 0.01;
    const start = spotPrice * 0.7;
    const end = spotPrice * 1.3;
    
    for (let price = start; price <= end; price += step) {
      priceRange.push(price);
    }
    
    const payoffPoints: PayoffPoint[] = priceRange.map(price => {
      let totalProfit = 0;
      
      legs.forEach(leg => {
        const intrinsicValue = leg.type === 'call' 
          ? Math.max(0, price - leg.strike)
          : Math.max(0, leg.strike - price);
        
        const profit = (intrinsicValue - leg.premium) * leg.quantity * 100;
        totalProfit += profit;
      });
      
      return { price, profit: totalProfit };
    });
    
    setPayoffData(payoffPoints);
    
    // Calculate max profit, max loss, and breakeven
    const profits = payoffPoints.map(p => p.profit);
    const maxP = Math.max(...profits);
    const maxL = Math.min(...profits);
    setMaxProfit(maxP);
    setMaxLoss(maxL);
    
    // Find breakeven points
    const breakevens: number[] = [];
    for (let i = 1; i < payoffPoints.length; i++) {
      if (payoffPoints[i - 1].profit * payoffPoints[i].profit <= 0) {
        // Linear interpolation for precise breakeven
        const ratio = Math.abs(payoffPoints[i - 1].profit) / (Math.abs(payoffPoints[i - 1].profit) + Math.abs(payoffPoints[i].profit));
        const be = payoffPoints[i - 1].price + ratio * (payoffPoints[i].price - payoffPoints[i - 1].price);
        breakevens.push(be);
      }
    }
    setBreakeven(breakevens);
    
    // Calculate probability of profit (simplified using normal distribution)
    const vol = volatility / 100;
    const timeToExp = daysToExpiry / 365;
    const drift = (riskFreeRate / 100 - 0.5 * vol * vol) * timeToExp;
    const diffusion = vol * Math.sqrt(timeToExp);
    
    let probProfit = 0;
    if (breakevens.length > 0) {
      const z = (Math.log(breakevens[0] / spotPrice) - drift) / diffusion;
      probProfit = selectedStrategy.includes('bear') ? normCDF(z) : 1 - normCDF(z);
    }
    setProbProfit(probProfit * 100);
  };

  const normCDF = (x: number) => {
    const erf = (x: number): number => {
      const sign = x >= 0 ? 1 : -1;
      x = Math.abs(x);
      const a1 =  0.254829592;
      const a2 = -0.284496736;
      const a3 =  1.421413741;
      const a4 = -1.453152027;
      const a5 =  1.061405429;
      const p  =  0.3275911;
      const t = 1.0 / (1.0 + p * x);
      const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
      return sign * y;
    };
    return 0.5 * (1 + erf(x / Math.sqrt(2)));
  };

  // Update calculations when parameters change
  useEffect(() => {
    calculateGreeks();
    calculatePayoff();
  }, [legs, spotPrice, volatility, riskFreeRate, daysToExpiry]);

  // Draw payoff diagram
  useEffect(() => {
    if (!payoffCanvasRef.current || payoffData.length === 0) return;
    
    const canvas = payoffCanvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Clear canvas
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Calculate bounds
    const minPrice = Math.min(...payoffData.map(p => p.price));
    const maxPrice = Math.max(...payoffData.map(p => p.price));
    const minProfit = Math.min(...payoffData.map(p => p.profit));
    const maxProfit = Math.max(...payoffData.map(p => p.profit));
    const profitRange = maxProfit - minProfit;
    
    const padding = 40;
    const width = canvas.width - 2 * padding;
    const height = canvas.height - 2 * padding;
    
    // Draw grid
    ctx.strokeStyle = '#1f2937';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      const x = padding + (i / 10) * width;
      const y = padding + (i / 10) * height;
      
      // Vertical lines
      ctx.beginPath();
      ctx.moveTo(x, padding);
      ctx.lineTo(x, canvas.height - padding);
      ctx.stroke();
      
      // Horizontal lines
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(canvas.width - padding, y);
      ctx.stroke();
    }
    
    // Draw axes
    ctx.strokeStyle = '#6b7280';
    ctx.lineWidth = 2;
    
    // X-axis (at y=0)
    const zeroY = padding + height * (maxProfit / profitRange);
    ctx.beginPath();
    ctx.moveTo(padding, zeroY);
    ctx.lineTo(canvas.width - padding, zeroY);
    ctx.stroke();
    
    // Y-axis
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, canvas.height - padding);
    ctx.stroke();
    
    // Draw payoff curve
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 3;
    ctx.beginPath();
    
    payoffData.forEach((point, i) => {
      const x = padding + ((point.price - minPrice) / (maxPrice - minPrice)) * width;
      const y = padding + height - ((point.profit - minProfit) / profitRange) * height;
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    
    ctx.stroke();
    
    // Draw current spot price line
    const spotX = padding + ((spotPrice - minPrice) / (maxPrice - minPrice)) * width;
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(spotX, padding);
    ctx.lineTo(spotX, canvas.height - padding);
    ctx.stroke();
    ctx.setLineDash([]);
    
    // Draw breakeven lines
    breakeven.forEach(be => {
      const beX = padding + ((be - minPrice) / (maxPrice - minPrice)) * width;
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 2;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(beX, padding);
      ctx.lineTo(beX, canvas.height - padding);
      ctx.stroke();
    });
    ctx.setLineDash([]);
    
    // Draw labels
    ctx.fillStyle = '#e5e7eb';
    ctx.font = '12px Inter';
    ctx.textAlign = 'center';
    
    // X-axis labels
    for (let i = 0; i <= 5; i++) {
      const price = minPrice + (i / 5) * (maxPrice - minPrice);
      const x = padding + (i / 5) * width;
      ctx.fillText(`$${price.toFixed(0)}`, x, canvas.height - padding + 20);
    }
    
    // Y-axis labels
    ctx.textAlign = 'right';
    for (let i = 0; i <= 5; i++) {
      const profit = minProfit + (i / 5) * profitRange;
      const y = padding + height - (i / 5) * height;
      ctx.fillText(`$${profit.toFixed(0)}`, padding - 10, y + 4);
    }
  }, [payoffData, spotPrice, breakeven]);

  // Draw volatility surface
  useEffect(() => {
    if (!volatilityCanvasRef.current) return;
    
    const canvas = volatilityCanvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Generate volatility surface data
    const strikes = [];
    const expiries = [];
    for (let i = 0.8; i <= 1.2; i += 0.05) {
      strikes.push(spotPrice * i);
    }
    for (let i = 7; i <= 90; i += 7) {
      expiries.push(i);
    }
    
    const cellWidth = canvas.width / strikes.length;
    const cellHeight = canvas.height / expiries.length;
    
    // Draw heatmap
    strikes.forEach((strike, i) => {
      expiries.forEach((expiry, j) => {
        const moneyness = strike / spotPrice;
        const timeRatio = expiry / 365;
        
        // Simulate IV smile/skew
        const baseIV = volatility / 100;
        const skew = 0.1 * Math.pow(moneyness - 1, 2);
        const termStructure = 0.05 * Math.sqrt(timeRatio);
        const iv = baseIV + skew + termStructure + (Math.random() - 0.5) * 0.02;
        
        const hue = 240 - iv * 1000; // Blue to red
        const lightness = 50 + iv * 100;
        ctx.fillStyle = `hsl(${hue}, 70%, ${lightness}%)`;
        ctx.fillRect(i * cellWidth, j * cellHeight, cellWidth - 1, cellHeight - 1);
      });
    });
    
    // Draw labels
    ctx.fillStyle = '#e5e7eb';
    ctx.font = '10px Inter';
    ctx.textAlign = 'center';
    
    // Strike labels
    strikes.forEach((strike, i) => {
      if (i % 2 === 0) {
        ctx.fillText(`${(strike / spotPrice).toFixed(2)}`, (i + 0.5) * cellWidth, canvas.height - 5);
      }
    });
    
    // Expiry labels
    ctx.textAlign = 'right';
    expiries.forEach((expiry, j) => {
      if (j % 2 === 0) {
        ctx.fillText(`${expiry}d`, 25, (j + 0.5) * cellHeight + 3);
      }
    });
  }, [spotPrice, volatility]);

  // Draw time decay chart
  useEffect(() => {
    if (!timeDecayCanvasRef.current) return;
    
    const canvas = timeDecayCanvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    const padding = 30;
    const width = canvas.width - 2 * padding;
    const height = canvas.height - 2 * padding;
    
    // Calculate option values at different times
    const times = [];
    const values = [];
    
    for (let days = daysToExpiry; days >= 0; days -= 1) {
      times.push(days);
      let totalValue = 0;
      
      legs.forEach(leg => {
        const T = days / 365;
        const r = riskFreeRate / 100;
        const sigma = volatility / 100;
        
        if (days === 0) {
          // At expiry
          const intrinsic = leg.type === 'call' 
            ? Math.max(0, spotPrice - leg.strike)
            : Math.max(0, leg.strike - spotPrice);
          totalValue += intrinsic * leg.quantity;
        } else {
          const optionValue = blackScholes(spotPrice, leg.strike, T, r, sigma, leg.type);
          totalValue += optionValue * leg.quantity;
        }
      });
      
      values.push(totalValue);
    }
    
    // Draw grid
    ctx.strokeStyle = '#1f2937';
    ctx.lineWidth = 1;
    
    for (let i = 0; i <= 5; i++) {
      const x = padding + (i / 5) * width;
      const y = padding + (i / 5) * height;
      
      ctx.beginPath();
      ctx.moveTo(x, padding);
      ctx.lineTo(x, canvas.height - padding);
      ctx.stroke();
      
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(canvas.width - padding, y);
      ctx.stroke();
    }
    
    // Draw decay curve
    const maxValue = Math.max(...values);
    const minValue = Math.min(...values);
    const valueRange = maxValue - minValue || 1;
    
    ctx.strokeStyle = '#8b5cf6';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    times.forEach((time, i) => {
      const x = padding + ((daysToExpiry - time) / daysToExpiry) * width;
      const y = padding + height - ((values[i] - minValue) / valueRange) * height;
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    
    ctx.stroke();
    
    // Draw labels
    ctx.fillStyle = '#e5e7eb';
    ctx.font = '11px Inter';
    ctx.textAlign = 'center';
    
    // X-axis (days to expiry)
    for (let i = 0; i <= 5; i++) {
      const days = daysToExpiry * (1 - i / 5);
      ctx.fillText(`${days.toFixed(0)}d`, padding + (i / 5) * width, canvas.height - padding + 15);
    }
    
    // Y-axis (value)
    ctx.textAlign = 'right';
    for (let i = 0; i <= 5; i++) {
      const value = minValue + (i / 5) * valueRange;
      ctx.fillText(`$${value.toFixed(0)}`, padding - 5, padding + height - (i / 5) * height + 3);
    }
  }, [legs, spotPrice, volatility, riskFreeRate, daysToExpiry]);

  // Draw volatility smile/skew
  useEffect(() => {
    if (!skewCanvasRef.current) return;
    
    const canvas = skewCanvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    const padding = 30;
    const width = canvas.width - 2 * padding;
    const height = canvas.height - 2 * padding;
    
    // Generate smile data
    const strikes = [];
    const ivs = [];
    
    for (let i = 0.7; i <= 1.3; i += 0.02) {
      const strike = spotPrice * i;
      strikes.push(strike);
      
      // Simulate realistic volatility smile
      const moneyness = i;
      const atmVol = volatility / 100;
      const skew = 0.15 * Math.pow(moneyness - 1, 2);
      const asymmetry = moneyness < 1 ? 0.05 * (1 - moneyness) : 0;
      const iv = atmVol + skew + asymmetry;
      
      ivs.push(iv * 100);
    }
    
    // Draw grid
    ctx.strokeStyle = '#1f2937';
    ctx.lineWidth = 1;
    
    for (let i = 0; i <= 5; i++) {
      const x = padding + (i / 5) * width;
      const y = padding + (i / 5) * height;
      
      ctx.beginPath();
      ctx.moveTo(x, padding);
      ctx.lineTo(x, canvas.height - padding);
      ctx.stroke();
      
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(canvas.width - padding, y);
      ctx.stroke();
    }
    
    // Draw smile curve
    const minIV = Math.min(...ivs);
    const maxIV = Math.max(...ivs);
    const ivRange = maxIV - minIV;
    
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    strikes.forEach((strike, i) => {
      const x = padding + ((strike - strikes[0]) / (strikes[strikes.length - 1] - strikes[0])) * width;
      const y = padding + height - ((ivs[i] - minIV) / ivRange) * height;
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    
    ctx.stroke();
    
    // Mark current spot
    const spotX = padding + ((spotPrice - strikes[0]) / (strikes[strikes.length - 1] - strikes[0])) * width;
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(spotX, padding);
    ctx.lineTo(spotX, canvas.height - padding);
    ctx.stroke();
    ctx.setLineDash([]);
    
    // Draw labels
    ctx.fillStyle = '#e5e7eb';
    ctx.font = '11px Inter';
    ctx.textAlign = 'center';
    
    // X-axis (strikes)
    for (let i = 0; i <= 5; i++) {
      const strike = strikes[0] + (i / 5) * (strikes[strikes.length - 1] - strikes[0]);
      ctx.fillText(`$${strike.toFixed(0)}`, padding + (i / 5) * width, canvas.height - padding + 15);
    }
    
    // Y-axis (IV)
    ctx.textAlign = 'right';
    for (let i = 0; i <= 5; i++) {
      const iv = minIV + (i / 5) * ivRange;
      ctx.fillText(`${iv.toFixed(1)}%`, padding - 5, padding + height - (i / 5) * height + 3);
    }
  }, [spotPrice, volatility]);

  const addLeg = () => {
    const newLeg: OptionLeg = {
      id: Math.random().toString(),
      type: 'call',
      position: 'long',
      strike: spotPrice,
      premium: 3.0,
      quantity: 1,
      expiry: '30'
    };
    setLegs([...legs, newLeg]);
  };

  const updateLeg = (id: string, field: keyof OptionLeg, value: any) => {
    setLegs(legs.map(leg => 
      leg.id === id ? { ...leg, [field]: value } : leg
    ));
  };

  const removeLeg = (id: string) => {
    setLegs(legs.filter(leg => leg.id !== id));
  };

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      {/* Header */}
      <div className="bg-gray-900 border-b border-gray-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link 
                href="/modules/stock-analysis/tools"
                className="inline-flex items-center gap-2 text-gray-400 hover:text-white transition-colors"
              >
                <ArrowLeft className="w-5 h-5" />
                <span>도구 목록</span>
              </Link>
              <div className="h-6 w-px bg-gray-700" />
              <h1 className="text-xl font-bold">Options Strategy Analyzer</h1>
              <span className="px-2 py-1 bg-purple-500/20 text-purple-400 rounded text-xs font-medium">
                Professional
              </span>
            </div>
            
            <div className="flex items-center gap-4">
              <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg font-medium transition-colors flex items-center gap-2">
                <Download className="w-4 h-4" />
                Export Analysis
              </button>
              <button className="p-2 hover:bg-gray-800 rounded-lg transition-colors">
                <Settings className="w-5 h-5" />
              </button>
              <button className="p-2 hover:bg-gray-800 rounded-lg transition-colors">
                <Maximize2 className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="flex h-[calc(100vh-64px)]">
        {/* Left Sidebar - Strategy Builder */}
        <div className="w-80 bg-gray-900 border-r border-gray-800 p-4 overflow-y-auto">
          <div className="space-y-4">
            {/* Strategy Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">Strategy Template</label>
              <select
                className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
                value={selectedStrategy}
                onChange={(e) => setSelectedStrategy(e.target.value)}
              >
                {Object.entries(strategies).map(([key, strategy]) => (
                  <option key={key} value={key}>{strategy.name}</option>
                ))}
              </select>
              <p className="text-xs text-gray-500 mt-1">
                {strategies[selectedStrategy]?.description}
              </p>
            </div>

            {/* Market Parameters */}
            <div className="space-y-3">
              <h3 className="text-sm font-medium text-gray-400">Market Parameters</h3>
              
              <div>
                <label className="text-xs text-gray-500">Spot Price</label>
                <input
                  type="number"
                  className="w-full px-3 py-1.5 bg-gray-800 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                  value={spotPrice}
                  onChange={(e) => setSpotPrice(Number(e.target.value))}
                />
              </div>
              
              <div>
                <label className="text-xs text-gray-500">Implied Volatility (%)</label>
                <input
                  type="number"
                  className="w-full px-3 py-1.5 bg-gray-800 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                  value={volatility}
                  onChange={(e) => setVolatility(Number(e.target.value))}
                />
              </div>
              
              <div>
                <label className="text-xs text-gray-500">Risk-Free Rate (%)</label>
                <input
                  type="number"
                  className="w-full px-3 py-1.5 bg-gray-800 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                  value={riskFreeRate}
                  onChange={(e) => setRiskFreeRate(Number(e.target.value))}
                />
              </div>
              
              <div>
                <label className="text-xs text-gray-500">Days to Expiry</label>
                <input
                  type="number"
                  className="w-full px-3 py-1.5 bg-gray-800 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
                  value={daysToExpiry}
                  onChange={(e) => setDaysToExpiry(Number(e.target.value))}
                />
              </div>
            </div>

            {/* Option Legs */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-medium text-gray-400">Option Legs</h3>
                <button
                  onClick={addLeg}
                  className="text-xs px-2 py-1 bg-blue-600 hover:bg-blue-700 rounded transition-colors"
                >
                  Add Leg
                </button>
              </div>
              
              <div className="space-y-2">
                {legs.map((leg, index) => (
                  <div key={leg.id} className="p-3 bg-gray-800 rounded-lg space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Leg {index + 1}</span>
                      <button
                        onClick={() => removeLeg(leg.id)}
                        className="text-red-400 hover:text-red-300 text-xs"
                      >
                        Remove
                      </button>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-2">
                      <select
                        className="px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs"
                        value={leg.type}
                        onChange={(e) => updateLeg(leg.id, 'type', e.target.value)}
                      >
                        <option value="call">Call</option>
                        <option value="put">Put</option>
                      </select>
                      
                      <select
                        className="px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs"
                        value={leg.position}
                        onChange={(e) => updateLeg(leg.id, 'position', e.target.value)}
                      >
                        <option value="long">Long</option>
                        <option value="short">Short</option>
                      </select>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <label className="text-xs text-gray-500">Strike</label>
                        <input
                          type="number"
                          className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs"
                          value={leg.strike}
                          onChange={(e) => updateLeg(leg.id, 'strike', Number(e.target.value))}
                        />
                      </div>
                      
                      <div>
                        <label className="text-xs text-gray-500">Premium</label>
                        <input
                          type="number"
                          step="0.1"
                          className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs"
                          value={leg.premium}
                          onChange={(e) => updateLeg(leg.id, 'premium', Number(e.target.value))}
                        />
                      </div>
                    </div>
                    
                    <div>
                      <label className="text-xs text-gray-500">Quantity</label>
                      <input
                        type="number"
                        className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs"
                        value={Math.abs(leg.quantity)}
                        onChange={(e) => updateLeg(leg.id, 'quantity', leg.position === 'short' ? -Math.abs(Number(e.target.value)) : Math.abs(Number(e.target.value)))}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 flex flex-col">
          {/* Tabs */}
          <div className="flex border-b border-gray-800 bg-gray-900">
            {['builder', 'greeks', 'analysis', 'comparison'].map(tab => (
              <button
                key={tab}
                onClick={() => setSelectedTab(tab)}
                className={`px-6 py-3 text-sm font-medium capitalize transition-colors ${
                  selectedTab === tab
                    ? 'text-blue-400 border-b-2 border-blue-400'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                {tab === 'builder' ? 'Payoff Diagram' : tab}
              </button>
            ))}
          </div>

          {/* Tab Content */}
          <div className="flex-1 p-4">
            {selectedTab === 'builder' && (
              <div className="grid grid-cols-2 gap-4 h-full">
                {/* Payoff Diagram */}
                <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
                  <div className="flex items-center justify-between mb-3">
                    <h2 className="text-lg font-semibold">Payoff Diagram</h2>
                    <div className="flex items-center gap-4 text-sm">
                      <span className="flex items-center gap-2">
                        <div className="w-3 h-3 bg-blue-500 rounded-sm"></div>
                        Strategy P&L
                      </span>
                      <span className="flex items-center gap-2">
                        <div className="w-3 h-3 bg-amber-500 rounded-sm"></div>
                        Current Price
                      </span>
                    </div>
                  </div>
                  <canvas
                    ref={payoffCanvasRef}
                    width={600}
                    height={400}
                    className="w-full h-full rounded"
                  />
                </div>

                {/* Key Metrics */}
                <div className="space-y-4">
                  <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
                    <h3 className="text-lg font-semibold mb-3">Strategy Metrics</h3>
                    <div className="space-y-3">
                      <div className="flex justify-between items-center p-3 bg-gray-800 rounded">
                        <span className="text-sm text-gray-400">Max Profit</span>
                        <span className={`font-mono font-medium ${maxProfit > 0 ? 'text-green-400' : 'text-red-400'}`}>
                          ${maxProfit.toFixed(2)}
                        </span>
                      </div>
                      
                      <div className="flex justify-between items-center p-3 bg-gray-800 rounded">
                        <span className="text-sm text-gray-400">Max Loss</span>
                        <span className={`font-mono font-medium ${maxLoss < 0 ? 'text-red-400' : 'text-green-400'}`}>
                          ${maxLoss.toFixed(2)}
                        </span>
                      </div>
                      
                      <div className="flex justify-between items-center p-3 bg-gray-800 rounded">
                        <span className="text-sm text-gray-400">Breakeven</span>
                        <span className="font-mono font-medium text-blue-400">
                          {breakeven.length > 0 ? breakeven.map(be => `$${be.toFixed(2)}`).join(', ') : 'N/A'}
                        </span>
                      </div>
                      
                      <div className="flex justify-between items-center p-3 bg-gray-800 rounded">
                        <span className="text-sm text-gray-400">Probability of Profit</span>
                        <span className={`font-mono font-medium ${probProfit > 50 ? 'text-green-400' : 'text-yellow-400'}`}>
                          {probProfit.toFixed(1)}%
                        </span>
                      </div>
                      
                      <div className="flex justify-between items-center p-3 bg-gray-800 rounded">
                        <span className="text-sm text-gray-400">Risk/Reward Ratio</span>
                        <span className="font-mono font-medium">
                          {maxLoss !== 0 ? `1:${Math.abs(maxProfit / maxLoss).toFixed(2)}` : 'N/A'}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Time Decay Chart */}
                  <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
                    <h3 className="text-lg font-semibold mb-3">Time Decay (Theta)</h3>
                    <canvas
                      ref={timeDecayCanvasRef}
                      width={400}
                      height={200}
                      className="w-full h-full rounded"
                    />
                  </div>
                </div>
              </div>
            )}

            {selectedTab === 'greeks' && (
              <div className="grid grid-cols-2 gap-4 h-full">
                {/* Greeks Display */}
                <div className="space-y-4">
                  <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
                    <h2 className="text-lg font-semibold mb-4">Position Greeks</h2>
                    <div className="space-y-4">
                      {[
                        { name: 'Delta', value: greeks.delta, description: 'Price sensitivity', icon: TrendingUp },
                        { name: 'Gamma', value: greeks.gamma, description: 'Delta rate of change', icon: Activity },
                        { name: 'Theta', value: greeks.theta, description: 'Time decay', icon: Clock },
                        { name: 'Vega', value: greeks.vega, description: 'Volatility sensitivity', icon: BarChart3 },
                        { name: 'Rho', value: greeks.rho, description: 'Interest rate sensitivity', icon: Percent }
                      ].map(greek => {
                        const Icon = greek.icon;
                        return (
                          <div key={greek.name} className="p-4 bg-gray-800 rounded-lg">
                            <div className="flex items-center justify-between mb-2">
                              <div className="flex items-center gap-3">
                                <Icon className="w-5 h-5 text-gray-400" />
                                <div>
                                  <h4 className="font-medium">{greek.name}</h4>
                                  <p className="text-xs text-gray-500">{greek.description}</p>
                                </div>
                              </div>
                              <span className={`text-xl font-mono font-bold ${
                                greek.value > 0 ? 'text-green-400' : greek.value < 0 ? 'text-red-400' : 'text-gray-400'
                              }`}>
                                {greek.value.toFixed(4)}
                              </span>
                            </div>
                            <div className="w-full h-1 bg-gray-700 rounded-full overflow-hidden">
                              <div 
                                className={`h-full transition-all ${
                                  greek.value > 0 ? 'bg-green-500' : 'bg-red-500'
                                }`}
                                style={{ width: `${Math.min(Math.abs(greek.value) * 20, 100)}%` }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Greeks Sensitivity */}
                  <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
                    <h3 className="text-lg font-semibold mb-3">Greeks Sensitivity Analysis</h3>
                    <div className="space-y-3 text-sm">
                      <div className="p-3 bg-gray-800 rounded">
                        <p className="text-gray-400">If underlying moves +$1:</p>
                        <p className="font-mono">P&L Change: ${(greeks.delta).toFixed(2)}</p>
                      </div>
                      <div className="p-3 bg-gray-800 rounded">
                        <p className="text-gray-400">If volatility increases 1%:</p>
                        <p className="font-mono">P&L Change: ${(greeks.vega).toFixed(2)}</p>
                      </div>
                      <div className="p-3 bg-gray-800 rounded">
                        <p className="text-gray-400">Daily time decay:</p>
                        <p className="font-mono">P&L Change: ${(greeks.theta).toFixed(2)}</p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Volatility Surface */}
                <div className="space-y-4">
                  <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
                    <h3 className="text-lg font-semibold mb-3">Implied Volatility Surface</h3>
                    <canvas
                      ref={volatilityCanvasRef}
                      width={500}
                      height={300}
                      className="w-full h-full rounded"
                    />
                    <div className="mt-3 flex items-center justify-center gap-4 text-xs text-gray-400">
                      <span>Moneyness →</span>
                      <span className="flex items-center gap-2">
                        <div className="w-3 h-3 bg-gradient-to-r from-blue-600 to-red-600 rounded"></div>
                        Low to High IV
                      </span>
                    </div>
                  </div>

                  {/* Volatility Smile */}
                  <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
                    <h3 className="text-lg font-semibold mb-3">Volatility Smile/Skew</h3>
                    <canvas
                      ref={skewCanvasRef}
                      width={500}
                      height={200}
                      className="w-full h-full rounded"
                    />
                  </div>
                </div>
              </div>
            )}

            {selectedTab === 'analysis' && (
              <div className="grid grid-cols-3 gap-4 h-full">
                {/* Scenario Analysis */}
                <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
                  <h3 className="text-lg font-semibold mb-3">Scenario Analysis</h3>
                  <div className="space-y-2">
                    {[
                      { scenario: 'Bull Market (+10%)', price: spotPrice * 1.1 },
                      { scenario: 'Mild Bull (+5%)', price: spotPrice * 1.05 },
                      { scenario: 'Neutral (0%)', price: spotPrice },
                      { scenario: 'Mild Bear (-5%)', price: spotPrice * 0.95 },
                      { scenario: 'Bear Market (-10%)', price: spotPrice * 0.9 },
                      { scenario: 'Market Crash (-20%)', price: spotPrice * 0.8 }
                    ].map(({ scenario, price }) => {
                      const profit = payoffData.find(p => Math.abs(p.price - price) < 1)?.profit || 0;
                      return (
                        <div key={scenario} className="p-3 bg-gray-800 rounded flex justify-between items-center">
                          <div>
                            <p className="text-sm font-medium">{scenario}</p>
                            <p className="text-xs text-gray-500">${price.toFixed(2)}</p>
                          </div>
                          <span className={`font-mono font-medium ${
                            profit > 0 ? 'text-green-400' : 'text-red-400'
                          }`}>
                            ${profit.toFixed(2)}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Risk Analysis */}
                <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
                  <h3 className="text-lg font-semibold mb-3">Risk Analysis</h3>
                  <div className="space-y-3">
                    <div className="p-3 bg-gray-800 rounded">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-gray-400">Value at Risk (95%)</span>
                        <AlertTriangle className="w-4 h-4 text-yellow-400" />
                      </div>
                      <p className="text-lg font-mono font-bold text-red-400">
                        ${(maxLoss * 0.95).toFixed(2)}
                      </p>
                    </div>
                    
                    <div className="p-3 bg-gray-800 rounded">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-gray-400">Expected Value</span>
                        <Target className="w-4 h-4 text-blue-400" />
                      </div>
                      <p className="text-lg font-mono font-bold">
                        ${((maxProfit * probProfit / 100) + (maxLoss * (1 - probProfit / 100))).toFixed(2)}
                      </p>
                    </div>
                    
                    <div className="p-3 bg-gray-800 rounded">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-gray-400">Sharpe Ratio</span>
                        <LineChart className="w-4 h-4 text-green-400" />
                      </div>
                      <p className="text-lg font-mono font-bold">
                        {((maxProfit + maxLoss) / Math.abs(maxLoss) * Math.sqrt(252 / daysToExpiry)).toFixed(2)}
                      </p>
                    </div>
                    
                    <div className="p-3 bg-gray-800 rounded">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-gray-400">Kelly Criterion</span>
                        <Calculator className="w-4 h-4 text-purple-400" />
                      </div>
                      <p className="text-lg font-mono font-bold">
                        {Math.max(0, (probProfit / 100 - (1 - probProfit / 100)) / (maxProfit / Math.abs(maxLoss)) * 100).toFixed(1)}%
                      </p>
                      <p className="text-xs text-gray-500 mt-1">Optimal position size</p>
                    </div>
                  </div>
                </div>

                {/* Market Conditions */}
                <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
                  <h3 className="text-lg font-semibold mb-3">Optimal Market Conditions</h3>
                  <div className="space-y-3">
                    <div className="p-3 bg-gray-800 rounded">
                      <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
                        <TrendingUp className="w-4 h-4" />
                        Direction
                      </h4>
                      <p className="text-sm text-gray-400">
                        {greeks.delta > 0.3 ? 'Bullish' : greeks.delta < -0.3 ? 'Bearish' : 'Neutral'}
                      </p>
                    </div>
                    
                    <div className="p-3 bg-gray-800 rounded">
                      <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
                        <Activity className="w-4 h-4" />
                        Volatility
                      </h4>
                      <p className="text-sm text-gray-400">
                        {greeks.vega > 0 ? 'Rising volatility favorable' : 'Falling volatility favorable'}
                      </p>
                    </div>
                    
                    <div className="p-3 bg-gray-800 rounded">
                      <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
                        <Clock className="w-4 h-4" />
                        Time Decay
                      </h4>
                      <p className="text-sm text-gray-400">
                        {greeks.theta > 0 ? 'Time decay favorable' : 'Time decay unfavorable'}
                      </p>
                    </div>
                    
                    <div className="p-3 bg-gray-800 rounded">
                      <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
                        <Shield className="w-4 h-4" />
                        Risk Profile
                      </h4>
                      <p className="text-sm text-gray-400">
                        {maxLoss === 0 ? 'Risk-free' : Math.abs(maxLoss) < 1000 ? 'Limited risk' : 'High risk'}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {selectedTab === 'comparison' && (
              <div className="bg-gray-900 rounded-lg border border-gray-800 p-6">
                <h2 className="text-lg font-semibold mb-4">Strategy Comparison</h2>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-gray-800">
                        <th className="text-left py-3 px-4">Strategy</th>
                        <th className="text-right py-3 px-4">Max Profit</th>
                        <th className="text-right py-3 px-4">Max Loss</th>
                        <th className="text-right py-3 px-4">Breakeven</th>
                        <th className="text-right py-3 px-4">Prob. of Profit</th>
                        <th className="text-right py-3 px-4">Delta</th>
                        <th className="text-right py-3 px-4">Theta</th>
                        <th className="text-center py-3 px-4">Complexity</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(strategies).map(([key, strategy]) => {
                        const isActive = key === selectedStrategy;
                        return (
                          <tr 
                            key={key} 
                            className={`border-b border-gray-800/50 cursor-pointer transition-colors ${
                              isActive ? 'bg-blue-500/10' : 'hover:bg-gray-800/50'
                            }`}
                            onClick={() => setSelectedStrategy(key)}
                          >
                            <td className="py-3 px-4">
                              <div>
                                <p className="font-medium">{strategy.name}</p>
                                <p className="text-xs text-gray-500">{strategy.legs.length} legs</p>
                              </div>
                            </td>
                            <td className="text-right py-3 px-4 font-mono text-green-400">
                              {isActive ? `$${maxProfit.toFixed(2)}` : '--'}
                            </td>
                            <td className="text-right py-3 px-4 font-mono text-red-400">
                              {isActive ? `$${maxLoss.toFixed(2)}` : '--'}
                            </td>
                            <td className="text-right py-3 px-4 font-mono">
                              {isActive && breakeven.length > 0 ? `$${breakeven[0].toFixed(2)}` : '--'}
                            </td>
                            <td className="text-right py-3 px-4 font-mono">
                              {isActive ? `${probProfit.toFixed(1)}%` : '--'}
                            </td>
                            <td className="text-right py-3 px-4 font-mono">
                              {isActive ? greeks.delta.toFixed(3) : '--'}
                            </td>
                            <td className="text-right py-3 px-4 font-mono">
                              {isActive ? greeks.theta.toFixed(3) : '--'}
                            </td>
                            <td className="text-center py-3 px-4">
                              <span className={`px-2 py-1 rounded text-xs ${
                                strategy.legs.length <= 2 ? 'bg-green-500/20 text-green-400' :
                                strategy.legs.length <= 4 ? 'bg-yellow-500/20 text-yellow-400' :
                                'bg-red-500/20 text-red-400'
                              }`}>
                                {strategy.legs.length <= 2 ? 'Simple' :
                                 strategy.legs.length <= 4 ? 'Moderate' : 'Complex'}
                              </span>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}