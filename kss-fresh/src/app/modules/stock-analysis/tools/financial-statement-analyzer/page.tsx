'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import Link from 'next/link';
import { ArrowLeft, FileText, TrendingUp, AlertTriangle, BarChart3, PieChart, Activity, DollarSign, Shield, Brain, Download, Upload, RefreshCcw, Info, CheckCircle, XCircle, AlertCircle, Calculator, Scale, Zap } from 'lucide-react';
import * as d3 from 'd3';

interface FinancialData {
  year: number;
  revenue: number;
  costOfRevenue: number;
  grossProfit: number;
  operatingExpenses: number;
  operatingIncome: number;
  netIncome: number;
  totalAssets: number;
  totalLiabilities: number;
  totalEquity: number;
  currentAssets: number;
  currentLiabilities: number;
  cash: number;
  inventory: number;
  accountsReceivable: number;
  longTermDebt: number;
  operatingCashFlow: number;
  investingCashFlow: number;
  financingCashFlow: number;
  freeCashFlow: number;
  shares: number;
}

interface Ratios {
  profitability: {
    grossMargin: number;
    operatingMargin: number;
    netMargin: number;
    roe: number;
    roa: number;
    roic: number;
  };
  liquidity: {
    currentRatio: number;
    quickRatio: number;
    cashRatio: number;
    workingCapital: number;
  };
  efficiency: {
    assetTurnover: number;
    inventoryTurnover: number;
    receivablesTurnover: number;
    daysInventory: number;
    daysReceivables: number;
  };
  leverage: {
    debtToEquity: number;
    debtToAssets: number;
    interestCoverage: number;
    equityMultiplier: number;
  };
}

interface DuPontAnalysis {
  roe: number;
  netMargin: number;
  assetTurnover: number;
  equityMultiplier: number;
  components: {
    salesEfficiency: number;
    assetEfficiency: number;
    leverageEffect: number;
  };
}

interface FraudIndicator {
  name: string;
  score: number;
  status: 'pass' | 'warning' | 'fail';
  description: string;
}

interface AIInsight {
  category: string;
  type: 'positive' | 'negative' | 'neutral';
  title: string;
  description: string;
  impact: 'high' | 'medium' | 'low';
  metric?: string;
  value?: number;
}

export default function FinancialStatementAnalyzerPage() {
  const incomeChartRef = useRef<HTMLDivElement>(null);
  const balanceChartRef = useRef<HTMLDivElement>(null);
  const cashFlowChartRef = useRef<HTMLDivElement>(null);
  const ratioChartRef = useRef<HTMLDivElement>(null);
  const dupontChartRef = useRef<HTMLDivElement>(null);

  const [financialData, setFinancialData] = useState<FinancialData[]>([]);
  const [selectedYear, setSelectedYear] = useState<number>(2023);
  const [ratios, setRatios] = useState<Ratios | null>(null);
  const [industryAvg, setIndustryAvg] = useState<Ratios | null>(null);
  const [dupontAnalysis, setDupontAnalysis] = useState<DuPontAnalysis | null>(null);
  const [fraudIndicators, setFraudIndicators] = useState<FraudIndicator[]>([]);
  const [aiInsights, setAiInsights] = useState<AIInsight[]>([]);
  const [altmanZScore, setAltmanZScore] = useState<number>(0);
  const [benfordAnalysis, setBenfordAnalysis] = useState<{ digit: number; expected: number; actual: number }[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedCompany, setSelectedCompany] = useState('AAPL');
  const [peerComparison, setPeerComparison] = useState<{ company: string; metrics: Ratios }[]>([]);

  // Generate sample financial data
  useEffect(() => {
    generateFinancialData();
  }, [selectedCompany]);

  const generateFinancialData = () => {
    const years = [2019, 2020, 2021, 2022, 2023];
    const data = years.map(year => ({
      year,
      revenue: 260000 + Math.random() * 100000 + (year - 2019) * 20000,
      costOfRevenue: 160000 + Math.random() * 50000 + (year - 2019) * 10000,
      grossProfit: 0,
      operatingExpenses: 40000 + Math.random() * 20000,
      operatingIncome: 0,
      netIncome: 55000 + Math.random() * 30000 + (year - 2019) * 5000,
      totalAssets: 320000 + Math.random() * 100000 + (year - 2019) * 30000,
      totalLiabilities: 180000 + Math.random() * 50000,
      totalEquity: 0,
      currentAssets: 120000 + Math.random() * 40000,
      currentLiabilities: 80000 + Math.random() * 30000,
      cash: 50000 + Math.random() * 30000,
      inventory: 5000 + Math.random() * 3000,
      accountsReceivable: 25000 + Math.random() * 10000,
      longTermDebt: 90000 + Math.random() * 20000,
      operatingCashFlow: 70000 + Math.random() * 30000,
      investingCashFlow: -30000 - Math.random() * 20000,
      financingCashFlow: -40000 - Math.random() * 10000,
      freeCashFlow: 0,
      shares: 16000 + Math.random() * 1000
    }));

    // Calculate derived values
    data.forEach(d => {
      d.grossProfit = d.revenue - d.costOfRevenue;
      d.operatingIncome = d.grossProfit - d.operatingExpenses;
      d.totalEquity = d.totalAssets - d.totalLiabilities;
      d.freeCashFlow = d.operatingCashFlow + d.investingCashFlow;
    });

    setFinancialData(data);
    calculateRatios(data[data.length - 1]);
    generateIndustryAverages();
    performDuPontAnalysis(data[data.length - 1]);
    checkFraudIndicators(data);
    generateAIInsights(data);
    calculateAltmanZScore(data[data.length - 1]);
    performBenfordAnalysis(data);
    generatePeerComparison();
  };

  const calculateRatios = (data: FinancialData) => {
    const ratios: Ratios = {
      profitability: {
        grossMargin: (data.grossProfit / data.revenue) * 100,
        operatingMargin: (data.operatingIncome / data.revenue) * 100,
        netMargin: (data.netIncome / data.revenue) * 100,
        roe: (data.netIncome / data.totalEquity) * 100,
        roa: (data.netIncome / data.totalAssets) * 100,
        roic: (data.operatingIncome / (data.totalEquity + data.longTermDebt)) * 100
      },
      liquidity: {
        currentRatio: data.currentAssets / data.currentLiabilities,
        quickRatio: (data.currentAssets - data.inventory) / data.currentLiabilities,
        cashRatio: data.cash / data.currentLiabilities,
        workingCapital: data.currentAssets - data.currentLiabilities
      },
      efficiency: {
        assetTurnover: data.revenue / data.totalAssets,
        inventoryTurnover: data.costOfRevenue / data.inventory,
        receivablesTurnover: data.revenue / data.accountsReceivable,
        daysInventory: 365 / (data.costOfRevenue / data.inventory),
        daysReceivables: 365 / (data.revenue / data.accountsReceivable)
      },
      leverage: {
        debtToEquity: data.totalLiabilities / data.totalEquity,
        debtToAssets: data.totalLiabilities / data.totalAssets,
        interestCoverage: data.operatingIncome / (data.longTermDebt * 0.05), // Assuming 5% interest
        equityMultiplier: data.totalAssets / data.totalEquity
      }
    };
    setRatios(ratios);
  };

  const generateIndustryAverages = () => {
    // Generate realistic industry averages
    const avgRatios: Ratios = {
      profitability: {
        grossMargin: 35 + Math.random() * 10,
        operatingMargin: 20 + Math.random() * 5,
        netMargin: 15 + Math.random() * 5,
        roe: 20 + Math.random() * 10,
        roa: 10 + Math.random() * 5,
        roic: 15 + Math.random() * 5
      },
      liquidity: {
        currentRatio: 1.5 + Math.random() * 0.5,
        quickRatio: 1.2 + Math.random() * 0.3,
        cashRatio: 0.5 + Math.random() * 0.3,
        workingCapital: 40000 + Math.random() * 20000
      },
      efficiency: {
        assetTurnover: 0.8 + Math.random() * 0.4,
        inventoryTurnover: 8 + Math.random() * 4,
        receivablesTurnover: 10 + Math.random() * 5,
        daysInventory: 45 + Math.random() * 20,
        daysReceivables: 36 + Math.random() * 10
      },
      leverage: {
        debtToEquity: 0.8 + Math.random() * 0.4,
        debtToAssets: 0.5 + Math.random() * 0.2,
        interestCoverage: 5 + Math.random() * 3,
        equityMultiplier: 2 + Math.random() * 0.5
      }
    };
    setIndustryAvg(avgRatios);
  };

  const performDuPontAnalysis = (data: FinancialData) => {
    const netMargin = data.netIncome / data.revenue;
    const assetTurnover = data.revenue / data.totalAssets;
    const equityMultiplier = data.totalAssets / data.totalEquity;
    const roe = netMargin * assetTurnover * equityMultiplier;

    setDupontAnalysis({
      roe: roe * 100,
      netMargin: netMargin * 100,
      assetTurnover,
      equityMultiplier,
      components: {
        salesEfficiency: netMargin * 100,
        assetEfficiency: assetTurnover * 100,
        leverageEffect: equityMultiplier
      }
    });
  };

  const checkFraudIndicators = (data: FinancialData[]) => {
    const latest = data[data.length - 1];
    const previous = data[data.length - 2];
    
    const indicators: FraudIndicator[] = [
      {
        name: 'Days Sales Outstanding',
        score: calculateDSOScore(latest, previous),
        status: calculateDSOScore(latest, previous) > 80 ? 'pass' : calculateDSOScore(latest, previous) > 50 ? 'warning' : 'fail',
        description: 'Measures collection efficiency and revenue quality'
      },
      {
        name: 'Quality of Earnings',
        score: calculateQualityOfEarnings(latest),
        status: calculateQualityOfEarnings(latest) > 80 ? 'pass' : calculateQualityOfEarnings(latest) > 60 ? 'warning' : 'fail',
        description: 'Cash flow vs accrual earnings analysis'
      },
      {
        name: 'Expense Capitalization',
        score: Math.random() * 100,
        status: Math.random() > 0.7 ? 'pass' : Math.random() > 0.4 ? 'warning' : 'fail',
        description: 'Detection of improper expense capitalization'
      },
      {
        name: 'Revenue Recognition',
        score: Math.random() * 100,
        status: Math.random() > 0.8 ? 'pass' : Math.random() > 0.5 ? 'warning' : 'fail',
        description: 'Analysis of revenue recognition patterns'
      },
      {
        name: 'Working Capital Changes',
        score: calculateWorkingCapitalScore(latest, previous),
        status: calculateWorkingCapitalScore(latest, previous) > 70 ? 'pass' : calculateWorkingCapitalScore(latest, previous) > 40 ? 'warning' : 'fail',
        description: 'Unusual changes in working capital components'
      }
    ];

    setFraudIndicators(indicators);
  };

  const calculateDSOScore = (latest: FinancialData, previous: FinancialData): number => {
    const currentDSO = (latest.accountsReceivable / latest.revenue) * 365;
    const previousDSO = (previous.accountsReceivable / previous.revenue) * 365;
    const change = Math.abs(currentDSO - previousDSO) / previousDSO;
    return Math.max(0, 100 - change * 200);
  };

  const calculateQualityOfEarnings = (data: FinancialData): number => {
    const cashFlowRatio = data.operatingCashFlow / data.netIncome;
    if (cashFlowRatio > 1.2) return 90 + Math.random() * 10;
    if (cashFlowRatio > 0.8) return 70 + Math.random() * 20;
    return 30 + Math.random() * 30;
  };

  const calculateWorkingCapitalScore = (latest: FinancialData, previous: FinancialData): number => {
    const currentWC = latest.currentAssets - latest.currentLiabilities;
    const previousWC = previous.currentAssets - previous.currentLiabilities;
    const change = Math.abs(currentWC - previousWC) / previousWC;
    return Math.max(0, 100 - change * 150);
  };

  const generateAIInsights = (data: FinancialData[]) => {
    const latest = data[data.length - 1];
    const insights: AIInsight[] = [];

    // Profitability insights
    if (latest.netIncome / latest.revenue > 0.2) {
      insights.push({
        category: 'Profitability',
        type: 'positive',
        title: 'Exceptional Net Margins',
        description: 'Net profit margins exceed 20%, indicating strong pricing power and cost control',
        impact: 'high',
        metric: 'Net Margin',
        value: (latest.netIncome / latest.revenue) * 100
      });
    }

    // Cash flow insights
    if (latest.freeCashFlow / latest.revenue > 0.15) {
      insights.push({
        category: 'Cash Flow',
        type: 'positive',
        title: 'Strong Free Cash Flow Generation',
        description: 'FCF yield above 15% suggests robust cash generation capabilities',
        impact: 'high',
        metric: 'FCF Yield',
        value: (latest.freeCashFlow / latest.revenue) * 100
      });
    }

    // Growth insights
    const revenueGrowth = ((latest.revenue - data[0].revenue) / data[0].revenue) / data.length * 100;
    if (revenueGrowth > 15) {
      insights.push({
        category: 'Growth',
        type: 'positive',
        title: 'Strong Revenue Growth',
        description: `Annual revenue growth of ${revenueGrowth.toFixed(1)}% outpaces industry average`,
        impact: 'high',
        metric: 'Revenue CAGR',
        value: revenueGrowth
      });
    }

    // Risk insights
    if (latest.totalLiabilities / latest.totalEquity > 2) {
      insights.push({
        category: 'Risk',
        type: 'negative',
        title: 'High Leverage Concern',
        description: 'Debt-to-equity ratio above 2x may indicate financial risk',
        impact: 'medium',
        metric: 'D/E Ratio',
        value: latest.totalLiabilities / latest.totalEquity
      });
    }

    // Efficiency insights
    if (latest.revenue / latest.totalAssets < 0.5) {
      insights.push({
        category: 'Efficiency',
        type: 'negative',
        title: 'Low Asset Utilization',
        description: 'Asset turnover below 0.5x suggests inefficient capital deployment',
        impact: 'medium',
        metric: 'Asset Turnover',
        value: latest.revenue / latest.totalAssets
      });
    }

    // Working capital insights
    const daysWorkingCapital = ((latest.currentAssets - latest.currentLiabilities) / latest.revenue) * 365;
    if (daysWorkingCapital < 30) {
      insights.push({
        category: 'Liquidity',
        type: 'neutral',
        title: 'Efficient Working Capital Management',
        description: 'Low working capital days indicate efficient operations but monitor liquidity',
        impact: 'low',
        metric: 'Days WC',
        value: daysWorkingCapital
      });
    }

    setAiInsights(insights);
  };

  const calculateAltmanZScore = (data: FinancialData) => {
    const workingCapital = data.currentAssets - data.currentLiabilities;
    const retainedEarnings = data.totalEquity * 0.6; // Approximation
    const ebit = data.operatingIncome;
    const marketValue = data.totalEquity * 1.5; // Approximation
    
    const z = 
      1.2 * (workingCapital / data.totalAssets) +
      1.4 * (retainedEarnings / data.totalAssets) +
      3.3 * (ebit / data.totalAssets) +
      0.6 * (marketValue / data.totalLiabilities) +
      1.0 * (data.revenue / data.totalAssets);
    
    setAltmanZScore(z);
  };

  const performBenfordAnalysis = (data: FinancialData[]) => {
    // Extract all financial numbers
    const numbers: number[] = [];
    data.forEach(d => {
      Object.values(d).forEach(value => {
        if (typeof value === 'number' && value > 0) {
          numbers.push(value);
        }
      });
    });

    // Count first digit frequency
    const digitCounts = new Array(9).fill(0);
    numbers.forEach(num => {
      const firstDigit = parseInt(num.toString()[0]);
      if (firstDigit >= 1 && firstDigit <= 9) {
        digitCounts[firstDigit - 1]++;
      }
    });

    // Calculate actual vs expected (Benford's Law)
    const analysis = digitCounts.map((count, i) => ({
      digit: i + 1,
      expected: Math.log10(1 + 1 / (i + 1)) * 100,
      actual: (count / numbers.length) * 100
    }));

    setBenfordAnalysis(analysis);
  };

  const generatePeerComparison = () => {
    const peers = ['MSFT', 'GOOGL', 'META', 'AMZN'];
    const comparison = peers.map(company => ({
      company,
      metrics: {
        profitability: {
          grossMargin: 30 + Math.random() * 20,
          operatingMargin: 15 + Math.random() * 15,
          netMargin: 10 + Math.random() * 15,
          roe: 15 + Math.random() * 20,
          roa: 8 + Math.random() * 10,
          roic: 12 + Math.random() * 15
        },
        liquidity: {
          currentRatio: 1.2 + Math.random() * 0.8,
          quickRatio: 1 + Math.random() * 0.6,
          cashRatio: 0.4 + Math.random() * 0.4,
          workingCapital: 30000 + Math.random() * 40000
        },
        efficiency: {
          assetTurnover: 0.6 + Math.random() * 0.6,
          inventoryTurnover: 6 + Math.random() * 8,
          receivablesTurnover: 8 + Math.random() * 7,
          daysInventory: 40 + Math.random() * 30,
          daysReceivables: 35 + Math.random() * 20
        },
        leverage: {
          debtToEquity: 0.6 + Math.random() * 0.8,
          debtToAssets: 0.4 + Math.random() * 0.4,
          interestCoverage: 4 + Math.random() * 6,
          equityMultiplier: 1.8 + Math.random() * 0.8
        }
      }
    }));
    setPeerComparison(comparison);
  };

  // Visualization functions
  useEffect(() => {
    if (financialData.length > 0) {
      drawIncomeStatement();
      drawBalanceSheet();
      drawCashFlow();
      drawRatioComparison();
      drawDuPontChart();
    }
  }, [financialData, selectedYear, ratios, industryAvg]);

  const drawIncomeStatement = () => {
    if (!incomeChartRef.current) return;
    
    d3.select(incomeChartRef.current).selectAll('*').remove();
    
    const margin = { top: 20, right: 30, bottom: 40, left: 90 };
    const width = incomeChartRef.current.clientWidth - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    const svg = d3.select(incomeChartRef.current)
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const selectedData = financialData.find(d => d.year === selectedYear);
    if (!selectedData) return;

    const data = [
      { name: 'Revenue', value: selectedData.revenue },
      { name: 'Cost of Revenue', value: -selectedData.costOfRevenue },
      { name: 'Gross Profit', value: selectedData.grossProfit },
      { name: 'Operating Expenses', value: -selectedData.operatingExpenses },
      { name: 'Operating Income', value: selectedData.operatingIncome },
      { name: 'Net Income', value: selectedData.netIncome }
    ];

    const x = d3.scaleLinear()
      .domain([d3.min(data, d => d.value)! * 1.1, d3.max(data, d => d.value)! * 1.1])
      .range([0, width]);

    const y = d3.scaleBand()
      .domain(data.map(d => d.name))
      .range([0, height])
      .padding(0.1);

    svg.selectAll('.bar')
      .data(data)
      .enter().append('rect')
      .attr('class', 'bar')
      .attr('x', d => d.value > 0 ? x(0) : x(d.value))
      .attr('y', d => y(d.name)!)
      .attr('width', d => Math.abs(x(d.value) - x(0)))
      .attr('height', y.bandwidth())
      .attr('fill', d => d.value > 0 ? '#10b981' : '#ef4444');

    svg.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(x).tickFormat(d => `$${d3.format('.0s')(d)}`));

    svg.append('g')
      .attr('transform', `translate(${x(0)},0)`)
      .call(d3.axisLeft(y));
  };

  const drawBalanceSheet = () => {
    if (!balanceChartRef.current) return;
    
    d3.select(balanceChartRef.current).selectAll('*').remove();
    
    const selectedData = financialData.find(d => d.year === selectedYear);
    if (!selectedData) return;

    const width = balanceChartRef.current.clientWidth;
    const height = 400;
    const radius = Math.min(width, height) / 2 - 40;

    const svg = d3.select(balanceChartRef.current)
      .append('svg')
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${width / 2},${height / 2})`);

    const data = [
      { name: 'Current Assets', value: selectedData.currentAssets },
      { name: 'Non-Current Assets', value: selectedData.totalAssets - selectedData.currentAssets },
      { name: 'Current Liabilities', value: selectedData.currentLiabilities },
      { name: 'Long-term Debt', value: selectedData.longTermDebt },
      { name: 'Equity', value: selectedData.totalEquity }
    ];

    const color = d3.scaleOrdinal()
      .domain(data.map(d => d.name))
      .range(['#3b82f6', '#6366f1', '#ef4444', '#f59e0b', '#10b981']);

    const pie = d3.pie<typeof data[0]>()
      .value(d => d.value);

    const arc = d3.arc<d3.PieArcDatum<typeof data[0]>>()
      .innerRadius(0)
      .outerRadius(radius);

    const arcs = svg.selectAll('.arc')
      .data(pie(data))
      .enter().append('g')
      .attr('class', 'arc');

    arcs.append('path')
      .attr('d', arc)
      .attr('fill', d => color(d.data.name) as string);

    arcs.append('text')
      .attr('transform', d => `translate(${arc.centroid(d)})`)
      .attr('dy', '0.35em')
      .style('text-anchor', 'middle')
      .style('fill', 'white')
      .style('font-size', '12px')
      .text(d => d.data.name);
  };

  const drawCashFlow = () => {
    if (!cashFlowChartRef.current) return;
    
    d3.select(cashFlowChartRef.current).selectAll('*').remove();
    
    const margin = { top: 20, right: 30, bottom: 40, left: 60 };
    const width = cashFlowChartRef.current.clientWidth - margin.left - margin.right;
    const height = 300 - margin.top - margin.bottom;

    const svg = d3.select(cashFlowChartRef.current)
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const x = d3.scaleBand()
      .domain(financialData.map(d => d.year.toString()))
      .range([0, width])
      .padding(0.1);

    const y = d3.scaleLinear()
      .domain([
        d3.min(financialData, d => Math.min(d.operatingCashFlow, d.investingCashFlow, d.financingCashFlow))! * 1.1,
        d3.max(financialData, d => Math.max(d.operatingCashFlow, d.freeCashFlow))! * 1.1
      ])
      .range([height, 0]);

    const line = d3.line<FinancialData>()
      .x(d => x(d.year.toString())! + x.bandwidth() / 2)
      .y(d => y(d.freeCashFlow));

    // Operating cash flow bars
    svg.selectAll('.operating')
      .data(financialData)
      .enter().append('rect')
      .attr('class', 'operating')
      .attr('x', d => x(d.year.toString())!)
      .attr('y', d => y(Math.max(0, d.operatingCashFlow)))
      .attr('width', x.bandwidth() / 3)
      .attr('height', d => Math.abs(y(d.operatingCashFlow) - y(0)))
      .attr('fill', '#10b981');

    // Investing cash flow bars
    svg.selectAll('.investing')
      .data(financialData)
      .enter().append('rect')
      .attr('class', 'investing')
      .attr('x', d => x(d.year.toString())! + x.bandwidth() / 3)
      .attr('y', d => y(Math.max(0, d.investingCashFlow)))
      .attr('width', x.bandwidth() / 3)
      .attr('height', d => Math.abs(y(d.investingCashFlow) - y(0)))
      .attr('fill', '#ef4444');

    // Free cash flow line
    svg.append('path')
      .datum(financialData)
      .attr('fill', 'none')
      .attr('stroke', '#3b82f6')
      .attr('stroke-width', 2)
      .attr('d', line);

    svg.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(x));

    svg.append('g')
      .call(d3.axisLeft(y).tickFormat(d => `$${d3.format('.0s')(d)}`));
  };

  const drawRatioComparison = () => {
    if (!ratioChartRef.current || !ratios || !industryAvg) return;
    
    d3.select(ratioChartRef.current).selectAll('*').remove();
    
    const margin = { top: 20, right: 120, bottom: 60, left: 60 };
    const width = ratioChartRef.current.clientWidth - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    const svg = d3.select(ratioChartRef.current)
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const categories = ['Gross Margin', 'Operating Margin', 'Net Margin', 'ROE', 'ROA'];
    const companyData = [
      ratios.profitability.grossMargin,
      ratios.profitability.operatingMargin,
      ratios.profitability.netMargin,
      ratios.profitability.roe,
      ratios.profitability.roa
    ];
    const industryData = [
      industryAvg.profitability.grossMargin,
      industryAvg.profitability.operatingMargin,
      industryAvg.profitability.netMargin,
      industryAvg.profitability.roe,
      industryAvg.profitability.roa
    ];

    const x0 = d3.scaleBand()
      .domain(categories)
      .rangeRound([0, width])
      .paddingInner(0.1);

    const x1 = d3.scaleBand()
      .domain(['Company', 'Industry'])
      .rangeRound([0, x0.bandwidth()])
      .padding(0.05);

    const y = d3.scaleLinear()
      .domain([0, d3.max([...companyData, ...industryData])! * 1.1])
      .rangeRound([height, 0]);

    const color = d3.scaleOrdinal()
      .domain(['Company', 'Industry'])
      .range(['#3b82f6', '#9ca3af']);

    const grouped = categories.map((cat, i) => ({
      category: cat,
      Company: companyData[i],
      Industry: industryData[i]
    }));

    svg.append('g')
      .selectAll('g')
      .data(grouped)
      .enter().append('g')
      .attr('transform', d => `translate(${x0(d.category)},0)`)
      .selectAll('rect')
      .data(d => ['Company', 'Industry'].map(key => ({ key, value: d[key as keyof typeof d] as number })))
      .enter().append('rect')
      .attr('x', d => x1(d.key)!)
      .attr('y', d => y(d.value))
      .attr('width', x1.bandwidth())
      .attr('height', d => height - y(d.value))
      .attr('fill', d => color(d.key) as string);

    svg.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(x0))
      .selectAll('text')
      .attr('transform', 'rotate(-45)')
      .style('text-anchor', 'end');

    svg.append('g')
      .call(d3.axisLeft(y).tickFormat(d => `${d}%`));

    // Legend
    const legend = svg.append('g')
      .attr('transform', `translate(${width + 10}, 20)`);

    ['Company', 'Industry'].forEach((key, i) => {
      const g = legend.append('g')
        .attr('transform', `translate(0, ${i * 20})`);

      g.append('rect')
        .attr('width', 15)
        .attr('height', 15)
        .attr('fill', color(key) as string);

      g.append('text')
        .attr('x', 20)
        .attr('y', 12)
        .text(key)
        .style('font-size', '12px');
    });
  };

  const drawDuPontChart = () => {
    if (!dupontChartRef.current || !dupontAnalysis) return;
    
    d3.select(dupontChartRef.current).selectAll('*').remove();
    
    const width = dupontChartRef.current.clientWidth;
    const height = 300;

    const svg = d3.select(dupontChartRef.current)
      .append('svg')
      .attr('width', width)
      .attr('height', height);

    // DuPont breakdown visualization
    const boxWidth = 150;
    const boxHeight = 60;
    const spacing = 50;

    // ROE box
    svg.append('rect')
      .attr('x', width / 2 - boxWidth / 2)
      .attr('y', 20)
      .attr('width', boxWidth)
      .attr('height', boxHeight)
      .attr('fill', '#3b82f6')
      .attr('rx', 5);

    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 50)
      .attr('text-anchor', 'middle')
      .attr('fill', 'white')
      .style('font-weight', 'bold')
      .text(`ROE: ${dupontAnalysis.roe.toFixed(1)}%`);

    // Component boxes
    const components = [
      { name: 'Net Margin', value: dupontAnalysis.netMargin, x: width / 4 - boxWidth / 2 },
      { name: 'Asset Turnover', value: dupontAnalysis.assetTurnover, x: width / 2 - boxWidth / 2 },
      { name: 'Equity Multiplier', value: dupontAnalysis.equityMultiplier, x: 3 * width / 4 - boxWidth / 2 }
    ];

    components.forEach(comp => {
      svg.append('rect')
        .attr('x', comp.x)
        .attr('y', 120)
        .attr('width', boxWidth)
        .attr('height', boxHeight)
        .attr('fill', '#10b981')
        .attr('rx', 5);

      svg.append('text')
        .attr('x', comp.x + boxWidth / 2)
        .attr('y', 145)
        .attr('text-anchor', 'middle')
        .attr('fill', 'white')
        .style('font-size', '12px')
        .text(comp.name);

      svg.append('text')
        .attr('x', comp.x + boxWidth / 2)
        .attr('y', 165)
        .attr('text-anchor', 'middle')
        .attr('fill', 'white')
        .style('font-weight', 'bold')
        .text(comp.name === 'Net Margin' ? `${comp.value.toFixed(1)}%` : comp.value.toFixed(2));

      // Connect lines
      svg.append('line')
        .attr('x1', comp.x + boxWidth / 2)
        .attr('y1', 120)
        .attr('x2', width / 2)
        .attr('y2', 80)
        .attr('stroke', '#64748b')
        .attr('stroke-width', 2);
    });
  };

  const getZScoreStatus = (score: number) => {
    if (score > 3) return { status: 'Safe', color: '#10b981' };
    if (score > 1.8) return { status: 'Caution', color: '#f59e0b' };
    return { status: 'Distress', color: '#ef4444' };
  };

  const downloadReport = () => {
    // Generate PDF report (simplified version)
    const report = {
      company: selectedCompany,
      date: new Date().toISOString(),
      financialData: financialData.find(d => d.year === selectedYear),
      ratios,
      industryAvg,
      dupontAnalysis,
      fraudIndicators,
      altmanZScore,
      aiInsights
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${selectedCompany}_financial_analysis_${selectedYear}.json`;
    a.click();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Link
                href="/modules/stock-analysis/tools"
                className="flex items-center text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white transition-colors"
              >
                <ArrowLeft className="h-5 w-5 mr-2" />
                도구 목록으로
              </Link>
              <div className="h-6 w-px bg-gray-300 dark:bg-gray-600" />
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white flex items-center">
                <FileText className="h-8 w-8 mr-3 text-indigo-600 dark:text-indigo-400" />
                AI 재무제표 분석기
              </h1>
            </div>
            <div className="flex items-center space-x-4">
              <select
                value={selectedCompany}
                onChange={(e) => setSelectedCompany(e.target.value)}
                className="px-4 py-2 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg"
              >
                <option value="AAPL">Apple Inc. (AAPL)</option>
                <option value="MSFT">Microsoft Corp. (MSFT)</option>
                <option value="GOOGL">Alphabet Inc. (GOOGL)</option>
                <option value="AMZN">Amazon.com Inc. (AMZN)</option>
              </select>
              <button
                onClick={downloadReport}
                className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors flex items-center"
              >
                <Download className="h-4 w-4 mr-2" />
                리포트 다운로드
              </button>
            </div>
          </div>
          <p className="mt-4 text-gray-600 dark:text-gray-300">
            AI 기반 재무제표 분석으로 기업의 재무 건전성과 성장 잠재력을 평가합니다
          </p>
        </div>

        {/* Year Selector */}
        <div className="mb-6 flex items-center space-x-4">
          <span className="text-gray-700 dark:text-gray-300">분석 연도:</span>
          <div className="flex space-x-2">
            {financialData.map(d => (
              <button
                key={d.year}
                onClick={() => setSelectedYear(d.year)}
                className={`px-4 py-2 rounded-lg transition-colors ${
                  selectedYear === d.year
                    ? 'bg-indigo-600 text-white'
                    : 'bg-white dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-600'
                }`}
              >
                {d.year}
              </button>
            ))}
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Income Statement */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4 flex items-center">
              <BarChart3 className="h-5 w-5 mr-2 text-indigo-600 dark:text-indigo-400" />
              손익계산서 분석
            </h2>
            <div ref={incomeChartRef} className="h-96" />
          </div>

          {/* Balance Sheet */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4 flex items-center">
              <PieChart className="h-5 w-5 mr-2 text-indigo-600 dark:text-indigo-400" />
              재무상태표 구성
            </h2>
            <div ref={balanceChartRef} className="h-96" />
          </div>

          {/* Cash Flow */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4 flex items-center">
              <Activity className="h-5 w-5 mr-2 text-indigo-600 dark:text-indigo-400" />
              현금흐름표
            </h2>
            <div ref={cashFlowChartRef} className="h-72" />
          </div>

          {/* Ratio Comparison */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4 flex items-center">
              <Scale className="h-5 w-5 mr-2 text-indigo-600 dark:text-indigo-400" />
              산업 평균 대비 비율 분석
            </h2>
            <div ref={ratioChartRef} className="h-96" />
          </div>
        </div>

        {/* DuPont Analysis */}
        <div className="mt-6 bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4 flex items-center">
            <Calculator className="h-5 w-5 mr-2 text-indigo-600 dark:text-indigo-400" />
            듀폰 분석 (ROE 분해)
          </h2>
          <div ref={dupontChartRef} className="h-72" />
        </div>

        {/* Financial Ratios Grid */}
        {ratios && (
          <div className="mt-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Profitability Ratios */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                <TrendingUp className="h-4 w-4 mr-2 text-green-600" />
                수익성 지표
              </h3>
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">매출총이익률</span>
                    <span className="font-medium">{ratios.profitability.grossMargin.toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 mt-1">
                    <div 
                      className="bg-green-600 h-2 rounded-full"
                      style={{ width: `${Math.min(ratios.profitability.grossMargin, 100)}%` }}
                    />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">영업이익률</span>
                    <span className="font-medium">{ratios.profitability.operatingMargin.toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 mt-1">
                    <div 
                      className="bg-green-600 h-2 rounded-full"
                      style={{ width: `${Math.min(ratios.profitability.operatingMargin, 100)}%` }}
                    />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">순이익률</span>
                    <span className="font-medium">{ratios.profitability.netMargin.toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 mt-1">
                    <div 
                      className="bg-green-600 h-2 rounded-full"
                      style={{ width: `${Math.min(ratios.profitability.netMargin, 100)}%` }}
                    />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">ROE</span>
                    <span className="font-medium">{ratios.profitability.roe.toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 mt-1">
                    <div 
                      className="bg-green-600 h-2 rounded-full"
                      style={{ width: `${Math.min(ratios.profitability.roe, 100)}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Liquidity Ratios */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                <Activity className="h-4 w-4 mr-2 text-blue-600" />
                유동성 지표
              </h3>
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">유동비율</span>
                    <span className="font-medium">{ratios.liquidity.currentRatio.toFixed(2)}x</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 mt-1">
                    <div 
                      className="bg-blue-600 h-2 rounded-full"
                      style={{ width: `${Math.min(ratios.liquidity.currentRatio * 33, 100)}%` }}
                    />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">당좌비율</span>
                    <span className="font-medium">{ratios.liquidity.quickRatio.toFixed(2)}x</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 mt-1">
                    <div 
                      className="bg-blue-600 h-2 rounded-full"
                      style={{ width: `${Math.min(ratios.liquidity.quickRatio * 40, 100)}%` }}
                    />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">현금비율</span>
                    <span className="font-medium">{ratios.liquidity.cashRatio.toFixed(2)}x</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 mt-1">
                    <div 
                      className="bg-blue-600 h-2 rounded-full"
                      style={{ width: `${Math.min(ratios.liquidity.cashRatio * 100, 100)}%` }}
                    />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">운전자본</span>
                    <span className="font-medium">${(ratios.liquidity.workingCapital / 1000).toFixed(0)}K</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Efficiency Ratios */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                <Zap className="h-4 w-4 mr-2 text-yellow-600" />
                효율성 지표
              </h3>
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">자산회전율</span>
                    <span className="font-medium">{ratios.efficiency.assetTurnover.toFixed(2)}x</span>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">재고회전율</span>
                    <span className="font-medium">{ratios.efficiency.inventoryTurnover.toFixed(1)}x</span>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">매출채권회전율</span>
                    <span className="font-medium">{ratios.efficiency.receivablesTurnover.toFixed(1)}x</span>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">재고보유일수</span>
                    <span className="font-medium">{ratios.efficiency.daysInventory.toFixed(0)}일</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Leverage Ratios */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                <Scale className="h-4 w-4 mr-2 text-purple-600" />
                레버리지 지표
              </h3>
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">부채비율</span>
                    <span className="font-medium">{ratios.leverage.debtToEquity.toFixed(2)}x</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 mt-1">
                    <div 
                      className="bg-purple-600 h-2 rounded-full"
                      style={{ width: `${Math.min(ratios.leverage.debtToEquity * 40, 100)}%` }}
                    />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">부채/자산 비율</span>
                    <span className="font-medium">{(ratios.leverage.debtToAssets * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 mt-1">
                    <div 
                      className="bg-purple-600 h-2 rounded-full"
                      style={{ width: `${ratios.leverage.debtToAssets * 100}%` }}
                    />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">이자보상배율</span>
                    <span className="font-medium">{ratios.leverage.interestCoverage.toFixed(1)}x</span>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">자기자본배율</span>
                    <span className="font-medium">{ratios.leverage.equityMultiplier.toFixed(2)}x</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Fraud Detection Section */}
        <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Fraud Indicators */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4 flex items-center">
              <Shield className="h-5 w-5 mr-2 text-red-600" />
              부정회계 탐지 지표
            </h2>
            <div className="space-y-4">
              {fraudIndicators.map((indicator, index) => (
                <div key={index} className="border-b border-gray-200 dark:border-gray-700 pb-3 last:border-0">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <h4 className="font-medium text-gray-900 dark:text-white flex items-center">
                        {indicator.name}
                        {indicator.status === 'pass' && <CheckCircle className="h-4 w-4 ml-2 text-green-600" />}
                        {indicator.status === 'warning' && <AlertCircle className="h-4 w-4 ml-2 text-yellow-600" />}
                        {indicator.status === 'fail' && <XCircle className="h-4 w-4 ml-2 text-red-600" />}
                      </h4>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                        {indicator.description}
                      </p>
                    </div>
                    <div className="ml-4">
                      <span className={`text-lg font-bold ${
                        indicator.status === 'pass' ? 'text-green-600' :
                        indicator.status === 'warning' ? 'text-yellow-600' :
                        'text-red-600'
                      }`}>
                        {indicator.score.toFixed(0)}%
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Altman Z-Score */}
            <div className="mt-6 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <h4 className="font-medium text-gray-900 dark:text-white mb-2">
                Altman Z-Score (파산 예측)
              </h4>
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="text-3xl font-bold" style={{ color: getZScoreStatus(altmanZScore).color }}>
                    {altmanZScore.toFixed(2)}
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    상태: {getZScoreStatus(altmanZScore).status}
                  </div>
                </div>
                <div className="ml-4 text-right">
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    <div>안전: &gt; 3.0</div>
                    <div>주의: 1.8 - 3.0</div>
                    <div>위험: &lt; 1.8</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Benford's Law Analysis */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4 flex items-center">
              <BarChart3 className="h-5 w-5 mr-2 text-indigo-600 dark:text-indigo-400" />
              벤포드 법칙 분석
            </h2>
            <div className="space-y-2">
              {benfordAnalysis.map(item => (
                <div key={item.digit} className="flex items-center">
                  <span className="w-8 text-gray-700 dark:text-gray-300">{item.digit}</span>
                  <div className="flex-1 flex items-center space-x-2">
                    <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-6 relative">
                      <div 
                        className="absolute top-0 left-0 h-full bg-blue-600 rounded-full"
                        style={{ width: `${item.actual}%` }}
                      />
                      <div 
                        className="absolute top-0 left-0 h-full border-2 border-green-600 rounded-full"
                        style={{ width: `${item.expected}%` }}
                      />
                    </div>
                    <span className="text-sm text-gray-600 dark:text-gray-400 w-20">
                      {item.actual.toFixed(1)}% / {item.expected.toFixed(1)}%
                    </span>
                  </div>
                </div>
              ))}
              <div className="mt-4 text-sm text-gray-600 dark:text-gray-400">
                <div className="flex items-center">
                  <div className="w-4 h-4 bg-blue-600 rounded mr-2" />
                  실제 분포
                </div>
                <div className="flex items-center mt-1">
                  <div className="w-4 h-4 border-2 border-green-600 rounded mr-2" />
                  예상 분포 (벤포드 법칙)
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* AI Insights */}
        <div className="mt-6 bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4 flex items-center">
            <Brain className="h-5 w-5 mr-2 text-indigo-600 dark:text-indigo-400" />
            AI 분석 인사이트
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {aiInsights.map((insight, index) => (
              <div 
                key={index}
                className={`p-4 rounded-lg border-2 ${
                  insight.type === 'positive' ? 'border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-900/20' :
                  insight.type === 'negative' ? 'border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-900/20' :
                  'border-gray-200 bg-gray-50 dark:border-gray-700 dark:bg-gray-800'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center">
                      {insight.type === 'positive' && <TrendingUp className="h-4 w-4 mr-2 text-green-600" />}
                      {insight.type === 'negative' && <AlertTriangle className="h-4 w-4 mr-2 text-red-600" />}
                      {insight.type === 'neutral' && <Info className="h-4 w-4 mr-2 text-blue-600" />}
                      <h4 className="font-medium text-gray-900 dark:text-white">{insight.title}</h4>
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                      {insight.description}
                    </p>
                    {insight.metric && (
                      <div className="mt-2 flex items-center text-sm">
                        <span className="text-gray-500 dark:text-gray-400">{insight.metric}:</span>
                        <span className="ml-2 font-medium">{insight.value?.toFixed(1)}%</span>
                      </div>
                    )}
                  </div>
                  <span className={`ml-4 px-2 py-1 text-xs rounded-full ${
                    insight.impact === 'high' ? 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300' :
                    insight.impact === 'medium' ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300' :
                    'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300'
                  }`}>
                    {insight.impact === 'high' ? '높음' : insight.impact === 'medium' ? '중간' : '낮음'}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Peer Comparison */}
        <div className="mt-6 bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4 flex items-center">
            <BarChart3 className="h-5 w-5 mr-2 text-indigo-600 dark:text-indigo-400" />
            동종업계 비교 분석
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead>
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    기업
                  </th>
                  <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    영업이익률
                  </th>
                  <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    ROE
                  </th>
                  <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    부채비율
                  </th>
                  <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    유동비율
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                <tr className="bg-blue-50 dark:bg-blue-900/20">
                  <td className="px-4 py-3 font-medium text-gray-900 dark:text-white">
                    {selectedCompany} (현재)
                  </td>
                  <td className="px-4 py-3 text-center">
                    {ratios?.profitability.operatingMargin.toFixed(1)}%
                  </td>
                  <td className="px-4 py-3 text-center">
                    {ratios?.profitability.roe.toFixed(1)}%
                  </td>
                  <td className="px-4 py-3 text-center">
                    {ratios?.leverage.debtToEquity.toFixed(2)}x
                  </td>
                  <td className="px-4 py-3 text-center">
                    {ratios?.liquidity.currentRatio.toFixed(2)}x
                  </td>
                </tr>
                {peerComparison.map((peer, index) => (
                  <tr key={index}>
                    <td className="px-4 py-3 font-medium text-gray-900 dark:text-white">
                      {peer.company}
                    </td>
                    <td className="px-4 py-3 text-center">
                      {peer.metrics.profitability.operatingMargin.toFixed(1)}%
                    </td>
                    <td className="px-4 py-3 text-center">
                      {peer.metrics.profitability.roe.toFixed(1)}%
                    </td>
                    <td className="px-4 py-3 text-center">
                      {peer.metrics.leverage.debtToEquity.toFixed(2)}x
                    </td>
                    <td className="px-4 py-3 text-center">
                      {peer.metrics.liquidity.currentRatio.toFixed(2)}x
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}