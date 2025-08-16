'use client';

import { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import { 
  ArrowLeft, 
  Newspaper, 
  TrendingUp, 
  TrendingDown, 
  AlertCircle, 
  Globe,
  Filter,
  Bell,
  BarChart3,
  Clock,
  Hash,
  Building2,
  User,
  Package,
  Shield,
  ChevronRight,
  Activity,
  Search,
  RefreshCw,
  Download,
  Settings,
  Info,
  ChevronUp,
  ChevronDown,
  Globe2,
  MessageSquare
} from 'lucide-react';
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, PointElement, LineElement, BarElement, RadialLinearScale } from 'chart.js';
import { Doughnut, Line, Bar, Radar } from 'react-chartjs-2';

ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, PointElement, LineElement, BarElement, RadialLinearScale);

interface NewsItem {
  id: string;
  title: string;
  source: string;
  sourceCredibility: number;
  timestamp: Date;
  sentiment: number;
  sentimentLabel: 'very_negative' | 'negative' | 'neutral' | 'positive' | 'very_positive';
  marketImpact: 'high' | 'medium' | 'low';
  entities: Entity[];
  keywords: string[];
  language: string;
  summary: string;
  readTime: number;
  reactions: {
    bullish: number;
    bearish: number;
    neutral: number;
  };
}

interface Entity {
  type: 'company' | 'person' | 'product' | 'location';
  name: string;
  sentiment: number;
  mentions: number;
}

interface SentimentAlert {
  id: string;
  type: 'positive_shift' | 'negative_shift' | 'high_volume';
  message: string;
  timestamp: Date;
  severity: 'high' | 'medium' | 'low';
}

export default function NewsSentimentAnalyzer() {
  const [newsItems, setNewsItems] = useState<NewsItem[]>([]);
  const [filteredItems, setFilteredItems] = useState<NewsItem[]>([]);
  const [selectedTimeframe, setSelectedTimeframe] = useState('1h');
  const [selectedSentiment, setSelectedSentiment] = useState('all');
  const [selectedImpact, setSelectedImpact] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [alerts, setAlerts] = useState<SentimentAlert[]>([]);
  const [isAutoRefresh, setIsAutoRefresh] = useState(true);
  const [selectedEntity, setSelectedEntity] = useState<string | null>(null);
  const [wordCloudData, setWordCloudData] = useState<{ word: string; count: number; sentiment: number }[]>([]);
  const newsContainerRef = useRef<HTMLDivElement>(null);

  // Initialize news data
  useEffect(() => {
    generateInitialNews();
    const interval = setInterval(() => {
      if (isAutoRefresh) {
        addNewNewsItem();
      }
    }, 5000);
    return () => clearInterval(interval);
  }, [isAutoRefresh]);

  // Filter news items
  useEffect(() => {
    let filtered = newsItems;

    if (searchQuery) {
      filtered = filtered.filter(item =>
        item.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        item.summary.toLowerCase().includes(searchQuery.toLowerCase()) ||
        item.entities.some(e => e.name.toLowerCase().includes(searchQuery.toLowerCase()))
      );
    }

    if (selectedSentiment !== 'all') {
      filtered = filtered.filter(item => item.sentimentLabel === selectedSentiment);
    }

    if (selectedImpact !== 'all') {
      filtered = filtered.filter(item => item.marketImpact === selectedImpact);
    }

    // Time filter
    const now = new Date();
    const timeframeHours = {
      '1h': 1,
      '4h': 4,
      '1d': 24,
      '1w': 168,
      '1m': 720
    };
    const hours = timeframeHours[selectedTimeframe as keyof typeof timeframeHours] || 1;
    filtered = filtered.filter(item => {
      const hoursDiff = (now.getTime() - item.timestamp.getTime()) / (1000 * 60 * 60);
      return hoursDiff <= hours;
    });

    setFilteredItems(filtered);

    // Update word cloud
    updateWordCloud(filtered);
  }, [newsItems, searchQuery, selectedSentiment, selectedImpact, selectedTimeframe]);

  const generateInitialNews = () => {
    const sources = [
      { name: 'Bloomberg', credibility: 0.95 },
      { name: 'Reuters', credibility: 0.94 },
      { name: 'WSJ', credibility: 0.93 },
      { name: 'Financial Times', credibility: 0.92 },
      { name: 'CNBC', credibility: 0.88 },
      { name: 'Forbes', credibility: 0.87 },
      { name: 'MarketWatch', credibility: 0.85 },
      { name: 'Seeking Alpha', credibility: 0.78 },
      { name: 'Reddit r/wallstreetbets', credibility: 0.45 },
      { name: 'Twitter', credibility: 0.55 }
    ];

    const headlines = [
      {
        title: "Federal Reserve Signals Potential Rate Cut in Next Quarter",
        sentiment: 0.7,
        impact: 'high',
        entities: [{ type: 'company', name: 'Federal Reserve', sentiment: 0.7, mentions: 3 }]
      },
      {
        title: "Apple Announces Record iPhone Sales, Beats Q4 Estimates",
        sentiment: 0.8,
        impact: 'high',
        entities: [
          { type: 'company', name: 'Apple Inc.', sentiment: 0.9, mentions: 5 },
          { type: 'product', name: 'iPhone', sentiment: 0.8, mentions: 3 }
        ]
      },
      {
        title: "Oil Prices Surge 5% on Middle East Tensions",
        sentiment: -0.6,
        impact: 'high',
        entities: [{ type: 'location', name: 'Middle East', sentiment: -0.7, mentions: 2 }]
      },
      {
        title: "Tesla Recalls 50,000 Vehicles Due to Safety Concerns",
        sentiment: -0.7,
        impact: 'medium',
        entities: [
          { type: 'company', name: 'Tesla Inc.', sentiment: -0.8, mentions: 4 },
          { type: 'person', name: 'Elon Musk', sentiment: -0.5, mentions: 2 }
        ]
      },
      {
        title: "Microsoft Cloud Revenue Grows 30% Year-over-Year",
        sentiment: 0.6,
        impact: 'medium',
        entities: [
          { type: 'company', name: 'Microsoft Corp.', sentiment: 0.7, mentions: 3 },
          { type: 'product', name: 'Azure', sentiment: 0.8, mentions: 2 }
        ]
      },
      {
        title: "Cryptocurrency Market Cap Drops Below $2 Trillion",
        sentiment: -0.5,
        impact: 'medium',
        entities: [
          { type: 'product', name: 'Bitcoin', sentiment: -0.6, mentions: 2 },
          { type: 'product', name: 'Ethereum', sentiment: -0.5, mentions: 1 }
        ]
      },
      {
        title: "Amazon Announces Major Expansion in Southeast Asia",
        sentiment: 0.5,
        impact: 'medium',
        entities: [
          { type: 'company', name: 'Amazon Inc.', sentiment: 0.6, mentions: 3 },
          { type: 'location', name: 'Southeast Asia', sentiment: 0.4, mentions: 2 }
        ]
      },
      {
        title: "Bank of America Downgrades Tech Sector Outlook",
        sentiment: -0.4,
        impact: 'medium',
        entities: [
          { type: 'company', name: 'Bank of America', sentiment: -0.3, mentions: 2 }
        ]
      }
    ];

    const initialNews: NewsItem[] = headlines.map((headline, index) => {
      const source = sources[Math.floor(Math.random() * sources.length)];
      const timestamp = new Date(Date.now() - Math.random() * 3600000); // Within last hour
      
      return {
        id: `news-${index}`,
        title: headline.title,
        source: source.name,
        sourceCredibility: source.credibility,
        timestamp,
        sentiment: headline.sentiment,
        sentimentLabel: getSentimentLabel(headline.sentiment),
        marketImpact: headline.impact as 'high' | 'medium' | 'low',
        entities: headline.entities as Entity[],
        keywords: extractKeywords(headline.title),
        language: Math.random() > 0.9 ? 'ko' : 'en',
        summary: generateSummary(headline.title),
        readTime: Math.floor(Math.random() * 5) + 1,
        reactions: {
          bullish: Math.floor(Math.random() * 1000),
          bearish: Math.floor(Math.random() * 1000),
          neutral: Math.floor(Math.random() * 500)
        }
      };
    });

    setNewsItems(initialNews.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime()));

    // Generate initial alerts
    const alert: SentimentAlert = {
      id: 'alert-1',
      type: 'negative_shift',
      message: 'Tech sector sentiment shifted negative in the last hour',
      timestamp: new Date(),
      severity: 'high'
    };
    setAlerts([alert]);
  };

  const addNewNewsItem = () => {
    const templates = [
      "Breaking: {company} Reports {movement}% {direction} in Pre-Market Trading",
      "{company} CEO {action} Amid {event}",
      "Analysts {sentiment} on {company} Following {catalyst}",
      "{sector} Sector {movement} as {event}",
      "Exclusive: {company} in Talks for {amount} {dealType}",
      "{regulator} Investigates {company} for {issue}",
      "{company} Launches {product} to Compete with {competitor}"
    ];

    const companies = ['Goldman Sachs', 'JPMorgan', 'Google', 'Meta', 'Netflix', 'AMD', 'NVIDIA', 'Berkshire Hathaway', 'Toyota', 'Samsung'];
    const movements = [2, 3, 4, 5, 6, 7, 8];
    const directions = ['Gain', 'Drop'];
    const actions = ['Steps Down', 'Announces Restructuring', 'Reveals Strategy'];
    const events = ['Market Volatility', 'Earnings Miss', 'Regulatory Pressure', 'Supply Chain Issues'];
    const sentiments = ['Upgrade', 'Downgrade', 'Maintain Rating'];
    const sectors = ['Financial', 'Technology', 'Healthcare', 'Energy', 'Consumer'];
    const amounts = ['$1B', '$5B', '$10B', '$20B'];
    const dealTypes = ['Acquisition', 'Investment Round', 'Partnership', 'Merger'];
    const regulators = ['SEC', 'FTC', 'DOJ', 'EU Commission'];
    const issues = ['Antitrust Concerns', 'Data Privacy', 'Market Manipulation', 'Accounting Practices'];
    const products = ['AI Platform', 'Cloud Service', 'Electric Vehicle', 'Streaming Service'];
    const competitors = ['Industry Leaders', 'Market Rivals', 'Tech Giants'];

    const template = templates[Math.floor(Math.random() * templates.length)];
    const company = companies[Math.floor(Math.random() * companies.length)];
    
    let title = template
      .replace('{company}', company)
      .replace('{movement}', movements[Math.floor(Math.random() * movements.length)].toString())
      .replace('{direction}', directions[Math.floor(Math.random() * directions.length)])
      .replace('{action}', actions[Math.floor(Math.random() * actions.length)])
      .replace('{event}', events[Math.floor(Math.random() * events.length)])
      .replace('{sentiment}', sentiments[Math.floor(Math.random() * sentiments.length)])
      .replace('{catalyst}', events[Math.floor(Math.random() * events.length)])
      .replace('{sector}', sectors[Math.floor(Math.random() * sectors.length)])
      .replace('{amount}', amounts[Math.floor(Math.random() * amounts.length)])
      .replace('{dealType}', dealTypes[Math.floor(Math.random() * dealTypes.length)])
      .replace('{regulator}', regulators[Math.floor(Math.random() * regulators.length)])
      .replace('{issue}', issues[Math.floor(Math.random() * issues.length)])
      .replace('{product}', products[Math.floor(Math.random() * products.length)])
      .replace('{competitor}', competitors[Math.floor(Math.random() * competitors.length)]);

    const sentiment = title.includes('Drop') || title.includes('Downgrade') || title.includes('Investigates') 
      ? -Math.random() * 0.8 - 0.2
      : Math.random() * 0.8 + 0.2;

    const sources = [
      { name: 'Bloomberg', credibility: 0.95 },
      { name: 'Reuters', credibility: 0.94 },
      { name: 'WSJ', credibility: 0.93 }
    ];
    const source = sources[Math.floor(Math.random() * sources.length)];

    const newItem: NewsItem = {
      id: `news-${Date.now()}`,
      title,
      source: source.name,
      sourceCredibility: source.credibility,
      timestamp: new Date(),
      sentiment,
      sentimentLabel: getSentimentLabel(sentiment),
      marketImpact: Math.abs(sentiment) > 0.6 ? 'high' : Math.abs(sentiment) > 0.3 ? 'medium' : 'low',
      entities: [{ type: 'company', name: company, sentiment, mentions: 2 }],
      keywords: extractKeywords(title),
      language: 'en',
      summary: generateSummary(title),
      readTime: Math.floor(Math.random() * 5) + 1,
      reactions: {
        bullish: Math.floor(Math.random() * 100),
        bearish: Math.floor(Math.random() * 100),
        neutral: Math.floor(Math.random() * 50)
      }
    };

    setNewsItems(prev => [newItem, ...prev].slice(0, 100)); // Keep last 100 items

    // Check for alerts
    if (Math.random() > 0.8) {
      const alertTypes: SentimentAlert['type'][] = ['positive_shift', 'negative_shift', 'high_volume'];
      const newAlert: SentimentAlert = {
        id: `alert-${Date.now()}`,
        type: alertTypes[Math.floor(Math.random() * alertTypes.length)],
        message: `Unusual sentiment activity detected for ${company}`,
        timestamp: new Date(),
        severity: Math.abs(sentiment) > 0.7 ? 'high' : 'medium'
      };
      setAlerts(prev => [newAlert, ...prev].slice(0, 10));
    }
  };

  const getSentimentLabel = (sentiment: number): NewsItem['sentimentLabel'] => {
    if (sentiment <= -0.6) return 'very_negative';
    if (sentiment <= -0.2) return 'negative';
    if (sentiment <= 0.2) return 'neutral';
    if (sentiment <= 0.6) return 'positive';
    return 'very_positive';
  };

  const extractKeywords = (text: string): string[] => {
    const stopWords = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'as', 'is', 'was', 'are', 'were'];
    return text.toLowerCase()
      .split(/\W+/)
      .filter(word => word.length > 3 && !stopWords.includes(word))
      .slice(0, 5);
  };

  const generateSummary = (title: string): string => {
    const summaries = [
      "Market analysts are closely monitoring this development for potential impacts on sector valuations.",
      "This news has triggered significant investor interest and could influence trading patterns.",
      "Industry experts suggest this could signal broader market trends in the coming weeks.",
      "The announcement comes amid heightened market sensitivity to corporate developments.",
      "Traders are adjusting positions based on the implications of this breaking news."
    ];
    return summaries[Math.floor(Math.random() * summaries.length)];
  };

  const updateWordCloud = (items: NewsItem[]) => {
    const wordFrequency: { [key: string]: { count: number; sentiment: number } } = {};
    
    items.forEach(item => {
      item.keywords.forEach(keyword => {
        if (wordFrequency[keyword]) {
          wordFrequency[keyword].count++;
          wordFrequency[keyword].sentiment = (wordFrequency[keyword].sentiment + item.sentiment) / 2;
        } else {
          wordFrequency[keyword] = { count: 1, sentiment: item.sentiment };
        }
      });
    });

    const cloudData = Object.entries(wordFrequency)
      .map(([word, data]) => ({ word, ...data }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 20);

    setWordCloudData(cloudData);
  };

  const getSentimentColor = (sentiment: number): string => {
    if (sentiment <= -0.6) return 'text-red-600 dark:text-red-400';
    if (sentiment <= -0.2) return 'text-orange-600 dark:text-orange-400';
    if (sentiment <= 0.2) return 'text-gray-600 dark:text-gray-400';
    if (sentiment <= 0.6) return 'text-green-600 dark:text-green-400';
    return 'text-emerald-600 dark:text-emerald-400';
  };

  const getSentimentBgColor = (sentiment: number): string => {
    if (sentiment <= -0.6) return 'bg-red-100 dark:bg-red-900/20';
    if (sentiment <= -0.2) return 'bg-orange-100 dark:bg-orange-900/20';
    if (sentiment <= 0.2) return 'bg-gray-100 dark:bg-gray-800/20';
    if (sentiment <= 0.6) return 'bg-green-100 dark:bg-green-900/20';
    return 'bg-emerald-100 dark:bg-emerald-900/20';
  };

  // Calculate aggregate sentiment
  const aggregateSentiment = filteredItems.length > 0
    ? filteredItems.reduce((sum, item) => sum + item.sentiment, 0) / filteredItems.length
    : 0;

  // Sentiment distribution data
  const sentimentDistribution = {
    labels: ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'],
    datasets: [{
      data: [
        filteredItems.filter(item => item.sentimentLabel === 'very_negative').length,
        filteredItems.filter(item => item.sentimentLabel === 'negative').length,
        filteredItems.filter(item => item.sentimentLabel === 'neutral').length,
        filteredItems.filter(item => item.sentimentLabel === 'positive').length,
        filteredItems.filter(item => item.sentimentLabel === 'very_positive').length
      ],
      backgroundColor: [
        'rgb(220, 38, 38)',
        'rgb(251, 146, 60)',
        'rgb(156, 163, 175)',
        'rgb(34, 197, 94)',
        'rgb(16, 185, 129)'
      ]
    }]
  };

  // Sentiment trend data
  const sentimentTrendData = {
    labels: Array.from({ length: 24 }, (_, i) => `${23 - i}h`),
    datasets: [{
      label: 'Average Sentiment',
      data: Array.from({ length: 24 }, () => Math.random() * 2 - 1),
      borderColor: 'rgb(59, 130, 246)',
      backgroundColor: 'rgba(59, 130, 246, 0.1)',
      tension: 0.4
    }]
  };

  // Source credibility data
  const sourceCredibilityData = {
    labels: ['Bloomberg', 'Reuters', 'WSJ', 'FT', 'CNBC', 'Forbes'],
    datasets: [{
      label: 'Credibility Score',
      data: [0.95, 0.94, 0.93, 0.92, 0.88, 0.87],
      backgroundColor: 'rgba(59, 130, 246, 0.8)'
    }]
  };

  // Entity sentiment radar
  const topEntities = Array.from(new Set(filteredItems.flatMap(item => item.entities.map(e => e.name))))
    .slice(0, 6)
    .map(name => {
      const entityItems = filteredItems.filter(item => 
        item.entities.some(e => e.name === name)
      );
      const avgSentiment = entityItems.reduce((sum, item) => {
        const entity = item.entities.find(e => e.name === name);
        return sum + (entity?.sentiment || 0);
      }, 0) / (entityItems.length || 1);
      return { name, sentiment: avgSentiment };
    });

  const entityRadarData = {
    labels: topEntities.map(e => e.name),
    datasets: [{
      label: 'Entity Sentiment',
      data: topEntities.map(e => (e.sentiment + 1) * 50), // Convert to 0-100 scale
      backgroundColor: 'rgba(59, 130, 246, 0.2)',
      borderColor: 'rgb(59, 130, 246)',
      pointBackgroundColor: 'rgb(59, 130, 246)'
    }]
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link 
                href="/modules/stock-analysis/tools"
                className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
              >
                <ArrowLeft className="w-5 h-5" />
                <span>도구 목록</span>
              </Link>
              <div className="h-6 w-px bg-gray-300 dark:bg-gray-600" />
              <div className="flex items-center gap-2">
                <Newspaper className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                <h1 className="text-xl font-bold text-gray-900 dark:text-white">
                  News Sentiment Analyzer
                </h1>
                <span className="px-2 py-1 text-xs font-medium bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 rounded">
                  NLP Powered
                </span>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={() => setIsAutoRefresh(!isAutoRefresh)}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  isAutoRefresh
                    ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                }`}
              >
                <RefreshCw className={`w-4 h-4 inline mr-2 ${isAutoRefresh ? 'animate-spin' : ''}`} />
                Auto Refresh
              </button>
              <button className="p-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors">
                <Download className="w-5 h-5" />
              </button>
              <button className="p-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors">
                <Settings className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="grid grid-cols-12 gap-6">
          {/* Left Column - Metrics and Controls */}
          <div className="col-span-3 space-y-6">
            {/* Aggregate Sentiment Gauge */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Market Sentiment
              </h3>
              <div className="relative">
                <div className="w-full h-32 relative">
                  <svg className="w-full h-full" viewBox="0 0 200 100">
                    {/* Gauge background */}
                    <path
                      d="M 20 80 A 60 60 0 0 1 180 80"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="8"
                      className="text-gray-200 dark:text-gray-700"
                    />
                    {/* Colored segments */}
                    <path
                      d="M 20 80 A 60 60 0 0 1 56 40"
                      fill="none"
                      stroke="rgb(220, 38, 38)"
                      strokeWidth="8"
                    />
                    <path
                      d="M 56 40 A 60 60 0 0 1 100 30"
                      fill="none"
                      stroke="rgb(251, 146, 60)"
                      strokeWidth="8"
                    />
                    <path
                      d="M 100 30 A 60 60 0 0 1 144 40"
                      fill="none"
                      stroke="rgb(34, 197, 94)"
                      strokeWidth="8"
                    />
                    <path
                      d="M 144 40 A 60 60 0 0 1 180 80"
                      fill="none"
                      stroke="rgb(16, 185, 129)"
                      strokeWidth="8"
                    />
                    {/* Needle */}
                    <line
                      x1="100"
                      y1="80"
                      x2={100 + 60 * Math.cos(Math.PI - (aggregateSentiment + 1) * Math.PI / 2)}
                      y2={80 - 60 * Math.sin(Math.PI - (aggregateSentiment + 1) * Math.PI / 2)}
                      stroke="currentColor"
                      strokeWidth="3"
                      className="text-gray-900 dark:text-white"
                    />
                    <circle cx="100" cy="80" r="4" fill="currentColor" className="text-gray-900 dark:text-white" />
                  </svg>
                </div>
                <div className="text-center mt-2">
                  <div className={`text-2xl font-bold ${getSentimentColor(aggregateSentiment)}`}>
                    {aggregateSentiment > 0 ? '+' : ''}{(aggregateSentiment * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-gray-500 dark:text-gray-400">
                    {getSentimentLabel(aggregateSentiment).replace('_', ' ').toUpperCase()}
                  </div>
                </div>
              </div>
            </div>

            {/* Sentiment Distribution */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Sentiment Distribution
              </h3>
              <div className="h-48">
                <Doughnut
                  data={sentimentDistribution}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        position: 'bottom',
                        labels: {
                          boxWidth: 12,
                          padding: 8,
                          font: { size: 11 }
                        }
                      }
                    }
                  }}
                />
              </div>
            </div>

            {/* Filters */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                <Filter className="w-5 h-5" />
                Filters
              </h3>
              <div className="space-y-4">
                <div>
                  <label className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-1 block">
                    Timeframe
                  </label>
                  <select
                    value={selectedTimeframe}
                    onChange={(e) => setSelectedTimeframe(e.target.value)}
                    className="w-full px-3 py-2 bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="1h">Last Hour</option>
                    <option value="4h">Last 4 Hours</option>
                    <option value="1d">Last 24 Hours</option>
                    <option value="1w">Last Week</option>
                    <option value="1m">Last Month</option>
                  </select>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-1 block">
                    Sentiment
                  </label>
                  <select
                    value={selectedSentiment}
                    onChange={(e) => setSelectedSentiment(e.target.value)}
                    className="w-full px-3 py-2 bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="all">All Sentiments</option>
                    <option value="very_positive">Very Positive</option>
                    <option value="positive">Positive</option>
                    <option value="neutral">Neutral</option>
                    <option value="negative">Negative</option>
                    <option value="very_negative">Very Negative</option>
                  </select>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-1 block">
                    Market Impact
                  </label>
                  <select
                    value={selectedImpact}
                    onChange={(e) => setSelectedImpact(e.target.value)}
                    className="w-full px-3 py-2 bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="all">All Impacts</option>
                    <option value="high">High Impact</option>
                    <option value="medium">Medium Impact</option>
                    <option value="low">Low Impact</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Alerts */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                <Bell className="w-5 h-5" />
                Sentiment Alerts
              </h3>
              <div className="space-y-3 max-h-64 overflow-y-auto">
                {alerts.map((alert) => (
                  <div
                    key={alert.id}
                    className={`p-3 rounded-lg text-sm ${
                      alert.severity === 'high'
                        ? 'bg-red-100 dark:bg-red-900/20 text-red-700 dark:text-red-400'
                        : alert.severity === 'medium'
                        ? 'bg-yellow-100 dark:bg-yellow-900/20 text-yellow-700 dark:text-yellow-400'
                        : 'bg-blue-100 dark:bg-blue-900/20 text-blue-700 dark:text-blue-400'
                    }`}
                  >
                    <div className="flex items-start gap-2">
                      <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                      <div className="flex-1">
                        <p className="font-medium">{alert.message}</p>
                        <p className="text-xs opacity-75 mt-1">
                          {new Date(alert.timestamp).toLocaleTimeString()}
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Middle Column - News Feed */}
          <div className="col-span-6 space-y-6">
            {/* Search Bar */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm">
              <div className="relative">
                <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search news, companies, keywords..."
                  className="w-full pl-12 pr-4 py-3 bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
              </div>
            </div>

            {/* News Feed */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm">
              <div className="p-4 border-b border-gray-200 dark:border-gray-700">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                    Real-time News Feed
                  </h3>
                  <span className="text-sm text-gray-500 dark:text-gray-400">
                    {filteredItems.length} articles
                  </span>
                </div>
              </div>
              <div ref={newsContainerRef} className="divide-y divide-gray-200 dark:divide-gray-700 max-h-[800px] overflow-y-auto">
                {filteredItems.map((item) => (
                  <div key={item.id} className="p-4 hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors">
                    <div className="flex items-start gap-4">
                      <div className={`w-1 h-full ${
                        item.sentiment > 0.6 ? 'bg-emerald-500' :
                        item.sentiment > 0.2 ? 'bg-green-500' :
                        item.sentiment > -0.2 ? 'bg-gray-400' :
                        item.sentiment > -0.6 ? 'bg-orange-500' :
                        'bg-red-500'
                      }`} />
                      <div className="flex-1">
                        <div className="flex items-start justify-between gap-4">
                          <div className="flex-1">
                            <h4 className="font-semibold text-gray-900 dark:text-white mb-1">
                              {item.title}
                            </h4>
                            <div className="flex items-center gap-3 text-sm text-gray-500 dark:text-gray-400 mb-2">
                              <span className="flex items-center gap-1">
                                <Globe2 className="w-3 h-3" />
                                {item.source}
                              </span>
                              <span className="flex items-center gap-1">
                                <Shield className="w-3 h-3" />
                                {(item.sourceCredibility * 100).toFixed(0)}%
                              </span>
                              <span className="flex items-center gap-1">
                                <Clock className="w-3 h-3" />
                                {new Date(item.timestamp).toLocaleTimeString()}
                              </span>
                              {item.language !== 'en' && (
                                <span className="px-2 py-0.5 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 rounded text-xs">
                                  {item.language.toUpperCase()}
                                </span>
                              )}
                            </div>
                            <p className="text-sm text-gray-600 dark:text-gray-300 mb-3">
                              {item.summary}
                            </p>
                            <div className="flex items-center gap-4">
                              <div className="flex items-center gap-2">
                                <span className={`px-3 py-1 rounded-full text-xs font-medium ${getSentimentBgColor(item.sentiment)} ${getSentimentColor(item.sentiment)}`}>
                                  {item.sentimentLabel.replace('_', ' ')}
                                </span>
                                <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                                  item.marketImpact === 'high' 
                                    ? 'bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400'
                                    : item.marketImpact === 'medium'
                                    ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400'
                                    : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-400'
                                }`}>
                                  {item.marketImpact} impact
                                </span>
                              </div>
                              <div className="flex items-center gap-3 text-xs text-gray-500 dark:text-gray-400">
                                <span className="flex items-center gap-1">
                                  <TrendingUp className="w-3 h-3" />
                                  {item.reactions.bullish}
                                </span>
                                <span className="flex items-center gap-1">
                                  <TrendingDown className="w-3 h-3" />
                                  {item.reactions.bearish}
                                </span>
                                <span className="flex items-center gap-1">
                                  <Activity className="w-3 h-3" />
                                  {item.reactions.neutral}
                                </span>
                              </div>
                            </div>
                          </div>
                        </div>
                        {item.entities.length > 0 && (
                          <div className="mt-3 flex flex-wrap gap-2">
                            {item.entities.map((entity, idx) => (
                              <button
                                key={idx}
                                onClick={() => setSelectedEntity(entity.name)}
                                className={`inline-flex items-center gap-1 px-2 py-1 rounded text-xs ${
                                  selectedEntity === entity.name
                                    ? 'bg-blue-600 text-white'
                                    : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                                } transition-colors`}
                              >
                                {entity.type === 'company' && <Building2 className="w-3 h-3" />}
                                {entity.type === 'person' && <User className="w-3 h-3" />}
                                {entity.type === 'product' && <Package className="w-3 h-3" />}
                                {entity.type === 'location' && <Globe className="w-3 h-3" />}
                                {entity.name}
                              </button>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Right Column - Analytics */}
          <div className="col-span-3 space-y-6">
            {/* Sentiment Trend */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                24h Sentiment Trend
              </h3>
              <div className="h-48">
                <Line
                  data={sentimentTrendData}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                      y: {
                        min: -1,
                        max: 1,
                        grid: {
                          color: 'rgba(0, 0, 0, 0.05)'
                        }
                      },
                      x: {
                        grid: {
                          display: false
                        }
                      }
                    },
                    plugins: {
                      legend: {
                        display: false
                      }
                    }
                  }}
                />
              </div>
            </div>

            {/* Word Cloud */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Trending Keywords
              </h3>
              <div className="space-y-2">
                {wordCloudData.slice(0, 10).map((word) => (
                  <div key={word.word} className="flex items-center justify-between">
                    <span
                      className={`text-sm font-medium ${getSentimentColor(word.sentiment)}`}
                      style={{ fontSize: `${Math.min(16, 12 + word.count * 0.5)}px` }}
                    >
                      {word.word}
                    </span>
                    <span className="text-xs text-gray-500 dark:text-gray-400">
                      {word.count}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Entity Sentiment Radar */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Entity Sentiment
              </h3>
              <div className="h-64">
                <Radar
                  data={entityRadarData}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                      r: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                          stepSize: 25
                        }
                      }
                    },
                    plugins: {
                      legend: {
                        display: false
                      }
                    }
                  }}
                />
              </div>
            </div>

            {/* Source Credibility */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Source Credibility
              </h3>
              <div className="h-48">
                <Bar
                  data={sourceCredibilityData}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    indexAxis: 'y',
                    scales: {
                      x: {
                        min: 0,
                        max: 1,
                        ticks: {
                          format: {
                            style: 'percent'
                          }
                        }
                      }
                    },
                    plugins: {
                      legend: {
                        display: false
                      }
                    }
                  }}
                />
              </div>
            </div>

            {/* Info Panel */}
            <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-4">
              <div className="flex items-start gap-3">
                <Info className="w-5 h-5 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
                <div className="text-sm">
                  <p className="font-semibold text-blue-900 dark:text-blue-300 mb-1">
                    About Sentiment Analysis
                  </p>
                  <p className="text-blue-800 dark:text-blue-400">
                    Our NLP engine analyzes news articles in real-time, extracting sentiment, 
                    entities, and market impact predictions. Scores range from -1 (very negative) 
                    to +1 (very positive).
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}