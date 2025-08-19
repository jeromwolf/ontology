'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { ArrowLeft, Filter, Search, TrendingUp, TrendingDown, AlertCircle, Download, RefreshCw, Save, Star, ChevronDown, ChevronUp, Info, BarChart3, DollarSign, Activity, PieChart } from 'lucide-react';

interface Stock {
  code: string;
  name: string;
  market: string;
  sector: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  marketCap: number;
  per: number;
  pbr: number;
  roe: number;
  eps: number;
  bps: number;
  dividend: number;
  dividendYield: number;
  foreignRate: number;
  institutionRate: number;
}

interface FilterCriteria {
  markets: string[];
  sectors: string[];
  priceMin: number | null;
  priceMax: number | null;
  marketCapMin: number | null;
  marketCapMax: number | null;
  perMin: number | null;
  perMax: number | null;
  pbrMin: number | null;
  pbrMax: number | null;
  roeMin: number | null;
  roeMax: number | null;
  dividendYieldMin: number | null;
  dividendYieldMax: number | null;
  volumeMin: number | null;
  volumeMax: number | null;
}

interface SavedScreen {
  id: string;
  name: string;
  criteria: FilterCriteria;
  createdAt: string;
}

// 샘플 데이터 생성
const generateSampleStocks = (): Stock[] => {
  const companies = [
    { code: '005930', name: '삼성전자', sector: '전기전자', marketCap: 450000 },
    { code: '000660', name: 'SK하이닉스', sector: '전기전자', marketCap: 120000 },
    { code: '035420', name: 'NAVER', sector: 'IT', marketCap: 65000 },
    { code: '035720', name: '카카오', sector: 'IT', marketCap: 25000 },
    { code: '207940', name: '삼성바이오로직스', sector: '바이오', marketCap: 55000 },
    { code: '068270', name: '셀트리온', sector: '바이오', marketCap: 30000 },
    { code: '005380', name: '현대차', sector: '자동차', marketCap: 45000 },
    { code: '051910', name: 'LG화학', sector: '화학', marketCap: 35000 },
    { code: '006400', name: '삼성SDI', sector: '전기전자', marketCap: 48000 },
    { code: '373220', name: 'LG에너지솔루션', sector: '전기전자', marketCap: 95000 },
    { code: '000270', name: '기아', sector: '자동차', marketCap: 32000 },
    { code: '012330', name: '현대모비스', sector: '자동차', marketCap: 20000 },
    { code: '105560', name: 'KB금융', sector: '금융', marketCap: 22000 },
    { code: '055550', name: '신한지주', sector: '금융', marketCap: 20000 },
    { code: '066570', name: 'LG전자', sector: '전기전자', marketCap: 18000 },
    { code: '034730', name: 'SK이노베이션', sector: '화학', marketCap: 15000 },
    { code: '015760', name: '한국전력', sector: '전기가스', marketCap: 13000 },
    { code: '032830', name: '삼성생명', sector: '보험', marketCap: 12000 },
    { code: '003550', name: 'LG', sector: '지주회사', marketCap: 11000 },
    { code: '017670', name: 'SK텔레콤', sector: '통신', marketCap: 10000 },
    { code: '316140', name: '우리금융지주', sector: '금융', marketCap: 9000 },
    { code: '030200', name: 'KT', sector: '통신', marketCap: 8500 },
    { code: '352820', name: '하이브', sector: '엔터테인먼트', marketCap: 8000 },
    { code: '005490', name: 'POSCO홀딩스', sector: '철강', marketCap: 18000 },
    { code: '028260', name: '삼성물산', sector: '유통', marketCap: 20000 },
    { code: '090430', name: '아모레퍼시픽', sector: '화장품', marketCap: 7000 },
    { code: '010130', name: '고려아연', sector: '비철금속', marketCap: 12000 },
    { code: '009150', name: '삼성전기', sector: '전기전자', marketCap: 10000 },
    { code: '086790', name: '하나금융지주', sector: '금융', marketCap: 13000 },
    { code: '018260', name: '삼성에스디에스', sector: 'IT', marketCap: 11000 }
  ];

  return companies.map(company => {
    const basePrice = Math.floor(Math.random() * 200000 + 50000);
    const change = (Math.random() - 0.5) * 0.1 * basePrice;
    const per = Math.random() * 30 + 5;
    const pbr = Math.random() * 3 + 0.5;
    const roe = pbr / per * 100;
    
    return {
      code: company.code,
      name: company.name,
      market: 'KOSPI',
      sector: company.sector,
      price: basePrice,
      change: Math.floor(change),
      changePercent: (change / basePrice) * 100,
      volume: Math.floor(Math.random() * 10000000 + 100000),
      marketCap: company.marketCap,
      per: parseFloat(per.toFixed(2)),
      pbr: parseFloat(pbr.toFixed(2)),
      roe: parseFloat(roe.toFixed(2)),
      eps: Math.floor(basePrice / per),
      bps: Math.floor(basePrice / pbr),
      dividend: Math.floor(Math.random() * 5000),
      dividendYield: parseFloat((Math.random() * 5).toFixed(2)),
      foreignRate: parseFloat((Math.random() * 50).toFixed(2)),
      institutionRate: parseFloat((Math.random() * 30).toFixed(2))
    };
  });
};

export default function StockScreenerPage() {
  const [stocks, setStocks] = useState<Stock[]>([]);
  const [filteredStocks, setFilteredStocks] = useState<Stock[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [showFilters, setShowFilters] = useState(true);
  const [sortBy, setSortBy] = useState<keyof Stock>('marketCap');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [selectedStocks, setSelectedStocks] = useState<Set<string>>(new Set());
  const [savedScreens, setSavedScreens] = useState<SavedScreen[]>([]);
  const [showSavedScreens, setShowSavedScreens] = useState(false);
  
  const [filters, setFilters] = useState<FilterCriteria>({
    markets: ['KOSPI'],
    sectors: [],
    priceMin: null,
    priceMax: null,
    marketCapMin: null,
    marketCapMax: null,
    perMin: null,
    perMax: null,
    pbrMin: null,
    pbrMax: null,
    roeMin: null,
    roeMax: null,
    dividendYieldMin: null,
    dividendYieldMax: null,
    volumeMin: null,
    volumeMax: null
  });

  const sectors = ['전기전자', 'IT', '바이오', '자동차', '화학', '금융', '보험', '철강', '유통', '통신', '엔터테인먼트', '화장품', '비철금속', '지주회사', '전기가스'];

  // 초기 데이터 로드
  useEffect(() => {
    const loadData = async () => {
      setIsLoading(true);
      // 실제로는 API 호출
      await new Promise(resolve => setTimeout(resolve, 1000));
      const sampleStocks = generateSampleStocks();
      setStocks(sampleStocks);
      setFilteredStocks(sampleStocks);
      setIsLoading(false);
    };
    loadData();

    // 저장된 스크린 불러오기
    const saved = localStorage.getItem('savedScreens');
    if (saved) {
      setSavedScreens(JSON.parse(saved));
    }
  }, []);

  // 필터 적용
  useEffect(() => {
    let filtered = [...stocks];

    // 시장 필터
    if (filters.markets.length > 0) {
      filtered = filtered.filter(stock => filters.markets.includes(stock.market));
    }

    // 섹터 필터
    if (filters.sectors.length > 0) {
      filtered = filtered.filter(stock => filters.sectors.includes(stock.sector));
    }

    // 가격 필터
    if (filters.priceMin !== null) {
      filtered = filtered.filter(stock => stock.price >= filters.priceMin!);
    }
    if (filters.priceMax !== null) {
      filtered = filtered.filter(stock => stock.price <= filters.priceMax!);
    }

    // 시가총액 필터
    if (filters.marketCapMin !== null) {
      filtered = filtered.filter(stock => stock.marketCap >= filters.marketCapMin!);
    }
    if (filters.marketCapMax !== null) {
      filtered = filtered.filter(stock => stock.marketCap <= filters.marketCapMax!);
    }

    // PER 필터
    if (filters.perMin !== null) {
      filtered = filtered.filter(stock => stock.per >= filters.perMin!);
    }
    if (filters.perMax !== null) {
      filtered = filtered.filter(stock => stock.per <= filters.perMax!);
    }

    // PBR 필터
    if (filters.pbrMin !== null) {
      filtered = filtered.filter(stock => stock.pbr >= filters.pbrMin!);
    }
    if (filters.pbrMax !== null) {
      filtered = filtered.filter(stock => stock.pbr <= filters.pbrMax!);
    }

    // ROE 필터
    if (filters.roeMin !== null) {
      filtered = filtered.filter(stock => stock.roe >= filters.roeMin!);
    }
    if (filters.roeMax !== null) {
      filtered = filtered.filter(stock => stock.roe <= filters.roeMax!);
    }

    // 배당수익률 필터
    if (filters.dividendYieldMin !== null) {
      filtered = filtered.filter(stock => stock.dividendYield >= filters.dividendYieldMin!);
    }
    if (filters.dividendYieldMax !== null) {
      filtered = filtered.filter(stock => stock.dividendYield <= filters.dividendYieldMax!);
    }

    // 거래량 필터
    if (filters.volumeMin !== null) {
      filtered = filtered.filter(stock => stock.volume >= filters.volumeMin!);
    }
    if (filters.volumeMax !== null) {
      filtered = filtered.filter(stock => stock.volume <= filters.volumeMax!);
    }

    // 정렬
    filtered.sort((a, b) => {
      const aVal = a[sortBy];
      const bVal = b[sortBy];
      if (sortOrder === 'asc') {
        return aVal > bVal ? 1 : -1;
      } else {
        return aVal < bVal ? 1 : -1;
      }
    });

    setFilteredStocks(filtered);
  }, [filters, stocks, sortBy, sortOrder]);

  // 정렬 변경
  const handleSort = (field: keyof Stock) => {
    if (sortBy === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(field);
      setSortOrder('desc');
    }
  };

  // 필터 초기화
  const resetFilters = () => {
    setFilters({
      markets: ['KOSPI'],
      sectors: [],
      priceMin: null,
      priceMax: null,
      marketCapMin: null,
      marketCapMax: null,
      perMin: null,
      perMax: null,
      pbrMin: null,
      pbrMax: null,
      roeMin: null,
      roeMax: null,
      dividendYieldMin: null,
      dividendYieldMax: null,
      volumeMin: null,
      volumeMax: null
    });
  };

  // 스크린 저장
  const saveScreen = () => {
    const name = prompt('스크린 이름을 입력하세요:');
    if (name) {
      const newScreen: SavedScreen = {
        id: Date.now().toString(),
        name,
        criteria: filters,
        createdAt: new Date().toISOString()
      };
      const updatedScreens = [...savedScreens, newScreen];
      setSavedScreens(updatedScreens);
      localStorage.setItem('savedScreens', JSON.stringify(updatedScreens));
      alert('스크린이 저장되었습니다.');
    }
  };

  // 저장된 스크린 불러오기
  const loadScreen = (screen: SavedScreen) => {
    setFilters(screen.criteria);
    setShowSavedScreens(false);
  };

  // 저장된 스크린 삭제
  const deleteScreen = (id: string) => {
    const updatedScreens = savedScreens.filter(s => s.id !== id);
    setSavedScreens(updatedScreens);
    localStorage.setItem('savedScreens', JSON.stringify(updatedScreens));
  };

  // 선택된 종목 관심종목에 추가
  const addToWatchlist = () => {
    if (selectedStocks.size === 0) {
      alert('선택된 종목이 없습니다.');
      return;
    }
    // 실제로는 API 호출
    alert(`${selectedStocks.size}개 종목이 관심종목에 추가되었습니다.`);
    setSelectedStocks(new Set());
  };

  // 결과 내보내기
  const exportResults = () => {
    const csv = [
      ['종목코드', '종목명', '섹터', '현재가', '등락률', 'PER', 'PBR', 'ROE', '배당수익률', '시가총액'].join(','),
      ...filteredStocks.map(stock => [
        stock.code,
        stock.name,
        stock.sector,
        stock.price,
        stock.changePercent.toFixed(2) + '%',
        stock.per,
        stock.pbr,
        stock.roe.toFixed(2) + '%',
        stock.dividendYield + '%',
        stock.marketCap + '억원'
      ].join(','))
    ].join('\n');

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `stock_screener_${new Date().toISOString().split('T')[0]}.csv`;
    link.click();
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600 dark:text-gray-400">데이터를 불러오는 중...</p>
        </div>
      </div>
    );
  }

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
              <div className="h-6 w-px bg-gray-300 dark:bg-gray-700" />
              <h1 className="text-xl font-bold text-gray-900 dark:text-white">종목 스크리너</h1>
              <span className="px-2 py-1 bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400 rounded text-xs font-medium">
                기초
              </span>
            </div>
            
            <div className="flex items-center gap-3">
              <button 
                onClick={() => setShowFilters(!showFilters)}
                className="px-4 py-2 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors flex items-center gap-2"
              >
                <Filter className="w-4 h-4" />
                필터 {showFilters ? '숨기기' : '보기'}
              </button>
              <button 
                onClick={() => setShowSavedScreens(!showSavedScreens)}
                className="px-4 py-2 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors flex items-center gap-2"
              >
                <Save className="w-4 h-4" />
                저장된 스크린
              </button>
              <button 
                onClick={exportResults}
                className="px-4 py-2 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors flex items-center gap-2"
              >
                <Download className="w-4 h-4" />
                내보내기
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Summary */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm p-4 mb-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-6">
              <div className="flex items-center gap-2">
                <Search className="w-5 h-5 text-gray-400" />
                <span className="text-gray-600 dark:text-gray-400">검색 결과:</span>
                <span className="font-bold text-gray-900 dark:text-white">{filteredStocks.length}개</span>
              </div>
              <div className="flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-red-500" />
                <span className="text-gray-600 dark:text-gray-400">상승:</span>
                <span className="font-bold text-red-600">{filteredStocks.filter(s => s.change > 0).length}개</span>
              </div>
              <div className="flex items-center gap-2">
                <TrendingDown className="w-5 h-5 text-blue-500" />
                <span className="text-gray-600 dark:text-gray-400">하락:</span>
                <span className="font-bold text-blue-600">{filteredStocks.filter(s => s.change < 0).length}개</span>
              </div>
            </div>
            {selectedStocks.size > 0 && (
              <button
                onClick={addToWatchlist}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2"
              >
                <Star className="w-4 h-4" />
                관심종목 추가 ({selectedStocks.size})
              </button>
            )}
          </div>
        </div>

        <div className="flex gap-6">
          {/* Filters */}
          {showFilters && (
            <div className="w-80 bg-white dark:bg-gray-800 rounded-lg shadow-sm p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-gray-900 dark:text-white">필터 설정</h3>
                <button
                  onClick={resetFilters}
                  className="text-sm text-blue-600 hover:text-blue-700"
                >
                  초기화
                </button>
              </div>

              <div className="space-y-6">
                {/* 시장 필터 */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    시장
                  </label>
                  <div className="space-y-2">
                    {['KOSPI', 'KOSDAQ'].map(market => (
                      <label key={market} className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={filters.markets.includes(market)}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setFilters({ ...filters, markets: [...filters.markets, market] });
                            } else {
                              setFilters({ ...filters, markets: filters.markets.filter(m => m !== market) });
                            }
                          }}
                          className="rounded text-blue-600"
                        />
                        <span className="text-sm">{market}</span>
                      </label>
                    ))}
                  </div>
                </div>

                {/* 섹터 필터 */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    섹터
                  </label>
                  <div className="max-h-40 overflow-y-auto space-y-2 border border-gray-200 dark:border-gray-700 rounded-lg p-2">
                    {sectors.map(sector => (
                      <label key={sector} className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={filters.sectors.includes(sector)}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setFilters({ ...filters, sectors: [...filters.sectors, sector] });
                            } else {
                              setFilters({ ...filters, sectors: filters.sectors.filter(s => s !== sector) });
                            }
                          }}
                          className="rounded text-blue-600"
                        />
                        <span className="text-sm">{sector}</span>
                      </label>
                    ))}
                  </div>
                </div>

                {/* 가격 필터 */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    주가 (원)
                  </label>
                  <div className="grid grid-cols-2 gap-2">
                    <input
                      type="number"
                      placeholder="최소"
                      value={filters.priceMin || ''}
                      onChange={(e) => setFilters({ ...filters, priceMin: e.target.value ? parseInt(e.target.value) : null })}
                      className="px-3 py-1 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-sm"
                    />
                    <input
                      type="number"
                      placeholder="최대"
                      value={filters.priceMax || ''}
                      onChange={(e) => setFilters({ ...filters, priceMax: e.target.value ? parseInt(e.target.value) : null })}
                      className="px-3 py-1 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-sm"
                    />
                  </div>
                </div>

                {/* 시가총액 필터 */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    시가총액 (억원)
                  </label>
                  <div className="grid grid-cols-2 gap-2">
                    <input
                      type="number"
                      placeholder="최소"
                      value={filters.marketCapMin || ''}
                      onChange={(e) => setFilters({ ...filters, marketCapMin: e.target.value ? parseInt(e.target.value) : null })}
                      className="px-3 py-1 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-sm"
                    />
                    <input
                      type="number"
                      placeholder="최대"
                      value={filters.marketCapMax || ''}
                      onChange={(e) => setFilters({ ...filters, marketCapMax: e.target.value ? parseInt(e.target.value) : null })}
                      className="px-3 py-1 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-sm"
                    />
                  </div>
                </div>

                {/* PER 필터 */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    PER
                  </label>
                  <div className="grid grid-cols-2 gap-2">
                    <input
                      type="number"
                      placeholder="최소"
                      value={filters.perMin || ''}
                      onChange={(e) => setFilters({ ...filters, perMin: e.target.value ? parseFloat(e.target.value) : null })}
                      className="px-3 py-1 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-sm"
                    />
                    <input
                      type="number"
                      placeholder="최대"
                      value={filters.perMax || ''}
                      onChange={(e) => setFilters({ ...filters, perMax: e.target.value ? parseFloat(e.target.value) : null })}
                      className="px-3 py-1 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-sm"
                    />
                  </div>
                </div>

                {/* ROE 필터 */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    ROE (%)
                  </label>
                  <div className="grid grid-cols-2 gap-2">
                    <input
                      type="number"
                      placeholder="최소"
                      value={filters.roeMin || ''}
                      onChange={(e) => setFilters({ ...filters, roeMin: e.target.value ? parseFloat(e.target.value) : null })}
                      className="px-3 py-1 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-sm"
                    />
                    <input
                      type="number"
                      placeholder="최대"
                      value={filters.roeMax || ''}
                      onChange={(e) => setFilters({ ...filters, roeMax: e.target.value ? parseFloat(e.target.value) : null })}
                      className="px-3 py-1 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-sm"
                    />
                  </div>
                </div>

                {/* 배당수익률 필터 */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    배당수익률 (%)
                  </label>
                  <div className="grid grid-cols-2 gap-2">
                    <input
                      type="number"
                      placeholder="최소"
                      value={filters.dividendYieldMin || ''}
                      onChange={(e) => setFilters({ ...filters, dividendYieldMin: e.target.value ? parseFloat(e.target.value) : null })}
                      className="px-3 py-1 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-sm"
                    />
                    <input
                      type="number"
                      placeholder="최대"
                      value={filters.dividendYieldMax || ''}
                      onChange={(e) => setFilters({ ...filters, dividendYieldMax: e.target.value ? parseFloat(e.target.value) : null })}
                      className="px-3 py-1 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-sm"
                    />
                  </div>
                </div>

                <button
                  onClick={saveScreen}
                  className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center justify-center gap-2"
                >
                  <Save className="w-4 h-4" />
                  현재 스크린 저장
                </button>
              </div>
            </div>
          )}

          {/* Results Table */}
          <div className="flex-1">
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="bg-gray-50 dark:bg-gray-700/50 text-xs uppercase tracking-wider">
                      <th className="px-4 py-3 text-center">
                        <input
                          type="checkbox"
                          checked={selectedStocks.size === filteredStocks.length && filteredStocks.length > 0}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setSelectedStocks(new Set(filteredStocks.map(s => s.code)));
                            } else {
                              setSelectedStocks(new Set());
                            }
                          }}
                          className="rounded text-blue-600"
                        />
                      </th>
                      <th 
                        className="px-4 py-3 text-left font-medium text-gray-700 dark:text-gray-300 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700"
                        onClick={() => handleSort('name')}
                      >
                        <div className="flex items-center gap-1">
                          종목명
                          {sortBy === 'name' && (sortOrder === 'asc' ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />)}
                        </div>
                      </th>
                      <th 
                        className="px-4 py-3 text-right font-medium text-gray-700 dark:text-gray-300 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700"
                        onClick={() => handleSort('price')}
                      >
                        <div className="flex items-center justify-end gap-1">
                          현재가
                          {sortBy === 'price' && (sortOrder === 'asc' ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />)}
                        </div>
                      </th>
                      <th 
                        className="px-4 py-3 text-right font-medium text-gray-700 dark:text-gray-300 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700"
                        onClick={() => handleSort('changePercent')}
                      >
                        <div className="flex items-center justify-end gap-1">
                          등락률
                          {sortBy === 'changePercent' && (sortOrder === 'asc' ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />)}
                        </div>
                      </th>
                      <th 
                        className="px-4 py-3 text-right font-medium text-gray-700 dark:text-gray-300 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700"
                        onClick={() => handleSort('volume')}
                      >
                        <div className="flex items-center justify-end gap-1">
                          거래량
                          {sortBy === 'volume' && (sortOrder === 'asc' ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />)}
                        </div>
                      </th>
                      <th 
                        className="px-4 py-3 text-right font-medium text-gray-700 dark:text-gray-300 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700"
                        onClick={() => handleSort('marketCap')}
                      >
                        <div className="flex items-center justify-end gap-1">
                          시가총액
                          {sortBy === 'marketCap' && (sortOrder === 'asc' ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />)}
                        </div>
                      </th>
                      <th 
                        className="px-4 py-3 text-right font-medium text-gray-700 dark:text-gray-300 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700"
                        onClick={() => handleSort('per')}
                      >
                        <div className="flex items-center justify-end gap-1">
                          PER
                          {sortBy === 'per' && (sortOrder === 'asc' ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />)}
                        </div>
                      </th>
                      <th 
                        className="px-4 py-3 text-right font-medium text-gray-700 dark:text-gray-300 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700"
                        onClick={() => handleSort('pbr')}
                      >
                        <div className="flex items-center justify-end gap-1">
                          PBR
                          {sortBy === 'pbr' && (sortOrder === 'asc' ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />)}
                        </div>
                      </th>
                      <th 
                        className="px-4 py-3 text-right font-medium text-gray-700 dark:text-gray-300 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700"
                        onClick={() => handleSort('roe')}
                      >
                        <div className="flex items-center justify-end gap-1">
                          ROE
                          {sortBy === 'roe' && (sortOrder === 'asc' ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />)}
                        </div>
                      </th>
                      <th 
                        className="px-4 py-3 text-right font-medium text-gray-700 dark:text-gray-300 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700"
                        onClick={() => handleSort('dividendYield')}
                      >
                        <div className="flex items-center justify-end gap-1">
                          배당
                          {sortBy === 'dividendYield' && (sortOrder === 'asc' ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />)}
                        </div>
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                    {filteredStocks.map((stock) => (
                      <tr key={stock.code} className="hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors">
                        <td className="px-4 py-3 text-center">
                          <input
                            type="checkbox"
                            checked={selectedStocks.has(stock.code)}
                            onChange={(e) => {
                              const newSelected = new Set(selectedStocks);
                              if (e.target.checked) {
                                newSelected.add(stock.code);
                              } else {
                                newSelected.delete(stock.code);
                              }
                              setSelectedStocks(newSelected);
                            }}
                            className="rounded text-blue-600"
                          />
                        </td>
                        <td className="px-4 py-3">
                          <div>
                            <p className="font-medium text-gray-900 dark:text-white">{stock.name}</p>
                            <p className="text-xs text-gray-500">{stock.code} • {stock.sector}</p>
                          </div>
                        </td>
                        <td className="px-4 py-3 text-right font-medium text-gray-900 dark:text-white">
                          {stock.price.toLocaleString()}
                        </td>
                        <td className="px-4 py-3 text-right">
                          <span className={`font-medium ${stock.change >= 0 ? 'text-red-600' : 'text-blue-600'}`}>
                            {stock.change >= 0 ? '+' : ''}{stock.changePercent.toFixed(2)}%
                          </span>
                        </td>
                        <td className="px-4 py-3 text-right text-gray-900 dark:text-white">
                          {(stock.volume / 1000).toFixed(0).toLocaleString()}K
                        </td>
                        <td className="px-4 py-3 text-right text-gray-900 dark:text-white">
                          {stock.marketCap.toLocaleString()}억
                        </td>
                        <td className="px-4 py-3 text-right text-gray-900 dark:text-white">
                          {stock.per.toFixed(2)}
                        </td>
                        <td className="px-4 py-3 text-right text-gray-900 dark:text-white">
                          {stock.pbr.toFixed(2)}
                        </td>
                        <td className="px-4 py-3 text-right text-gray-900 dark:text-white">
                          {stock.roe.toFixed(1)}%
                        </td>
                        <td className="px-4 py-3 text-right text-gray-900 dark:text-white">
                          {stock.dividendYield}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              
              {filteredStocks.length === 0 && (
                <div className="text-center py-12">
                  <AlertCircle className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-600 dark:text-gray-400">조건에 맞는 종목이 없습니다.</p>
                  <button
                    onClick={resetFilters}
                    className="mt-4 text-blue-600 hover:text-blue-700 text-sm font-medium"
                  >
                    필터 초기화
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Saved Screens Modal */}
        {showSavedScreens && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl p-6 w-full max-w-2xl max-h-[80vh] overflow-y-auto">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-bold text-gray-900 dark:text-white">저장된 스크린</h2>
                <button
                  onClick={() => setShowSavedScreens(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  ✕
                </button>
              </div>
              
              {savedScreens.length === 0 ? (
                <p className="text-center text-gray-500 py-8">저장된 스크린이 없습니다.</p>
              ) : (
                <div className="space-y-3">
                  {savedScreens.map((screen) => (
                    <div key={screen.id} className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 flex items-center justify-between">
                      <div>
                        <h3 className="font-medium text-gray-900 dark:text-white">{screen.name}</h3>
                        <p className="text-sm text-gray-500">
                          {new Date(screen.createdAt).toLocaleString()}
                        </p>
                      </div>
                      <div className="flex gap-2">
                        <button
                          onClick={() => loadScreen(screen)}
                          className="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700"
                        >
                          불러오기
                        </button>
                        <button
                          onClick={() => deleteScreen(screen.id)}
                          className="px-3 py-1 bg-red-600 text-white rounded text-sm hover:bg-red-700"
                        >
                          삭제
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}