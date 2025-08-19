'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { ArrowLeft, PieChart, Plus, Edit2, Trash2, TrendingUp, TrendingDown, DollarSign, Percent, Calendar, Download, RefreshCw, BarChart3, Activity, AlertCircle, Search, X, Check } from 'lucide-react';

interface Stock {
  id: string;
  symbolId: string;
  symbol: {
    code: string;
    name: string;
    sector?: string;
  };
  quantity: number;
  purchasePrice: number;
  currentPrice: number;
  purchaseDate: string;
  notes?: string;
}

interface Portfolio {
  id: string;
  name: string;
  description?: string;
  items: Stock[];
  cash: number;
  totalValue: number;
  totalCost: number;
  totalReturn: number;
  returnPercent: number;
  isActive: boolean;
  createdAt: string;
  updatedAt: string;
}

interface SearchResult {
  code: string;
  name: string;
  market: string;
  sector: string;
  display: string;
}

interface SectorAllocation {
  sector: string;
  value: number;
  percentage: number;
  color: string;
}

export default function PortfolioTrackerWithAPI() {
  const [portfolios, setPortfolios] = useState<Portfolio[]>([]);
  const [selectedPortfolio, setSelectedPortfolio] = useState<Portfolio | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  const [showAddModal, setShowAddModal] = useState(false);
  const [showCreatePortfolioModal, setShowCreatePortfolioModal] = useState(false);
  const [editingStock, setEditingStock] = useState<Stock | null>(null);
  
  // 종목 검색 관련
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [selectedStock, setSelectedStock] = useState<SearchResult | null>(null);
  
  const [newStock, setNewStock] = useState({
    symbolCode: '',
    symbolName: '',
    quantity: 0,
    purchasePrice: 0,
    purchaseDate: new Date().toISOString().split('T')[0],
    notes: ''
  });
  
  const [newPortfolio, setNewPortfolio] = useState({
    name: '',
    description: '',
    initialCash: 10000000
  });
  
  const [selectedView, setSelectedView] = useState<'overview' | 'holdings' | 'performance' | 'allocation'>('overview');

  // 섹터 색상 정의
  const sectorColors: Record<string, string> = {
    '전기전자': '#3b82f6',
    '금융': '#10b981',
    '바이오': '#f59e0b',
    'IT': '#ef4444',
    '자동차': '#8b5cf6',
    '화학': '#6366f1',
    '철강': '#ec4899',
    '유통': '#14b8a6',
    '보험': '#f97316',
    '통신': '#84cc16'
  };

  // 포트폴리오 목록 조회
  const fetchPortfolios = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/portfolio');
      if (!response.ok) throw new Error('Failed to fetch portfolios');
      const data = await response.json();
      setPortfolios(data);
      if (data.length > 0 && !selectedPortfolio) {
        setSelectedPortfolio(data[0]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load portfolios');
    } finally {
      setLoading(false);
    }
  };

  // 포트폴리오 생성
  const createPortfolio = async () => {
    try {
      const response = await fetch('/api/portfolio', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newPortfolio)
      });
      
      if (!response.ok) throw new Error('Failed to create portfolio');
      
      const created = await response.json();
      setPortfolios([...portfolios, created]);
      setSelectedPortfolio(created);
      setShowCreatePortfolioModal(false);
      setNewPortfolio({ name: '', description: '', initialCash: 10000000 });
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to create portfolio');
    }
  };

  // 종목 검색
  const searchStocks = async (query: string) => {
    if (query.length < 1) {
      setSearchResults([]);
      return;
    }
    
    try {
      setSearching(true);
      const response = await fetch(`/api/stock/search?q=${encodeURIComponent(query)}`);
      if (!response.ok) throw new Error('Failed to search stocks');
      const data = await response.json();
      setSearchResults(data.results || []);
    } catch (err) {
      console.error('Search error:', err);
    } finally {
      setSearching(false);
    }
  };

  // 종목 추가
  const addStock = async () => {
    if (!selectedPortfolio || !selectedStock) return;
    
    try {
      const response = await fetch('/api/portfolio/items', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          portfolioId: selectedPortfolio.id,
          symbolId: selectedStock.code, // 임시로 code를 ID로 사용
          quantity: newStock.quantity,
          purchasePrice: newStock.purchasePrice,
          purchaseDate: newStock.purchaseDate,
          notes: newStock.notes
        })
      });
      
      if (!response.ok) throw new Error('Failed to add stock');
      
      // 포트폴리오 새로고침
      await fetchPortfolios();
      
      // 모달 닫기 및 초기화
      setShowAddModal(false);
      setNewStock({
        symbolCode: '',
        symbolName: '',
        quantity: 0,
        purchasePrice: 0,
        purchaseDate: new Date().toISOString().split('T')[0],
        notes: ''
      });
      setSelectedStock(null);
      setSearchQuery('');
      setSearchResults([]);
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to add stock');
    }
  };

  // 종목 삭제
  const deleteStock = async (stockId: string) => {
    if (!confirm('정말로 이 종목을 삭제하시겠습니까?')) return;
    
    try {
      const response = await fetch(`/api/portfolio/items?id=${stockId}`, {
        method: 'DELETE'
      });
      
      if (!response.ok) throw new Error('Failed to delete stock');
      
      await fetchPortfolios();
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to delete stock');
    }
  };

  // 초기 데이터 로드
  useEffect(() => {
    fetchPortfolios();
  }, []);

  // 종목 검색 디바운싱
  useEffect(() => {
    const timer = setTimeout(() => {
      searchStocks(searchQuery);
    }, 300);
    
    return () => clearTimeout(timer);
  }, [searchQuery]);

  // 섹터별 자산 배분 계산
  const calculateSectorAllocation = (): SectorAllocation[] => {
    if (!selectedPortfolio) return [];
    
    const sectorMap = new Map<string, number>();
    
    selectedPortfolio.items.forEach(item => {
      const sector = item.symbol.sector || '기타';
      const value = item.quantity * item.currentPrice;
      sectorMap.set(sector, (sectorMap.get(sector) || 0) + value);
    });
    
    const totalStockValue = Array.from(sectorMap.values()).reduce((sum, val) => sum + val, 0);
    
    return Array.from(sectorMap.entries()).map(([sector, value]) => ({
      sector,
      value,
      percentage: (value / totalStockValue) * 100,
      color: sectorColors[sector] || '#6b7280'
    })).sort((a, b) => b.value - a.value);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link href="/modules/stock-analysis/tools" className="text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white">
                <ArrowLeft className="w-5 h-5" />
              </Link>
              <h1 className="text-xl font-bold text-gray-900 dark:text-white">포트폴리오 트래커</h1>
            </div>
            
            <div className="flex items-center gap-3">
              {portfolios.length > 0 && (
                <select 
                  value={selectedPortfolio?.id || ''}
                  onChange={(e) => {
                    const portfolio = portfolios.find(p => p.id === e.target.value);
                    setSelectedPortfolio(portfolio || null);
                  }}
                  className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                >
                  {portfolios.map(portfolio => (
                    <option key={portfolio.id} value={portfolio.id}>
                      {portfolio.name}
                    </option>
                  ))}
                </select>
              )}
              
              <button
                onClick={() => setShowCreatePortfolioModal(true)}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                새 포트폴리오
              </button>
              
              <button
                onClick={fetchPortfolios}
                className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
              >
                <RefreshCw className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </div>

      {error && (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-4">
          <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400 mt-0.5" />
            <div>
              <p className="text-sm text-red-800 dark:text-red-200">{error}</p>
            </div>
          </div>
        </div>
      )}

      {selectedPortfolio ? (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* 포트폴리오 요약 */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
              <div className="flex items-center justify-between mb-2">
                <p className="text-sm text-gray-600 dark:text-gray-400">총 자산</p>
                <DollarSign className="w-5 h-5 text-gray-400" />
              </div>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                ₩{selectedPortfolio.totalValue.toLocaleString()}
              </p>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
              <div className="flex items-center justify-between mb-2">
                <p className="text-sm text-gray-600 dark:text-gray-400">총 수익률</p>
                <Percent className="w-5 h-5 text-gray-400" />
              </div>
              <p className={`text-2xl font-bold ${selectedPortfolio.returnPercent >= 0 ? 'text-red-600' : 'text-blue-600'}`}>
                {selectedPortfolio.returnPercent >= 0 ? '+' : ''}{selectedPortfolio.returnPercent.toFixed(2)}%
              </p>
              <p className={`text-sm ${selectedPortfolio.totalReturn >= 0 ? 'text-red-600' : 'text-blue-600'}`}>
                {selectedPortfolio.totalReturn >= 0 ? '+' : ''}₩{Math.abs(selectedPortfolio.totalReturn).toLocaleString()}
              </p>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
              <div className="flex items-center justify-between mb-2">
                <p className="text-sm text-gray-600 dark:text-gray-400">보유 종목</p>
                <BarChart3 className="w-5 h-5 text-gray-400" />
              </div>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {selectedPortfolio.items.length}개
              </p>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
              <div className="flex items-center justify-between mb-2">
                <p className="text-sm text-gray-600 dark:text-gray-400">현금</p>
                <Activity className="w-5 h-5 text-gray-400" />
              </div>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                ₩{selectedPortfolio.cash.toLocaleString()}
              </p>
            </div>
          </div>

          {/* 탭 네비게이션 */}
          <div className="flex gap-4 mb-6">
            <button
              onClick={() => setSelectedView('overview')}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                selectedView === 'overview'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
              }`}
            >
              개요
            </button>
            <button
              onClick={() => setSelectedView('holdings')}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                selectedView === 'holdings'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
              }`}
            >
              보유 종목
            </button>
            <button
              onClick={() => setSelectedView('allocation')}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                selectedView === 'allocation'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
              }`}
            >
              자산 배분
            </button>
          </div>

          {/* 보유 종목 목록 */}
          {selectedView === 'holdings' && (
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm">
              <div className="p-6 border-b border-gray-200 dark:border-gray-700">
                <div className="flex items-center justify-between">
                  <h2 className="text-lg font-semibold text-gray-900 dark:text-white">보유 종목</h2>
                  <button
                    onClick={() => setShowAddModal(true)}
                    className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                  >
                    <Plus className="w-4 h-4" />
                    종목 추가
                  </button>
                </div>
              </div>
              
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-50 dark:bg-gray-700/50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">종목</th>
                      <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">수량</th>
                      <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">매입가</th>
                      <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">현재가</th>
                      <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">평가금액</th>
                      <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">수익률</th>
                      <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">작업</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                    {selectedPortfolio.items.map((stock) => {
                      const totalValue = stock.quantity * stock.currentPrice;
                      const totalCost = stock.quantity * stock.purchasePrice;
                      const profit = totalValue - totalCost;
                      const profitPercent = (profit / totalCost) * 100;
                      
                      return (
                        <tr key={stock.id}>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div>
                              <p className="text-sm font-medium text-gray-900 dark:text-white">{stock.symbol.name}</p>
                              <p className="text-xs text-gray-500">{stock.symbol.code}</p>
                            </div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-right text-sm text-gray-900 dark:text-white">
                            {stock.quantity.toLocaleString()}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-right text-sm text-gray-900 dark:text-white">
                            ₩{stock.purchasePrice.toLocaleString()}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-right text-sm text-gray-900 dark:text-white">
                            ₩{stock.currentPrice.toLocaleString()}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-right text-sm text-gray-900 dark:text-white">
                            ₩{totalValue.toLocaleString()}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-right">
                            <div className={`text-sm font-medium ${profitPercent >= 0 ? 'text-red-600' : 'text-blue-600'}`}>
                              {profitPercent >= 0 ? '+' : ''}{profitPercent.toFixed(2)}%
                            </div>
                            <div className={`text-xs ${profit >= 0 ? 'text-red-600' : 'text-blue-600'}`}>
                              {profit >= 0 ? '+' : ''}₩{Math.abs(profit).toLocaleString()}
                            </div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-center">
                            <button
                              onClick={() => deleteStock(stock.id)}
                              className="text-red-600 hover:text-red-800 dark:text-red-400 dark:hover:text-red-300"
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* 자산 배분 */}
          {selectedView === 'allocation' && (
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">섹터별 자산 배분</h2>
              <div className="space-y-4">
                {calculateSectorAllocation().map((sector) => (
                  <div key={sector.sector}>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-gray-700 dark:text-gray-300">{sector.sector}</span>
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        {sector.percentage.toFixed(1)}% (₩{sector.value.toLocaleString()})
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                      <div
                        className="h-2 rounded-full transition-all duration-500"
                        style={{
                          width: `${sector.percentage}%`,
                          backgroundColor: sector.color
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16 text-center">
          <p className="text-gray-500 dark:text-gray-400 mb-4">포트폴리오가 없습니다.</p>
          <button
            onClick={() => setShowCreatePortfolioModal(true)}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            첫 포트폴리오 만들기
          </button>
        </div>
      )}

      {/* 종목 추가 모달 */}
      {showAddModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 w-full max-w-md">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">종목 추가</h3>
              <button
                onClick={() => {
                  setShowAddModal(false);
                  setSelectedStock(null);
                  setSearchQuery('');
                  setSearchResults([]);
                }}
                className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="space-y-4">
              {/* 종목 검색 */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  종목 검색
                </label>
                <div className="relative">
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="종목코드 또는 종목명 입력"
                    className="w-full px-4 py-2 pl-10 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  />
                  <Search className="absolute left-3 top-2.5 w-4 h-4 text-gray-400" />
                </div>
                
                {/* 검색 결과 */}
                {searchResults.length > 0 && (
                  <div className="mt-2 max-h-40 overflow-y-auto bg-gray-50 dark:bg-gray-700 rounded-lg divide-y divide-gray-200 dark:divide-gray-600">
                    {searchResults.map((result) => (
                      <button
                        key={result.code}
                        onClick={() => {
                          setSelectedStock(result);
                          setNewStock({
                            ...newStock,
                            symbolCode: result.code,
                            symbolName: result.name
                          });
                          setSearchQuery(result.display);
                          setSearchResults([]);
                        }}
                        className="w-full px-4 py-2 text-left hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors"
                      >
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-sm font-medium text-gray-900 dark:text-white">{result.name}</p>
                            <p className="text-xs text-gray-500">{result.code} • {result.market}</p>
                          </div>
                          <span className="text-xs text-gray-400">{result.sector}</span>
                        </div>
                      </button>
                    ))}
                  </div>
                )}
                
                {searching && (
                  <p className="mt-2 text-sm text-gray-500">검색 중...</p>
                )}
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    수량
                  </label>
                  <input
                    type="number"
                    value={newStock.quantity}
                    onChange={(e) => setNewStock({ ...newStock, quantity: parseInt(e.target.value) || 0 })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    매입가
                  </label>
                  <input
                    type="number"
                    value={newStock.purchasePrice}
                    onChange={(e) => setNewStock({ ...newStock, purchasePrice: parseInt(e.target.value) || 0 })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  />
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  매입일
                </label>
                <input
                  type="date"
                  value={newStock.purchaseDate}
                  onChange={(e) => setNewStock({ ...newStock, purchaseDate: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  메모 (선택)
                </label>
                <textarea
                  value={newStock.notes}
                  onChange={(e) => setNewStock({ ...newStock, notes: e.target.value })}
                  rows={2}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                />
              </div>
            </div>
            
            <div className="flex gap-3 mt-6">
              <button
                onClick={addStock}
                disabled={!selectedStock || newStock.quantity <= 0 || newStock.purchasePrice <= 0}
                className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                추가
              </button>
              <button
                onClick={() => {
                  setShowAddModal(false);
                  setSelectedStock(null);
                  setSearchQuery('');
                  setSearchResults([]);
                }}
                className="flex-1 px-4 py-2 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
              >
                취소
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 포트폴리오 생성 모달 */}
      {showCreatePortfolioModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 w-full max-w-md">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">새 포트폴리오 만들기</h3>
              <button
                onClick={() => setShowCreatePortfolioModal(false)}
                className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  포트폴리오 이름
                </label>
                <input
                  type="text"
                  value={newPortfolio.name}
                  onChange={(e) => setNewPortfolio({ ...newPortfolio, name: e.target.value })}
                  placeholder="예: 성장주 포트폴리오"
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  설명 (선택)
                </label>
                <textarea
                  value={newPortfolio.description}
                  onChange={(e) => setNewPortfolio({ ...newPortfolio, description: e.target.value })}
                  rows={2}
                  placeholder="포트폴리오 전략이나 목표를 입력하세요"
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  초기 현금
                </label>
                <input
                  type="number"
                  value={newPortfolio.initialCash}
                  onChange={(e) => setNewPortfolio({ ...newPortfolio, initialCash: parseInt(e.target.value) || 0 })}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                />
              </div>
            </div>
            
            <div className="flex gap-3 mt-6">
              <button
                onClick={createPortfolio}
                disabled={!newPortfolio.name}
                className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                생성
              </button>
              <button
                onClick={() => setShowCreatePortfolioModal(false)}
                className="flex-1 px-4 py-2 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
              >
                취소
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}