'use client';

import { useState, useEffect } from 'react';
import {
  Database, Clock, GitBranch, RefreshCw,
  Search, Filter, Calendar, Code,
  TrendingUp, AlertCircle, CheckCircle, Zap
} from 'lucide-react';

interface Feature {
  id: string;
  name: string;
  type: 'online' | 'offline' | 'both';
  dataType: string;
  version: string;
  lastUpdated: Date;
  freshness: number;
  usage: number;
  description: string;
  transformation: string;
}

interface FeatureVersion {
  version: string;
  timestamp: Date;
  schema: Record<string, string>;
  status: 'active' | 'deprecated' | 'experimental';
}

export default function FeatureStoreExplorer() {
  const [features, setFeatures] = useState<Feature[]>([]);
  const [selectedFeature, setSelectedFeature] = useState<Feature | null>(null);
  const [filterType, setFilterType] = useState<'all' | 'online' | 'offline' | 'both'>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [versions, setVersions] = useState<FeatureVersion[]>([]);
  const [showTransformation, setShowTransformation] = useState(false);

  // Initialize features
  useEffect(() => {
    const sampleFeatures: Feature[] = [
      {
        id: 'user_age',
        name: 'user_age',
        type: 'both',
        dataType: 'int',
        version: 'v2.1',
        lastUpdated: new Date(Date.now() - 1000 * 60 * 30),
        freshness: 95,
        usage: 87,
        description: '사용자의 나이 (년)',
        transformation: 'SELECT FLOOR(DATEDIFF(CURDATE(), birth_date) / 365.25) AS user_age FROM users'
      },
      {
        id: 'user_purchase_count_7d',
        name: 'user_purchase_count_7d',
        type: 'online',
        dataType: 'int',
        version: 'v1.5',
        lastUpdated: new Date(Date.now() - 1000 * 60 * 5),
        freshness: 98,
        usage: 92,
        description: '최근 7일간 구매 횟수',
        transformation: 'SELECT COUNT(*) FROM purchases WHERE user_id = ? AND timestamp > NOW() - INTERVAL 7 DAY'
      },
      {
        id: 'user_avg_session_duration',
        name: 'user_avg_session_duration',
        type: 'offline',
        dataType: 'float',
        version: 'v1.0',
        lastUpdated: new Date(Date.now() - 1000 * 60 * 60 * 2),
        freshness: 75,
        usage: 68,
        description: '평균 세션 지속 시간 (분)',
        transformation: 'SELECT AVG(session_duration_minutes) FROM sessions WHERE user_id = ? GROUP BY user_id'
      },
      {
        id: 'item_embedding_vector',
        name: 'item_embedding_vector',
        type: 'both',
        dataType: 'array<float>',
        version: 'v3.0',
        lastUpdated: new Date(Date.now() - 1000 * 60 * 45),
        freshness: 88,
        usage: 95,
        description: '아이템 임베딩 벡터 (128차원)',
        transformation: 'model.encode(item_title + " " + item_description)'
      },
      {
        id: 'user_ltv_prediction',
        name: 'user_ltv_prediction',
        type: 'offline',
        dataType: 'float',
        version: 'v2.0',
        lastUpdated: new Date(Date.now() - 1000 * 60 * 60 * 24),
        freshness: 50,
        usage: 78,
        description: '예측된 고객 생애 가치 (LTV)',
        transformation: 'ltv_model.predict(user_features)'
      },
      {
        id: 'real_time_clicks_1h',
        name: 'real_time_clicks_1h',
        type: 'online',
        dataType: 'int',
        version: 'v1.0',
        lastUpdated: new Date(Date.now() - 1000 * 60 * 2),
        freshness: 99,
        usage: 85,
        description: '최근 1시간 클릭 수',
        transformation: 'SELECT COUNT(*) FROM clicks WHERE user_id = ? AND timestamp > NOW() - INTERVAL 1 HOUR'
      }
    ];

    setFeatures(sampleFeatures);
    setSelectedFeature(sampleFeatures[0]);
  }, []);

  // Generate version history
  useEffect(() => {
    if (!selectedFeature) return;

    const versionHistory: FeatureVersion[] = [];
    const currentVersion = parseInt(selectedFeature.version.substring(1));

    for (let i = currentVersion; i >= 1; i--) {
      versionHistory.push({
        version: `v${i}.0`,
        timestamp: new Date(Date.now() - (currentVersion - i) * 1000 * 60 * 60 * 24 * 30),
        schema: {
          name: selectedFeature.name,
          type: selectedFeature.dataType,
          nullable: i < currentVersion ? 'true' : 'false'
        },
        status: i === currentVersion ? 'active' : i === currentVersion - 1 ? 'deprecated' : 'experimental'
      });
    }

    setVersions(versionHistory);
  }, [selectedFeature]);

  const filteredFeatures = features.filter(f => {
    const matchesType = filterType === 'all' || f.type === filterType;
    const matchesSearch = f.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      f.description.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesType && matchesSearch;
  });

  const getFreshnessColor = (freshness: number) => {
    if (freshness >= 90) return 'text-green-400';
    if (freshness >= 70) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getFreshnessIcon = (freshness: number) => {
    if (freshness >= 90) return <CheckCircle className="w-4 h-4 text-green-400" />;
    if (freshness >= 70) return <AlertCircle className="w-4 h-4 text-yellow-400" />;
    return <AlertCircle className="w-4 h-4 text-red-400" />;
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'online': return 'bg-blue-600';
      case 'offline': return 'bg-purple-600';
      case 'both': return 'bg-green-600';
      default: return 'bg-slate-600';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-gray-900 text-white p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="bg-gradient-to-r from-slate-800 to-slate-700 rounded-xl p-6 border border-slate-600">
          <div className="flex items-center gap-3 mb-4">
            <div className="bg-cyan-500 p-3 rounded-lg">
              <Database className="w-8 h-8" />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Feature Store 탐색기</h1>
              <p className="text-slate-300">Feature Store Architecture Explorer</p>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-1">
                <Database className="w-4 h-4 text-cyan-400" />
                <span className="text-sm text-slate-300">총 피처</span>
              </div>
              <div className="text-2xl font-bold">{features.length}</div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-1">
                <Zap className="w-4 h-4 text-blue-400" />
                <span className="text-sm text-slate-300">Online</span>
              </div>
              <div className="text-2xl font-bold">
                {features.filter(f => f.type === 'online' || f.type === 'both').length}
              </div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-1">
                <Database className="w-4 h-4 text-purple-400" />
                <span className="text-sm text-slate-300">Offline</span>
              </div>
              <div className="text-2xl font-bold">
                {features.filter(f => f.type === 'offline' || f.type === 'both').length}
              </div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-1">
                <TrendingUp className="w-4 h-4 text-green-400" />
                <span className="text-sm text-slate-300">평균 사용률</span>
              </div>
              <div className="text-2xl font-bold">
                {(features.reduce((sum, f) => sum + f.usage, 0) / features.length).toFixed(0)}%
              </div>
            </div>
          </div>
        </div>

        {/* Search and Filter */}
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-semibold mb-2 flex items-center gap-2">
                <Search className="w-4 h-4 text-blue-400" />
                피처 검색
              </label>
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="피처 이름 또는 설명 검색..."
                className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2"
              />
            </div>
            <div>
              <label className="block text-sm font-semibold mb-2 flex items-center gap-2">
                <Filter className="w-4 h-4 text-purple-400" />
                타입 필터
              </label>
              <select
                value={filterType}
                onChange={(e) => setFilterType(e.target.value as any)}
                className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2"
              >
                <option value="all">전체</option>
                <option value="online">Online만</option>
                <option value="offline">Offline만</option>
                <option value="both">하이브리드</option>
              </select>
            </div>
          </div>
        </div>

        {/* Feature List and Details */}
        <div className="grid lg:grid-cols-2 gap-6">
          {/* Feature List */}
          <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
            <h2 className="text-xl font-bold mb-4">피처 카탈로그</h2>
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {filteredFeatures.map(feature => (
                <button
                  key={feature.id}
                  onClick={() => setSelectedFeature(feature)}
                  className={`w-full text-left p-4 rounded-lg border transition-all ${
                    selectedFeature?.id === feature.id
                      ? 'border-cyan-500 bg-slate-700'
                      : 'border-slate-700 bg-slate-800 hover:border-slate-600'
                  }`}
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex-1">
                      <div className="font-semibold text-lg">{feature.name}</div>
                      <div className="text-sm text-slate-400">{feature.description}</div>
                    </div>
                    <span className={`px-2 py-1 rounded text-xs font-semibold ${getTypeColor(feature.type)}`}>
                      {feature.type}
                    </span>
                  </div>
                  <div className="flex items-center gap-4 text-sm">
                    <div className="flex items-center gap-1">
                      {getFreshnessIcon(feature.freshness)}
                      <span className={getFreshnessColor(feature.freshness)}>
                        {feature.freshness}%
                      </span>
                    </div>
                    <div className="text-slate-400">
                      {feature.dataType}
                    </div>
                    <div className="text-slate-500">
                      {feature.version}
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Feature Details */}
          <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
            <h2 className="text-xl font-bold mb-4">피처 상세 정보</h2>
            {selectedFeature ? (
              <div className="space-y-4">
                <div>
                  <div className="text-sm text-slate-400 mb-1">피처 이름</div>
                  <div className="text-xl font-bold">{selectedFeature.name}</div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-sm text-slate-400 mb-1">데이터 타입</div>
                    <div className="font-semibold">{selectedFeature.dataType}</div>
                  </div>
                  <div>
                    <div className="text-sm text-slate-400 mb-1">버전</div>
                    <div className="font-semibold">{selectedFeature.version}</div>
                  </div>
                  <div>
                    <div className="text-sm text-slate-400 mb-1">저장소 타입</div>
                    <span className={`px-2 py-1 rounded text-sm font-semibold ${getTypeColor(selectedFeature.type)}`}>
                      {selectedFeature.type}
                    </span>
                  </div>
                  <div>
                    <div className="text-sm text-slate-400 mb-1">사용률</div>
                    <div className="font-semibold">{selectedFeature.usage}%</div>
                  </div>
                </div>

                <div>
                  <div className="text-sm text-slate-400 mb-1">설명</div>
                  <div className="text-slate-200">{selectedFeature.description}</div>
                </div>

                <div>
                  <div className="flex items-center justify-between mb-2">
                    <div className="text-sm text-slate-400">최신성</div>
                    <div className="flex items-center gap-2">
                      {getFreshnessIcon(selectedFeature.freshness)}
                      <span className={`font-semibold ${getFreshnessColor(selectedFeature.freshness)}`}>
                        {selectedFeature.freshness}%
                      </span>
                    </div>
                  </div>
                  <div className="bg-slate-700 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full ${
                        selectedFeature.freshness >= 90
                          ? 'bg-green-500'
                          : selectedFeature.freshness >= 70
                          ? 'bg-yellow-500'
                          : 'bg-red-500'
                      }`}
                      style={{ width: `${selectedFeature.freshness}%` }}
                    />
                  </div>
                </div>

                <div>
                  <div className="text-sm text-slate-400 mb-1">마지막 업데이트</div>
                  <div className="flex items-center gap-2">
                    <Clock className="w-4 h-4 text-blue-400" />
                    <span>{selectedFeature.lastUpdated.toLocaleString('ko-KR')}</span>
                  </div>
                </div>

                <div>
                  <button
                    onClick={() => setShowTransformation(!showTransformation)}
                    className="flex items-center gap-2 text-cyan-400 hover:text-cyan-300 font-semibold"
                  >
                    <Code className="w-4 h-4" />
                    {showTransformation ? '변환 로직 숨기기' : '변환 로직 보기'}
                  </button>
                  {showTransformation && (
                    <pre className="mt-2 bg-slate-900 rounded-lg p-4 text-sm overflow-x-auto">
                      <code className="text-green-400">{selectedFeature.transformation}</code>
                    </pre>
                  )}
                </div>
              </div>
            ) : (
              <div className="text-center text-slate-400 py-8">
                피처를 선택하세요
              </div>
            )}
          </div>
        </div>

        {/* Version History */}
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
            <GitBranch className="w-6 h-6 text-purple-400" />
            버전 히스토리
          </h2>
          {selectedFeature && versions.length > 0 ? (
            <div className="space-y-3">
              {versions.map((version, idx) => (
                <div
                  key={version.version}
                  className={`p-4 rounded-lg border ${
                    version.status === 'active'
                      ? 'border-green-500 bg-green-500/10'
                      : version.status === 'deprecated'
                      ? 'border-yellow-500 bg-yellow-500/10'
                      : 'border-slate-600 bg-slate-700/50'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-3">
                      <span className="font-bold text-lg">{version.version}</span>
                      <span className={`px-2 py-1 rounded text-xs font-semibold ${
                        version.status === 'active'
                          ? 'bg-green-600'
                          : version.status === 'deprecated'
                          ? 'bg-yellow-600'
                          : 'bg-slate-600'
                      }`}>
                        {version.status}
                      </span>
                    </div>
                    <div className="flex items-center gap-2 text-sm text-slate-400">
                      <Calendar className="w-4 h-4" />
                      {version.timestamp.toLocaleDateString('ko-KR')}
                    </div>
                  </div>
                  <div className="text-sm text-slate-300">
                    Schema: {JSON.stringify(version.schema, null, 2)}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center text-slate-400 py-8">
              피처를 선택하여 버전 히스토리를 확인하세요
            </div>
          )}
        </div>

        {/* Architecture Diagram */}
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <h2 className="text-xl font-bold mb-4">Feature Store 아키텍처</h2>
          <div className="bg-slate-900 rounded-lg p-6">
            <div className="grid md:grid-cols-3 gap-6">
              <div className="text-center">
                <div className="bg-blue-600 rounded-lg p-6 mb-3">
                  <Zap className="w-12 h-12 mx-auto" />
                </div>
                <h3 className="font-bold text-lg mb-2">Online Feature Store</h3>
                <p className="text-sm text-slate-400">
                  저지연 실시간 피처 서빙
                  <br />
                  Redis, DynamoDB
                </p>
              </div>
              <div className="text-center">
                <div className="bg-purple-600 rounded-lg p-6 mb-3">
                  <Database className="w-12 h-12 mx-auto" />
                </div>
                <h3 className="font-bold text-lg mb-2">Offline Feature Store</h3>
                <p className="text-sm text-slate-400">
                  대용량 배치 학습 데이터
                  <br />
                  S3, BigQuery, Snowflake
                </p>
              </div>
              <div className="text-center">
                <div className="bg-green-600 rounded-lg p-6 mb-3">
                  <RefreshCw className="w-12 h-12 mx-auto" />
                </div>
                <h3 className="font-bold text-lg mb-2">Sync Pipeline</h3>
                <p className="text-sm text-slate-400">
                  Offline → Online 동기화
                  <br />
                  Airflow, Spark
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
