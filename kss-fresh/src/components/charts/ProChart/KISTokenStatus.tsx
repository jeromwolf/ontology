/**
 * KIS API 토큰 상태 모니터링 컴포넌트
 * 토큰 유효성, 만료시간, 갱신 버튼 등을 제공
 */

'use client';

import { useState, useEffect } from 'react';
import { Shield, AlertTriangle, CheckCircle, RefreshCw, Clock, Wifi, WifiOff } from 'lucide-react';
import { kisTokenManager } from '@/lib/auth/kis-token-manager';
import { kisApiService } from '@/lib/services/kis-api-service';

interface TokenStatus {
  hasToken: boolean;
  isValid: boolean;
  expiresAt: Date | null;
  createdAt: Date | null;
  hoursUntilExpiry: number | null;
}

export default function KISTokenStatus() {
  const [tokenStatus, setTokenStatus] = useState<TokenStatus>({
    hasToken: false,
    isValid: false,
    expiresAt: null,
    createdAt: null,
    hoursUntilExpiry: null,
  });
  
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking');
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  // 토큰 상태 업데이트
  const updateTokenStatus = () => {
    const status = kisTokenManager.getTokenStatus();
    setTokenStatus(status);
    setLastUpdate(new Date());
  };

  // API 연결 상태 확인
  const checkConnection = async () => {
    setConnectionStatus('checking');
    try {
      const isConnected = await kisApiService.testConnection();
      setConnectionStatus(isConnected ? 'connected' : 'disconnected');
    } catch (error) {
      console.warn('API 연결 확인 실패 (데모 모드로 동작):', error);
      setConnectionStatus('disconnected');
    }
  };

  // 토큰 강제 갱신
  const handleRefreshToken = async () => {
    setIsRefreshing(true);
    try {
      await kisTokenManager.forceRefreshToken();
      updateTokenStatus();
      await checkConnection();
    } catch (error) {
      console.warn('토큰 갱신 실패 (데모 모드로 동작):', error);
    } finally {
      setIsRefreshing(false);
    }
  };

  // 토큰 삭제
  const handleClearToken = () => {
    kisTokenManager.clearToken();
    updateTokenStatus();
    setConnectionStatus('disconnected');
  };

  // 컴포넌트 마운트 시 초기화
  useEffect(() => {
    updateTokenStatus();
    
    // 환경변수가 있을 때만 연결 확인
    if (process.env.NEXT_PUBLIC_KIS_APP_KEY && process.env.NEXT_PUBLIC_KIS_APP_SECRET) {
      checkConnection();
    } else {
      setConnectionStatus('disconnected');
    }
    
    // 5분마다 토큰 상태 자동 업데이트
    const interval = setInterval(() => {
      updateTokenStatus();
    }, 5 * 60 * 1000);

    return () => clearInterval(interval);
  }, []);

  const getStatusIcon = () => {
    if (!tokenStatus.hasToken) {
      return <AlertTriangle className="w-4 h-4 text-red-400" />;
    }
    if (!tokenStatus.isValid) {
      return <AlertTriangle className="w-4 h-4 text-yellow-400" />;
    }
    return <CheckCircle className="w-4 h-4 text-green-400" />;
  };

  const getStatusText = () => {
    if (!tokenStatus.hasToken) return '토큰 없음';
    if (!tokenStatus.isValid) return '토큰 만료';
    return '정상';
  };

  const getStatusColor = () => {
    if (!tokenStatus.hasToken || !tokenStatus.isValid) {
      return 'text-red-400';
    }
    return 'text-green-400';
  };

  const formatTimeUntilExpiry = () => {
    if (!tokenStatus.hoursUntilExpiry) return '알 수 없음';
    
    const hours = Math.floor(tokenStatus.hoursUntilExpiry);
    const minutes = Math.floor((tokenStatus.hoursUntilExpiry - hours) * 60);
    
    if (hours > 0) {
      return `${hours}시간 ${minutes}분`;
    }
    return `${minutes}분`;
  };

  return (
    <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4 space-y-4">
      {/* 헤더 */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Shield className="w-5 h-5 text-blue-400" />
          <h3 className="text-sm font-semibold">KIS API 상태</h3>
        </div>
        <div className="flex items-center gap-2">
          {connectionStatus === 'checking' && <RefreshCw className="w-4 h-4 text-gray-400 animate-spin" />}
          {connectionStatus === 'connected' && <Wifi className="w-4 h-4 text-green-400" />}
          {connectionStatus === 'disconnected' && <WifiOff className="w-4 h-4 text-red-400" />}
          <span className="text-xs text-gray-400">
            {connectionStatus === 'checking' && '확인중'}
            {connectionStatus === 'connected' && '연결됨'}
            {connectionStatus === 'disconnected' && '연결안됨'}
          </span>
        </div>
      </div>

      {/* 토큰 상태 */}
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            {getStatusIcon()}
            <span className={`text-sm font-medium ${getStatusColor()}`}>
              {getStatusText()}
            </span>
          </div>
          
          {tokenStatus.hasToken && tokenStatus.isValid && (
            <div className="flex items-center gap-2 text-xs text-gray-400">
              <Clock className="w-3 h-3" />
              <span>만료까지: {formatTimeUntilExpiry()}</span>
            </div>
          )}
        </div>

        <div className="space-y-1 text-right">
          {tokenStatus.createdAt && (
            <div className="text-xs text-gray-400">
              생성: {tokenStatus.createdAt.toLocaleDateString()} {tokenStatus.createdAt.toLocaleTimeString()}
            </div>
          )}
          {tokenStatus.expiresAt && (
            <div className="text-xs text-gray-400">
              만료: {tokenStatus.expiresAt.toLocaleDateString()} {tokenStatus.expiresAt.toLocaleTimeString()}
            </div>
          )}
        </div>
      </div>

      {/* 상세 정보 */}
      {tokenStatus.hasToken && !tokenStatus.isValid && (
        <div className="bg-yellow-900/20 border border-yellow-500/30 rounded-lg p-3">
          <div className="flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 text-yellow-400 mt-0.5" />
            <div>
              <p className="text-sm text-yellow-200 font-medium">토큰 갱신 필요</p>
              <p className="text-xs text-yellow-300 mt-1">
                토큰이 만료되었거나 24시간이 지났습니다. 실시간 데이터를 받으려면 토큰을 갱신하세요.
              </p>
            </div>
          </div>
        </div>
      )}

      {!process.env.NEXT_PUBLIC_KIS_APP_KEY || !process.env.NEXT_PUBLIC_KIS_APP_SECRET ? (
        <div className="bg-blue-900/20 border border-blue-500/30 rounded-lg p-3">
          <div className="flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 text-blue-400 mt-0.5" />
            <div>
              <p className="text-sm text-blue-200 font-medium">데모 모드</p>
              <p className="text-xs text-blue-300 mt-1">
                KIS API 키가 설정되지 않아 데모 데이터로 동작합니다. 실제 데이터를 보려면 환경변수를 설정하세요.
              </p>
            </div>
          </div>
        </div>
      ) : !tokenStatus.hasToken && (
        <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-3">
          <div className="flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 text-red-400 mt-0.5" />
            <div>
              <p className="text-sm text-red-200 font-medium">토큰 없음</p>
              <p className="text-xs text-red-300 mt-1">
                KIS API 인증 토큰이 없습니다. 환경변수 설정을 확인하고 토큰을 생성하세요.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* 액션 버튼 */}
      <div className="flex gap-2">
        <button
          onClick={handleRefreshToken}
          disabled={isRefreshing}
          className="flex-1 flex items-center justify-center gap-2 px-3 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 disabled:opacity-50 rounded-lg text-sm font-medium transition-colors"
        >
          <RefreshCw className={`w-4 h-4 ${isRefreshing ? 'animate-spin' : ''}`} />
          {isRefreshing ? '갱신중...' : '토큰 갱신'}
        </button>
        
        <button
          onClick={checkConnection}
          disabled={connectionStatus === 'checking'}
          className="px-3 py-2 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:opacity-50 rounded-lg text-sm font-medium transition-colors"
        >
          연결 테스트
        </button>
        
        {tokenStatus.hasToken && (
          <button
            onClick={handleClearToken}
            className="px-3 py-2 bg-red-600 hover:bg-red-700 rounded-lg text-sm font-medium transition-colors"
          >
            토큰 삭제
          </button>
        )}
      </div>

      {/* 마지막 업데이트 시간 */}
      <div className="text-xs text-gray-500 text-center pt-2 border-t border-gray-700">
        마지막 업데이트: {lastUpdate.toLocaleTimeString()}
      </div>
    </div>
  );
}