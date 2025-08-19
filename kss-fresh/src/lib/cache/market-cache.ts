// 시장 데이터 캐싱 시스템
// 메모리 기반 캐시 (나중에 Redis로 업그레이드 가능)

interface CacheItem<T> {
  data: T;
  timestamp: number;
  ttl: number; // Time to live in milliseconds
}

class MarketDataCache {
  private cache: Map<string, CacheItem<any>> = new Map();
  private requestCount: Map<string, number> = new Map();
  private lastResetTime: number = Date.now();
  
  // 기본 캐시 설정
  private readonly DEFAULT_TTL = 60 * 1000; // 1분
  private readonly EXTENDED_TTL = 5 * 60 * 1000; // 5분 (장 마감 후)
  private readonly MAX_REQUESTS_PER_MINUTE = 50; // API 제한 고려
  
  // 캐시 키 생성
  private getCacheKey(type: string, params?: Record<string, any>): string {
    const sortedParams = params ? Object.keys(params).sort().map(k => `${k}:${params[k]}`).join('|') : '';
    return `${type}${sortedParams ? `|${sortedParams}` : ''}`;
  }
  
  // 현재 시간이 장중인지 확인 (한국 시간 기준)
  private isMarketOpen(): boolean {
    const now = new Date();
    const koreaTime = new Date(now.toLocaleString("en-US", {timeZone: "Asia/Seoul"}));
    const hours = koreaTime.getHours();
    const minutes = koreaTime.getMinutes();
    const day = koreaTime.getDay();
    
    // 평일 9:00 ~ 15:30
    if (day === 0 || day === 6) return false; // 주말
    if (hours < 9 || hours > 15) return false;
    if (hours === 15 && minutes > 30) return false;
    
    return true;
  }
  
  // 요청 제한 확인
  private checkRateLimit(key: string): boolean {
    const now = Date.now();
    
    // 1분마다 카운터 리셋
    if (now - this.lastResetTime > 60000) {
      this.requestCount.clear();
      this.lastResetTime = now;
    }
    
    const count = this.requestCount.get(key) || 0;
    if (count >= this.MAX_REQUESTS_PER_MINUTE) {
      console.warn(`Rate limit exceeded for ${key}`);
      return false;
    }
    
    this.requestCount.set(key, count + 1);
    return true;
  }
  
  // 캐시 조회
  get<T>(type: string, params?: Record<string, any>): T | null {
    const key = this.getCacheKey(type, params);
    const cached = this.cache.get(key);
    
    if (!cached) return null;
    
    const now = Date.now();
    if (now - cached.timestamp > cached.ttl) {
      // 캐시 만료
      this.cache.delete(key);
      return null;
    }
    
    console.log(`Cache hit for ${key}, age: ${Math.round((now - cached.timestamp) / 1000)}s`);
    return cached.data;
  }
  
  // 캐시 저장
  set<T>(type: string, data: T, params?: Record<string, any>, customTTL?: number): void {
    const key = this.getCacheKey(type, params);
    
    // 장중/장외 구분하여 TTL 설정
    const ttl = customTTL || (this.isMarketOpen() ? this.DEFAULT_TTL : this.EXTENDED_TTL);
    
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttl
    });
    
    console.log(`Cache set for ${key}, TTL: ${ttl / 1000}s`);
  }
  
  // 캐시 무효화
  invalidate(type?: string): void {
    if (type) {
      // 특정 타입의 캐시만 삭제
      const keysToDelete: string[] = [];
      this.cache.forEach((_, key) => {
        if (key.startsWith(type)) {
          keysToDelete.push(key);
        }
      });
      keysToDelete.forEach(key => this.cache.delete(key));
    } else {
      // 전체 캐시 삭제
      this.cache.clear();
    }
  }
  
  // 캐시 상태 조회
  getStats(): {
    size: number;
    keys: string[];
    requestCount: number;
    isMarketOpen: boolean;
  } {
    return {
      size: this.cache.size,
      keys: Array.from(this.cache.keys()),
      requestCount: Array.from(this.requestCount.values()).reduce((a, b) => a + b, 0),
      isMarketOpen: this.isMarketOpen()
    };
  }
  
  // 캐시된 데이터로 API 호출 래핑
  async fetchWithCache<T>(
    type: string,
    fetcher: () => Promise<T>,
    params?: Record<string, any>,
    options?: {
      ttl?: number;
      forceRefresh?: boolean;
    }
  ): Promise<T> {
    const key = this.getCacheKey(type, params);
    
    // 강제 새로고침이 아니면 캐시 확인
    if (!options?.forceRefresh) {
      const cached = this.get<T>(type, params);
      if (cached) return cached;
    }
    
    // Rate limit 확인
    if (!this.checkRateLimit(key)) {
      // Rate limit 초과 시 캐시된 데이터라도 반환
      const staleCache = this.cache.get(key);
      if (staleCache) {
        console.warn(`Rate limit exceeded, returning stale cache for ${key}`);
        return staleCache.data;
      }
      throw new Error('Rate limit exceeded and no cached data available');
    }
    
    try {
      // API 호출
      const data = await fetcher();
      
      // 캐시 저장
      this.set(type, data, params, options?.ttl);
      
      return data;
    } catch (error) {
      // 에러 발생 시 오래된 캐시라도 반환
      const staleCache = this.cache.get(key);
      if (staleCache) {
        console.warn(`API error, returning stale cache for ${key}`, error);
        return staleCache.data;
      }
      throw error;
    }
  }
}

// 싱글톤 인스턴스
export const marketCache = new MarketDataCache();

// 캐시 타입 상수
export const CACHE_TYPES = {
  MARKET_OVERVIEW: 'market-overview',
  STOCK_QUOTE: 'stock-quote',
  INDICES: 'indices',
  TOP_STOCKS: 'top-stocks',
  SECTOR_PERFORMANCE: 'sector-performance',
  HISTORICAL_DATA: 'historical-data'
} as const;

export default MarketDataCache;