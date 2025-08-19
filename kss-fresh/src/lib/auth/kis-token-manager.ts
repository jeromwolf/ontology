/**
 * 한국투자증권(KIS) API 토큰 관리 시스템
 * - 하루에 한 번만 토큰을 요청하여 저장
 * - 토큰 만료 시 자동 갱신
 * - 안전한 토큰 저장 및 관리
 */

interface KISTokenResponse {
  access_token: string;
  access_token_token_expired: string;
  token_type: string;
  expires_in: number;
}

interface StoredToken {
  access_token: string;
  expires_at: number;
  created_at: number;
}

export class KISTokenManager {
  private static instance: KISTokenManager;
  private currentToken: StoredToken | null = null;
  private readonly STORAGE_KEY = 'kis_token';
  private readonly API_BASE = 'https://openapi.koreainvestment.com:9443';
  
  // KIS API 인증 정보 (환경변수에서 로드)
  private readonly APP_KEY = process.env.NEXT_PUBLIC_KIS_APP_KEY || '';
  private readonly APP_SECRET = process.env.NEXT_PUBLIC_KIS_APP_SECRET || '';
  
  private constructor() {
    this.loadStoredToken();
  }
  
  public static getInstance(): KISTokenManager {
    if (!KISTokenManager.instance) {
      KISTokenManager.instance = new KISTokenManager();
    }
    return KISTokenManager.instance;
  }
  
  /**
   * 저장된 토큰 로드
   */
  private loadStoredToken(): void {
    try {
      const stored = localStorage.getItem(this.STORAGE_KEY);
      if (stored) {
        this.currentToken = JSON.parse(stored);
      }
    } catch (error) {
      console.error('토큰 로드 실패:', error);
      this.currentToken = null;
    }
  }
  
  /**
   * 토큰을 안전하게 저장
   */
  private saveToken(token: StoredToken): void {
    try {
      localStorage.setItem(this.STORAGE_KEY, JSON.stringify(token));
      this.currentToken = token;
    } catch (error) {
      console.error('토큰 저장 실패:', error);
    }
  }
  
  /**
   * 토큰이 유효한지 확인
   */
  private isTokenValid(): boolean {
    if (!this.currentToken) return false;
    
    const now = Date.now();
    const expiresAt = this.currentToken.expires_at;
    const createdAt = this.currentToken.created_at;
    
    // 24시간(하루) 경과 확인
    const oneDayInMs = 24 * 60 * 60 * 1000;
    const isDayOld = (now - createdAt) >= oneDayInMs;
    
    // 토큰 만료 시간 확인 (5분 여유)
    const isExpired = now >= (expiresAt - 5 * 60 * 1000);
    
    return !isDayOld && !isExpired;
  }
  
  /**
   * KIS API에서 새 토큰 요청
   */
  private async fetchNewToken(): Promise<StoredToken> {
    if (!this.APP_KEY || !this.APP_SECRET) {
      console.warn('KIS API 인증 정보가 설정되지 않았습니다. 데모 모드로 전환합니다.');
      // 데모용 가짜 토큰 반환
      const now = Date.now();
      return {
        access_token: 'demo_token_' + now,
        expires_at: now + 24 * 60 * 60 * 1000, // 24시간
        created_at: now,
      };
    }
    
    const response = await fetch(`${this.API_BASE}/oauth2/tokenP`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json; charset=utf-8',
      },
      body: JSON.stringify({
        grant_type: 'client_credentials',
        appkey: this.APP_KEY,
        appsecret: this.APP_SECRET,
      }),
    });
    
    if (!response.ok) {
      throw new Error(`토큰 요청 실패: ${response.status} ${response.statusText}`);
    }
    
    const tokenData: KISTokenResponse = await response.json();
    
    if (!tokenData.access_token) {
      throw new Error('유효하지 않은 토큰 응답');
    }
    
    const now = Date.now();
    const expiresIn = tokenData.expires_in * 1000; // 초를 밀리초로 변환
    
    return {
      access_token: tokenData.access_token,
      expires_at: now + expiresIn,
      created_at: now,
    };
  }
  
  /**
   * 유효한 토큰을 반환 (필요시 자동 갱신)
   */
  public async getValidToken(): Promise<string> {
    // 현재 토큰이 유효하면 그대로 사용
    if (this.isTokenValid()) {
      return this.currentToken!.access_token;
    }
    
    console.log('토큰이 만료되었거나 하루가 지났습니다. 새 토큰을 요청합니다...');
    
    try {
      const newToken = await this.fetchNewToken();
      this.saveToken(newToken);
      
      console.log('새 토큰이 성공적으로 생성되었습니다.');
      return newToken.access_token;
    } catch (error) {
      console.error('토큰 갱신 실패:', error);
      throw error;
    }
  }
  
  /**
   * 토큰 상태 정보 반환
   */
  public getTokenStatus(): {
    hasToken: boolean;
    isValid: boolean;
    expiresAt: Date | null;
    createdAt: Date | null;
    hoursUntilExpiry: number | null;
  } {
    if (!this.currentToken) {
      return {
        hasToken: false,
        isValid: false,
        expiresAt: null,
        createdAt: null,
        hoursUntilExpiry: null,
      };
    }
    
    const now = Date.now();
    const hoursUntilExpiry = Math.max(0, (this.currentToken.expires_at - now) / (1000 * 60 * 60));
    
    return {
      hasToken: true,
      isValid: this.isTokenValid(),
      expiresAt: new Date(this.currentToken.expires_at),
      createdAt: new Date(this.currentToken.created_at),
      hoursUntilExpiry,
    };
  }
  
  /**
   * 강제로 토큰 갱신
   */
  public async forceRefreshToken(): Promise<string> {
    console.log('토큰을 강제로 갱신합니다...');
    const newToken = await this.fetchNewToken();
    this.saveToken(newToken);
    return newToken.access_token;
  }
  
  /**
   * 저장된 토큰 삭제
   */
  public clearToken(): void {
    localStorage.removeItem(this.STORAGE_KEY);
    this.currentToken = null;
    console.log('토큰이 삭제되었습니다.');
  }
}

// 싱글톤 인스턴스 내보내기
export const kisTokenManager = KISTokenManager.getInstance();