/**
 * 한국투자증권(KIS) API 연동 서비스
 * 실제 주식 데이터를 가져오는 서비스 클래스
 */

import { kisTokenManager } from '../auth/kis-token-manager';

// KIS API 응답 타입 정의
interface KISQuoteResponse {
  rt_cd: string;
  msg_cd: string;
  msg1: string;
  output: {
    iscd_stat_cls_code: string;
    marg_rate: string;
    rprs_mrkt_kor_name: string;
    new_mkop_cls_code: string;
    bstp_kor_isnm: string;
    temp_stop_yn: string;
    oprc_rang_cont_yn: string;
    clpr_rang_cont_yn: string;
    crdt_able: string;
    grmn_rate_cls_code: string;
    elw_pblc_yn: string;
    stck_prpr: string;      // 현재가
    prdy_vrss: string;      // 전일 대비
    prdy_vrss_sign: string; // 전일 대비 부호
    prdy_ctrt: string;      // 전일 대비율
    acml_vol: string;       // 누적 거래량
    acml_tr_pbmn: string;   // 누적 거래 대금
    hts_kor_isnm: string;   // 종목명
    stck_mxpr: string;      // 상한가
    stck_llam: string;      // 하한가
    stck_oprc: string;      // 시가
    stck_hgpr: string;      // 고가
    stck_lwpr: string;      // 저가
    stck_prdy_clpr: string; // 전일 종가
  };
}

interface KISHistoryResponse {
  rt_cd: string;
  msg_cd: string;
  msg1: string;
  output2: Array<{
    stck_bsop_date: string; // 주식 영업 일자
    stck_clpr: string;      // 주식 종가
    stck_oprc: string;      // 주식 시가
    stck_hgpr: string;      // 주식 최고가
    stck_lwpr: string;      // 주식 최저가
    acml_vol: string;       // 누적 거래량
    acml_tr_pbmn: string;   // 누적 거래 대금
    flng_cls_code: string;  // 락 구분 코드
    prtt_rate: string;      // 분할 비율
    mod_yn: string;         // 분할변경여부
    prdy_vrss_sign: string; // 전일 대비 부호
    prdy_vrss: string;      // 전일 대비
    revl_issu_reas: string; // 재평가 사유
  }>;
}

export interface StockQuote {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  marketCap?: number;
  high: number;
  low: number;
  open: number;
  previousClose: number;
  timestamp: Date;
}

export interface StockCandle {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export class KISApiService {
  private static instance: KISApiService;
  private readonly API_BASE = 'https://openapi.koreainvestment.com:9443';
  private readonly TR_ID_QUOTE = 'FHKST01010100';
  private readonly TR_ID_HISTORY = 'FHKST03010100';
  
  private constructor() {}
  
  public static getInstance(): KISApiService {
    if (!KISApiService.instance) {
      KISApiService.instance = new KISApiService();
    }
    return KISApiService.instance;
  }
  
  /**
   * 공통 API 요청 헤더 생성
   */
  private async getHeaders(trId: string): Promise<Record<string, string>> {
    const token = await kisTokenManager.getValidToken();
    
    return {
      'Content-Type': 'application/json; charset=utf-8',
      'authorization': `Bearer ${token}`,
      'appKey': process.env.NEXT_PUBLIC_KIS_APP_KEY || '',
      'appSecret': process.env.NEXT_PUBLIC_KIS_APP_SECRET || '',
      'tr_id': trId,
      'custtype': 'P', // 개인
    };
  }
  
  /**
   * 실시간 주식 시세 조회
   */
  public async getStockQuote(symbol: string): Promise<StockQuote> {
    try {
      // 데모 모드인지 확인
      if (!process.env.NEXT_PUBLIC_KIS_APP_KEY || !process.env.NEXT_PUBLIC_KIS_APP_SECRET) {
        // 데모 데이터 반환
        return this.generateDemoQuote(symbol);
      }
      
      const headers = await this.getHeaders(this.TR_ID_QUOTE);
      const url = new URL(`${this.API_BASE}/uapi/domestic-stock/v1/quotations/inquire-price`);
      
      // URL 파라미터 설정
      url.searchParams.append('FID_COND_MRKT_DIV_CODE', 'J'); // 시장 구분
      url.searchParams.append('FID_INPUT_ISCD', symbol); // 종목코드
      
      const response = await fetch(url.toString(), {
        method: 'GET',
        headers,
      });
      
      if (!response.ok) {
        throw new Error(`KIS API 요청 실패: ${response.status} ${response.statusText}`);
      }
      
      const data: KISQuoteResponse = await response.json();
      
      if (data.rt_cd !== '0') {
        throw new Error(`KIS API 오류: ${data.msg1} (${data.msg_cd})`);
      }
      
      const output = data.output;
      const price = parseInt(output.stck_prpr);
      const change = parseInt(output.prdy_vrss);
      const changePercent = parseFloat(output.prdy_ctrt);
      
      return {
        symbol,
        name: output.hts_kor_isnm,
        price,
        change,
        changePercent,
        volume: parseInt(output.acml_vol),
        high: parseInt(output.stck_hgpr),
        low: parseInt(output.stck_lwpr),
        open: parseInt(output.stck_oprc),
        previousClose: parseInt(output.stck_prdy_clpr),
        timestamp: new Date(),
      };
    } catch (error) {
      console.error('주식 시세 조회 실패:', error);
      throw error;
    }
  }
  
  /**
   * 주식 차트 데이터 조회 (일봉)
   */
  public async getStockHistory(
    symbol: string, 
    period: 'D' | 'W' | 'M' = 'D',
    count: number = 100
  ): Promise<StockCandle[]> {
    try {
      const headers = await this.getHeaders(this.TR_ID_HISTORY);
      const url = new URL(`${this.API_BASE}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice`);
      
      // 종료일자 (오늘)
      const endDate = new Date().toISOString().slice(0, 10).replace(/-/g, '');
      
      // URL 파라미터 설정
      url.searchParams.append('FID_COND_MRKT_DIV_CODE', 'J');
      url.searchParams.append('FID_INPUT_ISCD', symbol);
      url.searchParams.append('FID_INPUT_DATE_1', '20240101'); // 시작일자
      url.searchParams.append('FID_INPUT_DATE_2', endDate);    // 종료일자
      url.searchParams.append('FID_PERIOD_DIV_CODE', period);  // 기간 구분
      url.searchParams.append('FID_ORG_ADJ_PRC', '0');         // 수정주가 구분
      
      const response = await fetch(url.toString(), {
        method: 'GET',
        headers,
      });
      
      if (!response.ok) {
        throw new Error(`KIS API 요청 실패: ${response.status} ${response.statusText}`);
      }
      
      const data: KISHistoryResponse = await response.json();
      
      if (data.rt_cd !== '0') {
        throw new Error(`KIS API 오류: ${data.msg1} (${data.msg_cd})`);
      }
      
      // 최신 데이터 순으로 정렬하고 요청된 개수만큼 반환
      const candles = data.output2
        .slice(0, count)
        .map(item => ({
          date: item.stck_bsop_date,
          open: parseInt(item.stck_oprc),
          high: parseInt(item.stck_hgpr),
          low: parseInt(item.stck_lwpr),
          close: parseInt(item.stck_clpr),
          volume: parseInt(item.acml_vol),
        }))
        .reverse(); // 오래된 것부터 최신 순으로 재정렬
      
      return candles;
    } catch (error) {
      console.error('주식 차트 데이터 조회 실패:', error);
      throw error;
    }
  }
  
  /**
   * 여러 종목 실시간 시세 조회
   */
  public async getMultipleQuotes(symbols: string[]): Promise<StockQuote[]> {
    const quotes: StockQuote[] = [];
    
    // 동시 요청 수 제한 (API 제한 고려)
    const batchSize = 5;
    
    for (let i = 0; i < symbols.length; i += batchSize) {
      const batch = symbols.slice(i, i + batchSize);
      const batchPromises = batch.map(symbol => 
        this.getStockQuote(symbol).catch(error => {
          console.error(`${symbol} 시세 조회 실패:`, error);
          return null;
        })
      );
      
      const batchResults = await Promise.all(batchPromises);
      const validResults = batchResults.filter(result => result !== null) as StockQuote[];
      quotes.push(...validResults);
      
      // API 요청 간격 조절 (Rate Limiting 방지)
      if (i + batchSize < symbols.length) {
        await new Promise(resolve => setTimeout(resolve, 200));
      }
    }
    
    return quotes;
  }
  
  /**
   * 주요 종목 목록 조회
   */
  public async getPopularStocks(): Promise<StockQuote[]> {
    const popularSymbols = [
      '005930', // 삼성전자
      '000660', // SK하이닉스
      '035720', // 카카오
      '035420', // NAVER
      '005380', // 현대차
      '051910', // LG화학
      '006400', // 삼성SDI
      '207940', // 삼성바이오로직스
      '068270', // 셀트리온
      '028260', // 삼성물산
    ];
    
    return this.getMultipleQuotes(popularSymbols);
  }
  
  /**
   * 데모 주식 시세 생성
   */
  private generateDemoQuote(symbol: string): StockQuote {
    const stockNames: { [key: string]: string } = {
      '005930': '삼성전자',
      '000660': 'SK하이닉스',
      '035720': '카카오',
      '035420': 'NAVER',
      '005380': '현대차',
      '051910': 'LG화학',
      '006400': '삼성SDI',
      '207940': '삼성바이오로직스',
    };

    const basePrice = 69800;
    const randomChange = (Math.random() - 0.5) * 1000;
    const price = Math.floor(basePrice + randomChange);
    const change = Math.floor((Math.random() - 0.5) * 500);
    const changePercent = (change / price) * 100;

    return {
      symbol,
      name: stockNames[symbol] || '데모 종목',
      price,
      change,
      changePercent,
      volume: Math.floor(Math.random() * 10000000) + 1000000,
      high: price + Math.floor(Math.random() * 500),
      low: price - Math.floor(Math.random() * 500),
      open: price + Math.floor((Math.random() - 0.5) * 200),
      previousClose: price - change,
      timestamp: new Date(),
    };
  }

  /**
   * API 연결 상태 테스트
   */
  public async testConnection(): Promise<boolean> {
    try {
      // 삼성전자로 연결 테스트
      await this.getStockQuote('005930');
      return true;
    } catch (error) {
      console.error('KIS API 연결 테스트 실패:', error);
      return false;
    }
  }
}

// 싱글톤 인스턴스 내보내기
export const kisApiService = KISApiService.getInstance();