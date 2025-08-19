import { NextResponse } from 'next/server';

export async function GET() {
  try {
    const APP_KEY = process.env.NEXT_PUBLIC_KIS_APP_KEY;
    const APP_SECRET = process.env.NEXT_PUBLIC_KIS_APP_SECRET;

    if (!APP_KEY || !APP_SECRET) {
      return NextResponse.json({ error: 'API keys not configured' });
    }

    // 간단한 API 테스트 - 해시키 생성
    const response = await fetch('https://openapi.koreainvestment.com:9443/uapi/hashkey', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json; charset=utf-8',
        'appkey': APP_KEY,
        'appsecret': APP_SECRET,
      },
      body: JSON.stringify({
        datas: {
          CANO: "50067891",
          ACNT_PRDT_CD: "01",
          PDNO: "005930",
          ORD_DVSN: "01",
          ORD_QTY: "10",
          ORD_UNPR: "0"
        }
      }),
    });

    const data = await response.text();
    
    return NextResponse.json({
      status: response.status,
      statusText: response.statusText,
      headers: Object.fromEntries(response.headers.entries()),
      body: data,
      keyInfo: {
        keyLength: APP_KEY.length,
        secretLength: APP_SECRET.length,
        keyPrefix: APP_KEY.substring(0, 10),
      }
    });
  } catch (error) {
    return NextResponse.json({ 
      error: error instanceof Error ? error.message : 'Unknown error' 
    });
  }
}