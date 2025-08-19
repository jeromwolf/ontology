import { NextResponse } from 'next/server';

export async function POST() {
  try {
    const APP_KEY = process.env.NEXT_PUBLIC_KIS_APP_KEY;
    const APP_SECRET = process.env.NEXT_PUBLIC_KIS_APP_SECRET;

    if (!APP_KEY || !APP_SECRET) {
      return NextResponse.json(
        { error: 'KIS API 인증 정보가 설정되지 않았습니다.' },
        { status: 400 }
      );
    }

    // 디버깅을 위해 키 길이 확인 (보안을 위해 키 자체는 로깅하지 않음)
    console.log('APP_KEY length:', APP_KEY.length);
    console.log('APP_SECRET length:', APP_SECRET.length);
    console.log('First 10 chars of APP_KEY:', APP_KEY.substring(0, 10) + '...');

    // KIS API 토큰 요청
    const response = await fetch('https://openapi.koreainvestment.com:9443/oauth2/tokenP', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json; charset=utf-8',
      },
      body: JSON.stringify({
        grant_type: 'client_credentials',
        appkey: APP_KEY,
        appsecret: APP_SECRET,
      }),
    });

    const responseText = await response.text();
    console.log('KIS API Response:', response.status, responseText);

    if (!response.ok) {
      return NextResponse.json(
        { error: `토큰 요청 실패: ${response.status} ${response.statusText}`, details: responseText },
        { status: response.status }
      );
    }

    try {
      const tokenData = JSON.parse(responseText);
      
      if (!tokenData.access_token) {
        return NextResponse.json(
          { error: '유효하지 않은 토큰 응답', details: tokenData },
          { status: 400 }
        );
      }

      return NextResponse.json({
        access_token: tokenData.access_token,
        access_token_token_expired: tokenData.access_token_token_expired,
        token_type: tokenData.token_type,
        expires_in: tokenData.expires_in,
      });
    } catch (parseError) {
      console.error('JSON Parse Error:', parseError);
      return NextResponse.json(
        { error: 'Invalid JSON response from KIS API', details: responseText },
        { status: 500 }
      );
    }
  } catch (error) {
    console.error('KIS Token Error:', error);
    return NextResponse.json(
      { error: '토큰 생성 중 오류가 발생했습니다.', details: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    );
  }
}