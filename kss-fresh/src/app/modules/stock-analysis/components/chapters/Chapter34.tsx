'use client';

import React from 'react';
import { Book, DollarSign, Globe, Shield, TrendingUp, CreditCard, FileText, AlertCircle } from 'lucide-react';

export default function Chapter34() {
  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-4xl font-bold mb-8">해외 증권사 계좌 개설</h1>
      
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-6 flex items-center gap-3">
          <Globe className="w-8 h-8 text-blue-500" />
          미국 주요 증권사 계좌 개설
        </h2>
        
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 mb-6">
          <h3 className="text-xl font-semibold mb-4">Interactive Brokers (IB)</h3>
          <div className="space-y-4">
            <div>
              <h4 className="font-medium mb-2">장점</h4>
              <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                <li>가장 저렴한 수수료 (주식 $0.005/주, 최소 $1)</li>
                <li>전 세계 135개 시장 접근 가능</li>
                <li>강력한 트레이딩 플랫폼 (TWS)</li>
                <li>마진 금리 최저 수준</li>
                <li>API 제공으로 자동매매 가능</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium mb-2">단점</h4>
              <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                <li>복잡한 인터페이스 (초보자 어려움)</li>
                <li>월 최소 수수료 $10 (자산 $100,000 미만)</li>
                <li>실시간 데이터 유료</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium mb-2">개설 절차</h4>
              <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
                <li>온라인 신청서 작성 (약 30분)</li>
                <li>신분증 업로드 (여권 필수)</li>
                <li>주소 증명서 제출 (영문 은행 거래내역서)</li>
                <li>투자 경험 및 자산 정보 입력</li>
                <li>W-8BEN 세금 양식 작성</li>
                <li>계좌 승인 대기 (1-3 영업일)</li>
              </ol>
            </div>
          </div>
        </div>

        <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6 mb-6">
          <h3 className="text-xl font-semibold mb-4">Charles Schwab</h3>
          <div className="space-y-4">
            <div>
              <h4 className="font-medium mb-2">장점</h4>
              <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                <li>무료 주식 거래 수수료</li>
                <li>우수한 리서치 자료 제공</li>
                <li>한국어 고객 서비스 지원</li>
                <li>사용하기 쉬운 플랫폼</li>
                <li>계좌 최소 금액 없음</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium mb-2">단점</h4>
              <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                <li>해외 시장 접근 제한적</li>
                <li>환전 수수료 다소 높음</li>
                <li>옵션 거래 수수료 있음</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium mb-2">개설 절차</h4>
              <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
                <li>Schwab International 계좌 신청</li>
                <li>온라인 양식 작성 및 서명</li>
                <li>여권 사본 공증 필요</li>
                <li>W-8BEN 작성 및 공증</li>
                <li>서류 국제우편 발송</li>
                <li>계좌 승인 (2-4주)</li>
              </ol>
            </div>
          </div>
        </div>

        <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
          <h3 className="text-xl font-semibold mb-4">TD Ameritrade (현 Schwab)</h3>
          <div className="space-y-4">
            <div>
              <h4 className="font-medium mb-2">장점</h4>
              <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                <li>thinkorswim 플랫폼 (최고 수준)</li>
                <li>무료 주식/ETF 거래</li>
                <li>풍부한 교육 자료</li>
                <li>paper trading 제공</li>
              </ul>
            </div>
            <div className="bg-yellow-100 dark:bg-yellow-900/20 rounded p-4 mt-4">
              <p className="text-sm">
                <AlertCircle className="inline w-4 h-4 mr-2" />
                2023년 Schwab에 인수되어 신규 계좌 개설은 Schwab로 통합되었습니다.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-6 flex items-center gap-3">
          <FileText className="w-8 h-8 text-green-500" />
          W-8BEN 세금 서류 작성
        </h2>
        
        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-6">
          <h3 className="text-xl font-semibold mb-4">W-8BEN이란?</h3>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            미국 비거주자가 미국 금융 기관에 제출하는 세금 양식으로, 한미 조세협약에 따른 
            원천징수세율 감면을 받기 위해 필수적입니다.
          </p>
          
          <div className="space-y-4">
            <div>
              <h4 className="font-medium mb-2">주요 작성 항목</h4>
              <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300">
                <li>Part I: 개인 정보 (이름, 주소, 생년월일)</li>
                <li>Part II: 조세협약 혜택 청구
                  <ul className="list-disc list-inside ml-6 mt-1">
                    <li>Line 9: 거주 국가 (Republic of Korea)</li>
                    <li>Line 10: Article 10 (배당), Article 11 (이자)</li>
                    <li>세율: 배당 15%, 이자 12.5%</li>
                  </ul>
                </li>
                <li>Part III: 서명 및 날짜</li>
              </ul>
            </div>
            
            <div className="bg-blue-100 dark:bg-blue-900/20 rounded p-4">
              <h4 className="font-medium mb-2">💡 작성 팁</h4>
              <ul className="list-disc list-inside space-y-1 text-sm">
                <li>반드시 영문으로 작성 (한글 불가)</li>
                <li>주소는 영문 주소 변환 서비스 이용</li>
                <li>TIN(납세자번호)은 주민등록번호 입력</li>
                <li>3년마다 갱신 필요</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-6 flex items-center gap-3">
          <DollarSign className="w-8 h-8 text-yellow-500" />
          효율적인 환전과 송금
        </h2>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4">전신송금 (Wire Transfer)</h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>✅ 대량 송금 시 유리</li>
              <li>✅ 안전하고 확실함</li>
              <li>❌ 송금 수수료 높음 ($30-50)</li>
              <li>❌ 중계은행 수수료 추가</li>
              <li>📍 추천: $10,000 이상 송금 시</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4">증권사 환전 서비스</h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>✅ 편리한 환전 프로세스</li>
              <li>✅ 소액도 가능</li>
              <li>❌ 환율 스프레드 존재</li>
              <li>❌ 실시간 환율 아님</li>
              <li>📍 추천: 소액 정기 투자 시</li>
            </ul>
          </div>
        </div>
        
        <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6 mt-6">
          <h3 className="text-lg font-semibold mb-3">🏆 추천 환전 전략</h3>
          <ol className="list-decimal list-inside space-y-2 text-gray-700 dark:text-gray-300">
            <li>환율이 유리할 때 일괄 송금 (전신송금)</li>
            <li>증권사 내 달러 보유</li>
            <li>필요시 분할 투자</li>
            <li>환율 1,300원 이상 시 환전 보류 검토</li>
          </ol>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-6 flex items-center gap-3">
          <Globe className="w-8 h-8 text-red-500" />
          기타 주요 시장 접근
        </h2>
        
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-3">🇯🇵 일본 시장</h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li><strong>SBI Securities</strong>: 일본 최대 온라인 증권사</li>
              <li><strong>Rakuten Securities</strong>: 영문 지원 우수</li>
              <li>특징: 저렴한 수수료, NISA 계좌 가능</li>
              <li>주의: 일본어 능력 필요, 세금 처리 복잡</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-3">🇨🇳 중국 시장</h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li><strong>후강통/선강통</strong>: 홍콩 계좌로 A주 투자</li>
              <li><strong>QFII 펀드</strong>: 간접 투자 방식</li>
              <li>특징: 직접 투자 제한적, 환전 규제</li>
              <li>추천: 중국 ETF로 우회 투자</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-3">🇪🇺 유럽 시장</h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li><strong>DEGIRO</strong>: 유럽 전역 저렴한 수수료</li>
              <li><strong>Saxo Bank</strong>: 프리미엄 서비스</li>
              <li>특징: 다양한 유럽 거래소 접근</li>
              <li>주의: 각국 세금 규정 상이</li>
            </ul>
          </div>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-6 flex items-center gap-3">
          <Shield className="w-8 h-8 text-purple-500" />
          계좌 관리 및 세금 신고
        </h2>
        
        <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
          <h3 className="text-xl font-semibold mb-4">해외금융계좌 신고</h3>
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 rounded p-4">
              <h4 className="font-medium mb-2">신고 대상</h4>
              <p className="text-gray-700 dark:text-gray-300">
                매월 말일 중 하루라도 모든 해외금융계좌 잔액의 합이 5억원을 초과하는 경우
              </p>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded p-4">
              <h4 className="font-medium mb-2">신고 방법</h4>
              <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                <li>신고 기한: 다음해 6월 30일까지</li>
                <li>신고 방법: 국세청 홈택스 또는 서면 신고</li>
                <li>미신고 과태료: 미신고 금액의 10-20%</li>
              </ul>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded p-4">
              <h4 className="font-medium mb-2">양도소득세 신고</h4>
              <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                <li>해외주식 양도차익: 22% (지방세 포함)</li>
                <li>연간 250만원 기본공제</li>
                <li>신고 기한: 다음해 5월 31일</li>
                <li>분기별 예납 가능 (선택사항)</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <div className="bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg p-8 text-white">
        <h2 className="text-2xl font-bold mb-4">해외 투자 시작 체크리스트</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h3 className="text-lg font-semibold mb-3">📝 준비 서류</h3>
            <ul className="space-y-2">
              <li>✓ 여권 (유효기간 6개월 이상)</li>
              <li>✓ 영문 주소 증명서</li>
              <li>✓ 영문 은행 거래내역서</li>
              <li>✓ W-8BEN 양식</li>
            </ul>
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-3">💰 투자 준비</h3>
            <ul className="space-y-2">
              <li>✓ 투자 목표 금액 설정</li>
              <li>✓ 환율 모니터링</li>
              <li>✓ 송금 한도 확인</li>
              <li>✓ 세금 규정 숙지</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}