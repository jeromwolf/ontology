'use client';

import { useState, useEffect } from 'react';
import { Volume2, Pause, MessageCircle, Users, Globe, Copy, CheckCircle, Play } from 'lucide-react';

export default function Chapter4() {
  const [copiedCode, setCopiedCode] = useState<string | null>(null)

  const copyCode = (code: string, id: string) => {
    navigator.clipboard.writeText(code)
    setCopiedCode(id)
    setTimeout(() => setCopiedCode(null), 2000)
  }

  const [activeTab, setActiveTab] = useState('airport')
  const [expandedTip, setExpandedTip] = useState<string | null>(null)

  const travelSections = [
    { id: 'airport', name: '공항', icon: '✈️' },
    { id: 'hotel', name: '호텔', icon: '🏨' },
    { id: 'restaurant', name: '레스토랑', icon: '🍽️' },
    { id: 'transportation', name: '교통', icon: '🚗' },
    { id: 'shopping', name: '쇼핑', icon: '🛍️' },
    { id: 'emergency', name: '응급상황', icon: '🆘' }
  ]

  const airportSituations = [
    {
      title: "체크인 카운터",
      expressions: [
        { eng: "I have a reservation under the name Smith.", kor: "스미스 이름으로 예약했습니다." },
        { eng: "I'd like a window seat, please.", kor: "창가 좌석으로 부탁드립니다." },
        { eng: "How many bags can I check in?", kor: "몇 개의 가방을 체크인할 수 있나요?" },
        { eng: "Is there an extra charge for overweight luggage?", kor: "수하물 초과 중량에 대한 추가 요금이 있나요?" },
        { eng: "Could I get an aisle seat instead?", kor: "대신 통로쪽 좌석으로 바꿀 수 있을까요?" }
      ]
    },
    {
      title: "보안검색대",
      expressions: [
        { eng: "Do I need to take off my shoes?", kor: "신발을 벗어야 하나요?" },
        { eng: "Can I keep my laptop in the bag?", kor: "노트북을 가방에 넣어둘 수 있나요?" },
        { eng: "Is this the line for international flights?", kor: "이것이 국제선 줄인가요?" },
        { eng: "Where should I put my liquids?", kor: "액체류는 어디에 두어야 하나요?" }
      ]
    },
    {
      title: "출입국 심사",
      expressions: [
        { eng: "I'm here for tourism/business.", kor: "관광/출장으로 왔습니다." },
        { eng: "I'll be staying for two weeks.", kor: "2주 동안 머물 예정입니다." },
        { eng: "This is my first time visiting your country.", kor: "귀하의 나라를 처음 방문합니다." },
        { eng: "I'm staying at the Hilton Hotel.", kor: "힐튼 호텔에 머물 예정입니다." }
      ]
    }
  ]

  const hotelSituations = [
    {
      title: "체크인",
      expressions: [
        { eng: "I have a reservation under Johnson.", kor: "존슨 이름으로 예약이 있습니다." },
        { eng: "Is breakfast included in the rate?", kor: "요금에 조식이 포함되어 있나요?" },
        { eng: "What time is checkout?", kor: "체크아웃 시간이 언제인가요?" },
        { eng: "Could I have a room on a higher floor?", kor: "더 높은 층의 방으로 가능할까요?" },
        { eng: "Is Wi-Fi available in the rooms?", kor: "객실에서 와이파이를 사용할 수 있나요?" }
      ]
    },
    {
      title: "호텔 서비스",
      expressions: [
        { eng: "Could you call me a taxi?", kor: "택시를 불러주실 수 있나요?" },
        { eng: "I'd like to extend my stay for one more night.", kor: "하루 더 연장하고 싶습니다." },
        { eng: "The air conditioning in my room isn't working.", kor: "제 방의 에어컨이 작동하지 않습니다." },
        { eng: "Could I get some extra towels?", kor: "수건을 더 받을 수 있을까요?" },
        { eng: "Is there a gym/pool in the hotel?", kor: "호텔에 헬스장/수영장이 있나요?" }
      ]
    }
  ]

  const emergencySituations = [
    {
      title: "의료 응급상황",
      expressions: [
        { eng: "I need to see a doctor immediately.", kor: "즉시 의사를 만나야 합니다." },
        { eng: "I'm having chest pain.", kor: "가슴이 아픕니다." },
        { eng: "I think I broke my arm.", kor: "팔이 부러진 것 같습니다." },
        { eng: "I'm allergic to penicillin.", kor: "저는 페니실린에 알레르기가 있습니다." },
        { eng: "Where is the nearest hospital?", kor: "가장 가까운 병원이 어디인가요?" }
      ]
    },
    {
      title: "경찰서/분실신고",
      expressions: [
        { eng: "I'd like to report a theft.", kor: "절도를 신고하고 싶습니다." },
        { eng: "My passport has been stolen.", kor: "여권을 도난당했습니다." },
        { eng: "I lost my wallet.", kor: "지갑을 잃어버렸습니다." },
        { eng: "Could you help me find the embassy?", kor: "대사관을 찾는 것을 도와주실 수 있나요?" },
        { eng: "I need to file a police report.", kor: "경찰서에 신고서를 작성해야 합니다." }
      ]
    }
  ]

  const culturalTips = [
    {
      country: "미국",
      tips: [
        "팁 문화: 레스토랑에서 15-20%, 택시에서 15-18% 팁이 관례입니다.",
        "개인공간: 대화할 때 팔 길이 정도의 거리를 유지하세요.",
        "인사: 악수가 일반적이며, 눈을 맞추는 것이 중요합니다.",
        "시간 관념: 약속 시간을 정확히 지키는 것이 매우 중요합니다."
      ]
    },
    {
      country: "영국",
      tips: [
        "줄서기: 영국인들은 줄서기를 매우 중요하게 생각합니다.",
        "예의: 'Please', 'Thank you', 'Sorry' 등의 표현을 자주 사용하세요.",
        "날씨 대화: 날씨에 대한 대화는 좋은 아이스브레이커입니다.",
        "펍 문화: 펍에서는 바에서 직접 주문하고 팁은 필수가 아닙니다."
      ]
    }
  ]

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-200 mb-4">
          완벽한 여행 영어 가이드
        </h2>
        <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
          해외여행의 모든 순간을 자신감 있게! 공항부터 호텔, 레스토랑, 쇼핑까지 
          여행의 전 과정에서 필요한 실전 영어 표현을 마스터하세요.
        </p>
      </div>

      {/* Travel Sections Navigation */}
      <div className="bg-gradient-to-r from-blue-50 to-cyan-50 dark:from-blue-950/20 dark:to-cyan-950/20 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
          🌍 여행 상황별 가이드
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          {travelSections.map(section => (
            <button
              key={section.id}
              onClick={() => setActiveTab(section.id)}
              className={`p-3 rounded-lg text-center transition-colors ${
                activeTab === section.id
                  ? 'bg-blue-500 text-white'
                  : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-blue-100 dark:hover:bg-blue-900/50'
              }`}
            >
              <div className="text-xl mb-1">{section.icon}</div>
              <div className="text-xs font-medium">{section.name}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Airport Content */}
      {activeTab === 'airport' && (
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
              ✈️ 공항에서 필요한 모든 표현
            </h3>
            
            {airportSituations.map((situation, idx) => (
              <div key={idx} className="mb-6">
                <button
                  onClick={() => setExpandedTip(expandedTip === situation.title ? null : situation.title)}
                  className="w-full text-left p-3 bg-blue-50 dark:bg-blue-950/20 rounded-lg hover:bg-blue-100 dark:hover:bg-blue-900/30 transition-colors"
                >
                  <h4 className="font-medium text-gray-800 dark:text-gray-200">
                    {idx + 1}. {situation.title}
                  </h4>
                </button>
                
                {expandedTip === situation.title && (
                  <div className="mt-3 space-y-3 pl-4">
                    {situation.expressions.map((expr, exprIdx) => (
                      <div key={exprIdx} className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                        <p className="font-medium text-gray-800 dark:text-gray-200 mb-1">
                          "{expr.eng}"
                        </p>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          {expr.kor}
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Airport Survival Tips */}
          <div className="bg-gradient-to-r from-amber-50 to-orange-50 dark:from-amber-950/20 dark:to-orange-950/20 rounded-xl p-6">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
              💡 공항 서바이벌 팁
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">📝 체크인 전 준비사항</h4>
                <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                  <li>• 여권과 e-ticket 준비</li>
                  <li>• 수하물 중량 제한 확인</li>
                  <li>• 좌석 선호도 미리 결정</li>
                  <li>• 특별식 요청사항 확인</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">🔍 보안검색 통과 요령</h4>
                <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                  <li>• 액체류는 100ml 이하로 준비</li>
                  <li>• 전자기기는 별도 트레이에</li>
                  <li>• 금속 액세서리 미리 제거</li>
                  <li>• 신발 벗기 준비</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Hotel Content */}
      {activeTab === 'hotel' && (
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
              🏨 호텔에서의 완벽한 소통
            </h3>
            
            {hotelSituations.map((situation, idx) => (
              <div key={idx} className="mb-6 p-4 bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-950/20 dark:to-emerald-950/20 rounded-lg">
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">
                  {situation.title}
                </h4>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                  {situation.expressions.map((expr, exprIdx) => (
                    <div key={exprIdx} className="bg-white dark:bg-gray-800 rounded-lg p-3">
                      <p className="font-medium text-gray-800 dark:text-gray-200 text-sm mb-1">
                        "{expr.eng}"
                      </p>
                      <p className="text-xs text-gray-600 dark:text-gray-400">
                        {expr.kor}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Emergency Content */}
      {activeTab === 'emergency' && (
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
              🆘 응급상황 대처 영어
            </h3>
            
            {emergencySituations.map((situation, idx) => (
              <div key={idx} className="mb-6 p-4 bg-gradient-to-r from-red-50 to-pink-50 dark:from-red-950/20 dark:to-pink-950/20 rounded-lg">
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">
                  {situation.title}
                </h4>
                <div className="space-y-2">
                  {situation.expressions.map((expr, exprIdx) => (
                    <div key={exprIdx} className="bg-white dark:bg-gray-800 rounded-lg p-3">
                      <p className="font-medium text-gray-800 dark:text-gray-200 mb-1">
                        "{expr.eng}"
                      </p>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        {expr.kor}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* Emergency Numbers */}
          <div className="bg-red-100 dark:bg-red-950/20 rounded-xl p-6 border border-red-200 dark:border-red-800">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
              📞 국가별 응급 전화번호
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">🇺🇸 미국</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">응급상황: 911</p>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">🇬🇧 영국</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">응급상황: 999</p>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">🇪🇺 유럽연합</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">응급상황: 112</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Cultural Tips */}
      <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-950/20 dark:to-indigo-950/20 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
          🌏 여행지별 문화 팁
        </h3>
        <div className="space-y-4">
          {culturalTips.map((country, idx) => (
            <div key={idx} className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">
                {country.country} 여행 시 알아두면 좋은 문화
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                {country.tips.map((tip, tipIdx) => (
                  <div key={tipIdx} className="flex items-start gap-2 text-sm text-gray-600 dark:text-gray-400">
                    <div className="w-1.5 h-1.5 bg-purple-500 rounded-full mt-2 flex-shrink-0" />
                    <span>{tip}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Travel Checklist */}
      <div className="bg-gradient-to-r from-teal-500 to-cyan-600 rounded-xl p-6 text-white">
        <h3 className="text-lg font-semibold mb-3">🎒 여행 영어 준비 체크리스트</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-teal-100">
          <div>
            <h4 className="font-semibold mb-2">📚 출발 전 준비</h4>
            <ul className="text-sm space-y-1">
              <li>✓ 기본 인사말 숙지</li>
              <li>✓ 숫자와 날짜 표현</li>
              <li>✓ 응급상황 표현</li>
              <li>✓ 방향과 교통 관련 표현</li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold mb-2">🗣️ 실전 연습</h4>
            <ul className="text-sm space-y-1">
              <li>✓ 호텔 체크인 역할극</li>
              <li>✓ 레스토랑 주문 연습</li>
              <li>✓ 길 묻기 시뮬레이션</li>
              <li>✓ 쇼핑 대화 연습</li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold mb-2">📱 유용한 앱</h4>
            <ul className="text-sm space-y-1">
              <li>✓ 번역 앱 다운로드</li>
              <li>✓ 지도 앱 오프라인 설정</li>
              <li>✓ 통화 변환 앱</li>
              <li>✓ 현지 교통 앱</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

