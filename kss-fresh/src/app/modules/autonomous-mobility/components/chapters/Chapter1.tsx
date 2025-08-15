'use client';

import { Car, Navigation, Cpu, MapPin, Route, Zap } from 'lucide-react';

export default function Chapter1() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          자율주행의 진화와 미래
        </h2>
        
        <div className="bg-gradient-to-r from-cyan-50 to-blue-50 dark:from-cyan-900/20 dark:to-blue-900/20 rounded-xl p-6 mb-6">
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
            자율주행은 단순한 기술 혁신을 넘어 인류의 이동 방식을 근본적으로 바꾸는 패러다임 시프트입니다.
            Tesla의 FSD, Waymo의 완전 무인 운행, 그리고 국내 카카오모빌리티의 상용화까지,
            우리는 SF에서나 봤던 미래가 현실이 되는 전환점에 서 있습니다.
          </p>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🚗 SAE 자율주행 레벨 체계
        </h3>
        
        <div className="grid md:grid-cols-3 gap-6 mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">Level 0-2: 운전자 중심</h4>
            <div className="space-y-3">
              <div className="flex items-start gap-3">
                <div className="w-6 h-6 bg-red-100 dark:bg-red-900/30 rounded-full flex items-center justify-center text-xs font-bold text-red-600 dark:text-red-400">0</div>
                <div>
                  <span className="font-semibold">No Automation</span>
                  <p className="text-sm text-gray-600 dark:text-gray-400">완전 수동 운전</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-6 h-6 bg-orange-100 dark:bg-orange-900/30 rounded-full flex items-center justify-center text-xs font-bold text-orange-600 dark:text-orange-400">1</div>
                <div>
                  <span className="font-semibold">Driver Assistance</span>
                  <p className="text-sm text-gray-600 dark:text-gray-400">어댑티브 크루즈 컨트롤</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-6 h-6 bg-yellow-100 dark:bg-yellow-900/30 rounded-full flex items-center justify-center text-xs font-bold text-yellow-600 dark:text-yellow-400">2</div>
                <div>
                  <span className="font-semibold">Partial Automation</span>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Tesla Autopilot, 현대 HDA</p>
                </div>
              </div>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">Level 3-4: 시스템 중심</h4>
            <div className="space-y-3">
              <div className="flex items-start gap-3">
                <div className="w-6 h-6 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center text-xs font-bold text-blue-600 dark:text-blue-400">3</div>
                <div>
                  <span className="font-semibold">Conditional Automation</span>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Audi Traffic Jam Pilot</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-6 h-6 bg-indigo-100 dark:bg-indigo-900/30 rounded-full flex items-center justify-center text-xs font-bold text-indigo-600 dark:text-indigo-400">4</div>
                <div>
                  <span className="font-semibold">High Automation</span>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Waymo One 상용 서비스</p>
                </div>
              </div>
            </div>
          </div>
          
          <div className="bg-gradient-to-br from-cyan-50 to-blue-50 dark:from-cyan-900/20 dark:to-blue-900/20 rounded-lg p-6 border border-cyan-200 dark:border-cyan-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">Level 5: 완전 자율</h4>
            <div className="space-y-3">
              <div className="flex items-start gap-3">
                <div className="w-6 h-6 bg-cyan-100 dark:bg-cyan-900/30 rounded-full flex items-center justify-center text-xs font-bold text-cyan-600 dark:text-cyan-400">5</div>
                <div>
                  <span className="font-semibold">Full Automation</span>
                  <p className="text-sm text-gray-600 dark:text-gray-400">운전대 없는 완전 무인차</p>
                </div>
              </div>
              <div className="bg-cyan-100 dark:bg-cyan-900/30 rounded-lg p-3 mt-3">
                <p className="text-xs text-cyan-700 dark:text-cyan-400">
                  🎯 목표: 2030년 상용화
                </p>
              </div>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🌍 글로벌 자율주행 생태계
        </h3>
        
        <div className="bg-gray-50 dark:bg-gray-900 rounded-xl p-6 mb-6">
          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <h4 className="font-bold text-gray-900 dark:text-white mb-4">🇺🇸 미국 빅테크</h4>
              <div className="space-y-3">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <div className="flex items-center gap-3 mb-2">
                    <div className="w-8 h-8 bg-red-100 dark:bg-red-900/30 rounded flex items-center justify-center">
                      <Car className="w-4 h-4 text-red-600 dark:text-red-400" />
                    </div>
                    <span className="font-bold">Tesla FSD</span>
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    비전 중심, 뉴럴넷 end-to-end 학습
                  </p>
                </div>
                
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <div className="flex items-center gap-3 mb-2">
                    <div className="w-8 h-8 bg-blue-100 dark:bg-blue-900/30 rounded flex items-center justify-center">
                      <Navigation className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                    </div>
                    <span className="font-bold">Waymo</span>
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    LiDAR 기반, 상용 로보택시 운영
                  </p>
                </div>
                
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <div className="flex items-center gap-3 mb-2">
                    <div className="w-8 h-8 bg-green-100 dark:bg-green-900/30 rounded flex items-center justify-center">
                      <Cpu className="w-4 h-4 text-green-600 dark:text-green-400" />
                    </div>
                    <span className="font-bold">NVIDIA DRIVE</span>
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    AI 컴퓨팅 플랫폼, 옴니버스 시뮬레이션
                  </p>
                </div>
              </div>
            </div>
            
            <div>
              <h4 className="font-bold text-gray-900 dark:text-white mb-4">🇰🇷 한국 & 🇨🇳 중국</h4>
              <div className="space-y-3">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <div className="flex items-center gap-3 mb-2">
                    <div className="w-8 h-8 bg-purple-100 dark:bg-purple-900/30 rounded flex items-center justify-center">
                      <MapPin className="w-4 h-4 text-purple-600 dark:text-purple-400" />
                    </div>
                    <span className="font-bold">42dot (현대)</span>
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    SDV OS, 아이오닉 6 기반 자율주행
                  </p>
                </div>
                
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <div className="flex items-center gap-3 mb-2">
                    <div className="w-8 h-8 bg-orange-100 dark:bg-orange-900/30 rounded flex items-center justify-center">
                      <Route className="w-4 h-4 text-orange-600 dark:text-orange-400" />
                    </div>
                    <span className="font-bold">카카오모빌리티</span>
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    제주도 자율주행 택시 상용화
                  </p>
                </div>
                
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                  <div className="flex items-center gap-3 mb-2">
                    <div className="w-8 h-8 bg-red-100 dark:bg-red-900/30 rounded flex items-center justify-center">
                      <Zap className="w-4 h-4 text-red-600 dark:text-red-400" />
                    </div>
                    <span className="font-bold">바이두 Apollo</span>
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    중국 최대 자율주행 플랫폼
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Tesla 로보택시 최신 정보 추가 */}
        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🚖 Tesla 로보택시: 2025년 현실이 되다
        </h3>
        
        <div className="bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 rounded-xl p-6 mb-6">
          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border-l-4 border-red-500">
              <h4 className="font-bold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
                <Car className="w-5 h-5 text-red-600" />
                2025년 6월: Austin에서 역사적인 첫 발걸음
              </h4>
              <div className="space-y-3 text-gray-700 dark:text-gray-300">
                <p>
                  <strong>2025년 6월 22일</strong>, Tesla가 텍사스 오스틴에서 로보택시 서비스를 공식 출시했습니다.
                  약 10대의 Model Y 차량으로 시작한 이 서비스는 <strong>전방 좌석에 안전 요원이 탑승</strong>한 상태로
                  운영되며, 한 번에 <strong>$4.20의 고정 요금</strong>으로 이용할 수 있습니다.
                </p>
                <div className="bg-red-50 dark:bg-red-900/20 rounded p-3">
                  <p className="text-sm">
                    <strong>🎯 주요 성과:</strong> 2025년 7월까지 7,000마일의 자율주행을 <strong>무사고</strong>로 달성
                  </p>
                </div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">📍 확장 계획 및 운영 현황</h4>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="space-y-3">
                  <h5 className="font-semibold text-gray-800 dark:text-gray-200">현재 운영 (2025년)</h5>
                  <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                    <li className="flex items-start gap-2">
                      <span className="text-green-500">✓</span>
                      <div>
                        <strong>오스틴, 텍사스:</strong> 10대 → 수천 대 확장 예정
                      </div>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-green-500">✓</span>
                      <div>
                        <strong>지오펜싱 구역:</strong> 제한된 지역 내 운행
                      </div>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-green-500">✓</span>
                      <div>
                        <strong>원격 모니터링:</strong> Tesla 직원이 실시간 감독
                      </div>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-green-500">✓</span>
                      <div>
                        <strong>FSD Unsupervised:</strong> 새로운 버전 테스트 중
                      </div>
                    </li>
                  </ul>
                </div>
                
                <div className="space-y-3">
                  <h5 className="font-semibold text-gray-800 dark:text-gray-200">확장 예정 도시</h5>
                  <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                    <li className="flex items-start gap-2">
                      <span className="text-blue-500">⏳</span>
                      <div>
                        <strong>로스앤젤레스 & 샌프란시스코:</strong> 2025년 하반기
                      </div>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-blue-500">⏳</span>
                      <div>
                        <strong>애리조나:</strong> 2025년 내 예정
                      </div>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-yellow-500">📅</span>
                      <div>
                        <strong>미국 전역:</strong> 2026년 목표
                      </div>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-yellow-500">📅</span>
                      <div>
                        <strong>유럽:</strong> 2026년 5월 진출 계획
                      </div>
                    </li>
                  </ul>
                </div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">🚗 차량 및 기술 사양</h4>
              <div className="space-y-3">
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-red-500 rounded-full mt-2"></div>
                  <div>
                    <strong>현재 차량:</strong> Model Y (FSD Unsupervised 탑재)
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-red-500 rounded-full mt-2"></div>
                  <div>
                    <strong>미래 차량:</strong> Cybercab (2026년 생산 시작 예정)
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-red-500 rounded-full mt-2"></div>
                  <div>
                    <strong>기술 방식:</strong> 카메라 기반 비전 시스템 + AI 칩
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-red-500 rounded-full mt-2"></div>
                  <div>
                    <strong>네트워크 확장:</strong> 개인 소유 Tesla 차량도 로보택시로 활용 계획
                  </div>
                </div>
              </div>
            </div>
            
            <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2 flex items-center gap-2">
                <Zap className="w-5 h-5 text-yellow-600" />
                경쟁 환경과 과제
              </h4>
              <div className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <p>
                  <strong>Waymo와의 경쟁:</strong> Waymo는 2024년에만 400만 건의 유료 승차를 완료했으며,
                  피닉스, 샌프란시스코, 로스앤젤레스에서 500평방마일 이상을 커버하고 있습니다.
                </p>
                <p>
                  <strong>규제 문제:</strong> 캘리포니아에서는 아직 상업용 로보택시 운영 허가를 신청하지 않은 상태입니다.
                  각 주마다 다른 규제 요구사항을 충족해야 합니다.
                </p>
                <p>
                  <strong>기술적 도전:</strong> LiDAR 없이 카메라만으로 Level 4 자율주행을 구현하는 것은
                  여전히 기술적 도전 과제입니다.
                </p>
              </div>
            </div>
            
            <div className="bg-gradient-to-r from-cyan-50 to-blue-50 dark:from-cyan-900/20 dark:to-blue-900/20 rounded-lg p-4">
              <p className="text-sm text-gray-700 dark:text-gray-300 italic">
                💡 <strong>미래 전망:</strong> Elon Musk는 "2025년 내에 미국 여러 도시에서 무인 FSD를 실현하고,
                2026년에는 전국적으로 확대할 것"이라고 발표했습니다. Tesla의 로보택시는 단순한 이동 수단을 넘어
                개인 차량 소유의 패러다임을 바꾸는 혁신이 될 것으로 기대됩니다.
              </p>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          📈 기술 발전 로드맵
        </h3>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700 mb-6">
          <div className="space-y-6">
            <div className="flex items-center gap-4">
              <div className="w-16 h-16 bg-green-100 dark:bg-green-900/30 rounded-full flex items-center justify-center">
                <span className="text-green-600 dark:text-green-400 font-bold">2024</span>
              </div>
              <div className="flex-1">
                <h4 className="font-bold text-gray-900 dark:text-white">Level 3 상용화</h4>
                <p className="text-gray-600 dark:text-gray-400">고속도로 자율주행, 조건부 무인화</p>
                <div className="flex gap-2 mt-2">
                  <span className="px-2 py-1 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 text-xs rounded">Mercedes EQS</span>
                  <span className="px-2 py-1 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 text-xs rounded">BMW iX</span>
                </div>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              <div className="w-16 h-16 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center">
                <span className="text-blue-600 dark:text-blue-400 font-bold">2027</span>
              </div>
              <div className="flex-1">
                <h4 className="font-bold text-gray-900 dark:text-white">Level 4 확산</h4>
                <p className="text-gray-600 dark:text-gray-400">도심 무인 택시, 물류 자동화</p>
                <div className="flex gap-2 mt-2">
                  <span className="px-2 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 text-xs rounded">Robotaxi</span>
                  <span className="px-2 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 text-xs rounded">자율배송</span>
                </div>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              <div className="w-16 h-16 bg-cyan-100 dark:bg-cyan-900/30 rounded-full flex items-center justify-center">
                <span className="text-cyan-600 dark:text-cyan-400 font-bold">2030</span>
              </div>
              <div className="flex-1">
                <h4 className="font-bold text-gray-900 dark:text-white">Level 5 실현</h4>
                <p className="text-gray-600 dark:text-gray-400">완전 무인차, 운전대 제거</p>
                <div className="flex gap-2 mt-2">
                  <span className="px-2 py-1 bg-cyan-100 dark:bg-cyan-900/30 text-cyan-700 dark:text-cyan-400 text-xs rounded">Full Autonomy</span>
                  <span className="px-2 py-1 bg-cyan-100 dark:bg-cyan-900/30 text-cyan-700 dark:text-cyan-400 text-xs rounded">MaaS 통합</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="border-t border-gray-200 dark:border-gray-700 pt-8">
        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          ⚖️ 법규 및 윤리적 이슈
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
            <h4 className="font-bold text-red-700 dark:text-red-400 mb-3">
              🚨 해결해야 할 과제들
            </h4>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• 사고 책임 소재 (제조사 vs 소유자)</li>
              <li>• 윤리적 딜레마 (트롤리 문제)</li>
              <li>• 사이버보안 위협</li>
              <li>• 일자리 대체 (운전업 종사자)</li>
              <li>• 데이터 프라이버시 보호</li>
            </ul>
          </div>
          
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
            <h4 className="font-bold text-green-700 dark:text-green-400 mb-3">
              ✅ 기대되는 효과들
            </h4>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• 교통사고 90% 감소 (인적 요인 제거)</li>
              <li>• 교통 효율성 40% 향상</li>
              <li>• 고령자, 장애인 이동권 확대</li>
              <li>• 주차공간 80% 절약</li>
              <li>• 배출가스 50% 감소 (전동화 연계)</li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  )
}