'use client'

import References from '@/components/common/References'

export default function Chapter9() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-8">
        Chapter 9: 이미지센서 & 디스플레이 반도체
      </h1>

      {/* CMOS 이미지센서 */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          9.1 CMOS 이미지센서 (CIS - CMOS Image Sensor)
        </h2>
        <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg mb-4">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            CMOS 이미지센서는 빛을 전기 신호로 변환하여 이미지를 포착하는 반도체입니다.
            스마트폰, 카메라, 자율주행차의 핵심 부품입니다.
          </p>

          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">
                픽셀 구조
              </h3>
              <svg className="w-full h-56" viewBox="0 0 280 240">
                {/* 픽셀 배열 */}
                <text x="100" y="20" fontSize="11" fontWeight="bold" fill="#3B82F6">
                  4-Transistor (4T) 픽셀
                </text>

                {/* 포토다이오드 */}
                <rect x="60" y="40" width="60" height="60" fill="#DBEAFE" stroke="#3B82F6" strokeWidth="2" />
                <text x="70" y="75" fontSize="10" fill="#2563EB" fontWeight="bold">PD</text>
                <text x="55" y="110" fontSize="8" fill="#6B7280">Photodiode</text>

                {/* 트랜지스터들 */}
                <rect x="140" y="40" width="25" height="20" fill="#FEE2E2" stroke="#EF4444" strokeWidth="1" />
                <text x="145" y="54" fontSize="8" fill="#DC2626">TX</text>
                <text x="170" y="54" fontSize="7" fill="#6B7280">Transfer</text>

                <rect x="180" y="40" width="25" height="20" fill="#FEE2E2" stroke="#EF4444" strokeWidth="1" />
                <text x="185" y="54" fontSize="8" fill="#DC2626">SF</text>
                <text x="210" y="54" fontSize="7" fill="#6B7280">Source Follower</text>

                <rect x="140" y="70" width="25" height="20" fill="#FEE2E2" stroke="#EF4444" strokeWidth="1" />
                <text x="142" y="84" fontSize="8" fill="#DC2626">RST</text>
                <text x="170" y="84" fontSize="7" fill="#6B7280">Reset</text>

                <rect x="180" y="70" width="25" height="20" fill="#FEE2E2" stroke="#EF4444" strokeWidth="1" />
                <text x="182" y="84" fontSize="8" fill="#DC2626">SEL</text>
                <text x="210" y="84" fontSize="7" fill="#6B7280">Select</text>

                {/* Bayer 필터 */}
                <text x="80" y="130" fontSize="10" fontWeight="bold" fill="#059669">
                  Bayer 컬러 필터
                </text>
                <rect x="60" y="140" width="30" height="30" fill="#EF4444" opacity="0.6" />
                <text x="70" y="160" fontSize="10" fill="white">R</text>

                <rect x="90" y="140" width="30" height="30" fill="#10B981" opacity="0.6" />
                <text x="100" y="160" fontSize="10" fill="white">G</text>

                <rect x="60" y="170" width="30" height="30" fill="#10B981" opacity="0.6" />
                <text x="70" y="190" fontSize="10" fill="white">G</text>

                <rect x="90" y="170" width="30" height="30" fill="#3B82F6" opacity="0.6" />
                <text x="100" y="190" fontSize="10" fill="white">B</text>

                {/* 마이크로렌즈 */}
                <ellipse cx="75" cy="215" rx="15" ry="8" fill="#A78BFA" opacity="0.5" />
                <text x="40" y="220" fontSize="8" fill="#7C3AED">Microlens</text>
              </svg>
            </div>

            <div className="space-y-3">
              <div className="bg-gray-800 text-white p-3 rounded text-xs">
                <h4 className="font-semibold mb-2">CIS 핵심 기술:</h4>
                <code>{`FSI (Front-Side Illumination)
- 빛이 회로층 통과 후 포토다이오드 도달
- 전통적 방식

BSI (Back-Side Illumination)
- 빛이 직접 포토다이오드 도달
- 감도 40% 향상
- 현대 스마트폰 표준

Stacked CMOS
- 픽셀층 + 로직층 TSV로 적층
- Sony Exmor RS
- 고속 readout (960fps 슬로우모션)

DTI (Deep Trench Isolation)
- 픽셀 간 빛 누화 방지
- 픽셀 크기 0.7μm까지 가능

Dual Pixel AF
- 위상차 검출 자동초점
- DSLR급 AF 속도`}</code>
              </div>

              <div className="bg-purple-50 dark:bg-purple-900/20 p-3 rounded">
                <h4 className="font-semibold text-purple-800 dark:text-purple-300 mb-2">
                  주요 기업
                </h4>
                <div className="space-y-1 text-xs text-gray-700 dark:text-gray-300">
                  <div className="flex justify-between">
                    <span>Sony</span>
                    <span className="text-blue-600">점유율 44% (1위)</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Samsung</span>
                    <span className="text-purple-600">점유율 29% (2위)</span>
                  </div>
                  <div className="flex justify-between">
                    <span>OmniVision</span>
                    <span className="text-green-600">중저가 시장</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
            <h3 className="font-semibold text-green-800 dark:text-green-300 mb-3">
              응용 분야
            </h3>
            <div className="grid md:grid-cols-4 gap-3 text-sm text-gray-700 dark:text-gray-300">
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <h4 className="font-semibold mb-2">스마트폰</h4>
                <p className="text-xs">108MP, 200MP 초고화소</p>
                <p className="text-xs text-gray-500">Pixel Binning</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <h4 className="font-semibold mb-2">자율주행</h4>
                <p className="text-xs">LED Flicker 억제</p>
                <p className="text-xs text-gray-500">HDR 120dB+</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <h4 className="font-semibold mb-2">산업용</h4>
                <p className="text-xs">Global Shutter</p>
                <p className="text-xs text-gray-500">NIR 감도</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <h4 className="font-semibold mb-2">의료/보안</h4>
                <p className="text-xs">저조도 성능</p>
                <p className="text-xs text-gray-500">UV/IR 센서</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 디스플레이 드라이버 IC */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          9.2 디스플레이 드라이버 IC (DDI)
        </h2>

        <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-lg mb-4">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            디스플레이 드라이버 IC는 디스플레이 패널의 각 픽셀을 제어하여 이미지를 표시합니다.
            LCD/OLED 패널의 필수 부품입니다.
          </p>

          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">
                OLED 드라이버 구조
              </h3>
              <svg className="w-full h-56" viewBox="0 0 280 240">
                {/* OLED 픽셀 */}
                <rect x="60" y="30" width="160" height="100" fill="#F3F4F6" stroke="#9CA3AF" strokeWidth="2" />
                <text x="90" y="25" fontSize="10" fontWeight="bold" fill="#374151">
                  OLED 픽셀 매트릭스
                </text>

                {/* 서브픽셀 */}
                <rect x="80" y="50" width="15" height="30" fill="#EF4444" opacity="0.7" />
                <text x="83" y="90" fontSize="8" fill="white">R</text>

                <rect x="100" y="50" width="15" height="30" fill="#10B981" opacity="0.7" />
                <text x="103" y="90" fontSize="8" fill="white">G</text>

                <rect x="120" y="50" width="15" height="30" fill="#3B82F6" opacity="0.7" />
                <text x="123" y="90" fontSize="8" fill="white">B</text>

                {/* TFT 백플레인 */}
                <rect x="70" y="100" width="10" height="10" fill="#DBEAFE" stroke="#3B82F6" strokeWidth="1" />
                <text x="65" y="118" fontSize="7" fill="#2563EB">TFT</text>

                <rect x="90" y="100" width="10" height="10" fill="#DBEAFE" stroke="#3B82F6" strokeWidth="1" />
                <rect x="110" y="100" width="10" height="10" fill="#DBEAFE" stroke="#3B82F6" strokeWidth="1" />

                {/* 드라이버 IC */}
                <rect x="60" y="150" width="80" height="30" fill="#7C3AED" stroke="#6B21A8" strokeWidth="2" />
                <text x="75" y="170" fontSize="10" fill="white" fontWeight="bold">
                  DDI (Driver IC)
                </text>

                {/* 연결선 */}
                <line x1="100" y1="130" x2="100" y2="150" stroke="#374151" strokeWidth="2" />
                <path d="M 95 145 L 100 150 L 105 145" fill="#374151" />

                {/* 신호 */}
                <text x="60" y="200" fontSize="8" fill="#6B7280">
                  • Gate 신호 (수평 스캔)
                </text>
                <text x="60" y="215" fontSize="8" fill="#6B7280">
                  • Data 신호 (픽셀 전압)
                </text>
                <text x="60" y="230" fontSize="8" fill="#6B7280">
                  • 전류 제어 (밝기 조절)
                </text>
              </svg>
            </div>

            <div className="space-y-3">
              <div className="bg-gray-800 text-white p-3 rounded text-xs">
                <h4 className="font-semibold mb-2">DDI 주요 기술:</h4>
                <code>{`OLED DDI
- 전류 구동 방식
- 높은 전압 (ELVDD/ELVSS)
- Compensation 회로 필수
- 120Hz+ 고주사율

LCD DDI
- 전압 구동 방식
- TFT-LCD: 백라이트 제어
- LTPS/a-Si/Oxide TFT

TDDI (Touch + Display)
- 터치센서 + 디스플레이 통합
- 원가 절감, 얇은 두께
- In-cell/On-cell 터치

COF (Chip On Film)
- Flexible 디스플레이용
- 폴더블폰 필수 기술

COG (Chip On Glass)
- 패널에 직접 본딩
- TV, 모니터용`}</code>
              </div>

              <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded">
                <h4 className="font-semibold text-blue-800 dark:text-blue-300 mb-2">
                  주요 기업 및 응용
                </h4>
                <div className="space-y-2 text-xs text-gray-700 dark:text-gray-300">
                  <div className="bg-white dark:bg-gray-800 p-2 rounded">
                    <div className="font-semibold mb-1">삼성전자 LSI</div>
                    <div className="text-gray-500">OLED DDI, TDDI</div>
                  </div>
                  <div className="bg-white dark:bg-gray-800 p-2 rounded">
                    <div className="font-semibold mb-1">Novatek</div>
                    <div className="text-gray-500">LCD DDI 1위</div>
                  </div>
                  <div className="bg-white dark:bg-gray-800 p-2 rounded">
                    <div className="font-semibold mb-1">Sitronix</div>
                    <div className="text-gray-500">중소형 OLED</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg">
            <h3 className="font-semibold text-yellow-800 dark:text-yellow-300 mb-3">
              차세대 기술
            </h3>
            <div className="grid md:grid-cols-3 gap-3 text-sm text-gray-700 dark:text-gray-300">
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <h4 className="font-semibold mb-2 text-blue-700 dark:text-blue-400">
                  Micro LED
                </h4>
                <p className="text-xs">
                  초소형 LED 자발광, 극한 밝기/명암비, 픽셀 개별 제어
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <h4 className="font-semibold mb-2 text-purple-700 dark:text-purple-400">
                  QD-OLED
                </h4>
                <p className="text-xs">
                  양자점 + OLED, 더 넓은 색재현율, 삼성 QD-Display
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <h4 className="font-semibold mb-2 text-green-700 dark:text-green-400">
                  Transparent Display
                </h4>
                <p className="text-xs">
                  투명 OLED, AR 글래스, 자동차 HUD, 미래 디스플레이
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 시장 동향 */}
      <section className="mb-8">
        <div className="bg-gradient-to-r from-blue-50 via-purple-50 to-pink-50 dark:from-blue-900/20 dark:via-purple-900/20 dark:to-pink-900/20 p-6 rounded-lg">
          <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
            시장 동향 및 미래 전망
          </h3>

          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-semibold text-blue-700 dark:text-blue-400 mb-3">
                CIS 시장
              </h4>
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li className="flex items-start gap-2">
                  <span className="text-green-600">▲</span>
                  <span>2025년 시장 규모: 250억 달러</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-600">▲</span>
                  <span>자율주행/ADAS 수요 폭발적 증가</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-600">●</span>
                  <span>스마트폰: 멀티 카메라 (4~5개)</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-purple-600">●</span>
                  <span>초고화소 경쟁: 200MP → 320MP</span>
                </li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-semibold text-purple-700 dark:text-purple-400 mb-3">
                DDI 시장
              </h4>
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li className="flex items-start gap-2">
                  <span className="text-green-600">▲</span>
                  <span>2025년 시장 규모: 80억 달러</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-600">▲</span>
                  <span>폴더블/롤러블 OLED DDI 성장</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-600">●</span>
                  <span>TDDI 일체화 트렌드</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-purple-600">●</span>
                  <span>LTPO OLED: 적응형 주사율 (1-120Hz)</span>
                </li>
              </ul>
            </div>
          </div>

          <div className="bg-gray-800 text-white p-4 rounded-lg">
            <h4 className="text-lg font-semibold mb-3 text-center">
              한국 반도체 산업의 강점 분야
            </h4>
            <div className="text-sm space-y-2">
              <p>✅ <strong>CIS</strong>: 삼성전자 세계 2위 (점유율 29%), 차량용 CIS 선도</p>
              <p>✅ <strong>OLED DDI</strong>: 삼성 LSI, LX세미콘 글로벌 Top 3</p>
              <p>✅ <strong>시스템 반도체</strong>: 메모리 외 차세대 성장 동력</p>
              <p>🎯 <strong>목표</strong>: 파운드리 + CIS/DDI 시너지로 종합 반도체 강국 도약</p>
            </div>
          </div>
        </div>
      </section>

      {/* 최종 요약 */}
      <section className="mb-8">
        <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 p-6 rounded-lg">
          <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
            핵심 요약
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li className="flex items-start">
              <span className="text-blue-600 dark:text-blue-400 mr-2">▪</span>
              <span>CIS는 BSI/Stacked 구조로 감도와 속도를 동시에 확보합니다</span>
            </li>
            <li className="flex items-start">
              <span className="text-blue-600 dark:text-blue-400 mr-2">▪</span>
              <span>자율주행 시대에 CIS는 차량당 8~12개 탑재되어 핵심 부품입니다</span>
            </li>
            <li className="flex items-start">
              <span className="text-blue-600 dark:text-blue-400 mr-2">▪</span>
              <span>OLED DDI는 픽셀별 전류 제어로 완벽한 블랙과 고명암비를 구현합니다</span>
            </li>
            <li className="flex items-start">
              <span className="text-blue-600 dark:text-blue-400 mr-2">▪</span>
              <span>TDDI 통합으로 얇고 저렴한 디스플레이 모듈이 가능해졌습니다</span>
            </li>
            <li className="flex items-start">
              <span className="text-blue-600 dark:text-blue-400 mr-2">▪</span>
              <span>Micro LED와 QD-OLED가 차세대 디스플레이 기술로 부상하고 있습니다</span>
            </li>
          </ul>
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 반도체 산업 리소스',
            icon: 'web' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'TSMC Technology',
                url: 'https://www.tsmc.com/english/dedicatedFoundry/technology',
                description: 'TSMC 파운드리 공정 기술 (3nm N3, 2nm N2 로드맵)'
              },
              {
                title: 'Samsung Foundry',
                url: 'https://www.samsungfoundry.com/',
                description: '삼성 파운드리 공정 기술 (3nm GAA, 2nm SF2)'
              },
              {
                title: 'Intel Process Technology',
                url: 'https://www.intel.com/content/www/us/en/silicon-innovations/intel-process-technology.html',
                description: 'Intel 프로세스 기술 (Intel 4, Intel 3, Intel 20A)'
              },
              {
                title: 'ASML - Lithography Systems',
                url: 'https://www.asml.com/en/technology/lithography-principles',
                description: 'EUV 리소그래피 기술 선도 기업'
              },
              {
                title: 'SEMI - Semiconductor Industry Association',
                url: 'https://www.semi.org/',
                description: '반도체 산업 협회 (시장 동향, 표준)'
              }
            ]
          },
          {
            title: '📖 핵심 교재 & 리소스',
            icon: 'research' as const,
            color: 'border-cyan-500',
            items: [
              {
                title: 'Semiconductor Device Fundamentals (Pierret)',
                url: 'https://www.pearson.com/en-us/subject-catalog/p/semiconductor-device-fundamentals/P200000003255',
                description: '반도체 소자 물리학 교과서 (1996)'
              },
              {
                title: 'VLSI Design (Weste & Harris)',
                url: 'https://www.pearson.com/en-us/subject-catalog/p/cmos-vlsi-design-a-circuits-and-systems-perspective/P200000003328',
                description: 'CMOS VLSI 설계 바이블 (4th Edition, 2010)'
              },
              {
                title: 'IEEE Electron Devices Society',
                url: 'https://eds.ieee.org/',
                description: 'IEEE 전자 소자 학회 (논문, 컨퍼런스)'
              },
              {
                title: 'Semiconductor Today',
                url: 'https://www.semiconductor-today.com/',
                description: '반도체 산업 뉴스 및 기술 동향'
              }
            ]
          },
          {
            title: '🛠️ 실전 도구',
            icon: 'tools' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'Cadence Virtuoso',
                url: 'https://www.cadence.com/en_US/home/tools/custom-ic-analog-rf-design/circuit-design/virtuoso-studio.html',
                description: 'Analog/Mixed-Signal IC 설계 도구'
              },
              {
                title: 'Synopsys Design Compiler',
                url: 'https://www.synopsys.com/implementation-and-signoff/rtl-synthesis-test/design-compiler-graphical.html',
                description: '디지털 IC 합성 도구'
              },
              {
                title: 'Mentor Graphics (Siemens EDA)',
                url: 'https://eda.sw.siemens.com/',
                description: 'IC 설계 및 검증 툴 (Calibre DRC/LVS)'
              },
              {
                title: 'SPICE (Simulation Program with Integrated Circuit Emphasis)',
                url: 'https://ngspice.sourceforge.io/',
                description: '회로 시뮬레이션 (ngspice 오픈소스)'
              },
              {
                title: 'Verilog / VHDL',
                url: 'https://www.ieee.org/communities/standards',
                description: '하드웨어 기술 언어 (HDL) IEEE 표준'
              }
            ]
          }
        ]}
      />
    </div>
  )
}
