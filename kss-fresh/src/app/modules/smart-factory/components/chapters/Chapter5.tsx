'use client';

import { 
  Wifi, Cloud, Server
} from 'lucide-react';
import CodeEditor from '../CodeEditor';
import Link from 'next/link';

export default function Chapter5() {
  return (
    <div className="space-y-8">
      {/* 산업용 IoT 센서 30가지 */}
      <div className="bg-white dark:bg-gray-800 p-8 border border-gray-200 dark:border-gray-700 rounded-lg">
        <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-6">🔧 산업용 IoT 센서 완전 가이드</h3>
        <div className="grid lg:grid-cols-2 gap-8">
          <div>
            <h4 className="font-semibold text-blue-800 dark:text-blue-300 mb-4">물리량 측정 센서 (15종)</h4>
            <div className="space-y-3 text-sm">
              {[
                { name: "온도 센서", type: "RTD, 열전대", range: "-200°C ~ 1800°C", accuracy: "±0.1°C", price: "5~50만원" },
                { name: "압력 센서", type: "압전, 스트레인게이지", range: "0 ~ 1000 bar", accuracy: "±0.25%", price: "20~200만원" },
                { name: "진동 센서", type: "가속도계, 자이로", range: "0.1 ~ 10kHz", accuracy: "±2%", price: "50~500만원" },
                { name: "유량 센서", type: "전자기, 초음파", range: "1L/h ~ 10000m³/h", accuracy: "±0.5%", price: "100~1000만원" },
                { name: "위치 센서", type: "엔코더, LVDT", range: "μm ~ 수십m", accuracy: "±0.001mm", price: "10~100만원" },
                { name: "토크 센서", type: "스트레인게이지", range: "1mNm ~ 100kNm", accuracy: "±0.1%", price: "200~2000만원" },
                { name: "힘 센서", type: "로드셀", range: "1g ~ 500톤", accuracy: "±0.02%", price: "50~500만원" }
              ].map((sensor, idx) => (
                <div key={idx} className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded border border-blue-200 dark:border-blue-800">
                  <div className="flex justify-between items-start mb-1">
                    <h5 className="font-medium text-blue-900 dark:text-blue-300">{sensor.name}</h5>
                    <span className="text-xs bg-blue-100 dark:bg-blue-800 text-blue-800 dark:text-blue-300 px-2 py-1 rounded">{sensor.price}</span>
                  </div>
                  <div className="text-xs text-blue-700 dark:text-blue-400 space-y-0.5">
                    <div><span className="font-medium">타입:</span> {sensor.type}</div>
                    <div><span className="font-medium">범위:</span> {sensor.range}</div>
                    <div><span className="font-medium">정확도:</span> {sensor.accuracy}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div>
            <h4 className="font-semibold text-green-800 dark:text-green-300 mb-4">환경/화학 센서 (15종)</h4>
            <div className="space-y-3 text-sm">
              {[
                { name: "가스 센서", type: "전기화학, 적외선", detect: "CO, CO₂, NH₃, H₂S", range: "ppm ~ %", price: "10~100만원" },
                { name: "습도 센서", type: "정전용량형", range: "0~100% RH", accuracy: "±2%", price: "5~30만원" },
                { name: "pH 센서", type: "유리전극", range: "0~14 pH", accuracy: "±0.02", price: "50~300만원" },
                { name: "전도도 센서", type: "2/4전극", range: "0.1~200,000 μS/cm", accuracy: "±1%", price: "30~200만원" },
                { name: "탁도 센서", type: "산란광", range: "0~4000 NTU", accuracy: "±2%", price: "100~500만원" },
                { name: "산소 센서", type: "전기화학", range: "ppb ~ %", accuracy: "±1%", price: "200~1000만원" },
                { name: "미세먼지 센서", type: "레이저 산란", detect: "PM2.5, PM10", accuracy: "±10%", price: "20~200만원" }
              ].map((sensor, idx) => (
                <div key={idx} className="bg-green-50 dark:bg-green-900/20 p-3 rounded border border-green-200 dark:border-green-800">
                  <div className="flex justify-between items-start mb-1">
                    <h5 className="font-medium text-green-900 dark:text-green-300">{sensor.name}</h5>
                    <span className="text-xs bg-green-100 dark:bg-green-800 text-green-800 dark:text-green-300 px-2 py-1 rounded">{sensor.price}</span>
                  </div>
                  <div className="text-xs text-green-700 dark:text-green-400 space-y-0.5">
                    <div><span className="font-medium">타입:</span> {sensor.type}</div>
                    <div><span className="font-medium">범위:</span> {sensor.range || sensor.detect}</div>
                    <div><span className="font-medium">정확도:</span> {sensor.accuracy}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* 시뮬레이터 체험 섹션 */}
      <div className="mt-8 p-6 bg-gradient-to-r from-cyan-50 to-blue-50 dark:from-cyan-900/20 dark:to-blue-900/20 rounded-xl border border-cyan-200 dark:border-cyan-800">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-cyan-900 dark:text-cyan-200 mb-2">
              🎮 센서 데이터 시뮬레이션 체험
            </h3>
            <p className="text-sm text-cyan-700 dark:text-cyan-300">
              다양한 센서의 물리적 특성과 데이터 수집 과정을 실시간으로 시뮬레이션해보세요.
            </p>
          </div>
          <Link
            href="/modules/smart-factory/simulators/physics-simulation?from=/modules/smart-factory/iot-sensor-networks"
            className="inline-flex items-center gap-2 px-4 py-2 bg-cyan-600 hover:bg-cyan-700 text-white rounded-lg transition-colors"
          >
            <span>시뮬레이터 체험</span>
            <span className="text-lg">→</span>
          </Link>
        </div>
      </div>

      {/* 산업 통신 프로토콜 */}
      <div className="bg-white dark:bg-gray-800 p-8 border border-gray-200 dark:border-gray-700 rounded-lg">
        <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-6">🌐 산업 통신 프로토콜 완전 가이드</h3>
        <div className="grid lg:grid-cols-3 gap-6">
          <div className="space-y-4">
            <h4 className="font-semibold text-purple-800 dark:text-purple-300">IT 기반 프로토콜</h4>
            <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded border border-purple-200 dark:border-purple-800">
              <h5 className="font-medium text-purple-900 dark:text-purple-300 mb-2">MQTT</h5>
              <ul className="text-sm text-purple-700 dark:text-purple-400 space-y-1">
                <li>• 경량 IoT 프로토콜</li>
                <li>• Publish/Subscribe 방식</li>
                <li>• QoS 0,1,2 지원</li>
                <li>• 배터리 효율성 우수</li>
              </ul>
              <div className="mt-2 text-xs bg-purple-100 dark:bg-purple-800 text-purple-800 dark:text-purple-300 p-2 rounded">
                <strong>적용:</strong> 센서 데이터 수집, 원격 제어
              </div>
            </div>
            <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded border border-purple-200 dark:border-purple-800">
              <h5 className="font-medium text-purple-900 dark:text-purple-300 mb-2">OPC-UA</h5>
              <ul className="text-sm text-purple-700 dark:text-purple-400 space-y-1">
                <li>• 산업 표준 인터페이스</li>
                <li>• 보안 기능 내장</li>
                <li>• 플랫폼 독립적</li>
                <li>• 의미론적 모델링</li>
              </ul>
              <div className="mt-2 text-xs bg-purple-100 dark:bg-purple-800 text-purple-800 dark:text-purple-300 p-2 rounded">
                <strong>적용:</strong> MES-ERP 연동, 시스템 통합
              </div>
            </div>
          </div>
          
          <div className="space-y-4">
            <h4 className="font-semibold text-orange-800 dark:text-orange-300">필드버스 프로토콜</h4>
            <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded border border-orange-200 dark:border-orange-800">
              <h5 className="font-medium text-orange-900 dark:text-orange-300 mb-2">Modbus RTU/TCP</h5>
              <ul className="text-sm text-orange-700 dark:text-orange-400 space-y-1">
                <li>• 가장 널리 사용</li>
                <li>• 단순하고 안정적</li>
                <li>• Master-Slave 구조</li>
                <li>• 최대 247개 디바이스</li>
              </ul>
              <div className="mt-2 text-xs bg-orange-100 dark:bg-orange-800 text-orange-800 dark:text-orange-300 p-2 rounded">
                <strong>적용:</strong> PLC 통신, 계측기 연결
              </div>
            </div>
            <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded border border-orange-200 dark:border-orange-800">
              <h5 className="font-medium text-orange-900 dark:text-orange-300 mb-2">PROFINET</h5>
              <ul className="text-sm text-orange-700 dark:text-orange-400 space-y-1">
                <li>• 지멘스 주도 표준</li>
                <li>• 실시간 이더넷</li>
                <li>• IRT (등시성) 지원</li>
                <li>• 진단 기능 우수</li>
              </ul>
              <div className="mt-2 text-xs bg-orange-100 dark:bg-orange-800 text-orange-800 dark:text-orange-300 p-2 rounded">
                <strong>적용:</strong> 자동화 시스템, 로봇 제어
              </div>
            </div>
          </div>
          
          <div className="space-y-4">
            <h4 className="font-semibold text-blue-800 dark:text-blue-300">실시간 이더넷</h4>
            <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded border border-blue-200 dark:border-blue-800">
              <h5 className="font-medium text-blue-900 dark:text-blue-300 mb-2">EtherCAT</h5>
              <ul className="text-sm text-blue-700 dark:text-blue-400 space-y-1">
                <li>• 초고속 통신 (≤100ns)</li>
                <li>• 토폴로지 자유도</li>
                <li>• Hot Connect 지원</li>
                <li>• 분산 시계 동기화</li>
              </ul>
              <div className="mt-2 text-xs bg-blue-100 dark:bg-blue-800 text-blue-800 dark:text-blue-300 p-2 rounded">
                <strong>적용:</strong> 고정밀 모션 제어, CNC
              </div>
            </div>
            <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded border border-blue-200 dark:border-blue-800">
              <h5 className="font-medium text-blue-900 dark:text-blue-300 mb-2">TSN (Time-Sensitive Networking)</h5>
              <ul className="text-sm text-blue-700 dark:text-blue-400 space-y-1">
                <li>• IEEE 802.1 표준</li>
                <li>• 결정론적 지연</li>
                <li>• QoS 보장</li>
                <li>• 미래 기술</li>
              </ul>
              <div className="mt-2 text-xs bg-blue-100 dark:bg-blue-800 text-blue-800 dark:text-blue-300 p-2 rounded">
                <strong>적용:</strong> 차세대 스마트팩토리
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 엣지 vs 클라우드 컴퓨팅 */}
      <div className="bg-white dark:bg-gray-800 p-8 border border-gray-200 dark:border-gray-700 rounded-lg">
        <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-6">☁️ 엣지 vs 클라우드 컴퓨팅 비교</h3>
        <div className="grid lg:grid-cols-2 gap-8">
          <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg border border-blue-200 dark:border-blue-800">
            <div className="flex items-center gap-3 mb-4">
              <Cloud className="w-8 h-8 text-blue-600 dark:text-blue-400" />
              <h4 className="font-semibold text-blue-800 dark:text-blue-300 text-lg">엣지 컴퓨팅</h4>
            </div>
            <div className="space-y-4">
              <div>
                <h5 className="font-medium text-blue-900 dark:text-blue-300 mb-2">장점</h5>
                <ul className="text-sm text-blue-700 dark:text-blue-400 space-y-1">
                  <li>• 초저지연: 1-10ms 응답시간</li>
                  <li>• 대역폭 절약: 로컬 처리로 90% 절약</li>
                  <li>• 보안성: 민감 데이터 로컬 보관</li>
                  <li>• 오프라인 동작: 네트워크 장애시 지속 운영</li>
                  <li>• 실시간 제어: 안전 시스템 적합</li>
                </ul>
              </div>
              <div>
                <h5 className="font-medium text-blue-900 dark:text-blue-300 mb-2">단점</h5>
                <ul className="text-sm text-blue-700 dark:text-blue-400 space-y-1">
                  <li>• 높은 초기 비용</li>
                  <li>• 제한적 연산 능력</li>
                  <li>• 현장 관리 복잡성</li>
                  <li>• 확장성 제약</li>
                </ul>
              </div>
              <div className="bg-blue-100 dark:bg-blue-800 p-3 rounded">
                <h6 className="font-medium text-blue-900 dark:text-blue-300 mb-1">적합한 용도</h6>
                <div className="text-xs text-blue-800 dark:text-blue-400">
                  • 실시간 품질 검사<br/>
                  • 안전 시스템 제어<br/>
                  • 예측 유지보수<br/>
                  • 모션 제어
                </div>
              </div>
            </div>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-lg border border-green-200 dark:border-green-800">
            <div className="flex items-center gap-3 mb-4">
              <Server className="w-8 h-8 text-green-600 dark:text-green-400" />
              <h4 className="font-semibold text-green-800 dark:text-green-300 text-lg">클라우드 컴퓨팅</h4>
            </div>
            <div className="space-y-4">
              <div>
                <h5 className="font-medium text-green-900 dark:text-green-300 mb-2">장점</h5>
                <ul className="text-sm text-green-700 dark:text-green-400 space-y-1">
                  <li>• 무제한 확장성: 필요시 즉시 확장</li>
                  <li>• 강력한 AI/ML: GPU 클러스터 활용</li>
                  <li>• 비용 효율성: 사용량 기반 과금</li>
                  <li>• 관리 편의성: 업데이트 자동화</li>
                  <li>• 백업/복구: 자동 데이터 보호</li>
                </ul>
              </div>
              <div>
                <h5 className="font-medium text-green-900 dark:text-green-300 mb-2">단점</h5>
                <ul className="text-sm text-green-700 dark:text-green-400 space-y-1">
                  <li>• 네트워크 의존성</li>
                  <li>• 지연시간: 50-200ms</li>
                  <li>• 데이터 보안 우려</li>
                  <li>• 대역폭 비용</li>
                </ul>
              </div>
              <div className="bg-green-100 dark:bg-green-800 p-3 rounded">
                <h6 className="font-medium text-green-900 dark:text-green-300 mb-1">적합한 용도</h6>
                <div className="text-xs text-green-800 dark:text-green-400">
                  • 빅데이터 분석<br/>
                  • 머신러닝 모델 훈련<br/>
                  • 장기 데이터 저장<br/>
                  • 복잡한 시뮬레이션
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 실습 예제 */}
      <div className="bg-slate-50 dark:bg-slate-800 p-8 border border-slate-200 dark:border-slate-700 rounded-lg">
        <h3 className="text-2xl font-semibold text-slate-800 dark:text-slate-200 mb-6">💻 실습: 아두이노/라즈베리파이 IoT 센서 네트워크</h3>
        
        {/* 하드웨어 구성 - 전체 너비로 상단에 배치 */}
        <div className="mb-8">
          <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-4">하드웨어 구성</h4>
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-slate-200 dark:border-slate-700">
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h5 className="font-medium text-slate-800 dark:text-slate-200 mb-3">필요 부품 리스트</h5>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between items-center p-2 bg-slate-50 dark:bg-slate-700/50 rounded">
                    <span className="text-slate-700 dark:text-slate-300">Arduino Uno R3</span>
                    <span className="font-mono text-slate-600 dark:text-slate-400">25,000원</span>
                  </div>
                  <div className="flex justify-between items-center p-2 bg-slate-50 dark:bg-slate-700/50 rounded">
                    <span className="text-slate-700 dark:text-slate-300">DHT22 온습도 센서</span>
                    <span className="font-mono text-slate-600 dark:text-slate-400">8,000원</span>
                  </div>
                  <div className="flex justify-between items-center p-2 bg-slate-50 dark:bg-slate-700/50 rounded">
                    <span className="text-slate-700 dark:text-slate-300">BMP280 압력 센서</span>
                    <span className="font-mono text-slate-600 dark:text-slate-400">12,000원</span>
                  </div>
                  <div className="flex justify-between items-center p-2 bg-slate-50 dark:bg-slate-700/50 rounded">
                    <span className="text-slate-700 dark:text-slate-300">MPU6050 자이로/가속도</span>
                    <span className="font-mono text-slate-600 dark:text-slate-400">15,000원</span>
                  </div>
                  <div className="flex justify-between items-center p-2 bg-slate-50 dark:bg-slate-700/50 rounded">
                    <span className="text-slate-700 dark:text-slate-300">ESP32 WiFi 모듈</span>
                    <span className="font-mono text-slate-600 dark:text-slate-400">20,000원</span>
                  </div>
                  <div className="flex justify-between items-center p-2 bg-slate-50 dark:bg-slate-700/50 rounded">
                    <span className="text-slate-700 dark:text-slate-300">OLED 디스플레이</span>
                    <span className="font-mono text-slate-600 dark:text-slate-400">10,000원</span>
                  </div>
                  <div className="flex justify-between items-center p-2 bg-slate-50 dark:bg-slate-700/50 rounded">
                    <span className="text-slate-700 dark:text-slate-300">브레드보드/점퍼선</span>
                    <span className="font-mono text-slate-600 dark:text-slate-400">15,000원</span>
                  </div>
                  <div className="flex justify-between items-center p-2 bg-blue-100 dark:bg-blue-900/30 rounded border-t-2 border-blue-400">
                    <span className="font-semibold text-blue-800 dark:text-blue-200">총 비용</span>
                    <span className="font-bold text-blue-700 dark:text-blue-300">105,000원</span>
                  </div>
                </div>
              </div>
              
              <div>
                <h5 className="font-medium text-slate-800 dark:text-slate-200 mb-3">연결 다이어그램</h5>
                <div className="bg-slate-100 dark:bg-slate-700/30 p-4 rounded-lg">
                  <div className="space-y-2 text-xs font-mono text-slate-600 dark:text-slate-400">
                    <div className="flex items-center gap-2">
                      <span className="text-orange-600 dark:text-orange-400">DHT22</span>
                      <span>→</span>
                      <span className="text-blue-600 dark:text-blue-400">D2 (Digital Pin 2)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-green-600 dark:text-green-400">BMP280</span>
                      <span>→</span>
                      <span className="text-blue-600 dark:text-blue-400">I2C (SDA/SCL)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-purple-600 dark:text-purple-400">MPU6050</span>
                      <span>→</span>
                      <span className="text-blue-600 dark:text-blue-400">I2C (0x68)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-red-600 dark:text-red-400">ESP32</span>
                      <span>→</span>
                      <span className="text-blue-600 dark:text-blue-400">Serial (TX/RX)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-yellow-600 dark:text-yellow-400">OLED</span>
                      <span>→</span>
                      <span className="text-blue-600 dark:text-blue-400">I2C (0x3C)</span>
                    </div>
                  </div>
                  <div className="mt-3 p-2 bg-yellow-50 dark:bg-yellow-900/20 rounded border border-yellow-200 dark:border-yellow-700">
                    <p className="text-xs text-yellow-700 dark:text-yellow-300">
                      ⚠️ 주의: I2C 디바이스들은 동일한 SDA/SCL 라인을 공유합니다
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Arduino 코드 예시 - 전체 너비로 하단에 배치 */}
        <div className="mb-8">
          <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-4">Arduino 코드 예시</h4>
          <CodeEditor
            code={`#include <WiFi.h>
#include <PubSubClient.h>
#include <DHT.h>

#define DHT_PIN 2
#define DHT_TYPE DHT22

DHT dht(DHT_PIN, DHT_TYPE);
WiFiClient espClient;
PubSubClient client(espClient);

void setup() {
  Serial.begin(115200);
  dht.begin();
  WiFi.begin("SSID", "PASSWORD");
  
  // WiFi 연결 대기
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("WiFi connected!");
  
  // MQTT 브로커 설정
  client.setServer("mqtt.broker.com", 1883);
}

void loop() {
  // WiFi 연결 확인
  if (!client.connected()) {
    reconnect();
  }
  client.loop();
  
  // 센서 데이터 읽기
  float temp = dht.readTemperature();
  float humid = dht.readHumidity();
  
  // 데이터 유효성 검사
  if (isnan(temp) || isnan(humid)) {
    Serial.println("Failed to read from DHT sensor!");
    return;
  }
  
  // JSON 형태로 데이터 전송
  String payload = "{\"temperature\":" + String(temp) + 
                   ",\"humidity\":" + String(humid) + 
                   ",\"timestamp\":" + String(millis()) + "}";
  
  // MQTT로 발행
  if (client.publish("factory/sensor/dht22", payload.c_str())) {
    Serial.println("Data sent: " + payload);
  } else {
    Serial.println("Failed to send data");
  }
  
  // 10초 간격으로 전송
  delay(10000);
}

void reconnect() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    if (client.connect("ArduinoClient")) {
      Serial.println("connected");
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      delay(5000);
    }
  }
}`}
            language="cpp"
            title="IoT_Sensor_Network.ino"
            maxHeight="500px"
          />
        </div>
        
        <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded border border-blue-200 dark:border-blue-800">
          <h5 className="font-semibold text-blue-800 dark:text-blue-300 mb-2">💡 실습 확장 아이디어</h5>
          <div className="grid md:grid-cols-3 gap-4 text-sm">
            <div>
              <h6 className="font-medium text-blue-900 dark:text-blue-300">Level 1: 기초</h6>
              <ul className="text-blue-700 dark:text-blue-400 space-y-1 mt-1">
                <li>• 센서 데이터 시리얼 출력</li>
                <li>• LED 상태 표시</li>
                <li>• 임계값 알람</li>
              </ul>
            </div>
            <div>
              <h6 className="font-medium text-blue-900 dark:text-blue-300">Level 2: 중급</h6>
              <ul className="text-blue-700 dark:text-blue-400 space-y-1 mt-1">
                <li>• WiFi 연결 및 MQTT 통신</li>
                <li>• 웹 대시보드 연동</li>
                <li>• 데이터 로깅</li>
              </ul>
            </div>
            <div>
              <h6 className="font-medium text-blue-900 dark:text-blue-300">Level 3: 고급</h6>
              <ul className="text-blue-700 dark:text-blue-400 space-y-1 mt-1">
                <li>• 다중 센서 네트워크</li>
                <li>• 엣지 AI 추론</li>
                <li>• OTA 업데이트</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}