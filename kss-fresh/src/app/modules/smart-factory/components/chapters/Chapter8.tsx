'use client';

import {
  Monitor, Building, Cog, Cpu, TestTube, AlertTriangle, TrendingUp, Settings, Code
} from 'lucide-react';
import CodeEditor from '../CodeEditor';
import Link from 'next/link';
import References from '@/components/common/References';

export default function Chapter8() {
  return (
    <div className="space-y-8">
      <div className="grid lg:grid-cols-2 gap-8">
        <div className="bg-white dark:bg-gray-800 p-6 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <Building className="w-6 h-6 text-slate-600" />
            디지털 트윈 5단계 구축 프로세스
          </h3>
          <div className="space-y-4">
            <div className="flex items-start gap-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-700">
              <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-white font-bold text-sm">1</div>
              <div>
                <h4 className="font-semibold text-blue-800 dark:text-blue-300 mb-2">설계 (Design)</h4>
                <p className="text-sm text-blue-700 dark:text-blue-400 mb-2">물리적 자산의 디지털 모델 설계</p>
                <ul className="text-xs text-blue-600 dark:text-blue-500 space-y-1">
                  <li>• CAD 도면 기반 3D 모델링</li>
                  <li>• 기하학적 형상과 물리적 특성 정의</li>
                  <li>• 센서 위치 및 데이터 포인트 설계</li>
                </ul>
              </div>
            </div>
            
            <div className="flex items-start gap-4 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-700">
              <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center text-white font-bold text-sm">2</div>
              <div>
                <h4 className="font-semibold text-green-800 dark:text-green-300 mb-2">모델링 (Modeling)</h4>
                <p className="text-sm text-green-700 dark:text-green-400 mb-2">물리 법칙과 동작 원리 구현</p>
                <ul className="text-xs text-green-600 dark:text-green-500 space-y-1">
                  <li>• 물리 엔진 통합 (중력, 마찰, 충돌)</li>
                  <li>• 열역학, 유체역학 시뮬레이션</li>
                  <li>• 기계 동작과 제어 로직 모델링</li>
                </ul>
              </div>
            </div>
            
            <div className="flex items-start gap-4 p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-700">
              <div className="w-8 h-8 bg-purple-500 rounded-full flex items-center justify-center text-white font-bold text-sm">3</div>
              <div>
                <h4 className="font-semibold text-purple-800 dark:text-purple-300 mb-2">연결 (Connect)</h4>
                <p className="text-sm text-purple-700 dark:text-purple-400 mb-2">실제 장비와 디지털 모델 동기화</p>
                <ul className="text-xs text-purple-600 dark:text-purple-500 space-y-1">
                  <li>• IoT 센서 데이터 실시간 수집</li>
                  <li>• MQTT, OPC-UA 통신 프로토콜</li>
                  <li>• 양방향 데이터 교환 시스템</li>
                </ul>
              </div>
            </div>
            
            <div className="flex items-start gap-4 p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg border border-orange-200 dark:border-orange-700">
              <div className="w-8 h-8 bg-orange-500 rounded-full flex items-center justify-center text-white font-bold text-sm">4</div>
              <div>
                <h4 className="font-semibold text-orange-800 dark:text-orange-300 mb-2">분석 (Analyze)</h4>
                <p className="text-sm text-orange-700 dark:text-orange-400 mb-2">데이터 분석과 패턴 인식</p>
                <ul className="text-xs text-orange-600 dark:text-orange-500 space-y-1">
                  <li>• 성능 지표 실시간 계산</li>
                  <li>• 이상 패턴 자동 감지</li>
                  <li>• 예측 분석과 트렌드 파악</li>
                </ul>
              </div>
            </div>
            
            <div className="flex items-start gap-4 p-4 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-700">
              <div className="w-8 h-8 bg-red-500 rounded-full flex items-center justify-center text-white font-bold text-sm">5</div>
              <div>
                <h4 className="font-semibold text-red-800 dark:text-red-300 mb-2">최적화 (Optimize)</h4>
                <p className="text-sm text-red-700 dark:text-red-400 mb-2">시뮬레이션 기반 운영 개선</p>
                <ul className="text-xs text-red-600 dark:text-red-500 space-y-1">
                  <li>• What-if 시나리오 분석</li>
                  <li>• 최적 운영 조건 도출</li>
                  <li>• 실제 장비에 피드백 적용</li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        {/* 시뮬레이터 체험 섹션 */}
        <div className="mt-8 p-6 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl border border-blue-200 dark:border-blue-800">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-blue-900 dark:text-blue-200 mb-2">
                🎮 디지털 트윈 팩토리 시뮬레이터
              </h3>
              <p className="text-sm text-blue-700 dark:text-blue-300">
                실제 공장의 디지털 복제본을 만들고 실시간 시뮬레이션을 체험해보세요.
              </p>
            </div>
            <Link
              href="/modules/smart-factory/simulators/digital-twin-factory?from=/modules/smart-factory/digital-twin-simulation"
              className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
            >
              <span>시뮬레이터 체험</span>
              <span className="text-lg">→</span>
            </Link>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 p-6 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <Cog className="w-6 h-6 text-slate-600" />
            3D 모델링 도구 비교
          </h3>
          <div className="space-y-6">
            <div className="p-4 bg-slate-50 dark:bg-slate-700/50 rounded-lg border">
              <div className="flex items-center justify-between mb-3">
                <h4 className="font-semibold text-slate-800 dark:text-slate-200">AutoCAD + Inventor</h4>
                <span className="text-xs bg-blue-100 dark:bg-blue-800 text-blue-700 dark:text-blue-300 px-2 py-1 rounded">Autodesk</span>
              </div>
              <div className="space-y-2 text-sm text-slate-600 dark:text-slate-400">
                <p><strong>장점:</strong> 정밀한 CAD 설계, 강력한 어셈블리 기능</p>
                <p><strong>용도:</strong> 기계 부품, 정밀 가공 장비</p>
                <p><strong>포맷:</strong> .dwg, .ipt, .iam → FBX/OBJ 변환</p>
              </div>
            </div>
            
            <div className="p-4 bg-slate-50 dark:bg-slate-700/50 rounded-lg border">
              <div className="flex items-center justify-between mb-3">
                <h4 className="font-semibold text-slate-800 dark:text-slate-200">SolidWorks</h4>
                <span className="text-xs bg-green-100 dark:bg-green-800 text-green-700 dark:text-green-300 px-2 py-1 rounded">Dassault</span>
              </div>
              <div className="space-y-2 text-sm text-slate-600 dark:text-slate-400">
                <p><strong>장점:</strong> 직관적 UI, 시뮬레이션 통합</p>
                <p><strong>용도:</strong> 산업 장비, 로봇 시스템</p>
                <p><strong>포맷:</strong> .sldprt, .sldasm → Unity/Unreal 직접 연동</p>
              </div>
            </div>
            
            <div className="p-4 bg-slate-50 dark:bg-slate-700/50 rounded-lg border">
              <div className="flex items-center justify-between mb-3">
                <h4 className="font-semibold text-slate-800 dark:text-slate-200">Blender</h4>
                <span className="text-xs bg-purple-100 dark:bg-purple-800 text-purple-700 dark:text-purple-300 px-2 py-1 rounded">오픈소스</span>
              </div>
              <div className="space-y-2 text-sm text-slate-600 dark:text-slate-400">
                <p><strong>장점:</strong> 무료, 유연한 모델링, Python 스크립팅</p>
                <p><strong>용도:</strong> 시각화, 애니메이션, 프로토타이핑</p>
                <p><strong>포맷:</strong> .blend → 모든 게임엔진 지원</p>
              </div>
            </div>
            
            <div className="p-4 bg-slate-50 dark:bg-slate-700/50 rounded-lg border">
              <div className="flex items-center justify-between mb-3">
                <h4 className="font-semibold text-slate-800 dark:text-slate-200">CATIA V6</h4>
                <span className="text-xs bg-yellow-100 dark:bg-yellow-800 text-yellow-700 dark:text-yellow-300 px-2 py-1 rounded">Enterprise</span>
              </div>
              <div className="space-y-2 text-sm text-slate-600 dark:text-slate-400">
                <p><strong>장점:</strong> 대규모 어셈블리, PLM 통합</p>
                <p><strong>용도:</strong> 자동차, 항공 산업</p>
                <p><strong>포맷:</strong> .CATProduct → 3DEXPERIENCE 플랫폼</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-gradient-to-r from-cyan-50 to-teal-50 dark:from-cyan-900/20 dark:to-teal-900/20 p-8 rounded-xl border border-cyan-200 dark:border-cyan-800">
        <h3 className="text-2xl font-bold text-cyan-900 dark:text-cyan-200 mb-6 flex items-center gap-3">
          <Cpu className="w-8 h-8" />
          Unity 3D 기반 실시간 시각화 시스템
        </h3>
        <div className="grid lg:grid-cols-2 gap-8">
          <div className="space-y-4">
            <h4 className="text-xl font-semibold text-cyan-800 dark:text-cyan-200 mb-4">핵심 기능</h4>
            <div className="space-y-3">
              <div className="p-4 bg-white dark:bg-cyan-800/30 rounded-lg border border-cyan-200 dark:border-cyan-600">
                <h5 className="font-semibold text-cyan-700 dark:text-cyan-300 mb-2">실시간 3D 렌더링</h5>
                <ul className="text-sm text-cyan-600 dark:text-cyan-400 space-y-1">
                  <li>• 60FPS 실시간 시뮬레이션</li>
                  <li>• PBR 머티리얼 기반 사실적 렌더링</li>
                  <li>• 동적 조명과 그림자 효과</li>
                </ul>
              </div>
              
              <div className="p-4 bg-white dark:bg-cyan-800/30 rounded-lg border border-cyan-200 dark:border-cyan-600">
                <h5 className="font-semibold text-cyan-700 dark:text-cyan-300 mb-2">물리 시뮬레이션</h5>
                <ul className="text-sm text-cyan-600 dark:text-cyan-400 space-y-1">
                  <li>• Rigidbody 기반 물리 계산</li>
                  <li>• 충돌 감지와 응답 시스템</li>
                  <li>• 관절(Joint) 시스템으로 기계 동작</li>
                </ul>
              </div>
              
              <div className="p-4 bg-white dark:bg-cyan-800/30 rounded-lg border border-cyan-200 dark:border-cyan-600">
                <h5 className="font-semibold text-cyan-700 dark:text-cyan-300 mb-2">데이터 시각화</h5>
                <ul className="text-sm text-cyan-600 dark:text-cyan-400 space-y-1">
                  <li>• 실시간 차트와 그래프</li>
                  <li>• 히트맵 기반 온도 분포</li>
                  <li>• 3D 공간의 데이터 오버레이</li>
                </ul>
              </div>
            </div>
          </div>
          
          <div className="space-y-4">
            <h4 className="text-xl font-semibold text-cyan-800 dark:text-cyan-200 mb-4">사용자 인터랙션</h4>
            <div className="space-y-3">
              <div className="p-3 bg-teal-50 dark:bg-teal-900/20 border-l-4 border-teal-400 rounded">
                <h5 className="font-semibold text-teal-700 dark:text-teal-300 text-sm">VR/AR 지원</h5>
                <p className="text-xs text-teal-600 dark:text-teal-400 mt-1">Oculus, HTC Vive를 통한 몰입형 체험</p>
              </div>
              
              <div className="p-3 bg-teal-50 dark:bg-teal-900/20 border-l-4 border-teal-400 rounded">
                <h5 className="font-semibold text-teal-700 dark:text-teal-300 text-sm">터치 인터랙션</h5>
                <p className="text-xs text-teal-600 dark:text-teal-400 mt-1">장비 클릭으로 상세 정보 표시</p>
              </div>
              
              <div className="p-3 bg-teal-50 dark:bg-teal-900/20 border-l-4 border-teal-400 rounded">
                <h5 className="font-semibold text-teal-700 dark:text-teal-300 text-sm">카메라 제어</h5>
                <p className="text-xs text-teal-600 dark:text-teal-400 mt-1">자유 시점, 고정 뷰, 추적 모드</p>
              </div>
              
              <div className="p-3 bg-teal-50 dark:bg-teal-900/20 border-l-4 border-teal-400 rounded">
                <h5 className="font-semibold text-teal-700 dark:text-teal-300 text-sm">시간 제어</h5>
                <p className="text-xs text-teal-600 dark:text-teal-400 mt-1">재생, 일시정지, 배속 조절</p>
              </div>
              
              <div className="p-3 bg-teal-50 dark:bg-teal-900/20 border-l-4 border-teal-400 rounded">
                <h5 className="font-semibold text-teal-700 dark:text-teal-300 text-sm">시나리오 분석</h5>
                <p className="text-xs text-teal-600 dark:text-teal-400 mt-1">다양한 조건별 결과 비교</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-800 p-8 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">
        <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <TestTube className="w-8 h-8 text-amber-600" />
          What-if 시나리오 분석
        </h3>
        <div className="grid md:grid-cols-3 gap-6">
          <div className="p-6 bg-gradient-to-br from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 rounded-lg border border-red-200 dark:border-red-700">
            <h4 className="font-bold text-red-800 dark:text-red-200 mb-4 flex items-center gap-2">
              <AlertTriangle className="w-5 h-5" />
              장비 고장 시나리오
            </h4>
            <div className="space-y-3 text-sm">
              <div className="bg-white dark:bg-red-800/30 p-3 rounded border border-red-200 dark:border-red-600">
                <h5 className="font-semibold text-red-700 dark:text-red-300 mb-1">컨베이어 벨트 정지</h5>
                <ul className="text-xs text-red-600 dark:text-red-400 space-y-1">
                  <li>• 생산량 80% 감소</li>
                  <li>• 재고 누적 발생</li>
                  <li>• 대체 라인 가동 필요</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-red-800/30 p-3 rounded border border-red-200 dark:border-red-600">
                <h5 className="font-semibold text-red-700 dark:text-red-300 mb-1">로봇 암 오류</h5>
                <ul className="text-xs text-red-600 dark:text-red-400 space-y-1">
                  <li>• 조립 공정 중단</li>
                  <li>• 인력 투입 필요</li>
                  <li>• 품질 편차 증가</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-red-800/30 p-3 rounded border border-red-200 dark:border-red-600">
                <h5 className="font-semibold text-red-700 dark:text-red-300 mb-1">센서 통신 장애</h5>
                <ul className="text-xs text-red-600 dark:text-red-400 space-y-1">
                  <li>• 실시간 모니터링 중단</li>
                  <li>• 수동 점검 전환</li>
                  <li>• 예측 기능 상실</li>
                </ul>
              </div>
            </div>
          </div>
          
          <div className="p-6 bg-gradient-to-br from-yellow-50 to-amber-50 dark:from-yellow-900/20 dark:to-amber-900/20 rounded-lg border border-yellow-200 dark:border-yellow-700">
            <h4 className="font-bold text-yellow-800 dark:text-yellow-200 mb-4 flex items-center gap-2">
              <TrendingUp className="w-5 h-5" />
              수요 변동 시나리오
            </h4>
            <div className="space-y-3 text-sm">
              <div className="bg-white dark:bg-yellow-800/30 p-3 rounded border border-yellow-200 dark:border-yellow-600">
                <h5 className="font-semibold text-yellow-700 dark:text-yellow-300 mb-1">급격한 수요 증가</h5>
                <ul className="text-xs text-yellow-600 dark:text-yellow-400 space-y-1">
                  <li>• 생산량 150% 증대</li>
                  <li>• 야간 시프트 가동</li>
                  <li>• 원자재 부족 위험</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-yellow-800/30 p-3 rounded border border-yellow-200 dark:border-yellow-600">
                <h5 className="font-semibold text-yellow-700 dark:text-yellow-300 mb-1">계절성 수요 변화</h5>
                <ul className="text-xs text-yellow-600 dark:text-yellow-400 space-y-1">
                  <li>• 제품 믹스 변경</li>
                  <li>• 라인 전환 시간</li>
                  <li>• 인력 재배치</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-yellow-800/30 p-3 rounded border border-yellow-200 dark:border-yellow-600">
                <h5 className="font-semibold text-yellow-700 dark:text-yellow-300 mb-1">맞춤형 주문 증가</h5>
                <ul className="text-xs text-yellow-600 dark:text-yellow-400 space-y-1">
                  <li>• 로트 사이즈 감소</li>
                  <li>• 설정 변경 빈도 증가</li>
                  <li>• 유연성 요구 증대</li>
                </ul>
              </div>
            </div>
          </div>
          
          <div className="p-6 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg border border-green-200 dark:border-green-700">
            <h4 className="font-bold text-green-800 dark:text-green-200 mb-4 flex items-center gap-2">
              <Settings className="w-5 h-5" />
              레이아웃 변경 시나리오
            </h4>
            <div className="space-y-3 text-sm">
              <div className="bg-white dark:bg-green-800/30 p-3 rounded border border-green-200 dark:border-green-600">
                <h5 className="font-semibold text-green-700 dark:text-green-300 mb-1">셀 방식 도입</h5>
                <ul className="text-xs text-green-600 dark:text-green-400 space-y-1">
                  <li>• 처리 시간 25% 단축</li>
                  <li>• 작업자 동선 최적화</li>
                  <li>• 재공재고 50% 감소</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-green-800/30 p-3 rounded border border-green-200 dark:border-green-600">
                <h5 className="font-semibold text-green-700 dark:text-green-300 mb-1">자동창고 확장</h5>
                <ul className="text-xs text-green-600 dark:text-green-400 space-y-1">
                  <li>• 보관 효율 300% 증대</li>
                  <li>• 피킹 시간 70% 단축</li>
                  <li>• 재고 정확도 99.9%</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-green-800/30 p-3 rounded border border-green-200 dark:border-green-600">
                <h5 className="font-semibold text-green-700 dark:text-green-300 mb-1">AGV 도입</h5>
                <ul className="text-xs text-green-600 dark:text-green-400 space-y-1">
                  <li>• 운반 비용 40% 절감</li>
                  <li>• 24시간 무인 운영</li>
                  <li>• 동선 충돌 방지</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-green-50 dark:bg-green-900/20 p-8 rounded-xl border border-green-200 dark:border-green-800">
        <h3 className="text-2xl font-bold text-green-900 dark:text-green-200 mb-6 flex items-center gap-3">
          <Code className="w-8 h-8" />
          실습: Unity로 생산라인 디지털 트윈 제작
        </h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-green-800/30 p-6 rounded-lg">
            <h4 className="font-bold text-green-800 dark:text-green-200 mb-4">1단계: Unity 프로젝트 설정</h4>
            <CodeEditor 
              code={`// Unity C# 스크립트
using UnityEngine;
using System.Collections;

public class ConveyorBelt : MonoBehaviour
{
    [Header("Conveyor Settings")]
    public float speed = 1.0f;
    public Vector3 direction = Vector3.forward;
    
    private Renderer beltRenderer;
    
    void Start()
    {
        beltRenderer = GetComponent<Renderer>();
    }
    
    void Update()
    {
        // 벨트 텍스처 애니메이션
        float offset = Time.time * speed * 0.1f;
        beltRenderer.material.mainTextureOffset = new Vector2(0, offset);
    }
    
    void OnTriggerStay(Collider other)
    {
        // 벨트 위 물체 이동
        if (other.attachedRigidbody != null)
        {
            Vector3 force = direction.normalized * speed;
            other.attachedRigidbody.AddForce(force, ForceMode.Force);
        }
    }
}`}
              language="csharp"
              title="컨베이어 벨트 제어"
              filename="ConveyorBelt.cs"
              maxHeight="400px"
            />
          </div>
          
          <div className="bg-white dark:bg-green-800/30 p-6 rounded-lg">
            <h4 className="font-bold text-green-800 dark:text-green-200 mb-4">2단계: IoT 데이터 연동</h4>
            <CodeEditor 
              code={`// IoT 센서 데이터 수신
using UnityEngine;
using System.Net.Sockets;
using System.Text;
using Newtonsoft.Json;

public class IoTDataReceiver : MonoBehaviour
{
    [Header("MQTT Settings")]
    public string brokerAddress = "192.168.1.100";
    public int brokerPort = 1883;
    public string topic = "factory/sensors";
    
    private UdpClient udpClient;
    
    void Start()
    {
        ConnectToMQTT();
    }
    
    void ConnectToMQTT()
    {
        udpClient = new UdpClient(brokerPort);
        StartCoroutine(ListenForData());
    }
    
    IEnumerator ListenForData()
    {
        while (true)
        {
            if (udpClient.Available > 0)
            {
                IPEndPoint endpoint = new IPEndPoint(IPAddress.Any, brokerPort);
                byte[] data = udpClient.Receive(ref endpoint);
                string json = Encoding.UTF8.GetString(data);
                
                SensorData sensorData = JsonConvert.DeserializeObject<SensorData>(json);
                UpdateDigitalTwin(sensorData);
            }
            yield return new WaitForSeconds(0.1f);
        }
    }
    
    void UpdateDigitalTwin(SensorData data)
    {
        // 실제 센서 값으로 디지털 트윈 업데이트
        GameObject machine = GameObject.FindWithTag("Machine_" + data.machineId);
        if (machine != null)
        {
            MachineController controller = machine.GetComponent<MachineController>();
            controller.UpdateTemperature(data.temperature);
            controller.UpdateVibration(data.vibration);
            controller.UpdateSpeed(data.rotationSpeed);
        }
    }
}

[System.Serializable]
public class SensorData
{
    public string machineId;
    public float temperature;
    public float vibration;
    public float rotationSpeed;
    public string timestamp;
}`}
              language="csharp"
              title="IoT 센서 데이터 연동"
              filename="IoTDataReceiver.cs"
              maxHeight="400px"
            />
          </div>
        </div>
      </div>

      <References
        sections={[
          {
            title: '📚 공식 표준 & 문서',
            icon: 'web' as const,
            color: 'border-emerald-500',
            items: [
              {
                title: 'ISO 23247 - Digital Twin Framework for Manufacturing',
                url: 'https://www.iso.org/standard/75066.html',
                description: '제조업을 위한 디지털 트윈 국제 표준 프레임워크 - 정의, 아키텍처, 구현 가이드라인 제공'
              },
              {
                title: 'Siemens Digital Twin Whitepaper',
                url: 'https://www.siemens.com/global/en/products/automation/topic-areas/digital-twin.html',
                description: '산업 자동화 선도기업의 디지털 트윈 구축 방법론 및 실제 적용 사례'
              },
              {
                title: 'Unity Industry Solutions',
                url: 'https://unity.com/solutions/industry',
                description: 'Unity 3D 기반 제조업 디지털 트윈 구축을 위한 공식 가이드 및 템플릿'
              },
              {
                title: 'Digital Twin Consortium',
                url: 'https://www.digitaltwinconsortium.org/',
                description: '글로벌 디지털 트윈 표준화 기구 - 백서, 유스케이스, 베스트 프랙티스 제공'
              },
              {
                title: 'AWS IoT TwinMaker Documentation',
                url: 'https://docs.aws.amazon.com/iot-twinmaker/',
                description: 'AWS 클라우드 기반 디지털 트윈 구축 서비스 공식 문서 및 튜토리얼'
              }
            ]
          },
          {
            title: '🔬 핵심 논문 & 연구',
            icon: 'research' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'GE Digital Twin: Predix Platform Architecture (2018)',
                url: 'https://www.ge.com/digital/applications/digital-twin',
                description: 'GE의 산업용 IoT 플랫폼 Predix 기반 디지털 트윈 아키텍처 연구'
              },
              {
                title: 'Digital Twin-driven Smart Manufacturing (IEEE Access, 2020)',
                url: 'https://ieeexplore.ieee.org/document/9184790',
                description: '디지털 트윈 기반 스마트 제조 프로세스 최적화 연구 - 실시간 시뮬레이션 검증'
              },
              {
                title: 'Physics-based Digital Twin for Predictive Maintenance (Nature, 2021)',
                url: 'https://www.nature.com/articles/s41598-021-89933-1',
                description: '물리 기반 디지털 트윈을 활용한 예측 정비 시스템 - NASA 공동 연구'
              },
              {
                title: '3DEXPERIENCE Platform - Dassault Systèmes Research',
                url: 'https://www.3ds.com/3dexperience',
                description: 'PLM 통합 디지털 트윈 플랫폼 연구 - CATIA, SIMULIA 연계 기술'
              }
            ]
          },
          {
            title: '🛠️ 실전 도구 & 플랫폼',
            icon: 'tools' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'ANSYS Twin Builder',
                url: 'https://www.ansys.com/products/digital-twin/ansys-twin-builder',
                description: '멀티피직스 시뮬레이션 기반 디지털 트윈 구축 도구 - ROM 생성 및 실시간 연동'
              },
              {
                title: 'Unity Reflect & PiXYZ',
                url: 'https://unity.com/products/unity-reflect',
                description: 'CAD 데이터를 Unity 3D 환경으로 변환하는 실시간 시각화 솔루션'
              },
              {
                title: 'Azure Digital Twins',
                url: 'https://azure.microsoft.com/en-us/products/digital-twins/',
                description: 'Microsoft 클라우드 기반 디지털 트윈 플랫폼 - DTDL 모델링 언어 지원'
              },
              {
                title: 'Eclipse Ditto - Open Source Digital Twin',
                url: 'https://www.eclipse.org/ditto/',
                description: 'IoT 디바이스를 위한 오픈소스 디지털 트윈 프레임워크 - 실시간 상태 동기화'
              },
              {
                title: 'Matterport for Digital Twin Visualization',
                url: 'https://matterport.com/industries/manufacturing',
                description: '3D 스캔 기반 공장 디지털 트윈 시각화 플랫폼 - VR/AR 지원'
              }
            ]
          }
        ]}
      />
    </div>
  );
}