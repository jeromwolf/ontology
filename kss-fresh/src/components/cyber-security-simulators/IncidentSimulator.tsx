import React, { useState } from 'react';
import { AlertTriangle, Clock, CheckCircle2, FileText, Activity } from 'lucide-react';
import SimulatorNav from './SimulatorNav';

interface IncidentPhase {
  id: number;
  name: string;
  description: string;
  actions: string[];
  completed: boolean;
}

interface Incident {
  id: string;
  type: string;
  severity: 'Critical' | 'High' | 'Medium';
  description: string;
  detectedAt: string;
}

export default function IncidentSimulator() {
  const [selectedIncident, setSelectedIncident] = useState<Incident | null>(null);
  const [currentPhaseIndex, setCurrentPhaseIndex] = useState<number>(0);
  const [timeline, setTimeline] = useState<string[]>([]);
  const [phases, setPhases] = useState<IncidentPhase[]>([
    {
      id: 1,
      name: 'Preparation (준비)',
      description: 'IR 팀 구성 및 도구 준비',
      actions: [
        'IR 팀 소집',
        'Incident Response Plan 확인',
        '통신 채널 수립',
        '필요 도구 및 시스템 접근 권한 확보',
      ],
      completed: false,
    },
    {
      id: 2,
      name: 'Detection & Analysis (탐지 및 분석)',
      description: '사고 탐지 및 초기 분석',
      actions: [
        'SIEM 로그 분석',
        '영향받은 시스템 식별',
        '공격 벡터 파악',
        '피해 범위 평가',
      ],
      completed: false,
    },
    {
      id: 3,
      name: 'Containment (격리)',
      description: '피해 확산 방지',
      actions: [
        '단기 격리: 네트워크 차단',
        '장기 격리: 영향받은 시스템 분리',
        '백업 시스템 활성화',
        '추가 피해 모니터링',
      ],
      completed: false,
    },
    {
      id: 4,
      name: 'Eradication (제거)',
      description: '위협 요소 완전 제거',
      actions: [
        '악성코드 제거',
        '취약점 패치 적용',
        '계정 권한 재설정',
        '보안 설정 강화',
      ],
      completed: false,
    },
    {
      id: 5,
      name: 'Recovery (복구)',
      description: '시스템 정상화',
      actions: [
        '클린 백업으로 복구',
        '시스템 기능 테스트',
        '모니터링 강화',
        '단계적 서비스 재개',
      ],
      completed: false,
    },
    {
      id: 6,
      name: 'Post-Incident (사후 분석)',
      description: '보고서 작성 및 개선',
      actions: [
        '상세 보고서 작성',
        'Lessons Learned 회의',
        '프로세스 개선안 도출',
        '보안 정책 업데이트',
      ],
      completed: false,
    },
  ]);

  const incidents: Incident[] = [
    {
      id: '1',
      type: '랜섬웨어 공격',
      severity: 'Critical',
      description: 'WannaCry 랜섬웨어가 내부 파일 서버에서 탐지됨',
      detectedAt: new Date().toLocaleString(),
    },
    {
      id: '2',
      type: 'DDoS 공격',
      severity: 'High',
      description: '웹 서버에 대규모 트래픽 유입 (500,000+ req/sec)',
      detectedAt: new Date().toLocaleString(),
    },
    {
      id: '3',
      type: '데이터 유출',
      severity: 'Critical',
      description: '고객 데이터베이스 비정상 접근 탐지',
      detectedAt: new Date().toLocaleString(),
    },
    {
      id: '4',
      type: 'APT 공격',
      severity: 'High',
      description: '내부 네트워크에서 의심스러운 Lateral Movement 탐지',
      detectedAt: new Date().toLocaleString(),
    },
  ];

  const startIncident = (incident: Incident) => {
    setSelectedIncident(incident);
    setCurrentPhaseIndex(0);
    setTimeline([`${new Date().toLocaleTimeString()} - 사고 발생: ${incident.type}`]);
    setPhases(phases.map((p) => ({ ...p, completed: false })));
  };

  const completePhase = () => {
    const newPhases = [...phases];
    newPhases[currentPhaseIndex].completed = true;
    setPhases(newPhases);

    const newTimeline = [
      ...timeline,
      `${new Date().toLocaleTimeString()} - ${phases[currentPhaseIndex].name} 완료`,
    ];
    setTimeline(newTimeline);

    if (currentPhaseIndex < phases.length - 1) {
      setCurrentPhaseIndex(currentPhaseIndex + 1);
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'Critical':
        return 'border-red-500 bg-red-900/30 text-red-200';
      case 'High':
        return 'border-orange-500 bg-orange-900/30 text-orange-200';
      case 'Medium':
        return 'border-yellow-500 bg-yellow-900/30 text-yellow-200';
      default:
        return 'border-gray-500 bg-gray-900/30 text-gray-200';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-slate-900 to-gray-900 text-white">
      <div className="max-w-7xl mx-auto px-4 py-8">
        <SimulatorNav />
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-4 flex items-center gap-3">
            <Activity className="w-10 h-10 text-orange-500" />
            Incident Response Simulator
          </h1>
          <p className="text-xl text-gray-300">보안 사고 대응 절차 시뮬레이션 (NIST 6단계)</p>
        </div>

        {!selectedIncident ? (
          /* Incident Selection */
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-gray-800 rounded-xl p-6">
              <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
                <AlertTriangle className="w-7 h-7 text-yellow-500" />
                발생 가능한 사고 유형
              </h2>
              <div className="space-y-4">
                {incidents.map((incident) => (
                  <div
                    key={incident.id}
                    className={`border-2 rounded-lg p-4 cursor-pointer hover:shadow-lg transition-all ${getSeverityColor(
                      incident.severity
                    )}`}
                    onClick={() => startIncident(incident)}
                  >
                    <div className="flex justify-between items-start mb-2">
                      <h3 className="font-bold text-lg">{incident.type}</h3>
                      <span className="text-xs px-2 py-1 rounded bg-black/30">
                        {incident.severity}
                      </span>
                    </div>
                    <p className="text-sm mb-2">{incident.description}</p>
                    <div className="text-xs text-gray-400">
                      <Clock className="w-3 h-3 inline mr-1" />
                      {incident.detectedAt}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-gray-800 rounded-xl p-6">
              <h2 className="text-2xl font-bold mb-6">💡 NIST IR Framework</h2>
              <div className="space-y-3">
                {phases.map((phase, idx) => (
                  <div key={phase.id} className="flex items-start gap-3">
                    <div className="flex-shrink-0 w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold text-sm">
                      {idx + 1}
                    </div>
                    <div>
                      <h3 className="font-bold text-blue-400">{phase.name}</h3>
                      <p className="text-sm text-gray-400">{phase.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : (
          /* Incident Response Flow */
          <div className="grid md:grid-cols-3 gap-6">
            {/* Current Incident Info */}
            <div className="md:col-span-3 bg-gray-800 rounded-xl p-6">
              <div className="flex justify-between items-start">
                <div>
                  <h2 className="text-2xl font-bold mb-2 flex items-center gap-2">
                    <AlertTriangle className="w-7 h-7 text-red-500" />
                    현재 사고: {selectedIncident.type}
                  </h2>
                  <p className="text-gray-300">{selectedIncident.description}</p>
                </div>
                <button
                  onClick={() => {
                    setSelectedIncident(null);
                    setTimeline([]);
                  }}
                  className="bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded text-sm"
                >
                  종료
                </button>
              </div>
            </div>

            {/* IR Phases Progress */}
            <div className="md:col-span-2 bg-gray-800 rounded-xl p-6">
              <h2 className="text-xl font-bold mb-6">대응 단계</h2>
              <div className="space-y-4">
                {phases.map((phase, idx) => (
                  <div
                    key={phase.id}
                    className={`border-2 rounded-lg p-4 ${
                      phase.completed
                        ? 'border-green-500 bg-green-900/20'
                        : idx === currentPhaseIndex
                        ? 'border-blue-500 bg-blue-900/20'
                        : 'border-gray-700 bg-gray-900/20'
                    }`}
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex items-center gap-3">
                        <div
                          className={`w-8 h-8 rounded-full flex items-center justify-center font-bold ${
                            phase.completed
                              ? 'bg-green-600'
                              : idx === currentPhaseIndex
                              ? 'bg-blue-600'
                              : 'bg-gray-700'
                          }`}
                        >
                          {phase.completed ? <CheckCircle2 className="w-5 h-5" /> : idx + 1}
                        </div>
                        <div>
                          <h3 className="font-bold">{phase.name}</h3>
                          <p className="text-xs text-gray-400">{phase.description}</p>
                        </div>
                      </div>
                    </div>

                    {idx === currentPhaseIndex && !phase.completed && (
                      <div className="mt-3 pt-3 border-t border-gray-700">
                        <h4 className="font-semibold mb-2 text-sm">수행 작업:</h4>
                        <ul className="space-y-1 mb-3">
                          {phase.actions.map((action, i) => (
                            <li key={i} className="text-sm text-gray-300">
                              • {action}
                            </li>
                          ))}
                        </ul>
                        <button
                          onClick={completePhase}
                          className="w-full bg-blue-600 hover:bg-blue-700 py-2 rounded font-semibold"
                        >
                          이 단계 완료
                        </button>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Timeline */}
            <div className="bg-gray-800 rounded-xl p-6">
              <h2 className="text-xl font-bold mb-6 flex items-center gap-2">
                <FileText className="w-6 h-6" />
                타임라인
              </h2>
              <div className="space-y-3 max-h-[600px] overflow-y-auto">
                {timeline.map((entry, idx) => (
                  <div key={idx} className="bg-gray-900 rounded p-3 text-sm border-l-4 border-blue-500">
                    {entry}
                  </div>
                ))}
              </div>

              {phases.every((p) => p.completed) && (
                <div className="mt-6 bg-green-900/30 border-2 border-green-500 rounded-lg p-4">
                  <h3 className="font-bold text-green-400 mb-2 flex items-center gap-2">
                    <CheckCircle2 className="w-5 h-5" />
                    사고 대응 완료!
                  </h3>
                  <p className="text-sm text-gray-300">
                    모든 IR 단계를 성공적으로 완료했습니다. 사후 분석 보고서를 작성하세요.
                  </p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
