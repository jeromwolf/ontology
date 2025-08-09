'use client';

import { useState, useEffect, useCallback } from 'react';
import { Shield, AlertTriangle, CheckCircle, XCircle, Lock, Unlock, Eye, Database } from 'lucide-react';

interface SecurityCheck {
  id: string;
  name: string;
  category: string;
  status: 'pass' | 'fail' | 'warning' | 'checking';
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
}

interface ModelInfo {
  name: string;
  type: string;
  size: string;
  parameters: string;
  lastModified: string;
}

const initialChecks: SecurityCheck[] = [
  {
    id: 'watermark',
    name: '모델 워터마킹',
    category: '소유권 보호',
    status: 'checking',
    severity: 'medium',
    description: '모델에 소유권 워터마크가 삽입되어 있는지 확인'
  },
  {
    id: 'backdoor',
    name: '백도어 탐지',
    category: '무결성',
    status: 'checking',
    severity: 'critical',
    description: '숨겨진 백도어 트리거 존재 여부 검사'
  },
  {
    id: 'robustness',
    name: '적대적 견고성',
    category: '견고성',
    status: 'checking',
    severity: 'high',
    description: '적대적 공격에 대한 모델의 저항성 평가'
  },
  {
    id: 'extraction',
    name: '모델 추출 방어',
    category: '기밀성',
    status: 'checking',
    severity: 'high',
    description: 'API를 통한 모델 추출 공격 방어 메커니즘'
  },
  {
    id: 'privacy',
    name: '프라이버시 유출',
    category: '프라이버시',
    status: 'checking',
    severity: 'high',
    description: '학습 데이터 정보 유출 가능성 검사'
  },
  {
    id: 'encryption',
    name: '모델 암호화',
    category: '기밀성',
    status: 'checking',
    severity: 'medium',
    description: '배포된 모델의 암호화 상태 확인'
  },
  {
    id: 'access',
    name: '접근 제어',
    category: '접근 관리',
    status: 'checking',
    severity: 'medium',
    description: 'API 접근 제어 및 인증 메커니즘'
  },
  {
    id: 'monitoring',
    name: '이상 탐지',
    category: '모니터링',
    status: 'checking',
    severity: 'low',
    description: '실시간 이상 패턴 탐지 시스템'
  }
];

export default function ModelSecurityAnalyzer() {
  const [modelInfo] = useState<ModelInfo>({
    name: 'ResNet-50-Secure',
    type: 'Image Classification',
    size: '98 MB',
    parameters: '25.6M',
    lastModified: '2024-01-15'
  });
  
  const [checks, setChecks] = useState<SecurityCheck[]>(initialChecks);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [overallScore, setOverallScore] = useState(0);
  const [selectedCheck, setSelectedCheck] = useState<SecurityCheck | null>(null);

  // 보안 분석 시뮬레이션
  const runSecurityAnalysis = useCallback(() => {
    setIsAnalyzing(true);
    setChecks(initialChecks);

    // 각 체크를 순차적으로 실행
    checks.forEach((check, index) => {
      setTimeout(() => {
        const results = ['pass', 'fail', 'warning'];
        const weights = {
          watermark: [0.7, 0.1, 0.2],
          backdoor: [0.8, 0.1, 0.1],
          robustness: [0.3, 0.3, 0.4],
          extraction: [0.5, 0.2, 0.3],
          privacy: [0.4, 0.3, 0.3],
          encryption: [0.6, 0.2, 0.2],
          access: [0.7, 0.1, 0.2],
          monitoring: [0.8, 0.1, 0.1]
        };

        const weight = weights[check.id as keyof typeof weights] || [0.5, 0.25, 0.25];
        const random = Math.random();
        let status: 'pass' | 'fail' | 'warning';
        
        if (random < weight[0]) status = 'pass';
        else if (random < weight[0] + weight[1]) status = 'fail';
        else status = 'warning';

        setChecks(prev => prev.map(c => 
          c.id === check.id ? { ...c, status } : c
        ));

        if (index === checks.length - 1) {
          setIsAnalyzing(false);
          calculateOverallScore();
        }
      }, 500 * (index + 1));
    });
  }, [checks]);

  // 전체 보안 점수 계산
  const calculateOverallScore = useCallback(() => {
    setTimeout(() => {
      setChecks(current => {
        const scores = current.map(check => {
          if (check.status === 'pass') return 100;
          if (check.status === 'warning') return 50;
          return 0;
        });
        const average = scores.reduce((a: number, b: number) => a + b, 0) / scores.length;
        setOverallScore(average);
        return current;
      });
    }, 100);
  }, []);

  // 카테고리별 그룹화
  const groupedChecks = checks.reduce((acc, check) => {
    if (!acc[check.category]) acc[check.category] = [];
    acc[check.category].push(check);
    return acc;
  }, {} as Record<string, SecurityCheck[]>);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pass':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'fail':
        return <XCircle className="w-5 h-5 text-red-500" />;
      case 'warning':
        return <AlertTriangle className="w-5 h-5 text-yellow-500" />;
      default:
        return <div className="w-5 h-5 rounded-full border-2 border-gray-300 animate-spin" />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'text-red-600 bg-red-50 dark:bg-red-900/30';
      case 'high':
        return 'text-orange-600 bg-orange-50 dark:bg-orange-900/30';
      case 'medium':
        return 'text-yellow-600 bg-yellow-50 dark:bg-yellow-900/30';
      case 'low':
        return 'text-blue-600 bg-blue-50 dark:bg-blue-900/30';
      default:
        return 'text-gray-600 bg-gray-50 dark:bg-gray-800';
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
      <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
        모델 보안 분석기
      </h3>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 mb-6">
            <h4 className="font-semibold mb-3">모델 정보</h4>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-gray-600 dark:text-gray-400">모델명:</span>
                <span className="ml-2 font-medium">{modelInfo.name}</span>
              </div>
              <div>
                <span className="text-gray-600 dark:text-gray-400">타입:</span>
                <span className="ml-2 font-medium">{modelInfo.type}</span>
              </div>
              <div>
                <span className="text-gray-600 dark:text-gray-400">크기:</span>
                <span className="ml-2 font-medium">{modelInfo.size}</span>
              </div>
              <div>
                <span className="text-gray-600 dark:text-gray-400">파라미터:</span>
                <span className="ml-2 font-medium">{modelInfo.parameters}</span>
              </div>
            </div>
          </div>

          <div className="space-y-4">
            {Object.entries(groupedChecks).map(([category, categoryChecks]) => (
              <div key={category} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                <h5 className="font-semibold mb-3 flex items-center">
                  <Shield className="w-4 h-4 mr-2" />
                  {category}
                </h5>
                <div className="space-y-2">
                  {categoryChecks.map(check => (
                    <div
                      key={check.id}
                      className="flex items-center justify-between p-2 hover:bg-gray-50 dark:hover:bg-gray-700 rounded cursor-pointer"
                      onClick={() => setSelectedCheck(check)}
                    >
                      <div className="flex items-center flex-1">
                        {getStatusIcon(check.status)}
                        <span className="ml-3 font-medium">{check.name}</span>
                        <span className={`ml-3 px-2 py-1 text-xs rounded ${getSeverityColor(check.severity)}`}>
                          {check.severity}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>

          <button
            onClick={runSecurityAnalysis}
            disabled={isAnalyzing}
            className="mt-6 w-full bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isAnalyzing ? '분석 중...' : '보안 분석 실행'}
          </button>
        </div>

        <div>
          <div className="bg-gradient-to-br from-red-50 to-gray-50 dark:from-gray-800 dark:to-red-900/30 rounded-lg p-6 mb-6">
            <h4 className="font-semibold mb-4">보안 점수</h4>
            <div className="relative w-32 h-32 mx-auto">
              <svg className="w-full h-full transform -rotate-90">
                <circle
                  cx="64"
                  cy="64"
                  r="56"
                  stroke="currentColor"
                  strokeWidth="12"
                  fill="none"
                  className="text-gray-200 dark:text-gray-700"
                />
                <circle
                  cx="64"
                  cy="64"
                  r="56"
                  stroke="currentColor"
                  strokeWidth="12"
                  fill="none"
                  strokeDasharray={`${2 * Math.PI * 56}`}
                  strokeDashoffset={`${2 * Math.PI * 56 * (1 - overallScore / 100)}`}
                  className={`transition-all duration-1000 ${
                    overallScore >= 80 ? 'text-green-500' :
                    overallScore >= 60 ? 'text-yellow-500' :
                    'text-red-500'
                  }`}
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-3xl font-bold">{Math.round(overallScore)}%</span>
              </div>
            </div>
            <p className="text-center mt-4 text-sm text-gray-600 dark:text-gray-400">
              {overallScore >= 80 ? '안전' :
               overallScore >= 60 ? '주의 필요' :
               '위험'}
            </p>
          </div>

          {selectedCheck && (
            <div className="bg-white dark:bg-gray-700 rounded-lg p-4 border border-gray-200 dark:border-gray-600">
              <h5 className="font-semibold mb-2">{selectedCheck.name}</h5>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                {selectedCheck.description}
              </p>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span>상태:</span>
                  <span className="flex items-center">
                    {getStatusIcon(selectedCheck.status)}
                    <span className="ml-2">
                      {selectedCheck.status === 'pass' ? '통과' :
                       selectedCheck.status === 'fail' ? '실패' :
                       selectedCheck.status === 'warning' ? '경고' : '검사 중'}
                    </span>
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>심각도:</span>
                  <span className={`px-2 py-1 text-xs rounded ${getSeverityColor(selectedCheck.severity)}`}>
                    {selectedCheck.severity}
                  </span>
                </div>
              </div>
              
              {selectedCheck.status === 'fail' && (
                <div className="mt-3 p-3 bg-red-50 dark:bg-red-900/30 rounded text-sm">
                  <p className="font-semibold text-red-700 dark:text-red-300 mb-1">권장 조치:</p>
                  <ul className="text-red-600 dark:text-red-400 space-y-1">
                    <li>• 보안 패치 적용</li>
                    <li>• 설정 검토 및 수정</li>
                    <li>• 추가 보안 계층 구현</li>
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}