import React, { useState } from 'react';
import { Terminal, Lock, Unlock, AlertTriangle, CheckCircle2 } from 'lucide-react';
import SimulatorNav from './SimulatorNav';

export default function HackingLab() {
  const [selectedChallenge, setSelectedChallenge] = useState<string>('sql-injection');
  const [terminalInput, setTerminalInput] = useState<string>('');
  const [terminalOutput, setTerminalOutput] = useState<string[]>([
    'Welcome to Cyber Security Hacking Lab',
    'Type your commands below to test security vulnerabilities...',
  ]);
  const [isVulnerable, setIsVulnerable] = useState<boolean>(true);
  const [exploitSuccess, setExploitSuccess] = useState<boolean>(false);

  const challenges = [
    {
      id: 'sql-injection',
      name: 'SQL Injection',
      desc: 'SQL 인젝션 공격 시뮬레이션',
      exploitCommand: "' OR '1'='1",
      hint: '로그인 폼에서 username에 특수 문자 입력 시도',
    },
    {
      id: 'xss',
      name: 'XSS Attack',
      desc: 'Cross-Site Scripting 공격',
      exploitCommand: '<script>alert("XSS")</script>',
      hint: '입력 필드에 스크립트 태그 삽입 시도',
    },
    {
      id: 'csrf',
      name: 'CSRF Attack',
      desc: 'Cross-Site Request Forgery',
      exploitCommand: 'csrf-token=fake',
      hint: 'CSRF 토큰 없이 요청 전송 시도',
    },
    {
      id: 'path-traversal',
      name: 'Path Traversal',
      desc: '경로 탐색 공격',
      exploitCommand: '../../etc/passwd',
      hint: '상위 디렉토리 접근 시도',
    },
  ];

  const currentChallenge = challenges.find((c) => c.id === selectedChallenge);

  const handleExecute = () => {
    if (!terminalInput.trim()) return;

    const newOutput = [...terminalOutput, `$ ${terminalInput}`];

    if (isVulnerable) {
      // 취약한 상태 - exploit 성공
      if (terminalInput.includes(currentChallenge?.exploitCommand || '')) {
        newOutput.push('⚠️ EXPLOIT SUCCESSFUL! 보안 취약점 발견!');
        newOutput.push('💡 이 시스템은 취약합니다. 보안 패치가 필요합니다.');
        setExploitSuccess(true);
      } else {
        newOutput.push('❌ 공격 실패: 올바른 exploit 시도가 아닙니다.');
        newOutput.push(`💡 Hint: ${currentChallenge?.hint}`);
      }
    } else {
      // 보안 패치 적용 상태 - exploit 차단
      newOutput.push('🛡️ 공격이 차단되었습니다!');
      newOutput.push('✅ 입력 검증 및 보안 패치가 활성화되어 있습니다.');
      setExploitSuccess(false);
    }

    setTerminalOutput(newOutput);
    setTerminalInput('');
  };

  const toggleVulnerability = () => {
    setIsVulnerable(!isVulnerable);
    setExploitSuccess(false);
    setTerminalOutput([
      `보안 모드 ${!isVulnerable ? '비활성화' : '활성화'}`,
      `현재 시스템은 ${!isVulnerable ? '취약한' : '보안된'} 상태입니다.`,
    ]);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-slate-900 to-gray-900 text-white">
      <div className="max-w-7xl mx-auto px-4 py-8">
        <SimulatorNav />
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-4 flex items-center gap-3">
            <Terminal className="w-10 h-10 text-red-500" />
            Hacking Lab
          </h1>
          <p className="text-xl text-gray-300">
            윤리적 해킹을 통한 보안 취약점 학습 시뮬레이터
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-6 mb-6">
          {/* Challenge Selection */}
          <div className="md:col-span-1 bg-gray-800 rounded-xl p-6">
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
              <AlertTriangle className="w-6 h-6 text-yellow-500" />
              Challenge 선택
            </h2>
            <div className="space-y-3">
              {challenges.map((challenge) => (
                <button
                  key={challenge.id}
                  onClick={() => {
                    setSelectedChallenge(challenge.id);
                    setExploitSuccess(false);
                    setTerminalOutput([
                      `Challenge 변경: ${challenge.name}`,
                      challenge.desc,
                      `Hint: ${challenge.hint}`,
                    ]);
                  }}
                  className={`w-full p-3 rounded-lg text-left transition-all ${
                    selectedChallenge === challenge.id
                      ? 'bg-red-600 border-2 border-red-400'
                      : 'bg-gray-700 hover:bg-gray-600'
                  }`}
                >
                  <div className="font-bold">{challenge.name}</div>
                  <div className="text-xs text-gray-400 mt-1">{challenge.desc}</div>
                </button>
              ))}
            </div>

            <div className="mt-6 pt-6 border-t border-gray-700">
              <h3 className="font-bold mb-3 flex items-center gap-2">
                {isVulnerable ? (
                  <Unlock className="w-5 h-5 text-red-500" />
                ) : (
                  <Lock className="w-5 h-5 text-green-500" />
                )}
                시스템 상태
              </h3>
              <button
                onClick={toggleVulnerability}
                className={`w-full py-2 px-4 rounded-lg font-semibold transition-all ${
                  isVulnerable
                    ? 'bg-red-600 hover:bg-red-700'
                    : 'bg-green-600 hover:bg-green-700'
                }`}
              >
                {isVulnerable ? '🔓 취약한 상태' : '🔒 보안 적용'}
              </button>
              <p className="text-xs text-gray-400 mt-2">
                {isVulnerable
                  ? '보안 패치가 적용되지 않은 취약한 시스템'
                  : '입력 검증 및 보안 패치 적용됨'}
              </p>
            </div>
          </div>

          {/* Terminal */}
          <div className="md:col-span-2 bg-gray-800 rounded-xl p-6">
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
              <Terminal className="w-6 h-6 text-green-500" />
              해킹 터미널
            </h2>

            {/* Terminal Output */}
            <div className="bg-black rounded-lg p-4 h-96 overflow-y-auto font-mono text-sm mb-4">
              {terminalOutput.map((line, idx) => (
                <div key={idx} className="mb-1">
                  {line}
                </div>
              ))}
            </div>

            {/* Terminal Input */}
            <div className="flex gap-2">
              <input
                type="text"
                value={terminalInput}
                onChange={(e) => setTerminalInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleExecute()}
                placeholder={`Try: ${currentChallenge?.exploitCommand}`}
                className="flex-1 bg-gray-900 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-red-500"
              />
              <button
                onClick={handleExecute}
                className="bg-red-600 hover:bg-red-700 px-6 py-2 rounded-lg font-semibold transition-all"
              >
                Execute
              </button>
            </div>

            {exploitSuccess && (
              <div className="mt-4 bg-yellow-900/50 border-2 border-yellow-500 rounded-lg p-4">
                <h3 className="font-bold text-yellow-400 mb-2 flex items-center gap-2">
                  <AlertTriangle className="w-5 h-5" />
                  보안 취약점 발견!
                </h3>
                <p className="text-sm text-gray-300 mb-2">
                  현재 {currentChallenge?.name} 공격에 취약합니다.
                </p>
                <button
                  onClick={toggleVulnerability}
                  className="bg-green-600 hover:bg-green-700 px-4 py-2 rounded-lg text-sm font-semibold flex items-center gap-2"
                >
                  <CheckCircle2 className="w-4 h-4" />
                  보안 패치 적용하기
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Info Section */}
        <div className="bg-gradient-to-r from-blue-900/50 to-purple-900/50 rounded-xl p-6">
          <h2 className="text-2xl font-bold mb-4">💡 학습 목표</h2>
          <div className="grid md:grid-cols-2 gap-4 text-sm">
            <div>
              <h3 className="font-bold text-blue-400 mb-2">공격 기법 이해</h3>
              <ul className="space-y-1 text-gray-300">
                <li>• SQL Injection 원리와 위험성</li>
                <li>• XSS 공격 메커니즘</li>
                <li>• CSRF 공격 시나리오</li>
                <li>• Path Traversal 기법</li>
              </ul>
            </div>
            <div>
              <h3 className="font-bold text-green-400 mb-2">방어 전략</h3>
              <ul className="space-y-1 text-gray-300">
                <li>• 입력 검증 및 필터링</li>
                <li>• Prepared Statement 사용</li>
                <li>• CSRF 토큰 구현</li>
                <li>• 최소 권한 원칙 적용</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
