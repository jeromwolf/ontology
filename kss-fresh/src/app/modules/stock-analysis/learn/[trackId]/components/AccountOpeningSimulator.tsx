'use client';

import React, { useState } from 'react';
import { CheckCircle, Circle, Trophy } from 'lucide-react';

export default function AccountOpeningSimulator() {
  const [step, setStep] = useState(0);
  
  const steps = [
    { title: '증권회사 선택', desc: '수수료와 서비스를 비교해보세요' },
    { title: '본인인증', desc: '휴대폰 인증 또는 공인인증서' },
    { title: '계좌 정보 입력', desc: '개인정보 및 투자 성향 조사' },
    { title: '계좌 개설 완료', desc: '입금 후 투자 시작!' }
  ];

  return (
    <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
      <h3 className="font-semibold mb-4">증권계좌 개설 체험</h3>
      <div className="space-y-3">
        {steps.map((s, i) => (
          <div
            key={i}
            className={`flex items-center gap-3 p-3 rounded-lg transition-colors ${
              i <= step ? 'bg-white dark:bg-gray-800' : 'opacity-50'
            }`}
          >
            {i < step ? (
              <CheckCircle className="w-5 h-5 text-green-500" />
            ) : i === step ? (
              <Circle className="w-5 h-5 text-blue-500" />
            ) : (
              <Circle className="w-5 h-5 text-gray-400" />
            )}
            <div>
              <div className={i <= step ? 'font-medium' : ''}>{s.title}</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">{s.desc}</div>
            </div>
          </div>
        ))}
      </div>
      {step < 3 && (
        <button
          onClick={() => setStep(step + 1)}
          className="mt-4 w-full py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
        >
          다음 단계
        </button>
      )}
      {step === 3 && (
        <div className="mt-4 p-3 bg-green-100 dark:bg-green-900/30 rounded-lg text-center">
          <Trophy className="w-8 h-8 text-green-600 mx-auto mb-2" />
          <p className="text-green-700 dark:text-green-300 font-medium">축하합니다! 계좌 개설 완료!</p>
        </div>
      )}
    </div>
  );
}