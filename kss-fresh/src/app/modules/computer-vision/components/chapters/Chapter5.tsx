'use client';

import { 
  Lightbulb,
  PlayCircle
} from 'lucide-react';

export default function Chapter5() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4">스테레오 비전</h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          스테레오 비전은 두 개 이상의 카메라를 사용하여 깊이 정보를 추출하는 기술입니다.
          인간의 양안 시차와 같은 원리로 3D 정보를 복원합니다.
        </p>

        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-6 mb-6">
          <div className="flex items-start gap-3">
            <Lightbulb className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5" />
            <div>
              <h3 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">핵심 원리</h3>
              <ul className="space-y-2 text-blue-800 dark:text-blue-200">
                <li>• 삼각측량: 두 카메라의 시차를 이용한 거리 계산</li>
                <li>• 에피폴라 기하학: 스테레오 매칭의 기하학적 제약</li>
                <li>• 디스패리티 맵: 픽셀별 깊이 정보를 담은 이미지</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">단일 이미지 깊이 추정</h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          최신 딥러닝 기술을 활용하면 단일 이미지에서도 깊이 정보를 추정할 수 있습니다.
          MiDaS, DPT 등의 모델이 대표적입니다.
        </p>

        <div className="flex items-center gap-3 p-4 bg-teal-50 dark:bg-teal-900/20 rounded-lg mb-6">
          <PlayCircle className="w-6 h-6 text-teal-600 dark:text-teal-400" />
          <p className="text-teal-800 dark:text-teal-200">
            2D to 3D Converter 시뮬레이터에서 실시간으로 깊이 추정을 체험해보세요!
          </p>
        </div>
      </section>
    </div>
  );
}