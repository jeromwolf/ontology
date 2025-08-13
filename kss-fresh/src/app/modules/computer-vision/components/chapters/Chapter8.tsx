'use client';

import { 
  CheckCircle,
  ExternalLink
} from 'lucide-react';

export default function Chapter8() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4">증강 현실 (AR)</h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          증강 현실은 실제 환경에 가상의 객체를 겹쳐서 보여주는 기술입니다.
          컴퓨터 비전은 AR의 핵심 기술로, 환경 인식과 추적을 담당합니다.
        </p>

        <div className="bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-lg p-6 mb-6">
          <h3 className="font-semibold text-purple-900 dark:text-purple-100 mb-3">AR의 핵심 기술</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h4 className="font-medium text-purple-800 dark:text-purple-200 mb-2">SLAM</h4>
              <p className="text-sm text-purple-700 dark:text-purple-300">
                Simultaneous Localization and Mapping - 동시에 위치를 추정하고 지도를 생성
              </p>
            </div>
            <div>
              <h4 className="font-medium text-purple-800 dark:text-purple-200 mb-2">마커 추적</h4>
              <p className="text-sm text-purple-700 dark:text-purple-300">
                ArUco, QR 코드 등의 마커를 인식하여 3D 콘텐츠 배치
              </p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">자율주행 비전</h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          자율주행 차량의 '눈' 역할을 하는 컴퓨터 비전은 도로, 차선, 신호등, 
          보행자, 다른 차량 등을 실시간으로 인식하고 추적합니다.
        </p>

        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-6">
          <h3 className="font-semibold mb-4">자율주행 비전 시스템 구성</h3>
          <div className="space-y-3">
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 bg-teal-100 dark:bg-teal-900/30 rounded-lg flex items-center justify-center flex-shrink-0">
                <span className="text-teal-600 dark:text-teal-400 font-bold">1</span>
              </div>
              <div>
                <h4 className="font-medium mb-1">다중 센서 융합</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  카메라, LiDAR, 레이더 데이터를 통합하여 정확한 환경 인식
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 bg-teal-100 dark:bg-teal-900/30 rounded-lg flex items-center justify-center flex-shrink-0">
                <span className="text-teal-600 dark:text-teal-400 font-bold">2</span>
              </div>
              <div>
                <h4 className="font-medium mb-1">실시간 객체 검출</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  차량, 보행자, 자전거, 신호등 등을 밀리초 단위로 검출
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 bg-teal-100 dark:bg-teal-900/30 rounded-lg flex items-center justify-center flex-shrink-0">
                <span className="text-teal-600 dark:text-teal-400 font-bold">3</span>
              </div>
              <div>
                <h4 className="font-medium mb-1">차선 및 도로 인식</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  세그멘테이션을 통한 주행 가능 영역 파악
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">의료 영상 분석</h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          딥러닝 기반 의료 영상 분석은 질병의 조기 진단과 정확한 판독에 기여하고 있습니다.
        </p>

        <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-6">
          <div className="flex items-start gap-3">
            <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400 mt-0.5" />
            <div>
              <h3 className="font-semibold text-green-900 dark:text-green-100 mb-2">적용 분야</h3>
              <ul className="space-y-2 text-green-800 dark:text-green-200">
                <li>• X-ray, CT, MRI 영상에서 종양 검출</li>
                <li>• 망막 사진을 통한 당뇨병성 망막병증 진단</li>
                <li>• 피부 사진을 통한 피부암 스크리닝</li>
                <li>• 병리 슬라이드 이미지 분석</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
          <div className="flex items-center gap-3">
            <ExternalLink className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            <p className="text-blue-800 dark:text-blue-200">
              각 시뮬레이터에서 이러한 기술들을 직접 체험하고 실습해보세요!
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}