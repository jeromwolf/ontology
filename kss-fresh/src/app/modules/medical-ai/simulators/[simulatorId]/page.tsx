'use client';

import React from 'react';
import { useParams } from 'next/navigation';
import dynamic from 'next/dynamic';

// Dynamic imports for all Medical AI simulators
const ChestXrayClassifier = dynamic(() => import('@/components/medical-ai-simulators/ChestXrayClassifier'), {
  ssr: false,
});
const TumorSegmentation = dynamic(
  () => import('@/components/medical-ai-simulators/TumorSegmentation'),
  { ssr: false }
);
const ECGAnomalyDetector = dynamic(
  () => import('@/components/medical-ai-simulators/ECGAnomalyDetector'),
  { ssr: false }
);
const MoleculeGenerator = dynamic(
  () => import('@/components/medical-ai-simulators/MoleculeGenerator'),
  { ssr: false }
);
const ClinicalNER = dynamic(
  () => import('@/components/medical-ai-simulators/ClinicalNER'),
  { ssr: false }
);
const SurvivalPredictor = dynamic(
  () => import('@/components/medical-ai-simulators/SurvivalPredictor'),
  { ssr: false }
);

export default function SimulatorPage() {
  const params = useParams();
  const simulatorId = params.simulatorId as string;

  const getSimulatorComponent = () => {
    switch (simulatorId) {
      case 'chest-xray-classifier':
        return <ChestXrayClassifier />;
      case 'tumor-segmentation':
        return <TumorSegmentation />;
      case 'ecg-anomaly-detector':
        return <ECGAnomalyDetector />;
      case 'molecule-generator':
        return <MoleculeGenerator />;
      case 'clinical-ner':
        return <ClinicalNER />;
      case 'survival-predictor':
        return <SurvivalPredictor />;
      default:
        return (
          <div className="min-h-screen bg-gradient-to-br from-pink-50 to-red-50 dark:from-gray-900 dark:to-pink-950 flex items-center justify-center">
            <div className="text-center">
              <h1 className="text-4xl font-bold mb-4 text-gray-900 dark:text-white">시뮬레이터를 찾을 수 없습니다</h1>
              <p className="text-gray-600 dark:text-gray-400 mb-6">요청하신 시뮬레이터가 존재하지 않습니다.</p>
              <a
                href="/modules/medical-ai"
                className="bg-gradient-to-r from-pink-500 to-red-600 hover:from-pink-600 hover:to-red-700 text-white px-6 py-3 rounded-lg font-semibold inline-block"
              >
                모듈로 돌아가기
              </a>
            </div>
          </div>
        );
    }
  };

  return getSimulatorComponent();
}
