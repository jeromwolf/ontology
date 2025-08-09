'use client';

import TwoDToThreeDConverter from '../../components/TwoDToThreeDConverter';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';

export default function TwoDToThreeDConverterPage() {
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <Link
            href="/modules/computer-vision"
            className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors mb-4"
          >
            <ArrowLeft className="w-4 h-4" />
            Computer Vision으로 돌아가기
          </Link>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            2D to 3D Converter
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            2D 이미지에서 깊이 정보를 추출하고 3D 모델을 생성하는 시뮬레이터
          </p>
        </div>

        {/* Simulator */}
        <TwoDToThreeDConverter />
      </div>
    </div>
  );
}