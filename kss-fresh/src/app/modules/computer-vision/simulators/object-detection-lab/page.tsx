'use client';

import React from 'react';
import { ArrowLeft } from 'lucide-react';
import Link from 'next/link';
import ObjectDetectionLab from '../../components/ObjectDetectionLab';

export default function ObjectDetectionLabPage() {
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <Link
            href="/modules/computer-vision"
            className="inline-flex items-center text-sm text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 mb-4"
          >
            <ArrowLeft className="w-4 h-4 mr-1" />
            컴퓨터 비전 모듈로 돌아가기
          </Link>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Object Detection Lab
          </h1>
          <p className="mt-2 text-gray-600 dark:text-gray-300">
            실시간 객체 탐지 시뮬레이터
          </p>
        </div>

        <ObjectDetectionLab />
      </div>
    </div>
  );
}