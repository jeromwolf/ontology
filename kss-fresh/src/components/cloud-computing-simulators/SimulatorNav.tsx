import React from 'react';
import Link from 'next/link';
import { ArrowLeft, Home } from 'lucide-react';

export default function SimulatorNav() {
  return (
    <div className="mb-4 flex items-center gap-4">
      <Link
        href="/modules/cloud-computing"
        className="flex items-center gap-2 px-4 py-2 text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white transition-colors rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700"
      >
        <ArrowLeft className="w-4 h-4" />
        <span>모듈로 돌아가기</span>
      </Link>
      <Link
        href="/"
        className="flex items-center gap-2 px-4 py-2 text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white transition-colors rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700"
      >
        <Home className="w-4 h-4" />
        <span>홈</span>
      </Link>
    </div>
  );
}
