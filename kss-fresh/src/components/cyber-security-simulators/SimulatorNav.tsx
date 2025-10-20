import React from 'react';
import Link from 'next/link';
import { ArrowLeft, Home } from 'lucide-react';

export default function SimulatorNav() {
  return (
    <div className="mb-6 flex items-center gap-4">
      <Link
        href="/modules/cyber-security"
        className="flex items-center gap-2 px-4 py-2 bg-gray-800/90 hover:bg-gray-700 text-white rounded-lg transition-colors backdrop-blur-sm shadow-lg"
      >
        <ArrowLeft className="w-4 h-4" />
        <span>모듈로 돌아가기</span>
      </Link>
      <Link
        href="/"
        className="flex items-center gap-2 px-4 py-2 bg-gray-800/90 hover:bg-gray-700 text-white rounded-lg transition-colors backdrop-blur-sm shadow-lg"
      >
        <Home className="w-4 h-4" />
        <span>홈</span>
      </Link>
    </div>
  );
}
