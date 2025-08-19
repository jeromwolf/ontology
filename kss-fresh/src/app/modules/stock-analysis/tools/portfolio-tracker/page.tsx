'use client';

import dynamic from 'next/dynamic';

// API 연동된 새로운 포트폴리오 트래커 컴포넌트를 동적으로 임포트
const PortfolioTrackerWithAPI = dynamic(
  () => import('./PortfolioTrackerWithAPI'),
  { ssr: false }
);

export default function PortfolioTrackerPage() {
  return <PortfolioTrackerWithAPI />;
}