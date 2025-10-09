'use client';

import React from 'react';
import dynamic from 'next/dynamic';

// Dynamic imports for sections
const Section1 = dynamic(() => import('./sections/Section1'), { ssr: false });
const Section2 = dynamic(() => import('./sections/Section2'), { ssr: false });
const Section3 = dynamic(() => import('./sections/Section3'), { ssr: false });

export default function Chapter3() {
  return (
    <div className="space-y-8">
      <Section1 />
      <Section2 />
      <Section3 />
    </div>
  );
}
