'use client';

import React from 'react';
import { Text } from '@react-three/drei';
import { extend } from '@react-three/fiber';
import * as THREE from 'three';

// Three.js 확장
extend({ TextGeometry: THREE.TextGeometry });

interface TextLabelProps {
  position: [number, number, number];
  text: string;
  color?: string;
  fontSize?: number;
  backgroundColor?: string;
}

export const TextLabel: React.FC<TextLabelProps> = ({
  position,
  text,
  color = '#ffffff',
  fontSize = 0.5,
  backgroundColor = '#000000'
}) => {
  return (
    <group position={position}>
      {/* 배경 평면 */}
      <mesh position={[0, 0, -0.01]}>
        <planeGeometry args={[text.length * fontSize * 0.6, fontSize * 1.5]} />
        <meshBasicMaterial color={backgroundColor} opacity={0.8} transparent />
      </mesh>
      
      {/* 텍스트 */}
      <Text
        color={color}
        fontSize={fontSize}
        maxWidth={200}
        lineHeight={1}
        letterSpacing={0.02}
        textAlign="center"
        font="/fonts/inter.woff"
        anchorX="center"
        anchorY="middle"
      >
        {text}
      </Text>
    </group>
  );
};