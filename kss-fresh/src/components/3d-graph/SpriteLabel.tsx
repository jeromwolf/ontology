'use client';

import React, { useMemo } from 'react';
import { Sprite, SpriteMaterial } from 'three';
import { useThree } from '@react-three/fiber';
import * as THREE from 'three';

interface SpriteLabelProps {
  position: [number, number, number];
  text: string;
  color?: string;
  backgroundColor?: string;
  fontSize?: number;
}

export const SpriteLabel: React.FC<SpriteLabelProps> = ({
  position,
  text,
  color = '#ffffff',
  backgroundColor = 'rgba(0, 0, 0, 0.9)',
  fontSize = 48
}) => {
  const { gl } = useThree();
  
  const texture = useMemo(() => {
    // Canvas 생성
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    if (!context) return null;
    
    // 먼저 폰트를 설정해야 measureText가 정확함
    context.font = `bold ${fontSize}px Arial`;
    
    // Canvas 크기 설정
    const metrics = context.measureText(text);
    const textWidth = metrics.width;
    const padding = 30;
    
    canvas.width = textWidth + padding * 2;
    canvas.height = fontSize + padding * 2;
    
    // 배경 그리기
    context.fillStyle = backgroundColor;
    context.fillRect(0, 0, canvas.width, canvas.height);
    
    // 테두리 그리기
    context.strokeStyle = 'rgba(255, 255, 255, 0.5)';
    context.lineWidth = 3;
    context.strokeRect(2, 2, canvas.width - 4, canvas.height - 4);
    
    // 그림자 효과
    context.shadowColor = 'rgba(0, 0, 0, 0.5)';
    context.shadowBlur = 4;
    context.shadowOffsetX = 2;
    context.shadowOffsetY = 2;
    
    // 텍스트 그리기
    context.font = `bold ${fontSize}px Arial`;
    context.fillStyle = color;
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    context.fillText(text, canvas.width / 2, canvas.height / 2);
    
    // 텍스처 생성
    const texture = new THREE.CanvasTexture(canvas);
    texture.needsUpdate = true;
    
    return texture;
  }, [text, color, backgroundColor, fontSize, gl]);
  
  if (!texture) return null;
  
  return (
    <sprite position={position} scale={[5, 2.5, 1]}>
      <spriteMaterial 
        map={texture} 
        sizeAttenuation={false}
        depthTest={false}
        transparent={true}
      />
    </sprite>
  );
};