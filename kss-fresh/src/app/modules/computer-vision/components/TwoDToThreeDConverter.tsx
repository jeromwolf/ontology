'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { 
  Upload, 
  Download, 
  RotateCw, 
  ZoomIn, 
  ZoomOut,
  Sliders,
  Play,
  Pause,
  RefreshCw,
  Box,
  Grid3X3,
  Layers
} from 'lucide-react';

interface Point3D {
  x: number;
  y: number;
  z: number;
  color: string;
}

type AlgorithmType = 'midas' | 'dpt' | 'nerf';
type VisualizationType = 'depth' | 'pointcloud' | 'mesh';

export default function TwoDToThreeDConverter() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [depthMap, setDepthMap] = useState<ImageData | null>(null);
  const [points3D, setPoints3D] = useState<Point3D[]>([]);
  const [algorithm, setAlgorithm] = useState<AlgorithmType>('midas');
  const [visualization, setVisualization] = useState<VisualizationType>('depth');
  const [isProcessing, setIsProcessing] = useState(false);
  const [isRotating, setIsRotating] = useState(true);
  
  // Parameters
  const [depthScale, setDepthScale] = useState(1.0);
  const [pointDensity, setPointDensity] = useState(0.5);
  const [smoothing, setSmoothing] = useState(0.5);
  const [zoom, setZoom] = useState(1.0);
  const [rotation, setRotation] = useState({ x: 0, y: 0 });
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const imageCanvasRef = useRef<HTMLCanvasElement>(null);
  const depthCanvasRef = useRef<HTMLCanvasElement>(null);
  const threeDCanvasRef = useRef<HTMLCanvasElement>(null);
  const dragRef = useRef({ isDragging: false, startX: 0, startY: 0 });
  
  // Handle file upload
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setSelectedImage(event.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };
  
  // Handle drag and drop
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };
  
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setSelectedImage(event.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };
  
  // Simulate depth estimation
  const estimateDepth = useCallback(() => {
    if (!selectedImage || !imageCanvasRef.current || !depthCanvasRef.current) return;
    
    setIsProcessing(true);
    
    const img = new Image();
    img.onload = () => {
      const imageCanvas = imageCanvasRef.current!;
      const depthCanvas = depthCanvasRef.current!;
      const ctx = imageCanvas.getContext('2d')!;
      const depthCtx = depthCanvas.getContext('2d')!;
      
      // Set canvas sizes
      imageCanvas.width = 400;
      imageCanvas.height = 300;
      depthCanvas.width = 400;
      depthCanvas.height = 300;
      
      // Draw original image
      ctx.drawImage(img, 0, 0, 400, 300);
      
      // Get image data
      const imageData = ctx.getImageData(0, 0, 400, 300);
      const depthData = depthCtx.createImageData(400, 300);
      
      // Simulate depth estimation based on algorithm
      for (let i = 0; i < imageData.data.length; i += 4) {
        const r = imageData.data[i];
        const g = imageData.data[i + 1];
        const b = imageData.data[i + 2];
        
        // Simple depth estimation based on luminance
        let depth = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
        
        // Apply algorithm-specific modifications
        if (algorithm === 'midas') {
          // Simulate MiDaS-style depth
          depth = Math.pow(depth, 2.2) * depthScale;
        } else if (algorithm === 'dpt') {
          // Simulate DPT-style depth with more detail
          depth = (Math.sin(depth * Math.PI * 2) + 1) / 2 * depthScale;
        } else if (algorithm === 'nerf') {
          // Simulate NeRF-style depth with noise
          depth = (depth + Math.random() * 0.1) * depthScale;
        }
        
        // Apply smoothing
        depth = depth * (1 - smoothing) + 0.5 * smoothing;
        
        // Convert depth to grayscale
        const depthValue = Math.floor(depth * 255);
        depthData.data[i] = depthValue;
        depthData.data[i + 1] = depthValue;
        depthData.data[i + 2] = depthValue;
        depthData.data[i + 3] = 255;
      }
      
      // Apply gaussian blur for smoothing
      if (smoothing > 0) {
        // Simple box blur simulation
        const tempData = new Uint8ClampedArray(depthData.data);
        const width = 400;
        const height = 300;
        const radius = Math.floor(smoothing * 5);
        
        for (let y = radius; y < height - radius; y++) {
          for (let x = radius; x < width - radius; x++) {
            let sum = 0;
            let count = 0;
            
            for (let dy = -radius; dy <= radius; dy++) {
              for (let dx = -radius; dx <= radius; dx++) {
                const idx = ((y + dy) * width + (x + dx)) * 4;
                sum += tempData[idx];
                count++;
              }
            }
            
            const idx = (y * width + x) * 4;
            const avg = Math.floor(sum / count);
            depthData.data[idx] = avg;
            depthData.data[idx + 1] = avg;
            depthData.data[idx + 2] = avg;
          }
        }
      }
      
      depthCtx.putImageData(depthData, 0, 0);
      setDepthMap(depthData);
      
      // Generate 3D points
      generate3DPoints(imageData, depthData);
      
      setIsProcessing(false);
    };
    
    img.src = selectedImage;
  }, [selectedImage, algorithm, depthScale, smoothing]);
  
  // Generate 3D point cloud
  const generate3DPoints = (imageData: ImageData, depthData: ImageData) => {
    const points: Point3D[] = [];
    const width = imageData.width;
    const height = imageData.height;
    const step = Math.floor(1 / pointDensity);
    
    for (let y = 0; y < height; y += step) {
      for (let x = 0; x < width; x += step) {
        const idx = (y * width + x) * 4;
        const depth = depthData.data[idx] / 255;
        
        if (depth > 0.1) { // Threshold to remove background
          const r = imageData.data[idx];
          const g = imageData.data[idx + 1];
          const b = imageData.data[idx + 2];
          
          points.push({
            x: (x - width / 2) / width * 2,
            y: -(y - height / 2) / height * 2,
            z: (depth - 0.5) * 2 * depthScale,
            color: `rgb(${r}, ${g}, ${b})`
          });
        }
      }
    }
    
    setPoints3D(points);
  };
  
  // Render 3D visualization
  const render3D = useCallback(() => {
    if (!threeDCanvasRef.current || points3D.length === 0) return;
    
    const canvas = threeDCanvasRef.current;
    const ctx = canvas.getContext('2d')!;
    canvas.width = 600;
    canvas.height = 450;
    
    // Clear canvas
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Transform and project points
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const scale = 150 * zoom;
    
    // Sort points by depth for proper rendering
    const rotatedPoints = points3D.map(point => {
      // Apply rotation
      const cosX = Math.cos(rotation.x);
      const sinX = Math.sin(rotation.x);
      const cosY = Math.cos(rotation.y);
      const sinY = Math.sin(rotation.y);
      
      // Rotate around Y axis
      const x1 = point.x * cosY - point.z * sinY;
      const z1 = point.x * sinY + point.z * cosY;
      
      // Rotate around X axis
      const y2 = point.y * cosX - z1 * sinX;
      const z2 = point.y * sinX + z1 * cosX;
      
      return {
        x: x1,
        y: y2,
        z: z2,
        color: point.color,
        screenX: centerX + x1 * scale,
        screenY: centerY + y2 * scale,
        size: Math.max(1, 4 * (1 + z2))
      };
    }).sort((a, b) => a.z - b.z);
    
    // Render based on visualization type
    if (visualization === 'pointcloud') {
      // Render point cloud
      rotatedPoints.forEach(point => {
        ctx.fillStyle = point.color;
        ctx.beginPath();
        ctx.arc(point.screenX, point.screenY, point.size / 2, 0, Math.PI * 2);
        ctx.fill();
      });
    } else if (visualization === 'mesh') {
      // Render wireframe mesh
      ctx.strokeStyle = '#00ff00';
      ctx.lineWidth = 0.5;
      
      // Simple grid-based mesh connection
      const gridSize = Math.floor(Math.sqrt(points3D.length));
      for (let i = 0; i < rotatedPoints.length - gridSize - 1; i++) {
        if (i % gridSize < gridSize - 1) {
          // Connect to right neighbor
          ctx.beginPath();
          ctx.moveTo(rotatedPoints[i].screenX, rotatedPoints[i].screenY);
          ctx.lineTo(rotatedPoints[i + 1].screenX, rotatedPoints[i + 1].screenY);
          ctx.stroke();
        }
        
        // Connect to bottom neighbor
        if (i + gridSize < rotatedPoints.length) {
          ctx.beginPath();
          ctx.moveTo(rotatedPoints[i].screenX, rotatedPoints[i].screenY);
          ctx.lineTo(rotatedPoints[i + gridSize].screenX, rotatedPoints[i + gridSize].screenY);
          ctx.stroke();
        }
      }
    }
    
    // Draw axes
    ctx.strokeStyle = '#666';
    ctx.lineWidth = 1;
    
    // X axis (red)
    ctx.strokeStyle = '#ff0000';
    ctx.beginPath();
    ctx.moveTo(centerX - 50, centerY);
    ctx.lineTo(centerX + 50, centerY);
    ctx.stroke();
    
    // Y axis (green)
    ctx.strokeStyle = '#00ff00';
    ctx.beginPath();
    ctx.moveTo(centerX, centerY - 50);
    ctx.lineTo(centerX, centerY + 50);
    ctx.stroke();
    
    // Z axis (blue)
    ctx.strokeStyle = '#0000ff';
    ctx.beginPath();
    ctx.moveTo(centerX - 25, centerY + 25);
    ctx.lineTo(centerX + 25, centerY - 25);
    ctx.stroke();
  }, [points3D, rotation, zoom, visualization]);
  
  // Auto-rotate animation
  useEffect(() => {
    if (isRotating && points3D.length > 0) {
      const interval = setInterval(() => {
        setRotation(prev => ({
          x: prev.x,
          y: prev.y + 0.02
        }));
      }, 50);
      
      return () => clearInterval(interval);
    }
  }, [isRotating, points3D.length]);
  
  // Render 3D when rotation or points change
  useEffect(() => {
    render3D();
  }, [render3D]);
  
  // Process image when selected
  useEffect(() => {
    if (selectedImage) {
      estimateDepth();
    }
  }, [selectedImage, estimateDepth]);
  
  // Mouse controls for 3D view
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    dragRef.current = {
      isDragging: true,
      startX: e.clientX,
      startY: e.clientY
    };
    setIsRotating(false);
  };
  
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (dragRef.current.isDragging) {
      const deltaX = e.clientX - dragRef.current.startX;
      const deltaY = e.clientY - dragRef.current.startY;
      
      setRotation(prev => ({
        x: prev.x + deltaY * 0.01,
        y: prev.y + deltaX * 0.01
      }));
      
      dragRef.current.startX = e.clientX;
      dragRef.current.startY = e.clientY;
    }
  };
  
  const handleMouseUp = () => {
    dragRef.current.isDragging = false;
  };
  
  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    setZoom(prev => Math.max(0.5, Math.min(3, prev - e.deltaY * 0.001)));
  };
  
  // Download 3D model
  const download3DModel = () => {
    if (points3D.length === 0) return;
    
    // Generate simple PLY format
    let plyContent = 'ply\n';
    plyContent += 'format ascii 1.0\n';
    plyContent += `element vertex ${points3D.length}\n`;
    plyContent += 'property float x\n';
    plyContent += 'property float y\n';
    plyContent += 'property float z\n';
    plyContent += 'property uchar red\n';
    plyContent += 'property uchar green\n';
    plyContent += 'property uchar blue\n';
    plyContent += 'end_header\n';
    
    points3D.forEach(point => {
      const rgb = point.color.match(/\d+/g);
      if (rgb) {
        plyContent += `${point.x} ${point.y} ${point.z} ${rgb[0]} ${rgb[1]} ${rgb[2]}\n`;
      }
    });
    
    const blob = new Blob([plyContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = '3d_model.ply';
    a.click();
    URL.revokeObjectURL(url);
  };
  
  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
      {/* Controls */}
      <div className="mb-6 space-y-4">
        <div className="flex flex-wrap gap-4">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileUpload}
            className="hidden"
          />
          
          <button
            onClick={() => fileInputRef.current?.click()}
            className="px-4 py-2 bg-teal-600 text-white rounded-lg hover:bg-teal-700 transition-colors flex items-center gap-2"
          >
            <Upload className="w-4 h-4" />
            이미지 업로드
          </button>
          
          {points3D.length > 0 && (
            <>
              <button
                onClick={download3DModel}
                className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors flex items-center gap-2"
              >
                <Download className="w-4 h-4" />
                3D 모델 다운로드
              </button>
              
              <button
                onClick={() => setIsRotating(!isRotating)}
                className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors flex items-center gap-2"
              >
                {isRotating ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                {isRotating ? '회전 정지' : '회전 시작'}
              </button>
              
              <button
                onClick={() => {
                  setRotation({ x: 0, y: 0 });
                  setZoom(1);
                }}
                className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors flex items-center gap-2"
              >
                <RefreshCw className="w-4 h-4" />
                뷰 리셋
              </button>
            </>
          )}
        </div>
        
        {/* Algorithm Selection */}
        <div className="flex flex-wrap gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              깊이 추정 알고리즘
            </label>
            <select
              value={algorithm}
              onChange={(e) => setAlgorithm(e.target.value as AlgorithmType)}
              className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value="midas">MiDaS</option>
              <option value="dpt">DPT (Dense Prediction Transformer)</option>
              <option value="nerf">NeRF Style</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              시각화 타입
            </label>
            <select
              value={visualization}
              onChange={(e) => setVisualization(e.target.value as VisualizationType)}
              className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value="depth">깊이 맵</option>
              <option value="pointcloud">포인트 클라우드</option>
              <option value="mesh">와이어프레임 메시</option>
            </select>
          </div>
        </div>
        
        {/* Parameters */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              깊이 스케일: {depthScale.toFixed(1)}
            </label>
            <input
              type="range"
              min="0.5"
              max="2.0"
              step="0.1"
              value={depthScale}
              onChange={(e) => setDepthScale(parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              포인트 밀도: {(pointDensity * 100).toFixed(0)}%
            </label>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.1"
              value={pointDensity}
              onChange={(e) => setPointDensity(parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              스무딩: {(smoothing * 100).toFixed(0)}%
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={smoothing}
              onChange={(e) => setSmoothing(parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
            />
          </div>
        </div>
      </div>
      
      {/* Main Content */}
      {!selectedImage ? (
        <div
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-20 text-center"
        >
          <Upload className="w-12 h-12 mx-auto mb-4 text-gray-400" />
          <p className="text-gray-600 dark:text-gray-400">
            이미지를 드래그하여 놓거나 클릭하여 업로드하세요
          </p>
          <p className="text-sm text-gray-500 dark:text-gray-500 mt-2">
            지원 형식: JPG, PNG, GIF
          </p>
        </div>
      ) : (
        <div className="space-y-6">
          {/* Processing indicator */}
          {isProcessing && (
            <div className="text-center py-4">
              <div className="inline-flex items-center gap-2 text-teal-600 dark:text-teal-400">
                <RotateCw className="w-5 h-5 animate-spin" />
                깊이 정보를 추정하고 있습니다...
              </div>
            </div>
          )}
          
          {/* Visualization Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Original Image */}
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2 flex items-center gap-2">
                <Layers className="w-5 h-5" />
                원본 이미지
              </h3>
              <div className="bg-gray-100 dark:bg-gray-700 rounded-lg p-2">
                <canvas
                  ref={imageCanvasRef}
                  className="w-full h-auto"
                  style={{ maxHeight: '300px' }}
                />
              </div>
            </div>
            
            {/* Depth Map */}
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2 flex items-center gap-2">
                <Grid3X3 className="w-5 h-5" />
                깊이 맵
              </h3>
              <div className="bg-gray-100 dark:bg-gray-700 rounded-lg p-2">
                <canvas
                  ref={depthCanvasRef}
                  className="w-full h-auto"
                  style={{ maxHeight: '300px' }}
                />
              </div>
            </div>
          </div>
          
          {/* 3D Visualization */}
          {points3D.length > 0 && (
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2 flex items-center gap-2">
                <Box className="w-5 h-5" />
                3D 시각화
                <span className="text-sm text-gray-500 ml-2">
                  (마우스로 회전, 스크롤로 줌)
                </span>
              </h3>
              <div className="bg-gray-900 rounded-lg p-2">
                <canvas
                  ref={threeDCanvasRef}
                  onMouseDown={handleMouseDown}
                  onMouseMove={handleMouseMove}
                  onMouseUp={handleMouseUp}
                  onMouseLeave={handleMouseUp}
                  onWheel={handleWheel}
                  className="w-full h-auto cursor-move"
                  style={{ maxHeight: '450px' }}
                />
              </div>
              
              {/* Zoom Controls */}
              <div className="flex items-center gap-4 mt-4">
                <button
                  onClick={() => setZoom(prev => Math.min(3, prev + 0.2))}
                  className="p-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
                >
                  <ZoomIn className="w-4 h-4" />
                </button>
                <span className="text-gray-600 dark:text-gray-400 min-w-[60px] text-center">
                  {(zoom * 100).toFixed(0)}%
                </span>
                <button
                  onClick={() => setZoom(prev => Math.max(0.5, prev - 0.2))}
                  className="p-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
                >
                  <ZoomOut className="w-4 h-4" />
                </button>
              </div>
            </div>
          )}
          
          {/* Statistics */}
          {points3D.length > 0 && (
            <div className="bg-gray-100 dark:bg-gray-700 rounded-lg p-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                3D 모델 정보
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <span className="text-gray-600 dark:text-gray-400">포인트 수:</span>
                  <span className="ml-2 font-medium text-gray-900 dark:text-white">
                    {points3D.length.toLocaleString()}
                  </span>
                </div>
                <div>
                  <span className="text-gray-600 dark:text-gray-400">알고리즘:</span>
                  <span className="ml-2 font-medium text-gray-900 dark:text-white">
                    {algorithm.toUpperCase()}
                  </span>
                </div>
                <div>
                  <span className="text-gray-600 dark:text-gray-400">시각화:</span>
                  <span className="ml-2 font-medium text-gray-900 dark:text-white">
                    {visualization === 'depth' ? '깊이 맵' : 
                     visualization === 'pointcloud' ? '포인트 클라우드' : '와이어프레임'}
                  </span>
                </div>
                <div>
                  <span className="text-gray-600 dark:text-gray-400">형식:</span>
                  <span className="ml-2 font-medium text-gray-900 dark:text-white">
                    PLY (Point Cloud)
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}