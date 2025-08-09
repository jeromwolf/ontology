'use client';

import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Upload, Camera, Download, Play, Pause, RefreshCw, Eye, EyeOff } from 'lucide-react';

// 객체 클래스 정의
const OBJECT_CLASSES = [
  { id: 'person', name: '사람', color: '#FF6B6B' },
  { id: 'car', name: '자동차', color: '#4ECDC4' },
  { id: 'bicycle', name: '자전거', color: '#45B7D1' },
  { id: 'motorcycle', name: '오토바이', color: '#96CEB4' },
  { id: 'bus', name: '버스', color: '#FECA57' },
  { id: 'truck', name: '트럭', color: '#FF9FF3' },
  { id: 'cat', name: '고양이', color: '#54A0FF' },
  { id: 'dog', name: '개', color: '#48DBFB' },
  { id: 'bird', name: '새', color: '#A29BFE' },
  { id: 'chair', name: '의자', color: '#FD79A8' },
  { id: 'laptop', name: '노트북', color: '#FDCB6E' },
  { id: 'phone', name: '휴대폰', color: '#6C5CE7' }
];

// 탐지 모델 정의
const DETECTION_MODELS = [
  { id: 'yolo', name: 'YOLO v5', speed: '빠름', accuracy: '높음' },
  { id: 'ssd', name: 'SSD MobileNet', speed: '매우 빠름', accuracy: '중간' },
  { id: 'frcnn', name: 'Faster R-CNN', speed: '느림', accuracy: '매우 높음' }
];

interface DetectedObject {
  class: string;
  confidence: number;
  bbox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
}

export default function ObjectDetectionLab() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const animationFrameRef = useRef<number>();
  
  const [sourceType, setSourceType] = useState<'image' | 'video' | 'webcam'>('image');
  const [selectedModel, setSelectedModel] = useState('yolo');
  const [isProcessing, setIsProcessing] = useState(false);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);
  const [nmsThreshold, setNmsThreshold] = useState(0.4);
  const [selectedClasses, setSelectedClasses] = useState<Set<string>>(new Set(OBJECT_CLASSES.map(c => c.id)));
  const [detectedObjects, setDetectedObjects] = useState<DetectedObject[]>([]);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [showLabels, setShowLabels] = useState(true);
  const [showConfidence, setShowConfidence] = useState(true);
  const [stats, setStats] = useState({
    totalObjects: 0,
    processingTime: 0,
    fps: 0
  });

  // 가상 객체 탐지 함수
  const detectObjects = useCallback((width: number, height: number): DetectedObject[] => {
    const objects: DetectedObject[] = [];
    const numObjects = Math.floor(Math.random() * 8) + 3;
    
    for (let i = 0; i < numObjects; i++) {
      const classIndex = Math.floor(Math.random() * OBJECT_CLASSES.length);
      const selectedClass = OBJECT_CLASSES[classIndex];
      
      if (!selectedClasses.has(selectedClass.id)) continue;
      
      const confidence = 0.5 + Math.random() * 0.5;
      if (confidence < confidenceThreshold) continue;
      
      objects.push({
        class: selectedClass.id,
        confidence,
        bbox: {
          x: Math.random() * (width - 100),
          y: Math.random() * (height - 100),
          width: 50 + Math.random() * 150,
          height: 50 + Math.random() * 150
        }
      });
    }
    
    return objects;
  }, [confidenceThreshold, selectedClasses]);

  // NMS (Non-Maximum Suppression) 적용
  const applyNMS = useCallback((objects: DetectedObject[]): DetectedObject[] => {
    const grouped = objects.reduce((acc, obj) => {
      if (!acc[obj.class]) acc[obj.class] = [];
      acc[obj.class].push(obj);
      return acc;
    }, {} as Record<string, DetectedObject[]>);
    
    const filtered: DetectedObject[] = [];
    
    Object.values(grouped).forEach(group => {
      const sorted = group.sort((a, b) => b.confidence - a.confidence);
      const kept: DetectedObject[] = [];
      
      sorted.forEach(obj => {
        let keep = true;
        
        for (const keptObj of kept) {
          const iou = calculateIOU(obj.bbox, keptObj.bbox);
          if (iou > nmsThreshold) {
            keep = false;
            break;
          }
        }
        
        if (keep) kept.push(obj);
      });
      
      filtered.push(...kept);
    });
    
    return filtered;
  }, [nmsThreshold]);

  // IOU (Intersection over Union) 계산
  const calculateIOU = (box1: any, box2: any): number => {
    const x1 = Math.max(box1.x, box2.x);
    const y1 = Math.max(box1.y, box2.y);
    const x2 = Math.min(box1.x + box1.width, box2.x + box2.width);
    const y2 = Math.min(box1.y + box1.height, box2.y + box2.height);
    
    if (x2 < x1 || y2 < y1) return 0;
    
    const intersection = (x2 - x1) * (y2 - y1);
    const area1 = box1.width * box1.height;
    const area2 = box2.width * box2.height;
    const union = area1 + area2 - intersection;
    
    return intersection / union;
  };

  // Canvas에 탐지 결과 그리기
  const drawDetections = useCallback((
    ctx: CanvasRenderingContext2D,
    objects: DetectedObject[],
    width: number,
    height: number
  ) => {
    ctx.clearRect(0, 0, width, height);
    
    // 원본 이미지/비디오 그리기
    if (imageUrl && sourceType === 'image') {
      const img = new Image();
      img.onload = () => {
        ctx.drawImage(img, 0, 0, width, height);
        drawBoundingBoxes(ctx, objects);
      };
      img.src = imageUrl;
    } else if (videoRef.current && (sourceType === 'video' || sourceType === 'webcam')) {
      ctx.drawImage(videoRef.current, 0, 0, width, height);
      drawBoundingBoxes(ctx, objects);
    }
  }, [imageUrl, sourceType, showLabels, showConfidence]);

  // 바운딩 박스 그리기
  const drawBoundingBoxes = (ctx: CanvasRenderingContext2D, objects: DetectedObject[]) => {
    objects.forEach(obj => {
      const classInfo = OBJECT_CLASSES.find(c => c.id === obj.class);
      if (!classInfo) return;
      
      // 바운딩 박스
      ctx.strokeStyle = classInfo.color;
      ctx.lineWidth = 2;
      ctx.strokeRect(obj.bbox.x, obj.bbox.y, obj.bbox.width, obj.bbox.height);
      
      // 라벨 배경
      if (showLabels || showConfidence) {
        const label = `${showLabels ? classInfo.name : ''} ${showConfidence ? `(${(obj.confidence * 100).toFixed(1)}%)` : ''}`.trim();
        ctx.font = '14px Inter';
        const textWidth = ctx.measureText(label).width;
        
        ctx.fillStyle = classInfo.color;
        ctx.fillRect(obj.bbox.x, obj.bbox.y - 25, textWidth + 10, 25);
        
        // 라벨 텍스트
        ctx.fillStyle = 'white';
        ctx.fillText(label, obj.bbox.x + 5, obj.bbox.y - 8);
      }
    });
  };

  // 이미지 처리
  const processImage = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const startTime = performance.now();
    const objects = detectObjects(canvas.width, canvas.height);
    const filteredObjects = applyNMS(objects);
    const endTime = performance.now();
    
    setDetectedObjects(filteredObjects);
    setStats({
      totalObjects: filteredObjects.length,
      processingTime: endTime - startTime,
      fps: 0
    });
    
    drawDetections(ctx, filteredObjects, canvas.width, canvas.height);
  }, [detectObjects, applyNMS, drawDetections]);

  // 비디오/웹캠 처리
  const processVideo = useCallback(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || !isProcessing) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const startTime = performance.now();
    const objects = detectObjects(canvas.width, canvas.height);
    const filteredObjects = applyNMS(objects);
    const endTime = performance.now();
    
    setDetectedObjects(filteredObjects);
    setStats({
      totalObjects: filteredObjects.length,
      processingTime: endTime - startTime,
      fps: Math.round(1000 / (endTime - startTime))
    });
    
    drawDetections(ctx, filteredObjects, canvas.width, canvas.height);
    
    animationFrameRef.current = requestAnimationFrame(processVideo);
  }, [isProcessing, detectObjects, applyNMS, drawDetections]);

  // 파일 업로드 처리
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (event) => {
      if (file.type.startsWith('image/')) {
        setSourceType('image');
        setImageUrl(event.target?.result as string);
        setTimeout(processImage, 100);
      } else if (file.type.startsWith('video/')) {
        setSourceType('video');
        if (videoRef.current) {
          videoRef.current.src = event.target?.result as string;
        }
      }
    };
    reader.readAsDataURL(file);
  };

  // 웹캠 시작
  const startWebcam = async () => {
    // 카메라 권한 확인
    if (!confirm('이 기능은 카메라 권한이 필요합니다. 객체 감지를 위해 카메라를 사용하시겠습니까?')) {
      return
    }
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setSourceType('webcam');
      }
    } catch (error) {
      console.error('웹캠 접근 실패:', error);
    }
  };

  // 결과 내보내기
  const exportResults = () => {
    const results = {
      model: selectedModel,
      timestamp: new Date().toISOString(),
      settings: {
        confidenceThreshold,
        nmsThreshold,
        selectedClasses: Array.from(selectedClasses)
      },
      detections: detectedObjects,
      stats
    };
    
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `detection-results-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // 클래스 토글
  const toggleClass = (classId: string) => {
    setSelectedClasses(prev => {
      const newSet = new Set(prev);
      if (newSet.has(classId)) {
        newSet.delete(classId);
      } else {
        newSet.add(classId);
      }
      return newSet;
    });
  };

  // 재처리
  useEffect(() => {
    if (sourceType === 'image' && imageUrl) {
      processImage();
    }
  }, [confidenceThreshold, nmsThreshold, selectedClasses, selectedModel]);

  // 비디오/웹캠 처리 시작/중지
  useEffect(() => {
    if (isProcessing && (sourceType === 'video' || sourceType === 'webcam')) {
      processVideo();
    } else {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    }
    
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isProcessing, processVideo]);

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
      {/* 상단 컨트롤 */}
      <div className="mb-6 grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* 소스 선택 */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            입력 소스
          </label>
          <div className="flex gap-2">
            <button
              onClick={() => fileInputRef.current?.click()}
              className="flex-1 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 
                       flex items-center justify-center gap-2"
            >
              <Upload className="w-4 h-4" />
              파일 업로드
            </button>
            <button
              onClick={startWebcam}
              className="flex-1 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 
                       flex items-center justify-center gap-2"
            >
              <Camera className="w-4 h-4" />
              웹캠
            </button>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*,video/*"
            onChange={handleFileUpload}
            className="hidden"
          />
        </div>

        {/* 모델 선택 */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            탐지 모델
          </label>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg 
                     bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
          >
            {DETECTION_MODELS.map(model => (
              <option key={model.id} value={model.id}>
                {model.name} (속도: {model.speed}, 정확도: {model.accuracy})
              </option>
            ))}
          </select>
        </div>

        {/* 액션 버튼 */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            컨트롤
          </label>
          <div className="flex gap-2">
            {(sourceType === 'video' || sourceType === 'webcam') && (
              <button
                onClick={() => setIsProcessing(!isProcessing)}
                className={`flex-1 px-4 py-2 rounded-lg text-white flex items-center justify-center gap-2
                         ${isProcessing ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'}`}
              >
                {isProcessing ? (
                  <>
                    <Pause className="w-4 h-4" />
                    정지
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4" />
                    시작
                  </>
                )}
              </button>
            )}
            {sourceType === 'image' && (
              <button
                onClick={processImage}
                className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 
                         flex items-center justify-center gap-2"
              >
                <RefreshCw className="w-4 h-4" />
                재처리
              </button>
            )}
            <button
              onClick={exportResults}
              className="flex-1 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 
                       flex items-center justify-center gap-2"
            >
              <Download className="w-4 h-4" />
              내보내기
            </button>
          </div>
        </div>
      </div>

      {/* 메인 컨텐츠 */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 캔버스 영역 */}
        <div className="lg:col-span-2">
          <div className="relative bg-gray-100 dark:bg-gray-900 rounded-lg overflow-hidden">
            <canvas
              ref={canvasRef}
              width={800}
              height={600}
              className="w-full h-auto"
            />
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="hidden"
            />
            
            {/* 통계 오버레이 */}
            <div className="absolute top-4 left-4 bg-black/50 text-white p-2 rounded-lg text-sm">
              <div>객체 수: {stats.totalObjects}</div>
              <div>처리 시간: {stats.processingTime.toFixed(2)}ms</div>
              {stats.fps > 0 && <div>FPS: {stats.fps}</div>}
            </div>
            
            {/* 표시 옵션 */}
            <div className="absolute top-4 right-4 flex gap-2">
              <button
                onClick={() => setShowLabels(!showLabels)}
                className={`p-2 rounded-lg ${showLabels ? 'bg-indigo-600' : 'bg-gray-600'} text-white`}
              >
                {showLabels ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
              </button>
              <button
                onClick={() => setShowConfidence(!showConfidence)}
                className={`p-2 rounded-lg ${showConfidence ? 'bg-indigo-600' : 'bg-gray-600'} text-white`}
              >
                %
              </button>
            </div>
          </div>
        </div>

        {/* 설정 패널 */}
        <div className="space-y-6">
          {/* 임계값 설정 */}
          <div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              임계값 설정
            </h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  신뢰도 임계값: {(confidenceThreshold * 100).toFixed(0)}%
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={confidenceThreshold}
                  onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  NMS 임계값: {(nmsThreshold * 100).toFixed(0)}%
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={nmsThreshold}
                  onChange={(e) => setNmsThreshold(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
            </div>
          </div>

          {/* 클래스 필터 */}
          <div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              객체 클래스 필터
            </h3>
            
            <div className="grid grid-cols-2 gap-2">
              {OBJECT_CLASSES.map(cls => (
                <button
                  key={cls.id}
                  onClick={() => toggleClass(cls.id)}
                  className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors
                           ${selectedClasses.has(cls.id)
                             ? 'bg-indigo-600 text-white'
                             : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                           }`}
                  style={{
                    backgroundColor: selectedClasses.has(cls.id) ? cls.color : undefined
                  }}
                >
                  {cls.name}
                </button>
              ))}
            </div>
          </div>

          {/* 탐지 결과 */}
          <div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              탐지 결과
            </h3>
            
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {detectedObjects.length === 0 ? (
                <p className="text-gray-500 dark:text-gray-400 text-sm">
                  탐지된 객체가 없습니다
                </p>
              ) : (
                detectedObjects.map((obj, idx) => {
                  const classInfo = OBJECT_CLASSES.find(c => c.id === obj.class);
                  return (
                    <div
                      key={idx}
                      className="flex items-center justify-between p-2 bg-gray-100 dark:bg-gray-700 rounded-lg"
                    >
                      <div className="flex items-center gap-2">
                        <div
                          className="w-4 h-4 rounded"
                          style={{ backgroundColor: classInfo?.color }}
                        />
                        <span className="text-sm font-medium text-gray-900 dark:text-white">
                          {classInfo?.name}
                        </span>
                      </div>
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        {(obj.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}