'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import { 
  Camera, Upload, Play, Pause, Activity, 
  User, Users, Download, Settings, 
  BarChart, Target, Zap, TrendingUp,
  Video, Image as ImageIcon, RefreshCw
} from 'lucide-react'

interface Keypoint {
  name: string
  x: number
  y: number
  confidence: number
}

interface Pose {
  id: string
  keypoints: Keypoint[]
  score: number
  activity?: string
}

interface PoseMetrics {
  fps: number
  detectionTime: number
  poseCount: number
  avgConfidence: number
  activities: { [key: string]: number }
}

// COCO 포즈 키포인트 정의
const COCO_KEYPOINTS = [
  'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
  'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
  'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
  'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

// 스켈레톤 연결 정의
const SKELETON_CONNECTIONS = [
  ['left_eye', 'nose'], ['right_eye', 'nose'],
  ['left_ear', 'left_eye'], ['right_ear', 'right_eye'],
  ['left_shoulder', 'right_shoulder'],
  ['left_shoulder', 'left_elbow'], ['left_elbow', 'left_wrist'],
  ['right_shoulder', 'right_elbow'], ['right_elbow', 'right_wrist'],
  ['left_shoulder', 'left_hip'], ['right_shoulder', 'right_hip'],
  ['left_hip', 'right_hip'],
  ['left_hip', 'left_knee'], ['left_knee', 'left_ankle'],
  ['right_hip', 'right_knee'], ['right_knee', 'right_ankle']
]

export default function PoseEstimationTracker() {
  const [selectedSource, setSelectedSource] = useState<'image' | 'video' | 'webcam'>('image')
  const [selectedFile, setSelectedFile] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [detectedPoses, setDetectedPoses] = useState<Pose[]>([])
  const [metrics, setMetrics] = useState<PoseMetrics>({
    fps: 0,
    detectionTime: 0,
    poseCount: 0,
    avgConfidence: 0,
    activities: {}
  })
  
  // Settings
  const [showSkeleton, setShowSkeleton] = useState(true)
  const [showKeypoints, setShowKeypoints] = useState(true)
  const [minConfidence, setMinConfidence] = useState(0.5)
  const [smoothing, setSmoothing] = useState(0.5)
  const [multiPerson, setMultiPerson] = useState(true)
  
  const fileInputRef = useRef<HTMLInputElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const webcamStreamRef = useRef<MediaStream | null>(null)
  const animationRef = useRef<number | null>(null)
  const lastFrameTime = useRef<number>(0)
  const frameCount = useRef<number>(0)

  // 파일 업로드 처리
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    
    const reader = new FileReader()
    reader.onload = (event) => {
      const url = event.target?.result as string
      setSelectedFile(url)
      
      if (file.type.startsWith('image/')) {
        setSelectedSource('image')
      } else if (file.type.startsWith('video/')) {
        setSelectedSource('video')
      }
    }
    reader.readAsDataURL(file)
  }

  // 웹캠 시작
  const startWebcam = async () => {
    // 카메라 권한 확인
    if (!confirm('이 기능은 카메라 권한이 필요합니다. 포즈 감지를 위해 카메라를 사용하시겠습니까?')) {
      return
    }
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480 } 
      })
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        webcamStreamRef.current = stream
        setSelectedSource('webcam')
        setSelectedFile(null)
      }
    } catch (error) {
      console.error('웹캠 접근 오류:', error)
    }
  }

  // 웹캠 중지
  const stopWebcam = () => {
    if (webcamStreamRef.current) {
      webcamStreamRef.current.getTracks().forEach(track => track.stop())
      webcamStreamRef.current = null
    }
  }

  // 포즈 감지 시뮬레이션
  const detectPoses = useCallback(() => {
    const canvas = canvasRef.current
    const video = videoRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = 640
    canvas.height = 480

    // 소스에 따라 그리기
    if (selectedSource === 'image' && selectedFile) {
      const img = new Image()
      img.onload = () => {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
        const poses = simulatePoseDetection()
        drawPoses(ctx, poses)
        updateMetrics(poses)
      }
      img.src = selectedFile
    } else if ((selectedSource === 'video' || selectedSource === 'webcam') && video) {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
      const poses = simulatePoseDetection()
      drawPoses(ctx, poses)
      updateMetrics(poses)
    }
  }, [selectedSource, selectedFile])

  // 포즈 감지 시뮬레이션 (실제로는 ML 모델 사용)
  const simulatePoseDetection = (): Pose[] => {
    const poses: Pose[] = []
    const numPeople = multiPerson ? Math.floor(Math.random() * 3) + 1 : 1

    for (let i = 0; i < numPeople; i++) {
      const centerX = 150 + Math.random() * 340
      const centerY = 100 + Math.random() * 280
      const scale = 0.8 + Math.random() * 0.4

      const keypoints: Keypoint[] = COCO_KEYPOINTS.map(name => {
        let x = centerX
        let y = centerY
        
        // 키포인트별 상대 위치 설정
        switch (name) {
          case 'nose':
            y -= 80 * scale
            break
          case 'left_eye':
            x -= 15 * scale
            y -= 85 * scale
            break
          case 'right_eye':
            x += 15 * scale
            y -= 85 * scale
            break
          case 'left_ear':
            x -= 30 * scale
            y -= 80 * scale
            break
          case 'right_ear':
            x += 30 * scale
            y -= 80 * scale
            break
          case 'left_shoulder':
            x -= 40 * scale
            y -= 40 * scale
            break
          case 'right_shoulder':
            x += 40 * scale
            y -= 40 * scale
            break
          case 'left_elbow':
            x -= 50 * scale
            y += 10 * scale
            break
          case 'right_elbow':
            x += 50 * scale
            y += 10 * scale
            break
          case 'left_wrist':
            x -= 60 * scale
            y += 60 * scale
            break
          case 'right_wrist':
            x += 60 * scale
            y += 60 * scale
            break
          case 'left_hip':
            x -= 20 * scale
            y += 40 * scale
            break
          case 'right_hip':
            x += 20 * scale
            y += 40 * scale
            break
          case 'left_knee':
            x -= 25 * scale
            y += 90 * scale
            break
          case 'right_knee':
            x += 25 * scale
            y += 90 * scale
            break
          case 'left_ankle':
            x -= 30 * scale
            y += 140 * scale
            break
          case 'right_ankle':
            x += 30 * scale
            y += 140 * scale
            break
        }

        // 약간의 랜덤 움직임 추가
        x += (Math.random() - 0.5) * 10
        y += (Math.random() - 0.5) * 10

        return {
          name,
          x,
          y,
          confidence: 0.7 + Math.random() * 0.3
        }
      })

      const pose: Pose = {
        id: `pose-${i}`,
        keypoints,
        score: keypoints.reduce((sum, kp) => sum + kp.confidence, 0) / keypoints.length,
        activity: detectActivity(keypoints)
      }

      poses.push(pose)
    }

    return poses
  }

  // 활동 감지 (간단한 규칙 기반)
  const detectActivity = (keypoints: Keypoint[]): string => {
    const getKeypoint = (name: string) => keypoints.find(kp => kp.name === name)
    
    const leftWrist = getKeypoint('left_wrist')
    const rightWrist = getKeypoint('right_wrist')
    const leftShoulder = getKeypoint('left_shoulder')
    const rightShoulder = getKeypoint('right_shoulder')
    const leftHip = getKeypoint('left_hip')
    const rightHip = getKeypoint('right_hip')
    
    if (leftWrist && rightWrist && leftShoulder && rightShoulder) {
      // 손이 어깨 위에 있으면 "손들기"
      if (leftWrist.y < leftShoulder.y && rightWrist.y < rightShoulder.y) {
        return '손들기'
      }
      
      // 한 손만 위에 있으면 "한손들기"
      if (leftWrist.y < leftShoulder.y || rightWrist.y < rightShoulder.y) {
        return '한손들기'
      }
    }
    
    if (leftHip && rightHip) {
      const hipDistance = Math.abs(leftHip.y - rightHip.y)
      if (hipDistance > 50) {
        return '앉기'
      }
    }
    
    return '서있기'
  }

  // 포즈 그리기
  const drawPoses = (ctx: CanvasRenderingContext2D, poses: Pose[]) => {
    poses.forEach((pose, index) => {
      const color = `hsl(${index * 120}, 70%, 50%)`
      
      // 스켈레톤 그리기
      if (showSkeleton) {
        ctx.strokeStyle = color
        ctx.lineWidth = 3
        
        SKELETON_CONNECTIONS.forEach(([from, to]) => {
          const fromKp = pose.keypoints.find(kp => kp.name === from)
          const toKp = pose.keypoints.find(kp => kp.name === to)
          
          if (fromKp && toKp && fromKp.confidence > minConfidence && toKp.confidence > minConfidence) {
            ctx.beginPath()
            ctx.moveTo(fromKp.x, fromKp.y)
            ctx.lineTo(toKp.x, toKp.y)
            ctx.stroke()
          }
        })
      }
      
      // 키포인트 그리기
      if (showKeypoints) {
        pose.keypoints.forEach(kp => {
          if (kp.confidence > minConfidence) {
            ctx.fillStyle = color
            ctx.beginPath()
            ctx.arc(kp.x, kp.y, 5, 0, Math.PI * 2)
            ctx.fill()
            
            // 신뢰도 표시
            if (kp.confidence < 0.8) {
              ctx.fillStyle = 'rgba(255, 255, 255, 0.8)'
              ctx.font = '10px Arial'
              ctx.fillText((kp.confidence * 100).toFixed(0) + '%', kp.x + 8, kp.y - 8)
            }
          }
        })
      }
      
      // 활동 레이블
      if (pose.activity) {
        const nose = pose.keypoints.find(kp => kp.name === 'nose')
        if (nose) {
          ctx.fillStyle = 'rgba(0, 0, 0, 0.7)'
          ctx.fillRect(nose.x - 40, nose.y - 120, 80, 25)
          ctx.fillStyle = 'white'
          ctx.font = '14px Arial'
          ctx.textAlign = 'center'
          ctx.fillText(pose.activity, nose.x, nose.y - 100)
        }
      }
    })
  }

  // 메트릭 업데이트
  const updateMetrics = (poses: Pose[]) => {
    const now = Date.now()
    const deltaTime = now - lastFrameTime.current
    lastFrameTime.current = now
    
    frameCount.current++
    
    // FPS 계산
    const fps = deltaTime > 0 ? 1000 / deltaTime : 0
    
    // 평균 신뢰도
    const totalConfidence = poses.reduce((sum, pose) => 
      sum + pose.keypoints.reduce((kpSum, kp) => kpSum + kp.confidence, 0) / pose.keypoints.length, 0
    )
    const avgConfidence = poses.length > 0 ? totalConfidence / poses.length : 0
    
    // 활동 카운트
    const activities: { [key: string]: number } = {}
    poses.forEach(pose => {
      if (pose.activity) {
        activities[pose.activity] = (activities[pose.activity] || 0) + 1
      }
    })
    
    setMetrics({
      fps: Math.round(fps),
      detectionTime: deltaTime,
      poseCount: poses.length,
      avgConfidence,
      activities
    })
    
    setDetectedPoses(poses)
  }

  // 애니메이션 루프
  useEffect(() => {
    if (isProcessing && (selectedFile || selectedSource === 'webcam')) {
      const animate = () => {
        detectPoses()
        animationRef.current = requestAnimationFrame(animate)
      }
      animate()
    } else {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isProcessing, selectedFile, selectedSource, detectPoses])

  // 비디오 재생 제어
  useEffect(() => {
    if (videoRef.current && selectedSource === 'video') {
      if (isPlaying) {
        videoRef.current.play()
      } else {
        videoRef.current.pause()
      }
    }
  }, [isPlaying, selectedSource])

  // 결과 내보내기
  const exportResults = () => {
    const results = {
      timestamp: new Date().toISOString(),
      source: selectedSource,
      metrics,
      poses: detectedPoses.map(pose => ({
        id: pose.id,
        score: pose.score,
        activity: pose.activity,
        keypoints: pose.keypoints.filter(kp => kp.confidence > minConfidence)
      }))
    }
    
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'pose_estimation_results.json'
    a.click()
  }

  return (
    <div className="space-y-6">
      {/* 툴바 */}
      <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
        <div className="flex flex-wrap items-center gap-4">
          <button
            onClick={() => fileInputRef.current?.click()}
            className="px-4 py-2 bg-teal-600 text-white rounded-lg hover:bg-teal-700 flex items-center gap-2"
          >
            <Upload className="w-4 h-4" />
            파일 업로드
          </button>
          
          <button
            onClick={selectedSource === 'webcam' ? stopWebcam : startWebcam}
            className={`px-4 py-2 rounded-lg flex items-center gap-2 ${
              selectedSource === 'webcam'
                ? 'bg-red-600 text-white hover:bg-red-700'
                : 'bg-blue-600 text-white hover:bg-blue-700'
            }`}
          >
            <Camera className="w-4 h-4" />
            {selectedSource === 'webcam' ? '웹캠 중지' : '웹캠 시작'}
          </button>
          
          <button
            onClick={() => setIsProcessing(!isProcessing)}
            className={`px-4 py-2 rounded-lg flex items-center gap-2 ${
              isProcessing
                ? 'bg-orange-600 text-white hover:bg-orange-700'
                : 'bg-green-600 text-white hover:bg-green-700'
            }`}
            disabled={!selectedFile && selectedSource !== 'webcam'}
          >
            {isProcessing ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            {isProcessing ? '중지' : '추적 시작'}
          </button>
          
          {selectedSource === 'video' && (
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 flex items-center gap-2"
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              {isPlaying ? '일시정지' : '재생'}
            </button>
          )}
          
          <button
            onClick={exportResults}
            disabled={detectedPoses.length === 0}
            className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 flex items-center gap-2 disabled:opacity-50 ml-auto"
          >
            <Download className="w-4 h-4" />
            결과 내보내기
          </button>
        </div>
      </div>

      {/* 메인 콘텐츠 */}
      <div className="grid lg:grid-cols-3 gap-6">
        {/* 비디오/캔버스 영역 */}
        <div className="lg:col-span-2">
          <div className="bg-black rounded-lg overflow-hidden relative">
            <canvas
              ref={canvasRef}
              className="w-full"
              style={{ maxHeight: '480px' }}
            />
            <video
              ref={videoRef}
              src={selectedSource === 'video' ? selectedFile || '' : ''}
              autoPlay={selectedSource === 'webcam'}
              playsInline
              muted
              loop
              className="hidden"
            />
            
            {!selectedFile && selectedSource !== 'webcam' && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center text-gray-400">
                  <Video className="w-16 h-16 mx-auto mb-4" />
                  <p>이미지나 비디오를 업로드하거나</p>
                  <p>웹캠을 시작하세요</p>
                </div>
              </div>
            )}
            
            {/* 실시간 메트릭 오버레이 */}
            {isProcessing && (
              <div className="absolute top-4 left-4 bg-black bg-opacity-70 text-white p-3 rounded-lg text-sm">
                <div className="flex items-center gap-2 mb-1">
                  <Activity className="w-4 h-4 text-green-400" />
                  <span>FPS: {metrics.fps}</span>
                </div>
                <div className="flex items-center gap-2">
                  <Users className="w-4 h-4 text-blue-400" />
                  <span>감지된 사람: {metrics.poseCount}</span>
                </div>
              </div>
            )}
          </div>

          {/* 감지된 포즈 목록 */}
          {detectedPoses.length > 0 && (
            <div className="mt-4 bg-white dark:bg-gray-800 rounded-lg p-4">
              <h3 className="font-semibold mb-3 flex items-center gap-2">
                <Users className="w-5 h-5 text-teal-600" />
                감지된 포즈 ({detectedPoses.length})
              </h3>
              
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {detectedPoses.map((pose, index) => (
                  <div
                    key={pose.id}
                    className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium">Person {index + 1}</span>
                      <span className="text-sm text-gray-500">
                        {(pose.score * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="text-sm">
                      <p className="text-teal-600 dark:text-teal-400 font-medium">
                        {pose.activity}
                      </p>
                      <p className="text-gray-600 dark:text-gray-400">
                        키포인트: {pose.keypoints.filter(kp => kp.confidence > minConfidence).length}/{COCO_KEYPOINTS.length}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* 사이드바 */}
        <div className="space-y-4">
          {/* 설정 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <Settings className="w-5 h-5 text-teal-600" />
              추적 설정
            </h3>
            
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">스켈레톤 표시</label>
                <button
                  onClick={() => setShowSkeleton(!showSkeleton)}
                  className={`w-12 h-6 rounded-full transition-colors ${
                    showSkeleton ? 'bg-teal-600' : 'bg-gray-300'
                  }`}
                >
                  <div className={`w-5 h-5 bg-white rounded-full transition-transform ${
                    showSkeleton ? 'translate-x-6' : 'translate-x-0.5'
                  }`} />
                </button>
              </div>
              
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">키포인트 표시</label>
                <button
                  onClick={() => setShowKeypoints(!showKeypoints)}
                  className={`w-12 h-6 rounded-full transition-colors ${
                    showKeypoints ? 'bg-teal-600' : 'bg-gray-300'
                  }`}
                >
                  <div className={`w-5 h-5 bg-white rounded-full transition-transform ${
                    showKeypoints ? 'translate-x-6' : 'translate-x-0.5'
                  }`} />
                </button>
              </div>
              
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">다중 감지</label>
                <button
                  onClick={() => setMultiPerson(!multiPerson)}
                  className={`w-12 h-6 rounded-full transition-colors ${
                    multiPerson ? 'bg-teal-600' : 'bg-gray-300'
                  }`}
                >
                  <div className={`w-5 h-5 bg-white rounded-full transition-transform ${
                    multiPerson ? 'translate-x-6' : 'translate-x-0.5'
                  }`} />
                </button>
              </div>
              
              <div>
                <label className="flex items-center justify-between text-sm font-medium mb-1">
                  <span>최소 신뢰도</span>
                  <span>{(minConfidence * 100).toFixed(0)}%</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={minConfidence}
                  onChange={(e) => setMinConfidence(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
              
              <div>
                <label className="flex items-center justify-between text-sm font-medium mb-1">
                  <span>스무딩</span>
                  <span>{(smoothing * 100).toFixed(0)}%</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={smoothing}
                  onChange={(e) => setSmoothing(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
            </div>
          </div>

          {/* 실시간 통계 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <BarChart className="w-5 h-5 text-teal-600" />
              실시간 통계
            </h3>
            
            <div className="space-y-3 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">프레임 레이트:</span>
                <span className="font-semibold">{metrics.fps} FPS</span>
              </div>
              
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">감지 시간:</span>
                <span className="font-semibold">{metrics.detectionTime.toFixed(1)}ms</span>
              </div>
              
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">평균 신뢰도:</span>
                <span className="font-semibold">{(metrics.avgConfidence * 100).toFixed(1)}%</span>
              </div>
              
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">총 프레임:</span>
                <span className="font-semibold">{frameCount.current}</span>
              </div>
            </div>
            
            {Object.keys(metrics.activities).length > 0 && (
              <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                <h4 className="font-medium mb-2">활동 분포</h4>
                {Object.entries(metrics.activities).map(([activity, count]) => (
                  <div key={activity} className="flex justify-between text-sm mb-1">
                    <span className="text-gray-600 dark:text-gray-400">{activity}:</span>
                    <span className="font-semibold">{count}명</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* 모델 정보 */}
          <div className="bg-teal-50 dark:bg-teal-900/20 rounded-lg p-4">
            <h3 className="font-semibold mb-3 flex items-center gap-2">
              <Zap className="w-5 h-5 text-teal-600" />
              포즈 추정 기술
            </h3>
            <div className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <p>• <strong>OpenPose</strong>: 다중 사람 실시간 추정</p>
              <p>• <strong>PoseNet</strong>: 브라우저 기반 경량 모델</p>
              <p>• <strong>MediaPipe</strong>: Google의 고성능 솔루션</p>
              <p>• <strong>MMPose</strong>: 최신 연구 모델 집합</p>
            </div>
          </div>
        </div>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*,video/*"
        onChange={handleFileUpload}
        className="hidden"
      />
    </div>
  )
}