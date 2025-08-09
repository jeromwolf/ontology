'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import { 
  Camera, Upload, User, Smile, Frown, Meh, 
  Heart, AlertCircle, Activity, Download,
  Play, Pause, RefreshCw, Eye, EyeOff,
  Users, Scan, Brain, Sparkles
} from 'lucide-react'

interface FaceData {
  id: string
  boundingBox: { x: number; y: number; width: number; height: number }
  landmarks: { [key: string]: { x: number; y: number } }
  emotion: string
  emotionConfidence: number
  age: number
  gender: string
  genderConfidence: number
  features: number[]
  name?: string
}

interface StoredFace {
  id: string
  name: string
  features: number[]
  images: string[]
}

export default function FaceRecognitionSystem() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [useWebcam, setUseWebcam] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [detectedFaces, setDetectedFaces] = useState<FaceData[]>([])
  const [storedFaces, setStoredFaces] = useState<StoredFace[]>([])
  const [showLandmarks, setShowLandmarks] = useState(true)
  const [showEmotions, setShowEmotions] = useState(true)
  const [showAttributes, setShowAttributes] = useState(true)
  const [selectedFace, setSelectedFace] = useState<string | null>(null)
  
  const fileInputRef = useRef<HTMLInputElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const webcamStreamRef = useRef<MediaStream | null>(null)
  const animationRef = useRef<number | null>(null)

  // 감정 아이콘 매핑
  const emotionIcons: { [key: string]: JSX.Element } = {
    'happy': <Smile className="w-5 h-5 text-yellow-500" />,
    'sad': <Frown className="w-5 h-5 text-blue-500" />,
    'neutral': <Meh className="w-5 h-5 text-gray-500" />,
    'surprised': <AlertCircle className="w-5 h-5 text-orange-500" />,
    'angry': <Heart className="w-5 h-5 text-red-500" />
  }

  // 얼굴 특징점 정의
  const landmarkPoints = [
    'leftEye', 'rightEye', 'nose', 'mouth', 
    'leftEyebrow', 'rightEyebrow', 'jawline'
  ]

  // 이미지 업로드 처리
  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader()
      reader.onload = (event) => {
        setSelectedImage(event.target?.result as string)
        setUseWebcam(false)
        stopWebcam()
      }
      reader.readAsDataURL(file)
    }
  }

  // 웹캠 시작
  const startWebcam = async () => {
    // 카메라 권한 확인
    if (!confirm('이 기능은 카메라 권한이 필요합니다. 얼굴 인식을 위해 카메라를 사용하시겠습니까?')) {
      return
    }
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480 } 
      })
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        webcamStreamRef.current = stream
        setUseWebcam(true)
        setSelectedImage(null)
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
    setUseWebcam(false)
  }

  // 얼굴 감지 시뮬레이션
  const detectFaces = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // 캔버스 크기 설정
    canvas.width = 640
    canvas.height = 480

    // 이미지 또는 비디오 그리기
    if (selectedImage) {
      const img = new Image()
      img.onload = () => {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
        simulateFaceDetection(ctx)
      }
      img.src = selectedImage
    } else if (useWebcam && videoRef.current) {
      ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height)
      simulateFaceDetection(ctx)
    }
  }, [selectedImage, useWebcam])

  // 얼굴 감지 시뮬레이션 로직
  const simulateFaceDetection = (ctx: CanvasRenderingContext2D) => {
    // 시뮬레이션된 얼굴 위치 (실제로는 AI 모델이 감지)
    const faces: FaceData[] = []
    
    // 랜덤하게 1-3개의 얼굴 생성
    const numFaces = Math.floor(Math.random() * 3) + 1
    
    for (let i = 0; i < numFaces; i++) {
      const x = 100 + Math.random() * 300
      const y = 50 + Math.random() * 200
      const size = 80 + Math.random() * 60
      
      const face: FaceData = {
        id: `face-${Date.now()}-${i}`,
        boundingBox: { x, y, width: size, height: size * 1.2 },
        landmarks: generateLandmarks(x, y, size),
        emotion: ['happy', 'neutral', 'sad', 'surprised', 'angry'][Math.floor(Math.random() * 5)],
        emotionConfidence: 0.7 + Math.random() * 0.3,
        age: Math.floor(20 + Math.random() * 40),
        gender: Math.random() > 0.5 ? 'male' : 'female',
        genderConfidence: 0.8 + Math.random() * 0.2,
        features: generateFeatures()
      }
      
      faces.push(face)
    }
    
    setDetectedFaces(faces)
    drawDetections(ctx, faces)
  }

  // 랜드마크 생성
  const generateLandmarks = (x: number, y: number, size: number) => {
    return {
      leftEye: { x: x + size * 0.3, y: y + size * 0.4 },
      rightEye: { x: x + size * 0.7, y: y + size * 0.4 },
      nose: { x: x + size * 0.5, y: y + size * 0.6 },
      mouth: { x: x + size * 0.5, y: y + size * 0.8 },
      leftEyebrow: { x: x + size * 0.3, y: y + size * 0.3 },
      rightEyebrow: { x: x + size * 0.7, y: y + size * 0.3 },
      jawline: { x: x + size * 0.5, y: y + size * 1.1 }
    }
  }

  // 특징 벡터 생성 (시뮬레이션)
  const generateFeatures = () => {
    return Array.from({ length: 128 }, () => Math.random())
  }

  // 감지 결과 그리기
  const drawDetections = (ctx: CanvasRenderingContext2D, faces: FaceData[]) => {
    faces.forEach(face => {
      const { x, y, width, height } = face.boundingBox
      
      // 바운딩 박스
      ctx.strokeStyle = selectedFace === face.id ? '#14b8a6' : '#10b981'
      ctx.lineWidth = selectedFace === face.id ? 3 : 2
      ctx.strokeRect(x, y, width, height)
      
      // 얼굴 ID/이름
      ctx.fillStyle = 'white'
      ctx.fillRect(x, y - 25, width, 20)
      ctx.fillStyle = 'black'
      ctx.font = '12px Arial'
      ctx.fillText(face.name || `Person ${faces.indexOf(face) + 1}`, x + 5, y - 10)
      
      // 랜드마크
      if (showLandmarks) {
        ctx.fillStyle = '#ef4444'
        Object.values(face.landmarks).forEach(point => {
          ctx.beginPath()
          ctx.arc(point.x, point.y, 3, 0, Math.PI * 2)
          ctx.fill()
        })
      }
      
      // 감정 표시
      if (showEmotions) {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.9)'
        ctx.fillRect(x, y + height + 5, width, 20)
        ctx.fillStyle = 'black'
        ctx.font = '11px Arial'
        ctx.fillText(`${face.emotion} (${(face.emotionConfidence * 100).toFixed(0)}%)`, x + 5, y + height + 18)
      }
      
      // 속성 표시
      if (showAttributes) {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.9)'
        ctx.fillRect(x + width + 5, y, 80, 40)
        ctx.fillStyle = 'black'
        ctx.font = '10px Arial'
        ctx.fillText(`Age: ${face.age}`, x + width + 10, y + 15)
        ctx.fillText(`Gender: ${face.gender}`, x + width + 10, y + 30)
      }
    })
  }

  // 얼굴 등록
  const registerFace = () => {
    if (!selectedFace) return
    
    const face = detectedFaces.find(f => f.id === selectedFace)
    if (!face) return
    
    const name = prompt('이름을 입력하세요:')
    if (!name) return
    
    const newStoredFace: StoredFace = {
      id: face.id,
      name,
      features: face.features,
      images: [selectedImage || '']
    }
    
    setStoredFaces(prev => [...prev, newStoredFace])
    face.name = name
    setDetectedFaces([...detectedFaces])
  }

  // 유사도 계산 (코사인 유사도)
  const calculateSimilarity = (features1: number[], features2: number[]) => {
    let dotProduct = 0
    let norm1 = 0
    let norm2 = 0
    
    for (let i = 0; i < features1.length; i++) {
      dotProduct += features1[i] * features2[i]
      norm1 += features1[i] * features1[i]
      norm2 += features2[i] * features2[i]
    }
    
    return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2))
  }

  // 최적 매칭 찾기
  const findBestMatch = (features: number[], stored: StoredFace[]): { face: StoredFace; similarity: number } | null => {
    let bestMatch: { face: StoredFace; similarity: number } | null = null
    let maxSimilarity = 0
    
    stored.forEach(storedFace => {
      const similarity = calculateSimilarity(features, storedFace.features)
      if (similarity > maxSimilarity) {
        maxSimilarity = similarity
        bestMatch = { face: storedFace, similarity }
      }
    })
    
    return bestMatch
  }

  // 얼굴 매칭
  const matchFaces = () => {
    detectedFaces.forEach(face => {
      const bestMatch = findBestMatch(face.features, storedFaces)
      if (bestMatch !== null && bestMatch.similarity > 0.8) {
        face.name = bestMatch.face.name
      }
    })
    setDetectedFaces([...detectedFaces])
  }

  // 결과 내보내기
  const exportResults = () => {
    const results = {
      timestamp: new Date().toISOString(),
      faces: detectedFaces.map(face => ({
        id: face.id,
        name: face.name,
        boundingBox: face.boundingBox,
        emotion: face.emotion,
        emotionConfidence: face.emotionConfidence,
        age: face.age,
        gender: face.gender,
        landmarks: face.landmarks
      })),
      storedFaces: storedFaces.map(face => ({
        name: face.name,
        registeredAt: new Date().toISOString()
      }))
    }
    
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'face_recognition_results.json'
    a.click()
  }

  // 애니메이션 루프
  useEffect(() => {
    if (isProcessing && (selectedImage || useWebcam)) {
      const animate = () => {
        detectFaces()
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
  }, [isProcessing, selectedImage, useWebcam, detectFaces])

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
            이미지 업로드
          </button>
          
          <button
            onClick={useWebcam ? stopWebcam : startWebcam}
            className={`px-4 py-2 rounded-lg flex items-center gap-2 ${
              useWebcam 
                ? 'bg-red-600 text-white hover:bg-red-700' 
                : 'bg-blue-600 text-white hover:bg-blue-700'
            }`}
          >
            <Camera className="w-4 h-4" />
            {useWebcam ? '웹캠 중지' : '웹캠 시작'}
          </button>
          
          <button
            onClick={() => setIsProcessing(!isProcessing)}
            className={`px-4 py-2 rounded-lg flex items-center gap-2 ${
              isProcessing
                ? 'bg-orange-600 text-white hover:bg-orange-700'
                : 'bg-green-600 text-white hover:bg-green-700'
            }`}
            disabled={!selectedImage && !useWebcam}
          >
            {isProcessing ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            {isProcessing ? '중지' : '감지 시작'}
          </button>
          
          <div className="flex gap-2 ml-auto">
            <button
              onClick={() => setShowLandmarks(!showLandmarks)}
              className={`p-2 rounded ${showLandmarks ? 'bg-teal-600 text-white' : 'bg-gray-200 dark:bg-gray-600'}`}
              title="랜드마크 표시"
            >
              <Scan className="w-5 h-5" />
            </button>
            
            <button
              onClick={() => setShowEmotions(!showEmotions)}
              className={`p-2 rounded ${showEmotions ? 'bg-teal-600 text-white' : 'bg-gray-200 dark:bg-gray-600'}`}
              title="감정 표시"
            >
              <Smile className="w-5 h-5" />
            </button>
            
            <button
              onClick={() => setShowAttributes(!showAttributes)}
              className={`p-2 rounded ${showAttributes ? 'bg-teal-600 text-white' : 'bg-gray-200 dark:bg-gray-600'}`}
              title="속성 표시"
            >
              <User className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>

      {/* 메인 컨텐츠 */}
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
              autoPlay
              playsInline
              muted
              className="hidden"
            />
            
            {!selectedImage && !useWebcam && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center text-gray-400">
                  <Camera className="w-16 h-16 mx-auto mb-4" />
                  <p>이미지를 업로드하거나 웹캠을 시작하세요</p>
                </div>
              </div>
            )}
          </div>

          {/* 감지된 얼굴 목록 */}
          {detectedFaces.length > 0 && (
            <div className="mt-4 bg-white dark:bg-gray-800 rounded-lg p-4">
              <h3 className="font-semibold mb-3 flex items-center gap-2">
                <Users className="w-5 h-5 text-teal-600" />
                감지된 얼굴 ({detectedFaces.length})
              </h3>
              
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {detectedFaces.map((face, index) => (
                  <div
                    key={face.id}
                    onClick={() => setSelectedFace(face.id)}
                    className={`p-3 rounded-lg border cursor-pointer transition-all ${
                      selectedFace === face.id
                        ? 'border-teal-500 bg-teal-50 dark:bg-teal-900/20'
                        : 'border-gray-200 dark:border-gray-700 hover:border-teal-300'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium">
                        {face.name || `Person ${index + 1}`}
                      </span>
                      {emotionIcons[face.emotion]}
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      <p>나이: {face.age}세</p>
                      <p>성별: {face.gender === 'male' ? '남성' : '여성'}</p>
                      <p>감정: {face.emotion} ({(face.emotionConfidence * 100).toFixed(0)}%)</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* 사이드바 */}
        <div className="space-y-4">
          {/* 얼굴 등록 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <Brain className="w-5 h-5 text-teal-600" />
              얼굴 인식 데이터베이스
            </h3>
            
            <button
              onClick={registerFace}
              disabled={!selectedFace}
              className="w-full mb-3 px-4 py-2 bg-teal-600 text-white rounded-lg hover:bg-teal-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              선택한 얼굴 등록
            </button>
            
            <button
              onClick={matchFaces}
              disabled={detectedFaces.length === 0 || storedFaces.length === 0}
              className="w-full mb-4 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              얼굴 매칭 실행
            </button>
            
            {/* 등록된 얼굴 목록 */}
            <div className="space-y-2 max-h-60 overflow-y-auto">
              {storedFaces.length === 0 ? (
                <p className="text-sm text-gray-500 text-center py-4">
                  등록된 얼굴이 없습니다
                </p>
              ) : (
                storedFaces.map(face => (
                  <div
                    key={face.id}
                    className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-700 rounded"
                  >
                    <span className="text-sm font-medium">{face.name}</span>
                    <button
                      onClick={() => setStoredFaces(prev => prev.filter(f => f.id !== face.id))}
                      className="text-red-500 hover:text-red-700"
                    >
                      삭제
                    </button>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* 통계 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <Activity className="w-5 h-5 text-teal-600" />
              분석 통계
            </h3>
            
            <div className="space-y-3 text-sm">
              <div className="flex justify-between">
                <span>총 감지된 얼굴:</span>
                <span className="font-semibold">{detectedFaces.length}</span>
              </div>
              
              {detectedFaces.length > 0 && (
                <>
                  <div className="flex justify-between">
                    <span>평균 나이:</span>
                    <span className="font-semibold">
                      {Math.round(detectedFaces.reduce((sum, f) => sum + f.age, 0) / detectedFaces.length)}세
                    </span>
                  </div>
                  
                  <div className="flex justify-between">
                    <span>주요 감정:</span>
                    <span className="font-semibold">
                      {(() => {
                        const emotions = detectedFaces.reduce((acc, face) => {
                          acc[face.emotion] = (acc[face.emotion] || 0) + 1
                          return acc
                        }, {} as Record<string, number>)
                        return Object.entries(emotions).sort((a, b) => b[1] - a[1])[0]?.[0] || 'N/A'
                      })()}
                    </span>
                  </div>
                  
                  <div className="flex justify-between">
                    <span>성별 분포:</span>
                    <span className="font-semibold">
                      남 {detectedFaces.filter(f => f.gender === 'male').length} : 
                      여 {detectedFaces.filter(f => f.gender === 'female').length}
                    </span>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* 내보내기 */}
          <button
            onClick={exportResults}
            disabled={detectedFaces.length === 0}
            className="w-full px-4 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Download className="w-4 h-4" />
            결과 내보내기
          </button>
        </div>
      </div>

      {/* 정보 패널 */}
      <div className="bg-teal-50 dark:bg-teal-900/20 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
          <Sparkles className="w-6 h-6 text-teal-600" />
          얼굴 인식 기술
        </h3>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          이 시뮬레이터는 딥러닝 기반 얼굴 인식 기술의 핵심 기능들을 보여줍니다.
          얼굴 감지, 랜드마크 추출, 감정 분석, 나이/성별 예측, 그리고 얼굴 매칭을 체험할 수 있습니다.
        </p>
        <div className="grid md:grid-cols-4 gap-4 text-sm">
          <div>
            <h4 className="font-semibold mb-1">얼굴 감지</h4>
            <p className="text-gray-600 dark:text-gray-400">MTCNN, RetinaFace</p>
          </div>
          <div>
            <h4 className="font-semibold mb-1">특징 추출</h4>
            <p className="text-gray-600 dark:text-gray-400">FaceNet, ArcFace</p>
          </div>
          <div>
            <h4 className="font-semibold mb-1">감정 분석</h4>
            <p className="text-gray-600 dark:text-gray-400">FER, DeepFace</p>
          </div>
          <div>
            <h4 className="font-semibold mb-1">얼굴 매칭</h4>
            <p className="text-gray-600 dark:text-gray-400">1:1, 1:N 검증</p>
          </div>
        </div>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleImageUpload}
        className="hidden"
      />
    </div>
  )
}