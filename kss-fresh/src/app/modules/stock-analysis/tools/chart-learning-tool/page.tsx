'use client';

import React, { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import { ArrowLeft, LineChart, Info, ChevronRight, ChevronLeft, CheckCircle, XCircle, RefreshCw, Trophy, BookOpen, TrendingUp, TrendingDown, Activity, Zap, Minus } from 'lucide-react';

interface Pattern {
  id: string;
  name: string;
  nameKo: string;
  description: string;
  type: 'bullish' | 'bearish' | 'neutral';
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  points: { x: number; y: number }[];
}

interface Question {
  id: number;
  pattern: Pattern;
  options: string[];
  correctAnswer: number;
  explanation: string;
}

export default function ChartLearningToolPage() {
  const [currentMode, setCurrentMode] = useState<'learn' | 'quiz' | 'practice'>('learn');
  const [selectedCategory, setSelectedCategory] = useState<'candlestick' | 'patterns' | 'indicators'>('candlestick');
  const [currentPatternIndex, setCurrentPatternIndex] = useState(0);
  const [quizScore, setQuizScore] = useState(0);
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null);
  const [showResult, setShowResult] = useState(false);
  const [userProgress, setUserProgress] = useState({
    completedLessons: [] as string[],
    quizScores: [] as number[],
    totalQuizzes: 0
  });
  
  // 기술적 분석 연습 상태
  const [practiceMode, setPracticeMode] = useState<'support-resistance' | 'moving-average' | 'volume' | 'indicators'>('support-resistance');
  const [practiceStep, setPracticeStep] = useState<'intro' | 'tutorial' | 'practice' | 'result'>('intro');
  const [tutorialStep, setTutorialStep] = useState(0);
  const [practiceScore, setPracticeScore] = useState(0);
  const [practiceData, setPracticeData] = useState<{ price: number; volume: number; time: number; date?: string }[]>([]);
  const [currentScenario, setCurrentScenario] = useState<'corona-crash' | 'earnings-surprise' | 'support-break' | null>(null);
  const [drawings, setDrawings] = useState<any[]>([]);
  const [selectedTool, setSelectedTool] = useState<'line' | 'horizontal' | 'trend' | null>(null);
  const [drawingTrend, setDrawingTrend] = useState<{ x1: number; y1: number } | null>(null);
  const [showIndicators, setShowIndicators] = useState({ ma20: false, ma50: false, volume: false, rsi: false, macd: false });
  const [showHint, setShowHint] = useState(false);
  const [practiceResults, setPracticeResults] = useState<{ correct: boolean; message: string }[]>([]);
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const quizCanvasRef = useRef<HTMLCanvasElement>(null);
  const practiceCanvasRef = useRef<HTMLCanvasElement>(null);
  const practiceVolumeCanvasRef = useRef<HTMLCanvasElement>(null);
  const practiceIndicatorCanvasRef = useRef<HTMLCanvasElement>(null);

  // 패턴 데이터
  const patterns: Pattern[] = [
    {
      id: 'doji',
      name: 'Doji',
      nameKo: '도지',
      description: '시가와 종가가 거의 같은 캔들로, 시장의 우유부단함을 나타냅니다.',
      type: 'neutral',
      difficulty: 'beginner',
      points: []
    },
    {
      id: 'hammer',
      name: 'Hammer',
      nameKo: '해머',
      description: '하락 추세 끝에 나타나는 반전 신호로, 긴 아래 꼬리와 작은 몸통이 특징입니다.',
      type: 'bullish',
      difficulty: 'beginner',
      points: []
    },
    {
      id: 'shooting-star',
      name: 'Shooting Star',
      nameKo: '유성',
      description: '상승 추세 끝에 나타나는 반전 신호로, 긴 위 꼬리와 작은 몸통이 특징입니다.',
      type: 'bearish',
      difficulty: 'beginner',
      points: []
    },
    {
      id: 'engulfing',
      name: 'Engulfing',
      nameKo: '장악형',
      description: '이전 캔들을 완전히 감싸는 캔들로, 강한 반전 신호입니다.',
      type: 'bullish',
      difficulty: 'intermediate',
      points: []
    },
    {
      id: 'head-shoulders',
      name: 'Head and Shoulders',
      nameKo: '헤드앤숄더',
      description: '상승 추세의 끝을 알리는 대표적인 반전 패턴입니다.',
      type: 'bearish',
      difficulty: 'advanced',
      points: []
    }
  ];

  // 퀴즈 문제 생성
  const generateQuizQuestions = (): Question[] => {
    return patterns.slice(0, 5).map((pattern, index) => ({
      id: index,
      pattern,
      options: [
        pattern.nameKo,
        patterns[(index + 1) % patterns.length].nameKo,
        patterns[(index + 2) % patterns.length].nameKo,
        patterns[(index + 3) % patterns.length].nameKo
      ].sort(() => Math.random() - 0.5),
      correctAnswer: 0, // Will be updated after shuffle
      explanation: pattern.description
    }));
  };

  const [quizQuestions] = useState<Question[]>(generateQuizQuestions());

  // 캔들스틱 그리기
  const drawCandlestick = (ctx: CanvasRenderingContext2D, x: number, open: number, high: number, low: number, close: number) => {
    const width = 20;
    const bodyHeight = Math.abs(close - open);
    const isGreen = close > open;
    
    // 그림자 그리기
    ctx.strokeStyle = '#666';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, high);
    ctx.lineTo(x, low);
    ctx.stroke();
    
    // 몸통 그리기
    ctx.fillStyle = isGreen ? '#22c55e' : '#ef4444';
    ctx.fillRect(x - width/2, Math.min(open, close), width, bodyHeight || 2);
    ctx.strokeRect(x - width/2, Math.min(open, close), width, bodyHeight || 2);
  };

  // 패턴 그리기 함수
  const drawPattern = (canvas: HTMLCanvasElement, patternId: string) => {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Clear canvas
    ctx.fillStyle = '#f9fafb';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw grid
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    for (let i = 0; i < canvas.width; i += 50) {
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i, canvas.height);
      ctx.stroke();
    }
    for (let i = 0; i < canvas.height; i += 50) {
      ctx.beginPath();
      ctx.moveTo(0, i);
      ctx.lineTo(canvas.width, i);
      ctx.stroke();
    }
    
    // Draw pattern based on ID
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    
    switch (patternId) {
      case 'doji':
        drawCandlestick(ctx, centerX, 150, 100, 200, 150);
        break;
      case 'hammer':
        drawCandlestick(ctx, centerX - 60, 170, 170, 170, 170);
        drawCandlestick(ctx, centerX, 160, 120, 200, 150);
        break;
      case 'shooting-star':
        drawCandlestick(ctx, centerX - 60, 130, 130, 130, 130);
        drawCandlestick(ctx, centerX, 150, 100, 180, 160);
        break;
      case 'engulfing':
        drawCandlestick(ctx, centerX - 30, 160, 150, 170, 140);
        drawCandlestick(ctx, centerX + 30, 130, 120, 180, 170);
        break;
      case 'head-shoulders':
        // Left shoulder
        drawCandlestick(ctx, 100, 150, 140, 160, 130);
        drawCandlestick(ctx, 130, 130, 120, 140, 120);
        // Head
        drawCandlestick(ctx, 160, 120, 110, 130, 100);
        drawCandlestick(ctx, 190, 100, 90, 110, 90);
        drawCandlestick(ctx, 220, 90, 80, 100, 100);
        // Right shoulder
        drawCandlestick(ctx, 250, 100, 90, 110, 120);
        drawCandlestick(ctx, 280, 120, 110, 130, 130);
        break;
    }
  };

  // 학습 모드 패턴 그리기
  useEffect(() => {
    if (currentMode === 'learn' && canvasRef.current) {
      drawPattern(canvasRef.current, patterns[currentPatternIndex].id);
    }
  }, [currentPatternIndex, currentMode]);

  // 퀴즈 모드 패턴 그리기
  useEffect(() => {
    if (currentMode === 'quiz' && quizCanvasRef.current && currentQuestion < quizQuestions.length) {
      drawPattern(quizCanvasRef.current, quizQuestions[currentQuestion].pattern.id);
    }
  }, [currentQuestion, currentMode, quizQuestions]);

  // 실습 모드 차트 그리기
  useEffect(() => {
    if (currentMode === 'practice' && practiceData.length > 0) {
      // 가격 차트 그리기
      if (practiceCanvasRef.current) {
        const ctx = practiceCanvasRef.current.getContext('2d');
        if (ctx) {
          // Clear canvas
          ctx.fillStyle = '#f9fafb';
          ctx.fillRect(0, 0, 800, 400);
          
          // 가격 데이터 그리기
          const minPrice = Math.min(...practiceData.map(d => d.price));
          const maxPrice = Math.max(...practiceData.map(d => d.price));
          const priceRange = maxPrice - minPrice;
          
          // 캔들스틱 그리기
          practiceData.forEach((data, i) => {
            const x = (i / practiceData.length) * 800 + 10;
            const y = 400 - ((data.price - minPrice) / priceRange) * 380;
            
            if (i > 0) {
              const prevX = ((i - 1) / practiceData.length) * 800 + 10;
              const prevY = 400 - ((practiceData[i - 1].price - minPrice) / priceRange) * 380;
              
              // 선 그리기
              ctx.strokeStyle = data.price > practiceData[i - 1].price ? '#22c55e' : '#ef4444';
              ctx.lineWidth = 2;
              ctx.beginPath();
              ctx.moveTo(prevX, prevY);
              ctx.lineTo(x, y);
              ctx.stroke();
            }
          });
          
          // 이동평균선 그리기
          if (showIndicators.ma20 && practiceData.length >= 20) {
            ctx.strokeStyle = '#3b82f6';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            for (let i = 19; i < practiceData.length; i++) {
              const ma20 = practiceData.slice(i - 19, i + 1).reduce((sum, d) => sum + d.price, 0) / 20;
              const x = (i / practiceData.length) * 800 + 10;
              const y = 400 - ((ma20 - minPrice) / priceRange) * 380;
              
              if (i === 19) ctx.moveTo(x, y);
              else ctx.lineTo(x, y);
            }
            ctx.stroke();
          }
          
          if (showIndicators.ma50 && practiceData.length >= 50) {
            ctx.strokeStyle = '#ef4444';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            for (let i = 49; i < practiceData.length; i++) {
              const ma50 = practiceData.slice(i - 49, i + 1).reduce((sum, d) => sum + d.price, 0) / 50;
              const x = (i / practiceData.length) * 800 + 10;
              const y = 400 - ((ma50 - minPrice) / priceRange) * 380;
              
              if (i === 49) ctx.moveTo(x, y);
              else ctx.lineTo(x, y);
            }
            ctx.stroke();
          }
          
          // 그린 선들 표시
          drawings.forEach((drawing, index) => {
            if (drawing.type === 'horizontal') {
              ctx.strokeStyle = drawing.color || '#6366f1';
              ctx.lineWidth = 2;
              ctx.setLineDash([5, 5]);
              ctx.beginPath();
              ctx.moveTo(0, drawing.y);
              ctx.lineTo(800, drawing.y);
              ctx.stroke();
              ctx.setLineDash([]);
              
              // 레이블 표시
              if (drawing.label) {
                const isSupport = drawing.label.includes('지지');
                ctx.fillStyle = isSupport ? '#22c55e' : '#ef4444';
                ctx.fillRect(5, drawing.y - 10, 100, 20);
                ctx.fillStyle = '#ffffff';
                ctx.font = 'bold 11px sans-serif';
                ctx.fillText(drawing.label, 10, drawing.y + 3);
              }
              
              // 힌트인 경우 반짝이는 효과
              if (drawing.isHint) {
                ctx.strokeStyle = '#fbbf24';
                ctx.lineWidth = 4;
                ctx.shadowColor = '#fbbf24';
                ctx.shadowBlur = 10;
                ctx.stroke();
                ctx.shadowBlur = 0;
              }
            } else if (drawing.type === 'trend') {
              ctx.strokeStyle = drawing.color || '#f59e0b';
              ctx.lineWidth = 2;
              ctx.beginPath();
              ctx.moveTo(drawing.x1, drawing.y1);
              ctx.lineTo(drawing.x2, drawing.y2);
              ctx.stroke();
              
              // 레이블 표시
              if (drawing.label) {
                const midX = (drawing.x1 + drawing.x2) / 2;
                const midY = (drawing.y1 + drawing.y2) / 2;
                ctx.fillStyle = drawing.color || '#f59e0b';
                ctx.fillRect(midX - 40, midY - 15, 80, 20);
                ctx.fillStyle = '#ffffff';
                ctx.font = 'bold 11px sans-serif';
                ctx.fillText(drawing.label, midX - 35, midY - 2);
              }
            }
          });
          
        }
      }
      
      // 거래량 차트 그리기
      if ((practiceMode === 'volume' || showIndicators.volume) && practiceVolumeCanvasRef.current) {
        const ctx = practiceVolumeCanvasRef.current.getContext('2d');
        if (ctx) {
          ctx.fillStyle = '#f9fafb';
          ctx.fillRect(0, 0, 800, 150);
          
          const maxVolume = Math.max(...practiceData.map(d => d.volume));
          
          practiceData.forEach((data, i) => {
            const x = (i / practiceData.length) * 800 + 10;
            const height = (data.volume / maxVolume) * 130;
            const width = (800 / practiceData.length) * 0.8;
            
            ctx.fillStyle = i > 0 && data.price > practiceData[i - 1].price ? '#22c55e' : '#ef4444';
            ctx.fillRect(x - width/2, 150 - height, width, height);
          });
        }
      }
      
      // 지표 차트 그리기
      if (practiceMode === 'indicators' && practiceIndicatorCanvasRef.current) {
        const ctx = practiceIndicatorCanvasRef.current.getContext('2d');
        if (ctx) {
          ctx.fillStyle = '#f9fafb';
          ctx.fillRect(0, 0, 800, 150);
          
          // RSI 그리기
          if (showIndicators.rsi && practiceData.length >= 14) {
            // RSI 계산 (간단한 버전)
            ctx.strokeStyle = '#8b5cf6';
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            for (let i = 14; i < practiceData.length; i++) {
              const rsi = 50 + (Math.random() - 0.5) * 40; // 시뮬레이션
              const x = (i / practiceData.length) * 800 + 10;
              const y = 150 - (rsi / 100) * 130;
              
              if (i === 14) ctx.moveTo(x, y);
              else ctx.lineTo(x, y);
            }
            ctx.stroke();
            
            // 과매수/과매도 선
            ctx.strokeStyle = '#999';
            ctx.lineWidth = 1;
            ctx.setLineDash([3, 3]);
            ctx.beginPath();
            ctx.moveTo(0, 150 - (70 / 100) * 130);
            ctx.lineTo(800, 150 - (70 / 100) * 130);
            ctx.moveTo(0, 150 - (30 / 100) * 130);
            ctx.lineTo(800, 150 - (30 / 100) * 130);
            ctx.stroke();
            ctx.setLineDash([]);
          }
        }
      }
    }
  }, [currentMode, practiceData, drawings, showIndicators, practiceMode]);

  // 학습 진도 저장
  const markLessonComplete = (patternId: string) => {
    if (!userProgress.completedLessons.includes(patternId)) {
      setUserProgress({
        ...userProgress,
        completedLessons: [...userProgress.completedLessons, patternId]
      });
    }
  };

  // 퀴즈 답변 처리
  const handleQuizAnswer = (answerIndex: number) => {
    setSelectedAnswer(answerIndex);
    setShowResult(true);
    
    const question = quizQuestions[currentQuestion];
    const correctIndex = question.options.findIndex(opt => opt === question.pattern.nameKo);
    
    if (answerIndex === correctIndex) {
      setQuizScore(quizScore + 1);
    }
  };

  // 다음 문제
  const nextQuestion = () => {
    if (currentQuestion < quizQuestions.length - 1) {
      setCurrentQuestion(currentQuestion + 1);
      setSelectedAnswer(null);
      setShowResult(false);
    } else {
      // 퀴즈 완료 - currentQuestion을 증가시켜 결과 화면으로 전환
      const finalScore = selectedAnswer !== null && 
        quizQuestions[currentQuestion].options[selectedAnswer] === quizQuestions[currentQuestion].pattern.nameKo
        ? quizScore : quizScore;
      
      setCurrentQuestion(currentQuestion + 1);
      setUserProgress({
        ...userProgress,
        quizScores: [...userProgress.quizScores, finalScore],
        totalQuizzes: userProgress.totalQuizzes + 1
      });
    }
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return 'text-green-600 bg-green-100';
      case 'intermediate': return 'text-blue-600 bg-blue-100';
      case 'advanced': return 'text-purple-600 bg-purple-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'bullish': return <TrendingUp className="w-4 h-4 text-green-500" />;
      case 'bearish': return <TrendingDown className="w-4 h-4 text-red-500" />;
      default: return <Activity className="w-4 h-4 text-gray-500" />;
    }
  };

  // 실제 시나리오 데이터 로드
  const loadRealScenario = (scenario: string) => {
    setDrawings([]); // 시나리오 변경 시 그려진 선들 초기화
    setCurrentScenario(scenario as any);
    setSelectedTool(null); // 도구 선택 초기화
    setDrawingTrend(null); // 추세선 그리기 상태 초기화
    setShowIndicators({ ma20: false, ma50: false, volume: false, rsi: false, macd: false }); // 지표 초기화
    setPracticeResults([]); // 연습 결과 초기화
    setShowHint(false); // 힌트 초기화
    
    let data = [];
    
    switch (scenario) {
      case 'corona-crash':
        // 삼성전자 2020년 코로나 시기 데이터 시뮬레이션
        let samsungPrice = 52000; // 시작 가격을 낮춤
        for (let i = 0; i < 60; i++) {
          const date = new Date(2020, 1, 1 + i);
          
          // 초기 안정기
          if (i < 20) {
            samsungPrice += (Math.random() - 0.5) * 300;
            // 52,000원 ~ 54,000원 사이 유지
            samsungPrice = Math.max(52000, Math.min(54000, samsungPrice));
          }
          // 2월 말 ~ 3월 폭락
          else if (i >= 20 && i < 35) {
            if (i < 28) {
              // 서서히 하락
              samsungPrice -= Math.random() * 800 + 200;
            } else {
              // 3월 중순 45,000원까지 급락
              if (samsungPrice > 45500) {
                samsungPrice -= Math.random() * 1500 + 500;
              }
              // 45,000원 근처에서 횡보
              if (samsungPrice < 45500) {
                samsungPrice = 45000 + Math.random() * 500;
              }
            }
          }
          // 3월 중순 이후 반등
          else if (i >= 35 && i < 50) {
            // 45,000원에서 강하게 반등
            if (i === 35) {
              samsungPrice = 45000 + Math.random() * 1000;
            } else {
              samsungPrice += Math.random() * 400 + 100;
            }
          }
          // 회복 추세
          else if (i >= 50) {
            samsungPrice += Math.random() * 300 + 100;
          }
          
          // 45,000원 지지선 효과
          if (samsungPrice < 45000) {
            samsungPrice = 45000 + Math.random() * 300;
          }
          
          data.push({
            price: samsungPrice,
            volume: i >= 20 && i < 40 ? Math.random() * 3000000 + 2000000 : Math.random() * 1500000 + 500000,
            time: i,
            date: date.toLocaleDateString()
          });
        }
        break;
        
      case 'earnings-surprise':
        // 카카오 실적 서프라이즈 시나리오
        let kakaoPrice = 80000;
        const supportLevel = 78000;
        const resistanceLevel = 85000;
        
        for (let i = 0; i < 60; i++) {
          // 박스권 횡보
          if (i < 40) {
            if (kakaoPrice < supportLevel + 1000) {
              kakaoPrice += Math.random() * 1000 + 200;
            } else if (kakaoPrice > resistanceLevel - 1000) {
              kakaoPrice -= Math.random() * 1000 + 200;
            } else {
              kakaoPrice += (Math.random() - 0.5) * 800;
            }
          }
          // 실적 발표 후 돌파
          else if (i === 40) {
            kakaoPrice = resistanceLevel + 2000;
          }
          // 상승 추세
          else {
            kakaoPrice += Math.random() * 1000;
          }
          
          data.push({
            price: kakaoPrice,
            volume: i === 40 ? 5000000 : Math.random() * 1000000 + 500000,
            time: i
          });
        }
        break;
    }
    
    setPracticeData(data);
  };

  // 정답 확인
  const checkAnswer = () => {
    if (!currentScenario || drawings.length === 0) return;
    
    let correct = false;
    let message = '';
    
    switch (currentScenario) {
      case 'corona-crash':
        // 45,000원 근처에 지지선을 그었는지 확인
        const supportDrawings = drawings.filter(d => 
          d.type === 'horizontal' && 
          Math.abs(d.y - 280) < 20 // y 좌표로 가격 추정
        );
        
        if (supportDrawings.length > 0) {
          correct = true;
          message = '정확합니다! 45,000원이 강력한 지지선 역할을 했습니다. 이후 이 가격에서 반등하여 회복했죠.';
        } else {
          message = '45,000원 근처를 다시 보세요. 3번 이상 반등한 곳이 있습니다.';
        }
        break;
        
      case 'earnings-surprise':
        // 78,000원 지지선과 85,000원 저항선을 찾았는지
        const resistanceDrawings = drawings.filter(d => 
          d.type === 'horizontal' && 
          Math.abs(d.y - 150) < 20
        );
        
        if (resistanceDrawings.length > 0) {
          correct = true;
          message = '훌륭합니다! 85,000원 저항선을 돌파하면서 상승 추세가 시작되었습니다.';
        } else {
          message = '차트 전반부에서 가격이 계속 부딪히던 상단 가격대를 찾아보세요.';
        }
        break;
    }
    
    setPracticeResults([...practiceResults, { correct, message }]);
    
    if (!correct) {
      setShowHint(true);
    }
  };

  // 힌트 표시
  const showAnswerHint = () => {
    if (!currentScenario) return;
    
    switch (currentScenario) {
      case 'corona-crash':
        // 45,000원에 반짝이는 선 추가
        setDrawings([...drawings, { 
          type: 'horizontal', 
          y: 280, 
          isHint: true 
        }]);
        break;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link 
                href="/modules/stock-analysis/tools"
                className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
              >
                <ArrowLeft className="w-5 h-5" />
                <span>도구 목록</span>
              </Link>
              <div className="h-6 w-px bg-gray-300 dark:bg-gray-700" />
              <h1 className="text-xl font-bold text-gray-900 dark:text-white">차트 기초 학습기</h1>
              <span className="px-2 py-1 bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400 rounded text-xs font-medium">
                Educational
              </span>
            </div>
            
            {/* Mode Selector & Progress Badge */}
            <div className="flex items-center gap-4">
              {/* Progress Badge */}
              <div className="flex items-center gap-2 px-3 py-1 bg-green-100 dark:bg-green-900/30 rounded-lg">
                <CheckCircle className="w-4 h-4 text-green-600 dark:text-green-400" />
                <span className="text-sm font-medium text-green-700 dark:text-green-300">
                  {userProgress.completedLessons.length}/{patterns.length} 완료
                </span>
              </div>
              
              {/* Mode Buttons */}
              <div className="flex items-center gap-2">
                {[
                  { id: 'learn', label: '학습', icon: BookOpen },
                  { id: 'practice', label: '실습', icon: Activity },
                  { id: 'quiz', label: '퀴즈', icon: Trophy }
                ].map(mode => {
                  const Icon = mode.icon;
                  return (
                    <button
                      key={mode.id}
                      onClick={() => setCurrentMode(mode.id as any)}
                      className={`px-4 py-2 rounded-lg font-medium transition-all flex items-center gap-2 ${
                        currentMode === mode.id
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200'
                      }`}
                    >
                      <Icon className="w-4 h-4" />
                      {mode.label}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {currentMode === 'learn' && (
          <div className="grid lg:grid-cols-3 gap-8">
            {/* Pattern List */}
            <div className="lg:col-span-1 space-y-4">
              <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  차트 패턴 목록
                </h2>
                <div className="space-y-2">
                  {patterns.map((pattern, index) => (
                    <button
                      key={pattern.id}
                      onClick={() => {
                        setCurrentPatternIndex(index);
                        markLessonComplete(pattern.id);
                      }}
                      className={`w-full p-3 rounded-lg text-left transition-all ${
                        currentPatternIndex === index
                          ? 'bg-blue-50 dark:bg-blue-900/20 border-2 border-blue-500'
                          : 'bg-gray-50 dark:bg-gray-700 border-2 border-transparent hover:bg-gray-100'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-medium text-gray-900 dark:text-white">
                          {pattern.nameKo}
                        </span>
                        {getTypeIcon(pattern.type)}
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-gray-500 dark:text-gray-400">
                          {pattern.name}
                        </span>
                        <span className={`text-xs px-2 py-0.5 rounded ${getDifficultyColor(pattern.difficulty)}`}>
                          {pattern.difficulty === 'beginner' && '초급'}
                          {pattern.difficulty === 'intermediate' && '중급'}
                          {pattern.difficulty === 'advanced' && '고급'}
                        </span>
                      </div>
                      {userProgress.completedLessons.includes(pattern.id) && (
                        <div className="mt-2 flex items-center gap-1 text-xs text-green-600">
                          <CheckCircle className="w-3 h-3" />
                          완료
                        </div>
                      )}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Pattern Display */}
            <div className="lg:col-span-2 space-y-6">
              <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xl font-bold text-gray-900 dark:text-white">
                    {patterns[currentPatternIndex].nameKo} 패턴
                  </h2>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setCurrentPatternIndex(Math.max(0, currentPatternIndex - 1))}
                      disabled={currentPatternIndex === 0}
                      className="p-2 rounded-lg bg-gray-100 dark:bg-gray-700 disabled:opacity-50 transition-colors"
                    >
                      <ChevronLeft className="w-5 h-5" />
                    </button>
                    <span className="text-sm font-medium px-3">
                      {currentPatternIndex + 1} / {patterns.length}
                    </span>
                    <button
                      onClick={() => setCurrentPatternIndex(Math.min(patterns.length - 1, currentPatternIndex + 1))}
                      disabled={currentPatternIndex === patterns.length - 1}
                      className="p-2 rounded-lg bg-gray-100 dark:bg-gray-700 disabled:opacity-50 transition-colors"
                    >
                      <ChevronRight className="w-5 h-5" />
                    </button>
                  </div>
                </div>

                {/* Canvas */}
                <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 mb-6">
                  <canvas 
                    ref={canvasRef}
                    width={400}
                    height={300}
                    className="w-full max-w-md mx-auto"
                  />
                </div>

                {/* Description */}
                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                  <div className="flex items-start gap-3">
                    <Info className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5" />
                    <div>
                      <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
                        패턴 설명
                      </h3>
                      <p className="text-gray-700 dark:text-gray-300">
                        {patterns[currentPatternIndex].description}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {currentMode === 'practice' && (
          <div className="space-y-6">
            {practiceStep === 'intro' && (
              <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-8 max-w-3xl mx-auto">
                <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 text-center">
                  실전 차트 분석 연습 💪
                </h2>
                
                <div className="grid md:grid-cols-2 gap-4 mb-8">
                  <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
                    <h3 className="font-bold text-lg mb-3">실제 시나리오로 배우기</h3>
                    <ul className="space-y-2 text-sm">
                      <li className="flex items-start gap-2">
                        <span className="text-blue-500">✓</span>
                        <span>삼성전자 코로나 폭락장</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-blue-500">✓</span>
                        <span>카카오 실적 서프라이즈</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-blue-500">✓</span>
                        <span>테슬라 지지선 돌파</span>
                      </li>
                    </ul>
                  </div>
                  
                  <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
                    <h3 className="font-bold text-lg mb-3">실습할 기술들</h3>
                    <ul className="space-y-2 text-sm">
                      <li className="flex items-start gap-2">
                        <span className="text-green-500">✓</span>
                        <span>지지선/저항선 찾기</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-green-500">✓</span>
                        <span>이동평균선 활용</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-green-500">✓</span>
                        <span>RSI, MACD 해석</span>
                      </li>
                    </ul>
                  </div>
                </div>
                
                <button
                  onClick={() => setPracticeStep('tutorial')}
                  className="w-full py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium"
                >
                  시작하기
                </button>
              </div>
            )}

            {practiceStep === 'tutorial' && (
              <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-8 max-w-4xl mx-auto">
                <div className="mb-6">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="text-xl font-bold">튜토리얼 {tutorialStep + 1}/3</h2>
                    <div className="flex gap-2">
                      {[0, 1, 2].map(i => (
                        <div
                          key={i}
                          className={`w-2 h-2 rounded-full ${
                            i === tutorialStep ? 'bg-blue-600' : 'bg-gray-300'
                          }`}
                        />
                      ))}
                    </div>
                  </div>
                  
                  {tutorialStep === 0 && (
                    <div className="space-y-4">
                      <h3 className="text-lg font-semibold">수평선과 추세선의 차이와 용도</h3>
                      
                      <div className="grid md:grid-cols-2 gap-4">
                        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                          <h4 className="font-bold mb-2 flex items-center gap-2">
                            <Minus className="w-5 h-5" />
                            수평선 (지지/저항선)
                          </h4>
                          <p className="text-sm mb-3">
                            <strong>용도:</strong> 특정 가격대에서 반복적으로 멈추는 곳
                          </p>
                          <ul className="text-sm space-y-1">
                            <li>• 가격이 여러 번 반등한 곳 = <span className="text-green-600 font-semibold">지지선</span></li>
                            <li>• 가격이 여러 번 막힌 곳 = <span className="text-red-600 font-semibold">저항선</span></li>
                            <li>• <strong>예:</strong> "삼성전자가 45,000원에서 3번 반등"</li>
                          </ul>
                        </div>
                        
                        <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-4">
                          <h4 className="font-bold mb-2 flex items-center gap-2">
                            <TrendingUp className="w-5 h-5" />
                            추세선 (경사선)
                          </h4>
                          <p className="text-sm mb-3">
                            <strong>용도:</strong> 가격의 전체적인 방향성 파악
                          </p>
                          <ul className="text-sm space-y-1">
                            <li>• 저점들을 연결 = <span className="text-green-600 font-semibold">상승 추세선</span></li>
                            <li>• 고점들을 연결 = <span className="text-red-600 font-semibold">하락 추세선</span></li>
                            <li>• <strong>예:</strong> "3개월간 꾸준히 상승 중"</li>
                          </ul>
                        </div>
                      </div>
                      
                      <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
                        <h4 className="font-semibold mb-2">🎯 실전 활용법</h4>
                        <div className="grid md:grid-cols-2 gap-3 text-sm">
                          <div>
                            <p className="font-semibold text-blue-600 mb-1">수평선 활용:</p>
                            <ul className="space-y-1">
                              <li>✓ 지지선 근처에서 매수 고려</li>
                              <li>✓ 저항선 근처에서 매도 고려</li>
                              <li>✓ 돌파 시 추세 전환 신호</li>
                            </ul>
                          </div>
                          <div>
                            <p className="font-semibold text-orange-600 mb-1">추세선 활용:</p>
                            <ul className="space-y-1">
                              <li>✓ 추세선 따라 매매 (추세 추종)</li>
                              <li>✓ 추세선 이탈 시 손절</li>
                              <li>✓ 장기 투자 방향 결정</li>
                            </ul>
                          </div>
                        </div>
                      </div>
                      
                      <div className="bg-gray-100 dark:bg-gray-700 rounded-lg p-4">
                        <p className="text-sm">
                          <strong>💡 핵심:</strong> 수평선은 "어느 가격에서" 사고팔지를 알려주고, 
                          추세선은 "어느 방향으로" 움직이는지를 알려줍니다!
                        </p>
                      </div>
                    </div>
                  )}
                  
                  {tutorialStep === 1 && (
                    <div className="space-y-4">
                      <h3 className="text-lg font-semibold">이동평균선 활용법</h3>
                      <div className="grid md:grid-cols-2 gap-4">
                        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                          <h4 className="font-semibold mb-2">MA20 (20일선)</h4>
                          <p className="text-sm">단기 추세를 보여줍니다. 가격이 20일선 위에 있으면 단기 상승세!</p>
                        </div>
                        <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-4">
                          <h4 className="font-semibold mb-2">MA50 (50일선)</h4>
                          <p className="text-sm">중기 추세를 보여줍니다. 가격이 50일선 아래면 중기 하락세!</p>
                        </div>
                      </div>
                      <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
                        <p className="text-sm">
                          🔍 <strong>골든크로스:</strong> 20일선이 50일선을 위로 돌파 = 매수 신호<br/>
                          🔍 <strong>데드크로스:</strong> 20일선이 50일선을 아래로 돌파 = 매도 신호
                        </p>
                      </div>
                    </div>
                  )}
                  
                  {tutorialStep === 2 && (
                    <div className="space-y-4">
                      <h3 className="text-lg font-semibold">RSI와 MACD 쉽게 이해하기</h3>
                      <div className="space-y-3">
                        <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
                          <h4 className="font-semibold mb-2">RSI (상대강도지수)</h4>
                          <p className="text-sm mb-2">주식의 "체온계"라고 생각하세요! 🌡️</p>
                          <ul className="text-sm space-y-1">
                            <li>• 70 이상: 과열! (곧 식을 수 있어요)</li>
                            <li>• 30-70: 정상 체온</li>
                            <li>• 30 이하: 저체온! (곧 회복할 수 있어요)</li>
                          </ul>
                        </div>
                        <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
                          <h4 className="font-semibold mb-2">MACD</h4>
                          <p className="text-sm mb-2">추세 전환을 알려주는 "신호등"입니다! 🚦</p>
                          <ul className="text-sm space-y-1">
                            <li>• MACD선이 시그널선 위로: 청신호 (상승)</li>
                            <li>• MACD선이 시그널선 아래로: 적신호 (하락)</li>
                          </ul>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
                
                <div className="flex justify-between">
                  <button
                    onClick={() => tutorialStep > 0 && setTutorialStep(tutorialStep - 1)}
                    disabled={tutorialStep === 0}
                    className="px-6 py-2 border border-gray-300 rounded-lg disabled:opacity-50"
                  >
                    이전
                  </button>
                  <button
                    onClick={() => {
                      if (tutorialStep < 2) {
                        setTutorialStep(tutorialStep + 1);
                      } else {
                        setPracticeStep('practice');
                        loadRealScenario('corona-crash');
                      }
                    }}
                    className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                  >
                    {tutorialStep < 2 ? '다음' : '실습 시작'}
                  </button>
                </div>
              </div>
            )}

            {practiceStep === 'practice' && (
              <div className="grid lg:grid-cols-4 gap-6">
                {/* Control Panel */}
                <div className="lg:col-span-1 space-y-4">
                  <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-4">
                    <h3 className="font-semibold mb-4">시나리오 선택</h3>
                    <div className="space-y-2">
                      <button
                        onClick={() => loadRealScenario('corona-crash')}
                        className={`w-full p-3 rounded-lg text-left ${
                          currentScenario === 'corona-crash'
                            ? 'bg-blue-50 dark:bg-blue-900/20 border-2 border-blue-500'
                            : 'bg-gray-50 dark:bg-gray-700 hover:bg-gray-100'
                        }`}
                      >
                        <div className="font-medium">삼성전자 코로나</div>
                        <div className="text-xs text-gray-500">지지선 찾기</div>
                      </button>
                      <button
                        onClick={() => loadRealScenario('earnings-surprise')}
                        className={`w-full p-3 rounded-lg text-left ${
                          currentScenario === 'earnings-surprise'
                            ? 'bg-blue-50 dark:bg-blue-900/20 border-2 border-blue-500'
                            : 'bg-gray-50 dark:bg-gray-700 hover:bg-gray-100'
                        }`}
                      >
                        <div className="font-medium">카카오 실적발표</div>
                        <div className="text-xs text-gray-500">저항선 돌파</div>
                      </button>
                    </div>
                  </div>

                  <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-4">
                    <h3 className="font-semibold mb-4">선 표시하기</h3>
                    <div className="space-y-2">
                      <button
                        onClick={() => {
                          if (!practiceData.length) return;
                          
                          const minPrice = Math.min(...practiceData.map(d => d.price));
                          const maxPrice = Math.max(...practiceData.map(d => d.price));
                          const priceRange = maxPrice - minPrice;
                          
                          // 삼성전자 코로나 시나리오: 45,000원 지지선 표시
                          if (currentScenario === 'corona-crash') {
                            // y = 400 - ((price - minPrice) / priceRange) * 380
                            const y = 400 - ((45000 - minPrice) / priceRange) * 380;
                            setDrawings([...drawings, { 
                              type: 'horizontal', 
                              y: y, // 정확한 45,000원 위치
                              label: '45,000원 지지선',
                              color: '#22c55e'
                            }]);
                          }
                          // 카카오 실적 시나리오: 85,000원 저항선 표시
                          else if (currentScenario === 'earnings-surprise') {
                            const y = 400 - ((85000 - minPrice) / priceRange) * 380;
                            setDrawings([...drawings, { 
                              type: 'horizontal', 
                              y: y, // 정확한 85,000원 위치
                              label: '85,000원 저항선',
                              color: '#ef4444'
                            }]);
                          }
                        }}
                        className="w-full p-3 rounded-lg text-left bg-blue-600 text-white hover:bg-blue-700"
                      >
                        <div className="flex items-center gap-2 mb-1">
                          <Minus className="w-4 h-4" />
                          <span className="font-medium">수평선 표시</span>
                        </div>
                        <div className="text-xs opacity-80">
                          지지/저항 가격대 보기
                        </div>
                      </button>
                      <button
                        onClick={() => {
                          if (!practiceData.length) return;
                          
                          const minPrice = Math.min(...practiceData.map(d => d.price));
                          const maxPrice = Math.max(...practiceData.map(d => d.price));
                          const priceRange = maxPrice - minPrice;
                          
                          // 삼성전자 코로나 시나리오: 상승 추세선
                          if (currentScenario === 'corona-crash') {
                            // 저점들을 찾아서 추세선 그리기
                            let lowIndex1 = 30; // 3월 중순쯤
                            let lowIndex2 = 50; // 4월 초쯤
                            
                            // 실제 저점 찾기
                            for (let i = 25; i < 35; i++) {
                              if (practiceData[i] && practiceData[i].price < practiceData[lowIndex1].price) {
                                lowIndex1 = i;
                              }
                            }
                            for (let i = 45; i < 55; i++) {
                              if (practiceData[i] && practiceData[i].price < practiceData[lowIndex2].price) {
                                lowIndex2 = i;
                              }
                            }
                            
                            const x1 = (lowIndex1 / practiceData.length) * 800 + 10;
                            const y1 = 400 - ((practiceData[lowIndex1].price - minPrice) / priceRange) * 380;
                            const x2 = (lowIndex2 / practiceData.length) * 800 + 10;
                            const y2 = 400 - ((practiceData[lowIndex2].price - minPrice) / priceRange) * 380;
                            
                            setDrawings([...drawings, { 
                              type: 'trend',
                              x1: x1, y1: y1,  // 첫 번째 저점
                              x2: x2, y2: y2,  // 두 번째 저점
                              label: '상승 추세선',
                              color: '#3b82f6'
                            }]);
                          }
                          // 카카오 실적 시나리오: 박스권 상단선
                          else if (currentScenario === 'earnings-surprise') {
                            // 85,000원 저항선과 같은 높이에 수평 추세선
                            const y = 400 - ((85000 - minPrice) / priceRange) * 380;
                            setDrawings([...drawings, { 
                              type: 'trend',
                              x1: 100, y1: y,
                              x2: 320, y2: y,
                              label: '박스권 상단',
                              color: '#f59e0b'
                            }]);
                          }
                        }}
                        className="w-full p-3 rounded-lg text-left bg-orange-600 text-white hover:bg-orange-700"
                      >
                        <div className="flex items-center gap-2 mb-1">
                          <TrendingUp className="w-4 h-4" />
                          <span className="font-medium">추세선 표시</span>
                        </div>
                        <div className="text-xs opacity-80">
                          가격 방향성 보기
                        </div>
                      </button>
                    </div>
                    
                    <div className="mt-4 p-3 bg-gray-50 dark:bg-gray-900 rounded-lg">
                      <p className="text-xs text-gray-600 dark:text-gray-400">
                        <strong>💡 학습 포인트:</strong> 
                        {currentScenario === 'corona-crash' ? 
                          "45,000원이 강력한 지지선 역할을 했습니다." : 
                          "85,000원이 저항선으로 작용했다가 돌파되었습니다."}
                      </p>
                    </div>
                    
                    {drawings.length > 0 && (
                      <button
                        onClick={() => setDrawings([])}
                        className="w-full mt-2 p-2 text-sm bg-gray-200 dark:bg-gray-700 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600"
                      >
                        모든 선 지우기
                      </button>
                    )}
                  </div>

                  <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-4">
                    <h3 className="font-semibold mb-4">지표</h3>
                    <div className="space-y-2">
                      <label className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={showIndicators.ma20}
                          onChange={(e) => setShowIndicators({...showIndicators, ma20: e.target.checked})}
                          className="rounded"
                        />
                        <span className="text-sm">MA20 (20일선)</span>
                      </label>
                      <label className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={showIndicators.ma50}
                          onChange={(e) => setShowIndicators({...showIndicators, ma50: e.target.checked})}
                          className="rounded"
                        />
                        <span className="text-sm">MA50 (50일선)</span>
                      </label>
                      <label className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={showIndicators.volume}
                          onChange={(e) => setShowIndicators({...showIndicators, volume: e.target.checked})}
                          className="rounded"
                        />
                        <span className="text-sm">거래량</span>
                      </label>
                    </div>
                    
                    {(showIndicators.ma20 || showIndicators.ma50) && (
                      <div className="mt-3 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                        <p className="text-xs text-blue-700 dark:text-blue-300">
                          <strong>💡 알아두기:</strong><br/>
                          MA50(50일선)이 더 짧게 보이는 이유는 50일치 데이터가 필요해서 차트 뒤쪽부터 시작되기 때문입니다.<br/>
                          <span className="text-blue-600 dark:text-blue-400">• MA20: 20일째부터 표시</span><br/>
                          <span className="text-red-600 dark:text-red-400">• MA50: 50일째부터 표시</span>
                        </p>
                      </div>
                    )}
                  </div>

                  <div className="space-y-2">
                    <button
                      onClick={checkAnswer}
                      className="w-full py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 font-medium"
                    >
                      정답 확인
                    </button>
                    {showHint && (
                      <button
                        onClick={showAnswerHint}
                        className="w-full py-2 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700 font-medium"
                      >
                        힌트 보기
                      </button>
                    )}
                  </div>
                </div>

                {/* Chart Area */}
                <div className="lg:col-span-3 space-y-4">
                  {currentScenario && (
                    <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                      <h3 className="font-semibold mb-2">
                        {currentScenario === 'corona-crash' && '🎯 미션: 삼성전자가 반등한 지지선을 찾아보세요!'}
                        {currentScenario === 'earnings-surprise' && '🎯 미션: 카카오가 돌파한 저항선을 찾아보세요!'}
                      </h3>
                      <div className="text-sm text-gray-600 dark:text-gray-400 space-y-2">
                        <p>
                          {currentScenario === 'corona-crash' && (
                            <>
                              <strong>수평선 연습:</strong> 45,000원 근처에서 3번 이상 반등한 곳을 찾아 수평선을 그어보세요.
                              <br />
                              <span className="text-xs">힌트: 3월 중순에 주목하세요!</span>
                            </>
                          )}
                          {currentScenario === 'earnings-surprise' && (
                            <>
                              <strong>수평선 연습:</strong> 85,000원 근처에서 계속 막히던 저항선을 찾아보세요.
                              <br />
                              <span className="text-xs">힌트: 실적 발표 전 40일간의 움직임을 보세요!</span>
                            </>
                          )}
                        </p>
                        <p className="pt-2 border-t border-gray-300 dark:border-gray-600">
                          <strong>추세선 연습:</strong> 전체적인 가격 흐름의 방향을 추세선으로 표시해보세요.
                          <br />
                          <span className="text-xs">
                            {currentScenario === 'corona-crash' ? 
                              "팁: 3월 저점들을 연결하면 상승 추세선이 보입니다." : 
                              "팁: 박스권 상단의 고점들을 연결해보세요."}
                          </span>
                        </p>
                      </div>
                    </div>
                  )}

                  <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-4">
                    <div className="bg-gray-100 dark:bg-gray-900 rounded-lg p-2">
                      <canvas
                        ref={practiceCanvasRef}
                        width={800}
                        height={400}
                        className="w-full"
                      />
                    </div>
                    
                    {showIndicators.volume && (
                      <div className="mt-4 bg-gray-100 dark:bg-gray-900 rounded-lg p-2">
                        <canvas
                          ref={practiceVolumeCanvasRef}
                          width={800}
                          height={150}
                          className="w-full"
                        />
                      </div>
                    )}
                  </div>

                  {practiceResults.length > 0 && (
                    <div className="space-y-2">
                      {practiceResults.map((result, i) => (
                        <div
                          key={i}
                          className={`p-4 rounded-lg ${
                            result.correct
                              ? 'bg-green-50 dark:bg-green-900/20 border border-green-300'
                              : 'bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-300'
                          }`}
                        >
                          <div className="flex items-start gap-2">
                            {result.correct ? (
                              <CheckCircle className="w-5 h-5 text-green-600 mt-0.5" />
                            ) : (
                              <Info className="w-5 h-5 text-yellow-600 mt-0.5" />
                            )}
                            <p className="text-sm">{result.message}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {currentMode === 'quiz' && (
          <div className="max-w-3xl mx-auto">
            {currentQuestion < quizQuestions.length ? (
              <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-8">
                <div className="mb-6">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="text-xl font-bold text-gray-900 dark:text-white">
                      문제 {currentQuestion + 1} / {quizQuestions.length}
                    </h2>
                    <div className="text-sm text-gray-500">
                      점수: {quizScore} / {currentQuestion}
                    </div>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div 
                      className="bg-blue-600 h-2 rounded-full transition-all"
                      style={{ width: `${((currentQuestion + 1) / quizQuestions.length) * 100}%` }}
                    />
                  </div>
                </div>

                {/* Quiz Canvas */}
                <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 mb-6">
                  <canvas 
                    ref={quizCanvasRef}
                    width={400}
                    height={300}
                    className="w-full max-w-md mx-auto"
                  />
                </div>

                <h3 className="text-lg font-semibold mb-4 text-center">
                  이 패턴의 이름은 무엇일까요?
                </h3>

                {/* Options */}
                <div className="grid grid-cols-2 gap-4 mb-6">
                  {quizQuestions[currentQuestion].options.map((option, index) => (
                    <button
                      key={index}
                      onClick={() => handleQuizAnswer(index)}
                      disabled={showResult}
                      className={`p-4 rounded-lg border-2 transition-all ${
                        showResult
                          ? index === quizQuestions[currentQuestion].options.findIndex(opt => opt === quizQuestions[currentQuestion].pattern.nameKo)
                            ? 'border-green-500 bg-green-50 dark:bg-green-900/20'
                            : selectedAnswer === index
                            ? 'border-red-500 bg-red-50 dark:bg-red-900/20'
                            : 'border-gray-200 dark:border-gray-700'
                          : selectedAnswer === index
                          ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                          : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
                      }`}
                    >
                      <span className="font-medium">{option}</span>
                    </button>
                  ))}
                </div>

                {/* Result */}
                {showResult && (
                  <div className={`p-4 rounded-lg mb-6 ${
                    selectedAnswer === quizQuestions[currentQuestion].options.findIndex(opt => opt === quizQuestions[currentQuestion].pattern.nameKo)
                      ? 'bg-green-50 dark:bg-green-900/20'
                      : 'bg-red-50 dark:bg-red-900/20'
                  }`}>
                    <div className="flex items-start gap-3">
                      {selectedAnswer === quizQuestions[currentQuestion].options.findIndex(opt => opt === quizQuestions[currentQuestion].pattern.nameKo) ? (
                        <CheckCircle className="w-5 h-5 text-green-600 mt-0.5" />
                      ) : (
                        <XCircle className="w-5 h-5 text-red-600 mt-0.5" />
                      )}
                      <div>
                        <p className="font-semibold mb-1">
                          {selectedAnswer === quizQuestions[currentQuestion].options.findIndex(opt => opt === quizQuestions[currentQuestion].pattern.nameKo)
                            ? '정답입니다!'
                            : '틀렸습니다.'}
                        </p>
                        <p className="text-sm text-gray-700 dark:text-gray-300">
                          {quizQuestions[currentQuestion].explanation}
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                <button
                  onClick={nextQuestion}
                  disabled={!showResult}
                  className="w-full py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                >
                  {currentQuestion === quizQuestions.length - 1 ? '결과 보기' : '다음 문제'}
                </button>
              </div>
            ) : (
              // Quiz Results
              <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-8 text-center">
                <Trophy className="w-16 h-16 text-yellow-500 mx-auto mb-4" />
                <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                  퀴즈 완료!
                </h2>
                <p className="text-4xl font-bold text-blue-600 mb-6">
                  {quizScore} / {quizQuestions.length}
                </p>
                
                <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 mb-6">
                  <div className="grid grid-cols-3 gap-4 text-center">
                    <div>
                      <p className="text-2xl font-bold text-green-600">
                        {Math.round((quizScore / quizQuestions.length) * 100)}%
                      </p>
                      <p className="text-sm text-gray-500">정답률</p>
                    </div>
                    <div>
                      <p className="text-2xl font-bold text-blue-600">
                        {userProgress.totalQuizzes}
                      </p>
                      <p className="text-sm text-gray-500">총 퀴즈 수</p>
                    </div>
                    <div>
                      <p className="text-2xl font-bold text-purple-600">
                        {userProgress.quizScores.length > 0
                          ? Math.round(userProgress.quizScores.reduce((a, b) => a + b, 0) / userProgress.quizScores.length)
                          : 0}
                      </p>
                      <p className="text-sm text-gray-500">평균 점수</p>
                    </div>
                  </div>
                </div>

                <div className="flex gap-4">
                  <button
                    onClick={() => {
                      setCurrentQuestion(0);
                      setQuizScore(0);
                      setSelectedAnswer(null);
                      setShowResult(false);
                    }}
                    className="flex-1 py-3 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
                  >
                    <RefreshCw className="w-5 h-5 inline mr-2" />
                    다시 도전
                  </button>
                  <button
                    onClick={() => setCurrentMode('learn')}
                    className="flex-1 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                  >
                    학습 모드로
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}