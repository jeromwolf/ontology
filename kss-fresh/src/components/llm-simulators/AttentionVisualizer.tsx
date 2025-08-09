'use client';

import { useState, useEffect, useRef } from 'react';
import styles from './Simulators.module.css';

interface AttentionScore {
  from: number;
  to: number;
  score: number;
}

const AttentionVisualizer = () => {
  const [inputTokens, setInputTokens] = useState(['나는', '오늘', '학교에', '갔다']);
  const [targetTokens, setTargetTokens] = useState(['I', 'went', 'to', 'school', 'today']); // For cross-attention
  const [attentionScores, setAttentionScores] = useState<AttentionScore[]>([]);
  const [selectedToken, setSelectedToken] = useState<number | null>(null);
  const [attentionType, setAttentionType] = useState<'self' | 'cross'>('self');
  const [animating, setAnimating] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [animationStep, setAnimationStep] = useState(0);
  const [hoveredCell, setHoveredCell] = useState<{row: number, col: number} | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | null>(null);
  const flowCanvasRef = useRef<HTMLCanvasElement>(null);

  const generateAttentionScores = () => {
    const scores: AttentionScore[] = [];
    
    if (attentionType === 'self') {
      // Self-Attention: 같은 시퀀스 내에서 토큰 간 관계
      const numTokens = inputTokens.length;
      
      for (let i = 0; i < numTokens; i++) {
        for (let j = 0; j < numTokens; j++) {
          let score = Math.random() * 0.3;
          
          // 대각선 강조 (자기 자신에 대한 attention)
          if (i === j) score = Math.min(score + 0.6, 1);
          
          // 인접 토큰 간 높은 attention
          const distance = Math.abs(i - j);
          if (distance === 1) score = Math.min(score + 0.4, 1);
          score = score * (1 - distance * 0.1);
          
          // 한국어 언어적 패턴
          if (inputTokens[i].includes('는') && inputTokens[j].includes('다')) {
            score = Math.min(score + 0.3, 1); // 주어-서술어
          }
          if (inputTokens[i].includes('에') && j === i + 1) {
            score = Math.min(score + 0.2, 1); // 조사-다음 단어
          }
          
          scores.push({ from: i, to: j, score: Math.max(0.1, score) });
        }
      }
    } else {
      // Cross-Attention: 다른 시퀀스 간 토큰 관계 (예: 번역)
      const numSourceTokens = inputTokens.length;
      const numTargetTokens = targetTokens.length;
      
      for (let i = 0; i < numSourceTokens; i++) {
        for (let j = 0; j < numTargetTokens; j++) {
          let score = Math.random() * 0.4;
          
          // 번역 정렬 패턴 시뮬레이션
          // 한국어-영어 어순 차이 반영
          if (inputTokens[i] === '나는' && targetTokens[j] === 'I') {
            score = 0.9; // 주어 매칭
          } else if (inputTokens[i] === '학교에' && targetTokens[j] === 'school') {
            score = 0.85; // 명사 매칭
          } else if (inputTokens[i] === '갔다' && targetTokens[j] === 'went') {
            score = 0.9; // 동사 매칭
          } else if (inputTokens[i] === '오늘' && targetTokens[j] === 'today') {
            score = 0.8; // 시간 부사 매칭
          }
          
          // 어순 차이로 인한 패턴
          const normalizedI = i / numSourceTokens;
          const normalizedJ = j / numTargetTokens;
          const alignmentScore = 1 - Math.abs(normalizedI - normalizedJ);
          score = score * 0.7 + alignmentScore * 0.3;
          
          scores.push({ from: i, to: j, score: Math.max(0.1, score) });
        }
      }
    }

    setAttentionScores(scores);
    startAnimation();
  };
  
  const startAnimation = () => {
    setAnimating(true);
    setAnimationStep(0);
    
    const animate = () => {
      setAnimationStep(prev => {
        const next = prev + 1;
        if (next >= inputTokens.length * inputTokens.length) {
          setAnimating(false);
          return 0;
        }
        return next;
      });
    };
    
    const interval = setInterval(animate, 50);
    setTimeout(() => {
      clearInterval(interval);
      setAnimating(false);
    }, inputTokens.length * inputTokens.length * 50 + 500);
  };
  
  const startAttentionFlow = () => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }
    
    const newIsPlaying = !isPlaying;
    setIsPlaying(newIsPlaying);
    
    if (newIsPlaying) {
      const animate = () => {
        drawAttentionMatrix();
        // Use ref to get current playing state
        if (animationRef.current !== null) {
          animationRef.current = requestAnimationFrame(animate);
        }
      };
      animationRef.current = requestAnimationFrame(animate);
    }
  };
  
  useEffect(() => {
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);
  
  const handleCanvasMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    const cellSize = 60;
    const padding = 60;
    
    const col = Math.floor((x - padding) / cellSize);
    const row = Math.floor((y - padding) / cellSize);
    
    const numCols = attentionType === 'self' ? inputTokens.length : targetTokens.length;
    const numRows = inputTokens.length;
    
    if (col >= 0 && col < numCols && row >= 0 && row < numRows) {
      setHoveredCell({ row, col });
    } else {
      setHoveredCell(null);
    }
  };
  
  const handleCanvasMouseLeave = () => {
    setHoveredCell(null);
  };

  useEffect(() => {
    generateAttentionScores();
  }, [inputTokens, targetTokens, attentionType]);

  useEffect(() => {
    drawAttentionMatrix();
  }, [attentionScores, selectedToken, isPlaying]);

  const drawAttentionMatrix = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const cellSize = 60;
    const padding = 60;
    const numRows = inputTokens.length;
    const numCols = attentionType === 'self' ? inputTokens.length : targetTokens.length;

    canvas.width = numCols * cellSize + padding * 2;
    canvas.height = numRows * cellSize + padding * 2;

    // Clear canvas
    const isDarkMode = document.documentElement.classList.contains('dark');
    ctx.fillStyle = isDarkMode ? '#1f2937' : '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw enhanced grid and scores
    for (let i = 0; i < numRows; i++) {
      for (let j = 0; j < numCols; j++) {
        const score = attentionScores.find(s => s.from === i && s.to === j)?.score || 0;
        const x = j * cellSize + padding;
        const y = i * cellSize + padding;

        // Enhanced highlighting logic
        const isHighlighted = selectedToken !== null && (i === selectedToken || j === selectedToken);
        const isHovered = hoveredCell && hoveredCell.row === i && hoveredCell.col === j;
        const isAnimated = animating && (i * numCols + j) <= animationStep;
        
        // Dynamic cell styling with gradients
        const baseOpacity = isHighlighted ? score : score * 0.7;
        const pulseEffect = isAnimated ? 0.3 * Math.sin(Date.now() * 0.01 + i + j) : 0;
        const finalOpacity = Math.min(1, baseOpacity + pulseEffect);
        
        // Create gradient for enhanced visual
        const gradient = ctx.createRadialGradient(
          x + cellSize/2, y + cellSize/2, 0,
          x + cellSize/2, y + cellSize/2, cellSize/2
        );
        
        if (isHovered) {
          gradient.addColorStop(0, `rgba(239, 68, 68, ${finalOpacity + 0.2})`);
          gradient.addColorStop(1, `rgba(220, 38, 38, ${finalOpacity})`);
        } else if (isHighlighted) {
          gradient.addColorStop(0, `rgba(248, 113, 113, ${finalOpacity})`);
          gradient.addColorStop(1, `rgba(239, 68, 68, ${finalOpacity})`);
        } else {
          gradient.addColorStop(0, `rgba(239, 68, 68, ${finalOpacity})`);
          gradient.addColorStop(1, `rgba(185, 28, 28, ${finalOpacity * 0.8})`);
        }
        
        ctx.fillStyle = gradient;
        
        // Enhanced cell shape with rounded corners
        const cornerRadius = 8;
        ctx.beginPath();
        ctx.roundRect(x + 1, y + 1, cellSize - 4, cellSize - 4, cornerRadius);
        ctx.fill();
        
        // Add glow effect for high scores
        if (score > 0.7) {
          ctx.shadowColor = 'rgba(239, 68, 68, 0.6)';
          ctx.shadowBlur = 15;
          ctx.strokeStyle = 'rgba(239, 68, 68, 0.8)';
          ctx.lineWidth = 2;
          ctx.stroke();
          ctx.shadowBlur = 0;
        }
        
        // Enhanced score text with better visibility
        const textColor = score > 0.5 ? '#ffffff' : isDarkMode ? '#e5e7eb' : '#374151';
        ctx.fillStyle = textColor;
        ctx.font = 'bold 11px Inter';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        
        // Add text shadow for better readability
        if (score > 0.5) {
          ctx.shadowColor = 'rgba(0, 0, 0, 0.7)';
          ctx.shadowBlur = 2;
          ctx.shadowOffsetX = 1;
          ctx.shadowOffsetY = 1;
        }
        
        ctx.fillText(score.toFixed(2), x + cellSize / 2, y + cellSize / 2);
        ctx.shadowBlur = 0;
        ctx.shadowOffsetX = 0;
        ctx.shadowOffsetY = 0;
        
        // Add connection lines for high attention scores
        if (score > 0.6 && selectedToken === i) {
          ctx.beginPath();
          ctx.moveTo(x + cellSize/2, y + cellSize/2);
          const targetX = j * cellSize + padding + cellSize/2;
          const targetY = i * cellSize + padding + cellSize/2;
          ctx.lineTo(targetX, targetY);
          ctx.strokeStyle = `rgba(239, 68, 68, ${score * 0.8})`;
          ctx.lineWidth = Math.max(1, score * 4);
          ctx.stroke();
        }
      }
    }

    // Enhanced labels with highlighting
    ctx.font = 'bold 14px "Noto Sans KR", Inter, sans-serif';
    
    // Top labels (columns) - target tokens for cross-attention
    const colTokens = attentionType === 'self' ? inputTokens : targetTokens;
    for (let i = 0; i < numCols; i++) {
      const isHighlighted = attentionType === 'self' && selectedToken === i;
      ctx.fillStyle = isHighlighted ? '#ef4444' : (isDarkMode ? '#e5e7eb' : '#374151');
      
      if (isHighlighted) {
        // Add background highlight
        const textWidth = ctx.measureText(colTokens[i]).width;
        ctx.fillStyle = 'rgba(239, 68, 68, 0.2)';
        ctx.fillRect(
          i * cellSize + padding + cellSize / 2 - textWidth/2 - 4,
          padding - 25,
          textWidth + 8,
          20
        );
        ctx.fillStyle = '#ef4444';
      }
      
      ctx.save();
      ctx.translate(i * cellSize + padding + cellSize / 2, padding - 15);
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText(colTokens[i], 0, 0);
      ctx.restore();
    }

    // Left labels (rows) - always input tokens
    for (let i = 0; i < numRows; i++) {
      const isHighlighted = selectedToken === i;
      ctx.fillStyle = isHighlighted ? '#ef4444' : (isDarkMode ? '#e5e7eb' : '#374151');
      
      if (isHighlighted) {
        // Add background highlight
        const textWidth = ctx.measureText(inputTokens[i]).width;
        ctx.fillStyle = 'rgba(239, 68, 68, 0.2)';
        ctx.fillRect(
          padding - 25 - textWidth,
          i * cellSize + padding + cellSize / 2 - 10,
          textWidth + 8,
          20
        );
        ctx.fillStyle = '#ef4444';
      }
      
      ctx.save();
      ctx.translate(padding - 15, i * cellSize + padding + cellSize / 2);
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(inputTokens[i], 0, 0);
      ctx.restore();
    }
    
    drawAttentionFlow();
  };
  
  const drawAttentionFlow = () => {
    const flowCanvas = flowCanvasRef.current;
    if (!flowCanvas) return;
    
    const ctx = flowCanvas.getContext('2d');
    if (!ctx) return;
    
    const mainCanvas = canvasRef.current;
    if (!mainCanvas) return;
    
    // Set canvas size to match main canvas
    if (flowCanvas.width !== mainCanvas.width || flowCanvas.height !== mainCanvas.height) {
      flowCanvas.width = mainCanvas.width;
      flowCanvas.height = mainCanvas.height;
    }
    
    ctx.clearRect(0, 0, flowCanvas.width, flowCanvas.height);
    
    // Only draw if playing
    if (!isPlaying) return;
    
    const cellSize = 60;
    const padding = 60;
    const numRows = inputTokens.length;
    const numCols = attentionType === 'self' ? inputTokens.length : targetTokens.length;
    const time = Date.now() * 0.001; // Adjusted speed
    
    // Draw attention flow animations
    if (selectedToken !== null) {
      // When a token is selected, show flows from that token
      attentionScores
        .filter(s => s.from === selectedToken && s.score > 0.3)
        .forEach((score, index) => {
          const fromX = selectedToken * cellSize + padding + cellSize/2;
          const fromY = selectedToken * cellSize + padding + cellSize/2;
          const toX = score.to * cellSize + padding + cellSize/2;
          const toY = attentionType === 'self' ? score.to * cellSize + padding + cellSize/2 : selectedToken * cellSize + padding + cellSize/2;
          
          // Multiple particles per connection
          for (let p = 0; p < 3; p++) {
            const offset = p / 3;
            const progress = (time + index * 0.2 + offset) % 1;
            
            // Curved path for visual interest
            const t = progress;
            const ctrlX = (fromX + toX) / 2;
            const ctrlY = fromY - 30; // Arc upward
            
            // Quadratic Bezier curve
            const x = (1-t)*(1-t)*fromX + 2*(1-t)*t*ctrlX + t*t*toX;
            const y = (1-t)*(1-t)*fromY + 2*(1-t)*t*ctrlY + t*t*toY;
            
            // Draw trail
            const trailLength = 5;
            ctx.beginPath();
            for (let i = 0; i < trailLength; i++) {
              const trailT = Math.max(0, t - i * 0.02);
              const trailX = (1-trailT)*(1-trailT)*fromX + 2*(1-trailT)*trailT*ctrlX + trailT*trailT*toX;
              const trailY = (1-trailT)*(1-trailT)*fromY + 2*(1-trailT)*trailT*ctrlY + trailT*trailT*toY;
              
              if (i === 0) {
                ctx.moveTo(trailX, trailY);
              } else {
                ctx.lineTo(trailX, trailY);
              }
            }
            
            const gradient = ctx.createLinearGradient(x - 10, y, x + 10, y);
            gradient.addColorStop(0, `rgba(239, 68, 68, 0)`);
            gradient.addColorStop(0.5, `rgba(239, 68, 68, ${score.score * 0.8})`);
            gradient.addColorStop(1, `rgba(239, 68, 68, 0)`);
            
            ctx.strokeStyle = gradient;
            ctx.lineWidth = 3 + score.score * 4;
            ctx.lineCap = 'round';
            ctx.stroke();
            
            // Draw particle
            const size = 5 + score.score * 7;
            
            // Outer glow
            ctx.beginPath();
            ctx.arc(x, y, size * 2.5, 0, Math.PI * 2);
            const glowGradient = ctx.createRadialGradient(x, y, 0, x, y, size * 2.5);
            glowGradient.addColorStop(0, `rgba(239, 68, 68, ${score.score * 0.4})`);
            glowGradient.addColorStop(1, 'rgba(239, 68, 68, 0)');
            ctx.fillStyle = glowGradient;
            ctx.fill();
            
            // Inner particle
            ctx.beginPath();
            ctx.arc(x, y, size, 0, Math.PI * 2);
            const particleGradient = ctx.createRadialGradient(x, y, 0, x, y, size);
            particleGradient.addColorStop(0, 'rgba(255, 255, 255, 1)');
            particleGradient.addColorStop(0.4, `rgba(248, 113, 113, ${score.score})`);
            particleGradient.addColorStop(1, `rgba(239, 68, 68, ${score.score * 0.9})`);
            ctx.fillStyle = particleGradient;
            ctx.fill();
          }
          
          // Draw connection strength indicator at destination
          const pulseSize = 12 + Math.sin(time * 4 + index) * 6;
          ctx.beginPath();
          ctx.arc(toX, toY, pulseSize, 0, Math.PI * 2);
          const pulseGradient = ctx.createRadialGradient(toX, toY, 0, toX, toY, pulseSize);
          pulseGradient.addColorStop(0, `rgba(239, 68, 68, ${score.score * 0.6})`);
          pulseGradient.addColorStop(1, 'rgba(239, 68, 68, 0)');
          ctx.fillStyle = pulseGradient;
          ctx.fill();
        });
    } else {
      // When no token is selected, show general flow pattern
      for (let i = 0; i < 5; i++) {
        const fromIdx = Math.floor(Math.random() * numRows);
        const toIdx = Math.floor(Math.random() * numCols);
        const score = attentionScores.find(s => s.from === fromIdx && s.to === toIdx);
        
        if (!score || score.score < 0.5) continue;
        
        const cellX = toIdx * cellSize + padding + cellSize/2;
        const fromY = fromIdx * cellSize + padding + cellSize/2;
        const toY = attentionType === 'self' ? toIdx * cellSize + padding + cellSize/2 : fromIdx * cellSize + padding + cellSize/2;
        
        const progress = (time + i * 0.2) % 1;
        const y = fromY + (toY - fromY) * progress;
        
        // Glowing orb
        const size = 8 + score.score * 6;
        ctx.beginPath();
        ctx.arc(cellX, y, size, 0, Math.PI * 2);
        
        const gradient = ctx.createRadialGradient(cellX, y, 0, cellX, y, size);
        gradient.addColorStop(0, 'rgba(255, 255, 255, 0.9)');
        gradient.addColorStop(0.5, `rgba(248, 113, 113, ${score.score * 0.8})`);
        gradient.addColorStop(1, 'rgba(239, 68, 68, 0)');
        
        ctx.fillStyle = gradient;
        ctx.shadowColor = 'rgba(239, 68, 68, 0.9)';
        ctx.shadowBlur = size * 3;
        ctx.fill();
        ctx.shadowBlur = 0;
      }
    }
    
    // Highlight selected token connections with enhanced visibility
    if (selectedToken !== null) {
      attentionScores
        .filter(s => s.from === selectedToken && s.score > 0.4)
        .forEach((score) => {
          const fromX = score.to * cellSize + padding + cellSize/2;
          const fromY = selectedToken * cellSize + padding + cellSize/2;
          
          // Draw glowing connection with red theme
          ctx.beginPath();
          ctx.arc(fromX, fromY, 10 + score.score * 12, 0, Math.PI * 2);
          const gradient = ctx.createRadialGradient(fromX, fromY, 0, fromX, fromY, 20);
          gradient.addColorStop(0, `rgba(239, 68, 68, ${score.score * 0.7})`);
          gradient.addColorStop(0.5, `rgba(239, 68, 68, ${score.score * 0.3})`);
          gradient.addColorStop(1, 'rgba(239, 68, 68, 0)');
          ctx.fillStyle = gradient;
          ctx.fill();
        });
    }
  };

  const handleTokensChange = (text: string) => {
    const tokens = text.split(' ').filter(t => t.length > 0);
    if (tokens.length > 0 && tokens.length <= 8) {
      setInputTokens(tokens);
    }
  };

  const exampleSentences = [
    '나는 오늘 학교에 갔다',
    'The cat sat on mat',
    'AI가 세상을 바꾼다',
    '코딩은 정말 재미있다'
  ];

  return (
    <div className={styles.simulator}>
      <div className={styles.header}>
        <h3>👁️ Attention 메커니즘 시각화</h3>
        <p>토큰 간의 attention 가중치를 시각적으로 확인해보세요</p>
      </div>

      <div className={styles.controls}>
        <div className={styles.inputSection}>
          <label>입력 토큰 (공백으로 구분, 최대 8개):</label>
          <input
            type="text"
            value={inputTokens.join(' ')}
            onChange={(e) => handleTokensChange(e.target.value)}
            placeholder="토큰을 입력하세요..."
          />
          {attentionType === 'cross' && (
            <>
              <label style={{ marginTop: '10px' }}>타겟 토큰 (Cross-Attention용):</label>
              <input
                type="text"
                value={targetTokens.join(' ')}
                onChange={(e) => {
                  const tokens = e.target.value.split(' ').filter(t => t.length > 0);
                  if (tokens.length > 0 && tokens.length <= 8) {
                    setTargetTokens(tokens);
                  }
                }}
                placeholder="타겟 토큰을 입력하세요..."
              />
            </>
          )}
          <div className={styles.exampleButtons}>
            {exampleSentences.map((sentence, index) => (
              <button
                key={index}
                className={styles.exampleBtn}
                onClick={() => {
                  handleTokensChange(sentence);
                  if (attentionType === 'cross') {
                    // 예시 번역 설정
                    if (sentence === '나는 오늘 학교에 갔다') {
                      setTargetTokens(['I', 'went', 'to', 'school', 'today']);
                    } else if (sentence === 'The cat sat on mat') {
                      setTargetTokens(['고양이가', '매트', '위에', '앉았다']);
                    } else if (sentence === 'AI가 세상을 바꾼다') {
                      setTargetTokens(['AI', 'changes', 'the', 'world']);
                    } else if (sentence === '코딩은 정말 재미있다') {
                      setTargetTokens(['Coding', 'is', 'really', 'fun']);
                    }
                  }
                }}
              >
                {sentence}
              </button>
            ))}
          </div>
        </div>

        <div className={styles.optionsRow}>
          <button 
            className={styles.regenerateBtn}
            onClick={generateAttentionScores}
          >
            🔄 Attention 재생성
          </button>
          <div className={styles.attentionTypeSelector}>
            <label>Attention 타입:</label>
            <select 
              value={attentionType} 
              onChange={(e) => setAttentionType(e.target.value as 'self' | 'cross')}
            >
              <option value="self">Self-Attention</option>
              <option value="cross">Cross-Attention</option>
            </select>
          </div>
        </div>
      </div>

      <div className={styles.results}>
        <div className={styles.attentionContainer}>
          <div className={styles.matrixSection}>
            <h4>Attention Matrix</h4>
            <div className={styles.canvasContainer} style={{ position: 'relative' }}>
              <canvas 
                ref={canvasRef}
                className={`${styles.attentionCanvas} ${animating ? styles.animating : ''}`}
                onMouseMove={handleCanvasMouseMove}
                onMouseLeave={handleCanvasMouseLeave}
              />
              <canvas 
                ref={flowCanvasRef}
                className={styles.flowCanvas}
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  pointerEvents: 'none',
                  zIndex: 1
                }}
              />
            </div>
            <div className={styles.canvasControls}>
              <button 
                className={`${styles.playBtn} ${isPlaying ? styles.playing : ''}`}
                onClick={startAttentionFlow}
              >
                {isPlaying ? '⏸️ 일시정지' : '▶️ 플로우 재생'}
              </button>
              <button 
                className={styles.resetBtn}
                onClick={() => {
                  setSelectedToken(null);
                  setHoveredCell(null);
                  setIsPlaying(false);
                }}
              >
                🔄 초기화
              </button>
            </div>
            <p className={styles.matrixExplanation}>
              {attentionType === 'self' 
                ? '각 셀은 행 토큰이 열 토큰에 주는 attention 점수를 나타냅니다. 대각선은 자기 자신에 대한 attention입니다.'
                : '각 셀은 소스 토큰(행)이 타겟 토큰(열)에 주는 attention 점수를 나타냅니다. 번역 정렬을 시각화합니다.'}
              진한 파란색일수록 높은 attention을 의미합니다.
              <br/><strong>💡 팁:</strong> 토큰을 클릭한 후 플로우 재생 버튼을 눌러보세요!
            </p>
          </div>

          <div className={styles.tokenInteraction}>
            <h4>토큰별 Attention 분석</h4>
            <div className={styles.tokenButtons}>
              {inputTokens.map((token, index) => (
                <button
                  key={index}
                  className={`${styles.tokenBtn} ${selectedToken === index ? styles.selected : ''}`}
                  onClick={() => setSelectedToken(selectedToken === index ? null : index)}
                >
                  {token}
                </button>
              ))}
            </div>
            {selectedToken !== null && (
              <div className={styles.tokenAnalysis}>
                <h5>"{inputTokens[selectedToken]}" 토큰 분석:</h5>
                <div className={styles.attentionDetails}>
                  <div>
                    <strong>{attentionType === 'self' ? '이 토큰이 주목하는 토큰들:' : '이 토큰이 주목하는 타겟 토큰들:'}</strong>
                    {attentionScores
                      .filter(s => s.from === selectedToken)
                      .sort((a, b) => b.score - a.score)
                      .slice(0, 3)
                      .map((s, i) => {
                        const targetToken = attentionType === 'self' ? inputTokens[s.to] : targetTokens[s.to];
                        return (
                          <div key={i} className={styles.scoreItem}>
                            {targetToken}: {(s.score * 100).toFixed(1)}%
                          </div>
                        );
                      })}
                  </div>
                  {attentionType === 'self' && (
                    <div>
                      <strong>이 토큰에 주목하는 토큰들:</strong>
                      {attentionScores
                        .filter(s => s.to === selectedToken)
                        .sort((a, b) => b.score - a.score)
                        .slice(0, 3)
                        .map((s, i) => (
                          <div key={i} className={styles.scoreItem}>
                            {inputTokens[s.from]}: {(s.score * 100).toFixed(1)}%
                          </div>
                        ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        <div className={styles.explanation}>
          <h4>Attention 메커니즘이란?</h4>
          <p>
            Attention은 모델이 입력 시퀀스의 어떤 부분에 "주목"해야 하는지 학습하는 메커니즘입니다.
            각 토큰은 다른 모든 토큰과의 관계를 계산하여, 문맥을 이해하는 데 필요한 정보를 선택적으로 활용합니다.
          </p>
          <ul>
            <li><strong>Self-Attention:</strong> 같은 시퀀스 내의 토큰들 간의 관계 (예: 문장 이해)</li>
            <li><strong>Cross-Attention:</strong> 다른 시퀀스 간의 토큰 관계 (예: 번역, 질의응답)</li>
            <li><strong>높은 점수:</strong> 두 토큰 간의 강한 연관성</li>
            <li><strong>Multi-Head:</strong> 여러 관점에서 동시에 attention 계산</li>
          </ul>
          {attentionType === 'cross' && (
            <div style={{ marginTop: '15px', padding: '10px', backgroundColor: 'rgba(59, 130, 246, 0.1)', borderRadius: '8px' }}>
              <strong>Cross-Attention 특징:</strong>
              <p style={{ marginTop: '5px', fontSize: '14px' }}>
                • 소스 언어(왼쪽)에서 타겟 언어(위쪽)로의 attention<br/>
                • 번역 시 어순 차이와 의미 정렬을 시각화<br/>
                • 기계 번역, 이미지 캡셔닝 등에서 핵심 역할
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AttentionVisualizer;