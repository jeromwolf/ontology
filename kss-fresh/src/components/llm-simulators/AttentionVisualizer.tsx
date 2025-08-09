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
    const numTokens = inputTokens.length;

    for (let i = 0; i < numTokens; i++) {
      for (let j = 0; j < numTokens; j++) {
        // Simulate more realistic attention patterns
        let score = Math.random();
        
        // Make diagonal stronger for self-attention
        if (i === j) score = Math.min(score + 0.5, 1);
        
        // Make nearby tokens have higher attention
        const distance = Math.abs(i - j);
        score = score * (1 - distance * 0.1);
        
        // Add some linguistic patterns
        if (inputTokens[i].includes('는') && inputTokens[j].includes('다')) {
          score = Math.min(score + 0.3, 1); // Subject-verb attention
        }
        
        scores.push({ from: i, to: j, score: Math.max(0.1, score) });
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
    }
    
    setIsPlaying(!isPlaying);
    
    if (!isPlaying) {
      const animate = () => {
        drawAttentionMatrix();
        if (isPlaying) {
          animationRef.current = requestAnimationFrame(animate);
        }
      };
      animate();
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
    
    if (col >= 0 && col < inputTokens.length && row >= 0 && row < inputTokens.length) {
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
  }, [inputTokens]);

  useEffect(() => {
    drawAttentionMatrix();
  }, [attentionScores, selectedToken]);

  const drawAttentionMatrix = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const cellSize = 60;
    const padding = 60;
    const numTokens = inputTokens.length;

    canvas.width = numTokens * cellSize + padding * 2;
    canvas.height = numTokens * cellSize + padding * 2;

    // Clear canvas
    const isDarkMode = document.documentElement.classList.contains('dark');
    ctx.fillStyle = isDarkMode ? '#1f2937' : '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw enhanced grid and scores
    for (let i = 0; i < numTokens; i++) {
      for (let j = 0; j < numTokens; j++) {
        const score = attentionScores.find(s => s.from === i && s.to === j)?.score || 0;
        const x = j * cellSize + padding;
        const y = i * cellSize + padding;

        // Enhanced highlighting logic
        const isHighlighted = selectedToken !== null && (i === selectedToken || j === selectedToken);
        const isHovered = hoveredCell && hoveredCell.row === i && hoveredCell.col === j;
        const isAnimated = animating && (i * numTokens + j) <= animationStep;
        
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
          gradient.addColorStop(0, `rgba(59, 130, 246, ${finalOpacity + 0.2})`);
          gradient.addColorStop(1, `rgba(37, 99, 235, ${finalOpacity})`);
        } else if (isHighlighted) {
          gradient.addColorStop(0, `rgba(99, 102, 241, ${finalOpacity})`);
          gradient.addColorStop(1, `rgba(37, 99, 235, ${finalOpacity})`);
        } else {
          gradient.addColorStop(0, `rgba(37, 99, 235, ${finalOpacity})`);
          gradient.addColorStop(1, `rgba(30, 70, 180, ${finalOpacity * 0.8})`);
        }
        
        ctx.fillStyle = gradient;
        
        // Enhanced cell shape with rounded corners
        const cornerRadius = 8;
        ctx.beginPath();
        ctx.roundRect(x + 1, y + 1, cellSize - 4, cellSize - 4, cornerRadius);
        ctx.fill();
        
        // Add glow effect for high scores
        if (score > 0.7) {
          ctx.shadowColor = 'rgba(59, 130, 246, 0.6)';
          ctx.shadowBlur = 15;
          ctx.strokeStyle = 'rgba(59, 130, 246, 0.8)';
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
          ctx.strokeStyle = `rgba(59, 130, 246, ${score * 0.8})`;
          ctx.lineWidth = Math.max(1, score * 4);
          ctx.stroke();
        }
      }
    }

    // Enhanced labels with highlighting
    ctx.font = 'bold 14px "Noto Sans KR", Inter, sans-serif';
    
    // Top labels with highlighting
    for (let i = 0; i < numTokens; i++) {
      const isHighlighted = selectedToken === i;
      ctx.fillStyle = isHighlighted ? '#3b82f6' : (isDarkMode ? '#e5e7eb' : '#374151');
      
      if (isHighlighted) {
        // Add background highlight
        const textWidth = ctx.measureText(inputTokens[i]).width;
        ctx.fillStyle = 'rgba(59, 130, 246, 0.2)';
        ctx.fillRect(
          i * cellSize + padding + cellSize / 2 - textWidth/2 - 4,
          padding - 25,
          textWidth + 8,
          20
        );
        ctx.fillStyle = '#3b82f6';
      }
      
      ctx.save();
      ctx.translate(i * cellSize + padding + cellSize / 2, padding - 15);
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText(inputTokens[i], 0, 0);
      ctx.restore();
    }

    // Left labels with highlighting
    for (let i = 0; i < numTokens; i++) {
      const isHighlighted = selectedToken === i;
      ctx.fillStyle = isHighlighted ? '#3b82f6' : (isDarkMode ? '#e5e7eb' : '#374151');
      
      if (isHighlighted) {
        // Add background highlight
        const textWidth = ctx.measureText(inputTokens[i]).width;
        ctx.fillStyle = 'rgba(59, 130, 246, 0.2)';
        ctx.fillRect(
          padding - 25 - textWidth,
          i * cellSize + padding + cellSize / 2 - 10,
          textWidth + 8,
          20
        );
        ctx.fillStyle = '#3b82f6';
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
    if (!flowCanvas || selectedToken === null) return;
    
    const ctx = flowCanvas.getContext('2d');
    if (!ctx) return;
    
    const mainCanvas = canvasRef.current;
    if (!mainCanvas) return;
    
    flowCanvas.width = mainCanvas.width;
    flowCanvas.height = mainCanvas.height;
    
    ctx.clearRect(0, 0, flowCanvas.width, flowCanvas.height);
    
    const cellSize = 60;
    const padding = 60;
    const numTokens = inputTokens.length;
    const time = Date.now() * 0.005;
    
    // Draw flowing particles for attention connections
    attentionScores
      .filter(s => s.from === selectedToken && s.score > 0.3)
      .forEach((score, index) => {
        const fromX = selectedToken * cellSize + padding + cellSize/2;
        const fromY = selectedToken * cellSize + padding + cellSize/2;
        const toX = score.to * cellSize + padding + cellSize/2;
        const toY = selectedToken * cellSize + padding + cellSize/2;
        
        // Draw flowing particles
        for (let i = 0; i < 5; i++) {
          const progress = (time + index * 0.5 + i * 0.2) % 1;
          const x = fromX + (toX - fromX) * progress;
          const y = fromY + (toY - fromY) * progress + Math.sin(progress * Math.PI * 2) * 5;
          
          const size = 3 + score.score * 4;
          const opacity = score.score * (1 - Math.abs(progress - 0.5) * 2);
          
          ctx.beginPath();
          ctx.arc(x, y, size, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(59, 130, 246, ${opacity})`;
          ctx.fill();
          
          // Add glow effect
          ctx.shadowColor = 'rgba(59, 130, 246, 0.6)';
          ctx.shadowBlur = size * 2;
          ctx.fill();
          ctx.shadowBlur = 0;
        }
      });
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
          <div className={styles.exampleButtons}>
            {exampleSentences.map((sentence, index) => (
              <button
                key={index}
                className={styles.exampleBtn}
                onClick={() => handleTokensChange(sentence)}
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
              각 셀은 행 토큰이 열 토큰에 주는 attention 점수를 나타냅니다.
              진한 파란색일수록 높은 attention을 의미합니다.
              토큰을 선택하면 attention 플로우를 확인할 수 있습니다.
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
                    <strong>이 토큰이 주목하는 토큰들:</strong>
                    {attentionScores
                      .filter(s => s.from === selectedToken)
                      .sort((a, b) => b.score - a.score)
                      .slice(0, 3)
                      .map((s, i) => (
                        <div key={i} className={styles.scoreItem}>
                          {inputTokens[s.to]}: {(s.score * 100).toFixed(1)}%
                        </div>
                      ))}
                  </div>
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
            <li><strong>Self-Attention:</strong> 같은 시퀀스 내의 토큰들 간의 관계</li>
            <li><strong>높은 점수:</strong> 두 토큰 간의 강한 연관성</li>
            <li><strong>Multi-Head:</strong> 여러 관점에서 동시에 attention 계산</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default AttentionVisualizer;