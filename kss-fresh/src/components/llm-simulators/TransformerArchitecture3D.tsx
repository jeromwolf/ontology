'use client';

import { useState, useEffect, useRef } from 'react';
import styles from './Simulators.module.css';

interface Token {
  id: number;
  text: string;
  embedding?: number[];
  attention?: number[];
  position: { x: number; y: number };
}

interface DataFlow {
  from: string;
  to: string;
  active: boolean;
  description: string;
}

const TransformerArchitecture3D = () => {
  const [step, setStep] = useState(0);
  const [selectedComponent, setSelectedComponent] = useState<string | null>(null);
  const [inputText, setInputText] = useState("The cat sat on the mat");
  const [tokens, setTokens] = useState<Token[]>([]);
  const [selectedTokenIndex, setSelectedTokenIndex] = useState(1); // Default to "cat"
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();

  // 학습 단계별 설명
  const steps = [
    {
      title: "1. 토큰화 (Tokenization)",
      description: "입력 텍스트를 작은 단위(토큰)로 분할합니다.",
      activeComponents: ['tokenizer'],
      highlight: ['The', 'cat', 'sat', 'on', 'the', 'mat']
    },
    {
      title: "2. 임베딩 (Embedding)",
      description: "각 토큰을 고차원 벡터로 변환합니다. 의미가 유사한 단어는 가까운 벡터가 됩니다.",
      activeComponents: ['embedding'],
      highlight: []
    },
    {
      title: "3. 위치 인코딩 (Positional Encoding)",
      description: "토큰의 순서 정보를 추가합니다. Transformer는 순차 처리가 아니므로 위치 정보가 필요합니다.",
      activeComponents: ['positional'],
      highlight: []
    },
    {
      title: "4. Self-Attention",
      description: "각 토큰이 다른 토큰들과 얼마나 관련있는지 계산합니다. 모든 토큰 쌍의 관계를 동시에 파악합니다.",
      activeComponents: ['attention'],
      highlight: []
    },
    {
      title: "5. Multi-Head Attention",
      description: "여러 관점(head)에서 attention을 계산합니다. 예: 문법적 관계, 의미적 관계 등",
      activeComponents: ['multihead'],
      highlight: []
    },
    {
      title: "6. Feed Forward Network",
      description: "각 토큰에 대해 독립적으로 비선형 변환을 적용합니다.",
      activeComponents: ['ffn'],
      highlight: []
    },
    {
      title: "7. 잔차 연결 & 정규화",
      description: "기울기 소실을 방지하고 학습을 안정화합니다.",
      activeComponents: ['residual', 'norm'],
      highlight: []
    },
    {
      title: "8. 출력 생성",
      description: "변환된 표현에서 다음 토큰을 예측하거나 분류를 수행합니다.",
      activeComponents: ['output'],
      highlight: []
    }
  ];

  // 토큰화
  useEffect(() => {
    const words = inputText.split(' ');
    const newTokens = words.map((word, index) => ({
      id: index,
      text: word,
      position: { x: 100 + index * 120, y: 100 }
    }));
    setTokens(newTokens);
  }, [inputText]);


  // Canvas 그리기
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const animate = () => {
      drawArchitecture(ctx, canvas);
      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [step, tokens, selectedComponent, selectedTokenIndex]);

  // Handle canvas click for token selection
  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (step !== 3) return; // Only active during Self-Attention step
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    // Check which token was clicked
    tokens.forEach((token, index) => {
      const tokenCount = tokens.length;
      const spacing = tokenCount > 6 ? 90 : 110;
      const startX = tokenCount > 6 ? 50 : 100;
      const tokenX = startX + index * spacing;
      const tokenY = 120;
      
      // Check if click is within token bounds
      if (x >= tokenX - 45 && x <= tokenX + 45 && 
          y >= tokenY - 20 && y <= tokenY + 20) {
        setSelectedTokenIndex(index);
      }
    });
  };

  const drawArchitecture = (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement) => {
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;

    // Clear canvas with gradient background
    const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
    gradient.addColorStop(0, '#ffffff');
    gradient.addColorStop(0.75, '#f0f9ff');
    gradient.addColorStop(0.77, '#f3f4f6');
    gradient.addColorStop(1, '#f3f4f6');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw separator line with gradient
    const separatorY = 290;
    const lineGradient = ctx.createLinearGradient(0, separatorY, canvas.width, separatorY);
    lineGradient.addColorStop(0, 'rgba(199, 210, 254, 0)');
    lineGradient.addColorStop(0.1, 'rgba(199, 210, 254, 1)');
    lineGradient.addColorStop(0.9, 'rgba(199, 210, 254, 1)');
    lineGradient.addColorStop(1, 'rgba(199, 210, 254, 0)');
    ctx.strokeStyle = lineGradient;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, separatorY);
    ctx.lineTo(canvas.width, separatorY);
    ctx.stroke();
    
    // Add area labels with better styling
    ctx.shadowColor = 'rgba(99, 102, 241, 0.1)';
    ctx.shadowBlur = 10;
    ctx.fillStyle = '#6366f1';
    ctx.font = 'bold 14px Inter';
    ctx.textAlign = 'left';
    ctx.fillText('입력 토큰', 30, 30);
    ctx.shadowBlur = 0;
    
    ctx.shadowColor = 'rgba(124, 58, 237, 0.1)';
    ctx.shadowBlur = 10;
    ctx.fillStyle = '#7c3aed';
    ctx.font = 'bold 12px Inter';
    ctx.fillText('Transformer 컴포넌트', 30, 305);
    ctx.shadowBlur = 0;

    const currentStep = steps[step];
    
    // Draw tokens
    tokens.forEach((token, index) => {
      const isHighlighted = currentStep.highlight.includes(token.text);
      const isSelected = step === 3 && index === selectedTokenIndex;
      
      // Adjust token position based on number of tokens
      const tokenCount = tokens.length;
      const spacing = tokenCount > 6 ? 90 : 110;
      const startX = tokenCount > 6 ? 50 : 100;
      token.position.y = 120;
      token.position.x = startX + index * spacing;
      
      // Token box with shadow
      ctx.shadowColor = 'rgba(0, 0, 0, 0.1)';
      ctx.shadowBlur = 4;
      ctx.shadowOffsetY = 2;
      
      if (isSelected) {
        ctx.fillStyle = '#3b82f6';
      } else if (isHighlighted) {
        ctx.fillStyle = '#10b981';
      } else {
        ctx.fillStyle = '#ffffff';
      }
      ctx.fillRect(token.position.x - 45, token.position.y - 20, 90, 40);
      
      // Reset shadow
      ctx.shadowColor = 'transparent';
      ctx.shadowBlur = 0;
      ctx.shadowOffsetY = 0;
      
      // Token border
      ctx.strokeStyle = isSelected ? '#1e40af' : isHighlighted ? '#059669' : '#d1d5db';
      ctx.lineWidth = isSelected || isHighlighted ? 2 : 1;
      ctx.strokeRect(token.position.x - 45, token.position.y - 20, 90, 40);
      
      // Add click hint for step 3
      if (step === 3) {
        ctx.globalAlpha = 0.7;
        ctx.fillStyle = '#6b7280';
        ctx.font = '10px Inter';
        ctx.fillText('클릭', token.position.x, token.position.y + 32);
        ctx.globalAlpha = 1;
      }
      
      // Token text
      ctx.fillStyle = isSelected ? '#ffffff' : '#212529';
      ctx.font = 'bold 14px Inter';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(token.text, token.position.x, token.position.y);

      // Draw embeddings for step 2
      if (step >= 1 && step <= 2) {
        // Visualize embedding vectors
        const vecY = token.position.y + 50;
        for (let i = 0; i < 8; i++) {
          const value = Math.sin(index + i) * 0.5 + 0.5;
          ctx.fillStyle = `hsl(${200 + value * 60}, 70%, 50%)`;
          ctx.fillRect(token.position.x - 35 + i * 9, vecY, 8, 20 * value);
        }
      }

      // Draw positional encoding for step 3
      if (step === 2) {
        ctx.strokeStyle = '#FF6B6B';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(token.position.x, token.position.y + 90, 20, 0, (index + 1) * Math.PI / 3);
        ctx.stroke();
      }
    });

    // Step 6: Feed Forward Network visualization
    if (step === 5 && tokens.length > 0) {
      tokens.forEach((token, index) => {
        const x = token.position.x;
        const y = token.position.y + 80;
        
        // Draw FFN layers
        ctx.fillStyle = '#8b5cf6';
        ctx.fillRect(x - 40, y, 80, 25);
        ctx.fillStyle = '#7c3aed';
        ctx.fillRect(x - 40, y + 30, 80, 25);
        
        ctx.fillStyle = '#ffffff';
        ctx.font = '11px Inter';
        ctx.textAlign = 'center';
        ctx.fillText('FC1', x, y + 12);
        ctx.fillText('FC2', x, y + 42);
        
        // Connection lines
        ctx.strokeStyle = '#8b5cf6';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x, token.position.y + 20);
        ctx.lineTo(x, y);
        ctx.stroke();
      });
    }
    
    // Step 7: Residual & Normalization visualization
    if (step === 6 && tokens.length > 0) {
      tokens.forEach((token, index) => {
        const x = token.position.x;
        const y = token.position.y;
        
        // Draw residual connection
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.arc(x, y, 60, -Math.PI/4, Math.PI/4);
        ctx.stroke();
        ctx.setLineDash([]);
        
        // Draw plus sign
        ctx.fillStyle = '#ef4444';
        ctx.font = 'bold 20px Inter';
        ctx.fillText('+', x + 50, y);
        
        // Draw normalization indicator
        ctx.fillStyle = '#10b981';
        ctx.font = '10px Inter';
        ctx.fillText('Norm', x, y + 70);
      });
    }
    
    // Step 8: Output generation visualization
    if (step === 7 && tokens.length > 0) {
      // Draw output probability distribution
      const outputY = 200;
      const lastToken = tokens[tokens.length - 1].text;
      
      // Title with context
      ctx.fillStyle = '#6366f1';
      ctx.font = 'bold 12px Inter';
      ctx.textAlign = 'center';
      ctx.fillText(`"${lastToken}" 다음에 올 단어/형태소 예측`, canvas.width / 2, outputY - 20);
      
      // Highlight last token
      const lastTokenObj = tokens[tokens.length - 1];
      ctx.strokeStyle = '#6366f1';
      ctx.lineWidth = 3;
      ctx.strokeRect(
        lastTokenObj.position.x - 47,
        lastTokenObj.position.y - 22,
        94,
        44
      );
      
      // Draw arrow from last token to prediction
      ctx.strokeStyle = '#6366f1';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(lastTokenObj.position.x, lastTokenObj.position.y + 20);
      ctx.lineTo(lastTokenObj.position.x, outputY - 40);
      ctx.lineTo(canvas.width / 2, outputY - 40);
      ctx.stroke();
      
      // Arrow head
      ctx.fillStyle = '#6366f1';
      ctx.beginPath();
      ctx.moveTo(canvas.width / 2, outputY - 40);
      ctx.lineTo(canvas.width / 2 - 8, outputY - 45);
      ctx.lineTo(canvas.width / 2 - 8, outputY - 35);
      ctx.closePath();
      ctx.fill();
      
      // Detect if input is Korean or English
      const isKorean = tokens.some(token => /[가-힣]/.test(token.text));
      
      let words, probs;
      if (isKorean) {
        // Korean predictions based on context - predict next morpheme/word
        const lastToken = tokens[tokens.length - 1].text;
        
        if (lastToken.includes('바꾼다')) {
          words = ['.', '고', '는', '며', '!', '면', '고,', '...'];
          probs = [0.35, 0.20, 0.15, 0.10, 0.08, 0.05, 0.04, 0.03];
        } else if (lastToken.includes('인공지능')) {
          words = ['이', '은', '의', '을', '에', '으로', '과', '기술'];
          probs = [0.30, 0.25, 0.15, 0.10, 0.08, 0.05, 0.04, 0.03];
        } else if (lastToken.includes('세상')) {
          words = ['을', '이', '은', '의', '에', '에서', '과', '속'];
          probs = [0.35, 0.20, 0.15, 0.10, 0.08, 0.05, 0.04, 0.03];
        } else if (lastToken.endsWith('이') || lastToken.endsWith('가')) {
          words = ['바꾼다', '변한다', '온다', '있다', '된다', '만든다', '시작한다', '발전한다'];
          probs = [0.25, 0.20, 0.15, 0.10, 0.10, 0.08, 0.07, 0.05];
        } else {
          words = ['이', '을', '은', '의', '에', '와', '로', '다'];
          probs = [0.20, 0.18, 0.15, 0.12, 0.10, 0.10, 0.08, 0.07];
        }
      } else {
        // English predictions based on context
        if (inputText.toLowerCase().includes('cat')) {
          words = ['sat', 'is', 'was', 'sleeps', 'runs', 'plays', 'meows', 'jumps'];
          probs = [0.25, 0.20, 0.15, 0.10, 0.08, 0.08, 0.07, 0.07];
        } else {
          words = ['The', 'cat', 'dog', 'mat', 'sits', 'runs', 'jumps', 'sleeps'];
          probs = [0.05, 0.15, 0.08, 0.35, 0.12, 0.08, 0.10, 0.07];
        }
      }
      
      words.forEach((word, i) => {
        const x = 200 + i * 80;
        const barHeight = probs[i] * 100;
        
        // Draw bar
        const barGradient = ctx.createLinearGradient(x, outputY + 50 - barHeight, x, outputY + 50);
        barGradient.addColorStop(0, '#818cf8');
        barGradient.addColorStop(1, '#6366f1');
        ctx.fillStyle = barGradient;
        ctx.fillRect(x - 25, outputY + 50 - barHeight, 50, barHeight);
        
        // Draw word
        ctx.fillStyle = '#374151';
        ctx.font = '11px Inter';
        ctx.fillText(word, x, outputY + 65);
        
        // Draw probability
        ctx.fillStyle = '#6366f1';
        ctx.font = 'bold 10px Inter';
        ctx.fillText((probs[i] * 100).toFixed(0) + '%', x, outputY + 40 - barHeight);
      });
    }

    // Draw attention connections for step 4
    if (step === 3 && tokens.length > 0) {
      // Draw attention arrows for selected token
      const selectedToken = tokens[selectedTokenIndex];
      
      if (selectedToken) {
        // Draw attention from "cat" to all other tokens
        tokens.forEach((targetToken, j) => {
          if (j !== selectedTokenIndex) {
            // Dynamic attention calculation based on content
            let attention = 0.1;
            
            // Check for semantic relationships
            const selectedText = selectedToken.text.toLowerCase();
            const targetText = targetToken.text.toLowerCase();
            
            // Korean language patterns
            if (selectedText.includes('인공지능') && targetText.includes('바꾼다')) attention = 0.8;
            else if (selectedText.includes('세상') && targetText.includes('바꾼다')) attention = 0.7;
            else if (selectedText.includes('인공지능') && targetText.includes('세상')) attention = 0.6;
            
            // English language patterns
            else if (selectedText === 'cat' && targetText === 'mat') attention = 0.8;
            else if (selectedText === 'sat' && (targetText === 'cat' || targetText === 'mat')) attention = 0.6;
            else if (selectedText === 'the' && j === selectedTokenIndex + 1) attention = 0.5;
            else if (selectedText === 'attention' && targetText === 'need') attention = 0.7;
            else if (selectedText === 'is' && targetText === 'all') attention = 0.6;
            
            // More Korean patterns
            else if ((selectedText.includes('이') || selectedText.includes('가')) && 
                     (targetText.includes('다') || targetText.includes('니다'))) attention = 0.5;
            
            // General patterns
            else if (Math.abs(j - selectedTokenIndex) === 1) attention = 0.3; // Adjacent words
            else if (j === tokens.length - 1 && selectedTokenIndex === 0) attention = 0.4; // First-last connection
            else if (selectedText === targetText) attention = 0.9; // Same word
            
            // Only draw significant connections
            if (attention > 0.2) {
              // Calculate curve control point
              const midX = (selectedToken.position.x + targetToken.position.x) / 2;
              const midY = selectedToken.position.y - 50;
              
              // Draw curved attention line
              ctx.strokeStyle = `rgba(78, 205, 196, ${attention})`;
              ctx.lineWidth = attention * 6;
              ctx.beginPath();
              ctx.moveTo(selectedToken.position.x, selectedToken.position.y - 20);
              ctx.quadraticCurveTo(midX, midY, targetToken.position.x, targetToken.position.y - 20);
              ctx.stroke();
              
              // Draw arrow head
              const angle = Math.atan2(
                targetToken.position.y - midY,
                targetToken.position.x - midX
              );
              const arrowSize = 8;
              ctx.fillStyle = `rgba(78, 205, 196, ${attention})`;
              ctx.beginPath();
              ctx.moveTo(targetToken.position.x, targetToken.position.y - 20);
              ctx.lineTo(
                targetToken.position.x - arrowSize * Math.cos(angle - Math.PI / 6),
                targetToken.position.y - 20 - arrowSize * Math.sin(angle - Math.PI / 6)
              );
              ctx.lineTo(
                targetToken.position.x - arrowSize * Math.cos(angle + Math.PI / 6),
                targetToken.position.y - 20 - arrowSize * Math.sin(angle + Math.PI / 6)
              );
              ctx.closePath();
              ctx.fill();
              
              // Draw attention score
              ctx.fillStyle = '#374151';
              ctx.font = 'bold 11px Inter';
              ctx.textAlign = 'center';
              ctx.fillText(attention.toFixed(1), midX, midY + 20);
            }
          }
        });
        
        // Highlight selected token
        ctx.strokeStyle = '#3b82f6';
        ctx.lineWidth = 3;
        ctx.strokeRect(
          selectedToken.position.x - 47,
          selectedToken.position.y - 22,
          94,
          44
        );
        
        // Add label
        ctx.fillStyle = '#3b82f6';
        ctx.font = 'bold 12px Inter';
        ctx.textAlign = 'center';
        ctx.fillText('Query', selectedToken.position.x, selectedToken.position.y - 35);
      }
    }

    // Draw multi-head attention for step 5
    if (step === 4) {
      const heads = ['문법', '의미', '문맥', '스타일'];
      heads.forEach((head, i) => {
        ctx.fillStyle = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#9B5DE5'][i];
        ctx.fillRect(200 + i * 90, 150, 70, 30);
        ctx.fillStyle = '#fff';
        ctx.font = '11px Inter';
        ctx.textAlign = 'center';
        ctx.fillText(head, 235 + i * 90, 165);
      });
    }

    // Draw components - better layout with larger text
    const components = [
      { id: 'tokenizer', x: 50, y: 325, width: 85, height: 35, label: 'Tokenizer' },
      { id: 'embedding', x: 145, y: 325, width: 85, height: 35, label: 'Embedding' },
      { id: 'positional', x: 240, y: 325, width: 85, height: 35, label: 'Positional' },
      { id: 'attention', x: 335, y: 325, width: 85, height: 35, label: 'Attention' },
      { id: 'multihead', x: 430, y: 325, width: 85, height: 35, label: 'Multi-Head' },
      { id: 'ffn', x: 525, y: 325, width: 85, height: 35, label: 'FFN' },
      { id: 'residual', x: 620, y: 325, width: 85, height: 35, label: 'Residual' },
      { id: 'norm', x: 715, y: 325, width: 85, height: 35, label: 'LayerNorm' },
    ];

    components.forEach(comp => {
      const isActive = currentStep.activeComponents.includes(comp.id);
      const isSelected = selectedComponent === comp.id;
      
      // Component box with shadow
      ctx.shadowColor = 'rgba(0, 0, 0, 0.08)';
      ctx.shadowBlur = 3;
      ctx.shadowOffsetY = 1;
      
      if (isActive) {
        // Active component gradient
        const compGradient = ctx.createLinearGradient(comp.x, comp.y, comp.x, comp.y + comp.height);
        compGradient.addColorStop(0, '#06b6d4');
        compGradient.addColorStop(1, '#0891b2');
        ctx.fillStyle = compGradient;
      } else if (isSelected) {
        ctx.fillStyle = '#60a5fa';
      } else {
        ctx.fillStyle = '#ffffff';
      }
      
      ctx.fillRect(comp.x, comp.y, comp.width, comp.height);
      
      // Reset shadow
      ctx.shadowColor = 'transparent';
      ctx.shadowBlur = 0;
      ctx.shadowOffsetY = 0;
      
      // Component border
      ctx.strokeStyle = isActive ? '#0891b2' : isSelected ? '#3b82f6' : '#e5e7eb';
      ctx.lineWidth = isActive || isSelected ? 2 : 1;
      ctx.strokeRect(comp.x, comp.y, comp.width, comp.height);
      
      // Component text - larger and clearer
      ctx.fillStyle = isActive || isSelected ? '#ffffff' : '#374151';
      ctx.font = isActive ? 'bold 12px Inter' : '600 12px Inter';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(comp.label, comp.x + comp.width / 2, comp.y + comp.height / 2);
    });
  };

  // 인터랙티브 예제
  const interactiveExamples = [
    { text: "The cat sat on the mat", description: "기본 예제" },
    { text: "Attention is all you need", description: "논문 제목" },
    { text: "인공지능이 세상을 바꾼다", description: "한국어 예제" },
  ];

  return (
    <div className={styles.simulator}>
      <div className={styles.compactHeader}>
        <div>
          <h3>🏗️ Transformer 아키텍처 인터랙티브 학습</h3>
          <div className={styles.stepInfo}>
            <h4>{steps[step].title}</h4>
            <p>{steps[step].description}</p>
          </div>
        </div>
        
        <div className={styles.compactControls}>
          <div className={styles.inputGroup}>
            <input
              type="text"
              value={inputText}
              onChange={(e) => {
                const words = e.target.value.split(' ');
                if (words.length <= 8) {
                  setInputText(e.target.value);
                }
              }}
              className={styles.compactInput}
              placeholder="텍스트 입력... (최대 8단어)"
              maxLength={50}
            />
            <select 
              onChange={(e) => setInputText(e.target.value)}
              className={styles.exampleSelect}
            >
              <option value="">예제 선택</option>
              {interactiveExamples.map((ex, i) => (
                <option key={i} value={ex.text}>{ex.description}</option>
              ))}
            </select>
          </div>
          
          <div className={styles.stepBtns}>
            <button
              onClick={() => setStep(Math.max(0, step - 1))}
              disabled={step === 0}
            >
              ◀
            </button>
            <span>{step + 1}/{steps.length}</span>
            <button
              onClick={() => setStep((step + 1) % steps.length)}
              disabled={step === steps.length - 1}
            >
              ▶
            </button>
          </div>
        </div>
      </div>

      <div className={styles.mainContent}>
        <div className={styles.canvasContainer}>
          <canvas
            ref={canvasRef}
            className={styles.compactCanvas}
            onClick={handleCanvasClick}
            style={{ cursor: step === 3 ? 'pointer' : 'default' }}
            width={900}
            height={380}
          />
        </div>

        {step === 3 && (
          <div className={styles.attentionInfo}>
            <h5>Attention Matrix</h5>
            <div className={styles.miniMatrix}>
              {tokens.slice(0, 6).map((token1, i) => (
                <div key={i} className={styles.matrixRow}>
                  {tokens.slice(0, 6).map((token2, j) => {
                    // Dynamic scores based on selected token
                    let score = 0.05;
                    if (i === selectedTokenIndex) {
                      const token1Text = tokens[i].text.toLowerCase();
                      const token2Text = tokens[j].text.toLowerCase();
                      
                      // Same patterns as canvas visualization
                      if (token1Text.includes('인공지능') && token2Text.includes('바꾼다')) score = 0.8;
                      else if (token1Text.includes('세상') && token2Text.includes('바꾼다')) score = 0.7;
                      else if (token1Text.includes('인공지능') && token2Text.includes('세상')) score = 0.6;
                      else if (token1Text === 'cat' && token2Text === 'mat') score = 0.8;
                      else if (token1Text === 'sat' && (token2Text === 'cat' || token2Text === 'mat')) score = 0.6;
                      else if (token1Text === 'the' && j === i + 1) score = 0.5;
                      else if (token1Text === 'attention' && token2Text === 'need') score = 0.7;
                      else if (token1Text === 'is' && token2Text === 'all') score = 0.6;
                      else if ((token1Text.includes('이') || token1Text.includes('가')) && 
                               (token2Text.includes('다') || token2Text.includes('니다'))) score = 0.5;
                      else if (Math.abs(j - i) === 1) score = 0.3;
                      else if (j === tokens.length - 1 && i === 0) score = 0.4;
                      else if (token1Text === token2Text) score = 0.9;
                      else score = 0.1;
                    }
                    
                    const isSelected = i === selectedTokenIndex;
                    
                    return (
                      <div
                        key={j}
                        className={styles.miniCell}
                        style={{ 
                          backgroundColor: isSelected ? `rgba(78, 205, 196, ${score})` : `rgba(156, 163, 175, ${score})`,
                          color: score > 0.5 ? '#fff' : '#000',
                          border: isSelected ? '2px solid #3b82f6' : '1px solid #e5e7eb'
                        }}
                        title={`${token1.text} → ${token2.text}`}
                      >
                        {score.toFixed(1)}
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
            <p className={styles.miniDesc}>
              선택된 토큰 "{tokens[selectedTokenIndex]?.text}"의 attention 패턴
            </p>
          </div>
        )}
        
        {step === 4 && (
          <div className={styles.multiHeadInfo}>
            <h5>Multi-Head Attention</h5>
            <div className={styles.headList}>
              <span className={styles.headItem}>문법</span>
              <span className={styles.headItem}>의미</span>
              <span className={styles.headItem}>문맥</span>
              <span className={styles.headItem}>스타일</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default TransformerArchitecture3D;