'use client';

import { useState, useEffect, useRef } from 'react';
import styles from './Simulators.module.css';

interface TrainingMetrics {
  epoch: number;
  loss: number;
  accuracy: number;
  learningRate: number;
  gradientNorm: number;
}

interface TrainingHistory {
  id: string;
  timestamp: Date;
  hyperparameters: {
    batchSize: number;
    learningRate: number;
    epochs: number;
    optimizer: string;
  };
  finalMetrics: {
    loss: number;
    accuracy: number;
  };
  completed: boolean;
}

const TrainingSimulator = () => {
  const [isTraining, setIsTraining] = useState(false);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [metrics, setMetrics] = useState<TrainingMetrics[]>([]);
  const [hyperparameters, setHyperparameters] = useState({
    batchSize: 32,
    learningRate: 0.001,
    epochs: 50,
    optimizer: 'adam',
    warmupSteps: 1000
  });
  const [selectedMetric, setSelectedMetric] = useState<'loss' | 'accuracy'>('loss');
  const [trainingHistory, setTrainingHistory] = useState<TrainingHistory[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [currentTrainingId, setCurrentTrainingId] = useState<string>('');
  const chartRef = useRef<HTMLCanvasElement>(null);
  const intervalRef = useRef<NodeJS.Timeout>();

  useEffect(() => {
    drawChart();
  }, [metrics, selectedMetric]);

  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  const startTraining = () => {
    setIsTraining(true);
    setCurrentEpoch(0);
    setMetrics([]);
    
    // Create new training session ID
    const trainingId = `training_${Date.now()}`;
    setCurrentTrainingId(trainingId);
    
    // Add to history with initial state
    const newHistory: TrainingHistory = {
      id: trainingId,
      timestamp: new Date(),
      hyperparameters: { ...hyperparameters },
      finalMetrics: { loss: 0, accuracy: 0 },
      completed: false
    };
    setTrainingHistory(prev => [...prev, newHistory]);

    let epoch = 0;
    const baseLosse = 2.5;
    const targetLoss = 0.1;
    const currentSessionId = trainingId; // Capture trainingId for interval closure

    intervalRef.current = setInterval(() => {
      epoch++;
      
      // Simulate training metrics with hyperparameter influence
      const progress = epoch / hyperparameters.epochs;
      const noise = (Math.random() - 0.5) * 0.1;
      
      // Optimizer influence on convergence speed
      const optimizerMultiplier = {
        'adam': 1.0,
        'adamw': 0.95,
        'sgd': 1.5,
        'rmsprop': 1.2,
        'adagrad': 1.3
      }[hyperparameters.optimizer] || 1.0;
      
      // Learning rate influence
      const lrEffect = Math.sqrt(hyperparameters.learningRate / 0.001);
      
      // Batch size influence (larger batch = more stable but slower)
      const batchEffect = Math.sqrt(32 / hyperparameters.batchSize);
      
      // Loss decreases with hyperparameter influence
      const effectiveProgress = progress * lrEffect * batchEffect;
      const loss = baseLosse * Math.exp(-effectiveProgress * 3 / optimizerMultiplier) + targetLoss + noise;
      
      // High learning rate can cause instability
      const instability = hyperparameters.learningRate > 0.005 ? Math.random() * 0.2 : 0;
      const finalLoss = Math.max(targetLoss, loss + instability);
      
      // Accuracy increases
      const accuracy = Math.min(0.98, 1 - finalLoss / baseLosse + noise * 0.05);
      
      // Learning rate with warmup
      let learningRate = hyperparameters.learningRate;
      if (epoch < hyperparameters.warmupSteps / hyperparameters.batchSize) {
        learningRate *= epoch / (hyperparameters.warmupSteps / hyperparameters.batchSize);
      }
      
      // Gradient norm (influenced by optimizer)
      const gradientNorm = 10 * Math.exp(-progress * 2) * optimizerMultiplier + Math.random() * 2;

      const newMetric: TrainingMetrics = {
        epoch,
        loss: finalLoss,
        accuracy,
        learningRate,
        gradientNorm
      };

      setMetrics(prev => [...prev, newMetric]);
      setCurrentEpoch(epoch);

      if (epoch >= hyperparameters.epochs) {
        // Update history with final metrics using captured session ID
        setTrainingHistory(prev => prev.map(h => 
          h.id === currentSessionId 
            ? { ...h, finalMetrics: { loss: finalLoss, accuracy }, completed: true }
            : h
        ));
        stopTraining(false);  // Don't update history again in stopTraining
      }
    }, 200);
  };

  const stopTraining = (updateHistory = true) => {
    setIsTraining(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    
    // If stopped early, update history (unless it's from reset)
    if (updateHistory && metrics.length > 0 && currentTrainingId) {
      const lastMetric = metrics[metrics.length - 1];
      setTrainingHistory(prev => prev.map(h => 
        h.id === currentTrainingId 
          ? { ...h, finalMetrics: { loss: lastMetric.loss, accuracy: lastMetric.accuracy }, completed: false }
          : h
      ));
    }
  };

  const drawChart = () => {
    const canvas = chartRef.current;
    if (!canvas || metrics.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const padding = 40;
    const width = canvas.width = canvas.clientWidth;
    const height = canvas.height = canvas.clientHeight;
    const chartWidth = width - padding * 2;
    const chartHeight = height - padding * 2;

    // Clear canvas with dark mode support
    const isDarkMode = document.documentElement.classList.contains('dark');
    ctx.fillStyle = isDarkMode ? '#1f2937' : '#ffffff';
    ctx.fillRect(0, 0, width, height);

    // Draw axes
    ctx.strokeStyle = isDarkMode ? '#4b5563' : '#e0e0e0';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();

    // Get data range
    const values = metrics.map(m => selectedMetric === 'loss' ? m.loss : m.accuracy);
    const maxValue = Math.max(...values) * 1.1;
    const minValue = Math.min(...values) * 0.9;

    // Draw grid lines
    ctx.strokeStyle = isDarkMode ? '#374151' : '#f0f0f0';
    for (let i = 0; i <= 5; i++) {
      const y = padding + (chartHeight * i) / 5;
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();

      // Y-axis labels
      ctx.fillStyle = isDarkMode ? '#e5e7eb' : '#666';
      ctx.font = '12px Inter';
      ctx.textAlign = 'right';
      const value = maxValue - (maxValue - minValue) * (i / 5);
      ctx.fillText(value.toFixed(3), padding - 10, y + 4);
    }

    // Draw data
    ctx.strokeStyle = selectedMetric === 'loss' ? '#ef4444' : '#22c55e';
    ctx.lineWidth = 2;
    ctx.beginPath();

    metrics.forEach((metric, index) => {
      const x = padding + (chartWidth * index) / (hyperparameters.epochs - 1);
      const value = selectedMetric === 'loss' ? metric.loss : metric.accuracy;
      const y = padding + chartHeight * (1 - (value - minValue) / (maxValue - minValue));

      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    // Draw current point
    if (metrics.length > 0) {
      const lastMetric = metrics[metrics.length - 1];
      const x = padding + (chartWidth * (metrics.length - 1)) / (hyperparameters.epochs - 1);
      const value = selectedMetric === 'loss' ? lastMetric.loss : lastMetric.accuracy;
      const y = padding + chartHeight * (1 - (value - minValue) / (maxValue - minValue));

      ctx.fillStyle = selectedMetric === 'loss' ? '#ef4444' : '#22c55e';
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    }

    // X-axis labels
    ctx.fillStyle = isDarkMode ? '#e5e7eb' : '#666';
    ctx.textAlign = 'center';
    for (let i = 0; i <= 5; i++) {
      const epoch = Math.floor((hyperparameters.epochs * i) / 5);
      const x = padding + (chartWidth * i) / 5;
      ctx.fillText(`Epoch ${epoch}`, x, height - padding + 20);
    }

    // Chart title
    ctx.fillStyle = isDarkMode ? '#f9fafb' : '#000';
    ctx.font = 'bold 14px Inter';
    ctx.textAlign = 'center';
    ctx.fillText(
      selectedMetric === 'loss' ? 'Training Loss' : 'Training Accuracy',
      width / 2,
      20
    );
  };

  const resetSimulation = () => {
    stopTraining(false); // Don't update history when resetting
    setCurrentEpoch(0);
    setMetrics([]);
    setCurrentTrainingId('');
  };

  return (
    <div className={styles.simulator}>
      <div className={styles.header}>
        <h3>🎯 학습 과정 시뮬레이터</h3>
        <p>LLM의 학습 과정을 실시간으로 관찰해보세요</p>
      </div>

      <div className={styles.controls}>
        <div className={styles.hyperparameters}>
          <h4>학습 시뮬레이션 설정</h4>
          <div className={styles.paramGrid}>
            <div className={styles.paramControl}>
              <label className={styles.paramLabel}>배치 크기</label>
              <input
                type="number"
                min="8"
                max="128"
                step="8"
                value={hyperparameters.batchSize}
                onChange={(e) => setHyperparameters({...hyperparameters, batchSize: parseInt(e.target.value) || 32})}
                disabled={isTraining}
                className={styles.paramInput}
              />
            </div>
            <div className={styles.paramControl}>
              <label className={styles.paramLabel}>학습률</label>
              <select
                value={hyperparameters.learningRate}
                onChange={(e) => setHyperparameters({...hyperparameters, learningRate: parseFloat(e.target.value)})}
                disabled={isTraining}
                className={styles.paramSelect}
              >
                <option value="0.0001">0.0001</option>
                <option value="0.0005">0.0005</option>
                <option value="0.001">0.001</option>
                <option value="0.005">0.005</option>
                <option value="0.01">0.01</option>
              </select>
            </div>
            <div className={styles.paramControl}>
              <label className={styles.paramLabel}>에폭 수</label>
              <input
                type="number"
                min="10"
                max="200"
                step="10"
                value={hyperparameters.epochs}
                onChange={(e) => setHyperparameters({...hyperparameters, epochs: parseInt(e.target.value) || 50})}
                disabled={isTraining}
                className={styles.paramInput}
              />
            </div>
            <div className={styles.paramControl}>
              <label className={styles.paramLabel}>옵티마이저</label>
              <select
                value={hyperparameters.optimizer}
                onChange={(e) => setHyperparameters({...hyperparameters, optimizer: e.target.value})}
                disabled={isTraining}
                className={styles.paramSelect}
              >
                <option value="adam">ADAM</option>
                <option value="sgd">SGD</option>
                <option value="rmsprop">RMSprop</option>
                <option value="adagrad">AdaGrad</option>
                <option value="adamw">AdamW</option>
              </select>
            </div>
          </div>
          <p className={styles.simulationNote}>
            실제 GPT-3 학습을 기반으로 한 시뮬레이션입니다
          </p>
        </div>

        <div className={styles.trainingControls}>
          {!isTraining ? (
            <button className={styles.startBtn} onClick={startTraining}>
              학습 시작
            </button>
          ) : (
            <button className={styles.stopBtn} onClick={() => stopTraining()}>
              학습 중지
            </button>
          )}
          <button className={styles.resetBtn} onClick={resetSimulation}>
            초기화
          </button>
          <button 
            className={styles.historyBtn} 
            onClick={() => setShowHistory(!showHistory)}
          >
            {showHistory ? '📊 차트 보기' : '📜 히스토리'}
          </button>
        </div>
      </div>

      <div className={styles.results}>
        <div className={styles.metricsDisplay}>
          <div className={styles.currentMetrics}>
            <h4>현재 상태</h4>
            <div className={styles.metricCards}>
              <div className={styles.metricCard}>
                <span className={styles.metricLabel}>Epoch</span>
                <span className={styles.metricValue}>{currentEpoch} / {hyperparameters.epochs}</span>
              </div>
              {metrics.length > 0 && (
                <>
                  <div className={styles.metricCard}>
                    <span className={styles.metricLabel}>Loss</span>
                    <span className={styles.metricValue}>
                      {metrics[metrics.length - 1].loss.toFixed(4)}
                    </span>
                  </div>
                  <div className={styles.metricCard}>
                    <span className={styles.metricLabel}>Accuracy</span>
                    <span className={styles.metricValue}>
                      {(metrics[metrics.length - 1].accuracy * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className={styles.metricCard}>
                    <span className={styles.metricLabel}>Learning Rate</span>
                    <span className={styles.metricValue}>
                      {metrics[metrics.length - 1].learningRate.toFixed(6)}
                    </span>
                  </div>
                </>
              )}
            </div>
          </div>

          {!showHistory ? (
            <div className={styles.chartSection}>
              <div className={styles.chartControls}>
                <button
                  className={selectedMetric === 'loss' ? styles.active : ''}
                  onClick={() => setSelectedMetric('loss')}
                >
                  Loss
                </button>
                <button
                  className={selectedMetric === 'accuracy' ? styles.active : ''}
                  onClick={() => setSelectedMetric('accuracy')}
                >
                  Accuracy
                </button>
              </div>
              <canvas
                ref={chartRef}
                className={styles.trainingChart}
                width={600}
                height={300}
              />
            </div>
          ) : (
            <div className={styles.historySection}>
              <h4>🏆 학습 히스토리</h4>
              {trainingHistory.length === 0 ? (
                <p className={styles.noHistory}>아직 학습 기록이 없습니다.</p>
              ) : (
                <div className={styles.historyTable}>
                  <table>
                    <thead>
                      <tr>
                        <th>시간</th>
                        <th>배치크기</th>
                        <th>학습률</th>
                        <th>에폭</th>
                        <th>옵티마이저</th>
                        <th>최종 Loss</th>
                        <th>최종 정확도</th>
                        <th>상태</th>
                      </tr>
                    </thead>
                    <tbody>
                      {trainingHistory.slice().reverse().map((history) => (
                        <tr key={history.id}>
                          <td>{new Date(history.timestamp).toLocaleTimeString()}</td>
                          <td>{history.hyperparameters.batchSize}</td>
                          <td>{history.hyperparameters.learningRate}</td>
                          <td>{history.hyperparameters.epochs}</td>
                          <td>{history.hyperparameters.optimizer.toUpperCase()}</td>
                          <td className={styles.lossValue}>
                            {history.finalMetrics.loss === 0 && !history.completed 
                              ? '-' 
                              : history.finalMetrics.loss.toFixed(4)}
                          </td>
                          <td className={styles.accuracyValue}>
                            {history.finalMetrics.accuracy === 0 && !history.completed 
                              ? '-' 
                              : `${(history.finalMetrics.accuracy * 100).toFixed(2)}%`}
                          </td>
                          <td>
                            <span className={
                              history.completed ? styles.completed : 
                              (history.finalMetrics.loss === 0 && history.finalMetrics.accuracy === 0 ? styles.running : styles.stopped)
                            }>
                              {history.completed ? '✅ 완료' : 
                               (history.finalMetrics.loss === 0 && history.finalMetrics.accuracy === 0 ? '⏳ 진행중' : '⏸️ 중단')}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </div>

        <div className={styles.trainingProgress}>
          <h4>학습 진행률</h4>
          <div className={styles.progressBar}>
            <div
              className={styles.progressFill}
              style={{ width: `${(currentEpoch / hyperparameters.epochs) * 100}%` }}
            />
          </div>
          <p>{((currentEpoch / hyperparameters.epochs) * 100).toFixed(1)}% 완료</p>
        </div>

        <div className={styles.explanation}>
          <h4>학습 과정 이해하기</h4>
          <ul>
            <li><strong>Loss:</strong> 모델의 예측과 실제 값의 차이. 낮을수록 좋음</li>
            <li><strong>Accuracy:</strong> 올바른 예측의 비율. 높을수록 좋음</li>
            <li><strong>Learning Rate:</strong> 가중치 업데이트 크기. Warmup으로 점진적 증가</li>
            <li><strong>Batch Size:</strong> 한 번에 처리하는 데이터 수. 메모리와 속도에 영향</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default TrainingSimulator;