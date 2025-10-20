'use client';

import { useState, useEffect, useRef } from 'react';
import {
  Image as ImageIcon, Type, Search, Sparkles,
  Upload, RefreshCw, Maximize2, TrendingUp, Grid3x3
} from 'lucide-react';

interface EmbeddingPoint {
  id: string;
  type: 'image' | 'text';
  content: string;
  embedding: number[];
  x: number;
  y: number;
}

interface SimilarityResult {
  id: string;
  content: string;
  type: 'image' | 'text';
  similarity: number;
}

export default function CLIPExplorer() {
  const [selectedImage, setSelectedImage] = useState<string>('cat.jpg');
  const [queryText, setQueryText] = useState('a cute cat');
  const [embeddings, setEmbeddings] = useState<EmbeddingPoint[]>([]);
  const [similarityResults, setSimilarityResults] = useState<SimilarityResult[]>([]);
  const [visualizationMode, setVisualizationMode] = useState<'2d' | '3d'>('2d');
  const [isComputing, setIsComputing] = useState(false);
  const [nearestK, setNearestK] = useState(5);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const sampleImages = [
    { id: 'cat.jpg', label: '고양이 사진', emoji: '🐱' },
    { id: 'dog.jpg', label: '강아지 사진', emoji: '🐶' },
    { id: 'car.jpg', label: '자동차 사진', emoji: '🚗' },
    { id: 'sunset.jpg', label: '일몰 사진', emoji: '🌅' },
    { id: 'food.jpg', label: '음식 사진', emoji: '🍕' },
    { id: 'mountain.jpg', label: '산 풍경', emoji: '⛰️' }
  ];

  const sampleTexts = [
    'a cute cat sitting',
    'a friendly dog playing',
    'a red sports car',
    'beautiful sunset over ocean',
    'delicious pizza with toppings',
    'snow-capped mountain peak'
  ];

  // Generate mock CLIP embeddings (512-dimensional, reduced to 2D/3D for visualization)
  const generateEmbedding = (content: string, type: 'image' | 'text'): number[] => {
    const seed = content.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
    const random = (i: number) => {
      const x = Math.sin(seed + i) * 10000;
      return x - Math.floor(x);
    };

    return Array.from({ length: 512 }, (_, i) => random(i) * 2 - 1);
  };

  // Cosine similarity between two embeddings
  const cosineSimilarity = (a: number[], b: number[]): number => {
    const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const magA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const magB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
    return dotProduct / (magA * magB);
  };

  // PCA dimension reduction (mock implementation for visualization)
  const reduceDimensions = (embedding: number[]): { x: number; y: number } => {
    // Simple mock: use first two principal components
    const x = embedding.slice(0, 100).reduce((sum, val) => sum + val, 0) / 100;
    const y = embedding.slice(100, 200).reduce((sum, val) => sum + val, 0) / 100;
    return {
      x: (x + 1) * 200 + 50, // Scale to canvas coordinates
      y: (y + 1) * 150 + 50
    };
  };

  // Initialize embeddings
  useEffect(() => {
    const points: EmbeddingPoint[] = [];

    // Add image embeddings
    sampleImages.forEach((img) => {
      const emb = generateEmbedding(img.label, 'image');
      const pos = reduceDimensions(emb);
      points.push({
        id: img.id,
        type: 'image',
        content: img.label,
        embedding: emb,
        ...pos
      });
    });

    // Add text embeddings
    sampleTexts.forEach((text, idx) => {
      const emb = generateEmbedding(text, 'text');
      const pos = reduceDimensions(emb);
      points.push({
        id: `text-${idx}`,
        type: 'text',
        content: text,
        embedding: emb,
        ...pos
      });
    });

    setEmbeddings(points);
  }, []);

  // Draw embedding space
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || embeddings.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    // Clear
    ctx.fillStyle = '#f9fafb';
    ctx.fillRect(0, 0, rect.width, rect.height);

    // Draw grid
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    for (let i = 0; i <= rect.width; i += 50) {
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i, rect.height);
      ctx.stroke();
    }
    for (let i = 0; i <= rect.height; i += 50) {
      ctx.beginPath();
      ctx.moveTo(0, i);
      ctx.lineTo(rect.width, i);
      ctx.stroke();
    }

    // Draw points
    embeddings.forEach(point => {
      if (point.type === 'image') {
        // Images as circles
        ctx.fillStyle = '#3b82f6';
        ctx.beginPath();
        ctx.arc(point.x, point.y, 8, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#1e40af';
        ctx.lineWidth = 2;
        ctx.stroke();
      } else {
        // Text as squares
        ctx.fillStyle = '#10b981';
        ctx.fillRect(point.x - 6, point.y - 6, 12, 12);
        ctx.strokeStyle = '#047857';
        ctx.lineWidth = 2;
        ctx.strokeRect(point.x - 6, point.y - 6, 12, 12);
      }
    });

    // Draw query point if exists
    const queryEmb = generateEmbedding(queryText, 'text');
    const queryPos = reduceDimensions(queryEmb);
    ctx.fillStyle = '#ef4444';
    ctx.beginPath();
    ctx.arc(queryPos.x, queryPos.y, 10, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#991b1b';
    ctx.lineWidth = 3;
    ctx.stroke();

    // Draw similarity lines
    if (similarityResults.length > 0) {
      similarityResults.slice(0, 3).forEach(result => {
        const point = embeddings.find(p => p.id === result.id);
        if (point) {
          ctx.strokeStyle = `rgba(139, 92, 246, ${result.similarity})`;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(queryPos.x, queryPos.y);
          ctx.lineTo(point.x, point.y);
          ctx.stroke();
        }
      });
    }
  }, [embeddings, queryText, similarityResults]);

  // Compute similarities
  const computeSimilarity = () => {
    setIsComputing(true);

    setTimeout(() => {
      const queryEmb = generateEmbedding(queryText, 'text');
      const results: SimilarityResult[] = embeddings.map(point => ({
        id: point.id,
        content: point.content,
        type: point.type,
        similarity: cosineSimilarity(queryEmb, point.embedding)
      }));

      results.sort((a, b) => b.similarity - a.similarity);
      setSimilarityResults(results.slice(0, nearestK));
      setIsComputing(false);
    }, 500);
  };

  useEffect(() => {
    if (queryText) {
      computeSimilarity();
    }
  }, [queryText, nearestK]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-violet-500 to-purple-600 rounded-xl p-6 text-white">
        <div className="flex items-center gap-3 mb-2">
          <Sparkles size={32} />
          <h2 className="text-2xl font-bold">CLIP 임베딩 탐색기</h2>
        </div>
        <p className="text-violet-100">
          텍스트와 이미지를 공통 임베딩 공간에서 탐색하고 유사도를 계산하세요
        </p>
      </div>

      {/* Input Controls */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Text Query */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
          <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
            <Type className="text-green-600" />
            텍스트 쿼리
          </h3>
          <textarea
            value={queryText}
            onChange={(e) => setQueryText(e.target.value)}
            placeholder="검색할 텍스트를 입력하세요..."
            className="w-full px-4 py-3 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg resize-none focus:ring-2 focus:ring-purple-500"
            rows={3}
          />
          <div className="mt-3 flex flex-wrap gap-2">
            {sampleTexts.slice(0, 3).map((text, idx) => (
              <button
                key={idx}
                onClick={() => setQueryText(text)}
                className="px-3 py-1 bg-green-50 dark:bg-green-900/30 text-green-700 dark:text-green-300 text-sm rounded-full hover:bg-green-100 dark:hover:bg-green-900/50 transition-colors"
              >
                {text}
              </button>
            ))}
          </div>
        </div>

        {/* Image Selection */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
          <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
            <ImageIcon className="text-blue-600" />
            이미지 선택
          </h3>
          <div className="grid grid-cols-3 gap-3">
            {sampleImages.map((img) => (
              <button
                key={img.id}
                onClick={() => setSelectedImage(img.id)}
                className={`p-4 rounded-lg border-2 transition-all text-center ${
                  selectedImage === img.id
                    ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/30'
                    : 'border-gray-200 dark:border-gray-700 hover:border-blue-300'
                }`}
              >
                <div className="text-3xl mb-1">{img.emoji}</div>
                <div className="text-xs">{img.label}</div>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Embedding Space Visualization */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-bold flex items-center gap-2">
            <Grid3x3 className="text-purple-600" />
            임베딩 공간 시각화
          </h3>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 text-sm">
              <div className="w-4 h-4 bg-blue-500 rounded-full"></div>
              <span>이미지</span>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <div className="w-4 h-4 bg-green-500"></div>
              <span>텍스트</span>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <div className="w-4 h-4 bg-red-500 rounded-full"></div>
              <span>쿼리</span>
            </div>
          </div>
        </div>

        <canvas
          ref={canvasRef}
          className="w-full bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700"
          style={{ height: 400 }}
        />

        <div className="mt-4 text-sm text-gray-600 dark:text-gray-400">
          💡 <strong>PCA로 2차원으로 축소된 512차원 CLIP 임베딩 공간</strong>
          <br />
          유사한 의미를 가진 이미지와 텍스트가 가까운 위치에 배치됩니다.
        </div>
      </div>

      {/* Similarity Results */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-bold flex items-center gap-2">
            <TrendingUp className="text-purple-600" />
            유사도 결과 (Top-{nearestK})
          </h3>
          <div className="flex items-center gap-2">
            <label className="text-sm font-semibold">K:</label>
            <input
              type="number"
              value={nearestK}
              onChange={(e) => setNearestK(parseInt(e.target.value))}
              className="w-20 px-3 py-1 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg text-sm"
              min={1}
              max={10}
            />
            <button
              onClick={computeSimilarity}
              disabled={isComputing}
              className="flex items-center gap-2 px-3 py-1 bg-purple-500 hover:bg-purple-600 disabled:bg-gray-400 text-white rounded-lg transition-colors"
            >
              <RefreshCw size={16} className={isComputing ? 'animate-spin' : ''} />
              재계산
            </button>
          </div>
        </div>

        {similarityResults.length > 0 ? (
          <div className="space-y-3">
            {similarityResults.map((result, idx) => {
              const emoji = sampleImages.find(img => img.id === result.id)?.emoji || '📄';
              return (
                <div
                  key={result.id}
                  className="flex items-center gap-4 p-4 bg-gray-50 dark:bg-gray-900 rounded-lg"
                >
                  <div className="text-2xl font-bold text-gray-400 w-8">{idx + 1}</div>
                  <div className="text-3xl">{emoji}</div>
                  <div className="flex-1">
                    <div className="font-semibold">{result.content}</div>
                    <div className="text-sm text-gray-500">
                      Type: {result.type === 'image' ? '이미지' : '텍스트'}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-2xl font-bold text-purple-600">
                      {(result.similarity * 100).toFixed(1)}%
                    </div>
                    <div className="text-xs text-gray-500">코사인 유사도</div>
                  </div>
                  <div className="w-32">
                    <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-purple-500 to-pink-600 transition-all duration-500"
                        style={{ width: `${result.similarity * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="text-center py-12 text-gray-400">
            쿼리를 입력하여 유사도 검색을 시작하세요
          </div>
        )}
      </div>

      {/* Technical Details */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-bold mb-4">🔍 CLIP 작동 원리</h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-semibold mb-2 text-purple-600">1. Dual Encoder 구조</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Vision Encoder (ViT)와 Text Encoder (Transformer)가 각각 이미지와 텍스트를 512차원 벡터로 임베딩합니다.
            </p>
          </div>
          <div>
            <h4 className="font-semibold mb-2 text-purple-600">2. Contrastive Learning</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              4억 개의 (이미지, 텍스트) 쌍으로 학습하여 의미적으로 유사한 것들을 가까운 공간에 배치합니다.
            </p>
          </div>
          <div>
            <h4 className="font-semibold mb-2 text-purple-600">3. Zero-Shot Classification</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              별도의 학습 없이 텍스트 프롬프트만으로 이미지 분류 및 검색이 가능합니다.
            </p>
          </div>
          <div>
            <h4 className="font-semibold mb-2 text-purple-600">4. Cosine Similarity</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              임베딩 벡터 간 코사인 유사도로 의미적 관련성을 측정합니다 (범위: -1 ~ 1).
            </p>
          </div>
        </div>
      </div>

      {/* Code Example */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-bold mb-4">💻 Python 구현 예제</h3>
        <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
{`import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Prepare inputs
image = Image.open("cat.jpg")
text = ["${queryText}", "a dog", "a car"]

# Get embeddings
inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)

# Compute similarity
image_embeds = outputs.image_embeds  # [1, 512]
text_embeds = outputs.text_embeds    # [3, 512]

# Cosine similarity
similarity = torch.nn.functional.cosine_similarity(
    image_embeds.unsqueeze(1),
    text_embeds.unsqueeze(0),
    dim=-1
)

print(f"Similarities: {similarity[0].tolist()}")
# Output: [${similarityResults.slice(0, 3).map(r => (r.similarity).toFixed(3)).join(', ')}]`}
        </pre>
      </div>
    </div>
  );
}
