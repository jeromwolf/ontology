'use client';

import { useState, useRef, useEffect } from 'react';
import {
  Image as ImageIcon, MessageSquare, Send, Sparkles,
  Eye, Brain, Lightbulb, RefreshCw, Camera
} from 'lucide-react';

interface ImageItem {
  id: string;
  label: string;
  emoji: string;
  description: string;
  exampleQuestions: string[];
}

interface VQAResult {
  question: string;
  answer: string;
  confidence: number;
  attentionRegions: { x: number; y: number; intensity: number }[];
  reasoning: string;
}

export default function VQASystem() {
  const [selectedImage, setSelectedImage] = useState<string>('cat');
  const [question, setQuestion] = useState('');
  const [vqaResults, setVqaResults] = useState<VQAResult[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [showAttention, setShowAttention] = useState(true);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const imageGallery: ImageItem[] = [
    {
      id: 'cat',
      label: '고양이',
      emoji: '🐱',
      description: '소파에 앉아있는 주황색 고양이',
      exampleQuestions: [
        'What color is the cat?',
        'Where is the cat sitting?',
        'Is the cat sleeping?',
        'How many cats are in the image?'
      ]
    },
    {
      id: 'beach',
      label: '해변',
      emoji: '🏖️',
      description: '야자수와 맑은 물이 있는 열대 해변',
      exampleQuestions: [
        'What is the weather like?',
        'Are there palm trees?',
        'Is it a tropical beach?',
        'What color is the water?'
      ]
    },
    {
      id: 'city',
      label: '도시',
      emoji: '🌃',
      description: '밤의 현대적인 도시 스카이라인',
      exampleQuestions: [
        'Is it day or night?',
        'How many buildings are there?',
        'Are the lights on?',
        'What kind of city is this?'
      ]
    },
    {
      id: 'food',
      label: '음식',
      emoji: '🍕',
      description: '치즈와 토핑이 있는 피자',
      exampleQuestions: [
        'What type of food is this?',
        'What toppings are on the pizza?',
        'Is this Italian food?',
        'Does it look delicious?'
      ]
    },
    {
      id: 'mountain',
      label: '산',
      emoji: '⛰️',
      description: '파란 하늘 아래 눈 덮인 산봉우리',
      exampleQuestions: [
        'Is there snow on the mountain?',
        'What is the weather like?',
        'Is it a high mountain?',
        'What color is the sky?'
      ]
    },
    {
      id: 'car',
      label: '자동차',
      emoji: '🚗',
      description: '산길의 빨간 스포츠카',
      exampleQuestions: [
        'What color is the car?',
        'What type of car is it?',
        'Where is the car located?',
        'Is it moving?'
      ]
    }
  ];

  // Simulate VQA processing
  const processVQA = async (q: string) => {
    setIsProcessing(true);

    // Simulate processing delay
    await new Promise(resolve => setTimeout(resolve, 1000));

    const selectedImg = imageGallery.find(img => img.id === selectedImage)!;

    // Generate simulated answer based on question and image
    let answer = '';
    let confidence = 0.85 + Math.random() * 0.1;
    let reasoning = '';

    const qLower = q.toLowerCase();

    // Simple rule-based answering (simulating VQA model)
    if (qLower.includes('color')) {
      if (selectedImage === 'cat') {
        answer = 'The cat is orange/ginger colored.';
        reasoning = 'Detected dominant warm color tones in the main object (cat).';
      } else if (selectedImage === 'car') {
        answer = 'The car is red.';
        reasoning = 'Identified red color in the vehicle region.';
      } else if (selectedImage === 'beach') {
        answer = 'The water is turquoise blue and the sand is white.';
        reasoning = 'Analyzed color distribution in ocean and beach areas.';
      } else {
        answer = 'The image contains various colors.';
        reasoning = 'Multiple color regions detected.';
      }
    } else if (qLower.includes('how many')) {
      if (qLower.includes('cat')) {
        answer = 'There is one cat in the image.';
        reasoning = 'Object detection identified a single cat instance.';
      } else if (qLower.includes('building')) {
        answer = 'There are approximately 5-7 buildings visible.';
        reasoning = 'Counted building structures in the skyline.';
      } else {
        answer = 'Multiple objects are present.';
        reasoning = 'Performed object counting in the scene.';
      }
    } else if (qLower.includes('where')) {
      answer = `The ${selectedImg.label} is ${selectedImg.description.split(' ').slice(-3).join(' ')}.`;
      reasoning = 'Analyzed spatial relationships and scene context.';
    } else if (qLower.includes('is it') || qLower.includes('is the') || qLower.includes('are there')) {
      // Yes/No questions
      const keywords = ['yes', 'no', 'probably', 'likely'];
      answer = `${keywords[Math.floor(Math.random() * 2)]}, ${selectedImg.description.split(' ').slice(0, 5).join(' ')}.`;
      reasoning = 'Binary classification based on visual features.';
      confidence = 0.9 + Math.random() * 0.05;
    } else {
      answer = selectedImg.description;
      reasoning = 'Generated description using image captioning model.';
    }

    // Generate attention map (simulated)
    const attentionRegions: { x: number; y: number; intensity: number }[] = [];
    for (let i = 0; i < 20; i++) {
      attentionRegions.push({
        x: Math.random() * 400,
        y: Math.random() * 300,
        intensity: Math.random() * 0.8 + 0.2
      });
    }

    const result: VQAResult = {
      question: q,
      answer,
      confidence,
      attentionRegions,
      reasoning
    };

    setVqaResults([result, ...vqaResults].slice(0, 5));
    setIsProcessing(false);
  };

  const handleSubmit = () => {
    if (!question.trim()) {
      alert('Please enter a question.');
      return;
    }
    processVQA(question);
  };

  // Draw attention map
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !showAttention || vqaResults.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    // Clear
    ctx.clearRect(0, 0, rect.width, rect.height);

    // Draw base image representation
    const selectedImg = imageGallery.find(img => img.id === selectedImage)!;
    ctx.fillStyle = '#1f2937';
    ctx.fillRect(0, 0, rect.width, rect.height);
    ctx.font = 'bold 80px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(selectedImg.emoji, rect.width / 2, rect.height / 2);

    // Draw attention regions
    const latestResult = vqaResults[0];
    latestResult.attentionRegions.forEach(region => {
      const gradient = ctx.createRadialGradient(
        region.x, region.y, 0,
        region.x, region.y, 30
      );
      gradient.addColorStop(0, `rgba(255, 0, 0, ${region.intensity * 0.6})`);
      gradient.addColorStop(1, 'rgba(255, 0, 0, 0)');
      ctx.fillStyle = gradient;
      ctx.fillRect(region.x - 30, region.y - 30, 60, 60);
    });
  }, [selectedImage, vqaResults, showAttention]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-violet-500 to-purple-600 rounded-xl p-6 text-white">
        <div className="flex items-center gap-3 mb-2">
          <Eye size={32} />
          <h2 className="text-2xl font-bold">Visual Question Answering 시스템</h2>
        </div>
        <p className="text-violet-100">
          이미지에 대한 자연어 질문에 AI가 답변합니다
        </p>
      </div>

      {/* Image Gallery */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
          <ImageIcon className="text-blue-600" />
          이미지 갤러리
        </h3>
        <div className="grid grid-cols-3 md:grid-cols-6 gap-3">
          {imageGallery.map(img => (
            <button
              key={img.id}
              onClick={() => {
                setSelectedImage(img.id);
                setVqaResults([]);
              }}
              className={`p-4 rounded-lg border-2 transition-all text-center ${
                selectedImage === img.id
                  ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/30'
                  : 'border-gray-200 dark:border-gray-700 hover:border-purple-300'
              }`}
            >
              <div className="text-4xl mb-2">{img.emoji}</div>
              <div className="text-xs font-semibold">{img.label}</div>
            </button>
          ))}
        </div>
        <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
          <p className="text-sm text-gray-600 dark:text-gray-400">
            <strong>선택된 이미지:</strong> {imageGallery.find(img => img.id === selectedImage)?.description}
          </p>
        </div>
      </div>

      {/* Question Input */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
          <MessageSquare className="text-green-600" />
          질문 입력
        </h3>
        <div className="flex gap-3">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSubmit()}
            placeholder="Ask a question about the image..."
            className="flex-1 px-4 py-3 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-purple-500"
          />
          <button
            onClick={handleSubmit}
            disabled={isProcessing}
            className="flex items-center gap-2 px-6 py-3 bg-purple-500 hover:bg-purple-600 disabled:bg-gray-400 text-white rounded-lg font-semibold transition-colors"
          >
            <Send size={18} />
            {isProcessing ? '처리중...' : '질문'}
          </button>
        </div>

        {/* Example Questions */}
        <div className="mt-4">
          <p className="text-sm font-semibold text-gray-600 dark:text-gray-400 mb-2">예시 질문:</p>
          <div className="flex flex-wrap gap-2">
            {imageGallery.find(img => img.id === selectedImage)?.exampleQuestions.map((q, idx) => (
              <button
                key={idx}
                onClick={() => setQuestion(q)}
                className="px-3 py-1 bg-purple-50 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 text-sm rounded-full hover:bg-purple-100 dark:hover:bg-purple-900/50 transition-colors"
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Attention Map */}
      {vqaResults.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-bold flex items-center gap-2">
              <Brain className="text-purple-600" />
              어텐션 맵 (Attention Map)
            </h3>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={showAttention}
                onChange={(e) => setShowAttention(e.target.checked)}
                className="w-4 h-4"
              />
              <span className="text-sm">표시</span>
            </label>
          </div>
          <canvas
            ref={canvasRef}
            className="w-full bg-gray-900 rounded-lg border-2 border-purple-300"
            style={{ height: 300 }}
          />
          <p className="mt-3 text-sm text-gray-600 dark:text-gray-400">
            🔴 빨간 영역은 AI가 질문에 답하기 위해 집중한 이미지 부분을 나타냅니다.
          </p>
        </div>
      )}

      {/* VQA Results */}
      {vqaResults.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
          <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
            <Sparkles className="text-purple-600" />
            질문-답변 결과
          </h3>
          <div className="space-y-4">
            {vqaResults.map((result, idx) => (
              <div
                key={idx}
                className={`p-4 rounded-lg border-2 ${
                  idx === 0
                    ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/30'
                    : 'border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900'
                }`}
              >
                <div className="flex items-start gap-3 mb-3">
                  <MessageSquare className="text-blue-600 flex-shrink-0" size={20} />
                  <div className="flex-1">
                    <p className="font-semibold text-blue-600 mb-1">질문:</p>
                    <p>{result.question}</p>
                  </div>
                </div>
                <div className="flex items-start gap-3 mb-3">
                  <Lightbulb className="text-green-600 flex-shrink-0" size={20} />
                  <div className="flex-1">
                    <p className="font-semibold text-green-600 mb-1">답변:</p>
                    <p>{result.answer}</p>
                  </div>
                </div>
                <div className="flex items-center justify-between mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
                  <div className="flex items-center gap-4 text-sm">
                    <span className="text-gray-500">
                      신뢰도: <strong className="text-purple-600">{(result.confidence * 100).toFixed(1)}%</strong>
                    </span>
                    <div className="h-2 w-32 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-purple-500 to-pink-600"
                        style={{ width: `${result.confidence * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
                <div className="mt-3 p-3 bg-white dark:bg-gray-800 rounded text-sm">
                  <p className="text-gray-500 mb-1"><strong>추론 과정:</strong></p>
                  <p className="text-gray-600 dark:text-gray-400">{result.reasoning}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {vqaResults.length === 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-12 shadow-lg text-center">
          <Camera size={48} className="mx-auto text-gray-400 mb-3" />
          <p className="text-gray-500">이미지를 선택하고 질문을 입력하여 시작하세요.</p>
        </div>
      )}

      {/* How VQA Works */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-bold mb-4">🧠 VQA 시스템 작동 원리</h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-semibold mb-2 text-purple-600">1. 이미지 인코딩</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              CNN 또는 Vision Transformer로 이미지를 고차원 특징 벡터로 변환합니다.
            </p>
          </div>
          <div>
            <h4 className="font-semibold mb-2 text-purple-600">2. 질문 인코딩</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              BERT, GPT 등의 언어 모델로 질문을 임베딩합니다.
            </p>
          </div>
          <div>
            <h4 className="font-semibold mb-2 text-purple-600">3. 멀티모달 융합</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Cross-Attention으로 이미지와 질문을 결합하여 관련 영역을 파악합니다.
            </p>
          </div>
          <div>
            <h4 className="font-semibold mb-2 text-purple-600">4. 답변 생성</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              분류기 또는 생성 모델로 자연어 답변을 출력합니다.
            </p>
          </div>
        </div>
      </div>

      {/* Code Example */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-bold mb-4">💻 Python 구현 예제</h3>
        <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
{`from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image

# Load VQA model
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# Prepare inputs
image = Image.open("${selectedImage}.jpg")
question = "${vqaResults.length > 0 ? vqaResults[0].question : 'What is in the image?'}"

# Process
inputs = processor(image, question, return_tensors="pt")
outputs = model(**inputs)

# Get answer
logits = outputs.logits
predicted_idx = logits.argmax(-1).item()
answer = model.config.id2label[predicted_idx]

print(f"Q: {question}")
print(f"A: {answer}")

# Visualize attention (requires additional processing)
attention_weights = outputs.attentions[-1]  # Last layer attention
# ... plot attention map`}
        </pre>
      </div>

      {/* Stats */}
      <div className="grid md:grid-cols-4 gap-4">
        <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg p-4 text-white">
          <div className="text-3xl font-bold">{imageGallery.length}</div>
          <div className="text-sm text-blue-100">이미지 갤러리</div>
        </div>
        <div className="bg-gradient-to-br from-green-500 to-green-600 rounded-lg p-4 text-white">
          <div className="text-3xl font-bold">{vqaResults.length}</div>
          <div className="text-sm text-green-100">질문-답변 기록</div>
        </div>
        <div className="bg-gradient-to-br from-purple-500 to-purple-600 rounded-lg p-4 text-white">
          <div className="text-3xl font-bold">
            {vqaResults.length > 0 ? (vqaResults[0].confidence * 100).toFixed(0) : 0}%
          </div>
          <div className="text-sm text-purple-100">최근 신뢰도</div>
        </div>
        <div className="bg-gradient-to-br from-orange-500 to-orange-600 rounded-lg p-4 text-white">
          <div className="text-3xl font-bold">
            {vqaResults.reduce((sum, r) => sum + r.attentionRegions.length, 0)}
          </div>
          <div className="text-sm text-orange-100">어텐션 포인트</div>
        </div>
      </div>
    </div>
  );
}
