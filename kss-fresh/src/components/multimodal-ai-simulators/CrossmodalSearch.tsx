'use client';

import { useState } from 'react';
import {
  Search, Image as ImageIcon, Type, ArrowRight,
  Filter, SlidersHorizontal, TrendingUp, Sparkles
} from 'lucide-react';

interface MediaItem {
  id: string;
  type: 'image' | 'text';
  content: string;
  caption?: string;
  tags: string[];
  emoji: string;
}

interface SearchResult extends MediaItem {
  score: number;
  matchedQuery: string;
}

export default function CrossmodalSearch() {
  const [searchMode, setSearchMode] = useState<'text-to-image' | 'image-to-text'>('text-to-image');
  const [queryText, setQueryText] = useState('');
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [filterModality, setFilterModality] = useState<'all' | 'image' | 'text'>('all');
  const [minScore, setMinScore] = useState(0.5);
  const [isSearching, setIsSearching] = useState(false);

  const mediaDatabase: MediaItem[] = [
    { id: 'img1', type: 'image', content: 'cat.jpg', caption: 'A fluffy orange cat sleeping on a couch', tags: ['animal', 'cat', 'sleeping'], emoji: '🐱' },
    { id: 'img2', type: 'image', content: 'dog.jpg', caption: 'A golden retriever playing in the park', tags: ['animal', 'dog', 'playing'], emoji: '🐶' },
    { id: 'img3', type: 'image', content: 'sunset.jpg', caption: 'Beautiful sunset over the ocean with orange sky', tags: ['nature', 'sunset', 'ocean'], emoji: '🌅' },
    { id: 'img4', type: 'image', content: 'mountain.jpg', caption: 'Snow-capped mountain peaks under blue sky', tags: ['nature', 'mountain', 'snow'], emoji: '⛰️' },
    { id: 'img5', type: 'image', content: 'city.jpg', caption: 'Modern city skyline at night with tall buildings', tags: ['urban', 'city', 'night'], emoji: '🌃' },
    { id: 'img6', type: 'image', content: 'food.jpg', caption: 'Delicious pizza with cheese and toppings', tags: ['food', 'pizza', 'italian'], emoji: '🍕' },
    { id: 'img7', type: 'image', content: 'car.jpg', caption: 'Red sports car on a mountain road', tags: ['vehicle', 'car', 'sports'], emoji: '🚗' },
    { id: 'img8', type: 'image', content: 'beach.jpg', caption: 'Tropical beach with palm trees and clear water', tags: ['nature', 'beach', 'tropical'], emoji: '🏖️' },
    { id: 'text1', type: 'text', content: 'The cat is resting peacefully on the soft cushion', tags: ['animal', 'cat', 'rest'], emoji: '📝' },
    { id: 'text2', type: 'text', content: 'Dogs are known for their loyalty and playful nature', tags: ['animal', 'dog', 'behavior'], emoji: '📝' },
    { id: 'text3', type: 'text', content: 'Watching the sunset is one of the most peaceful experiences', tags: ['nature', 'sunset', 'peace'], emoji: '📝' },
    { id: 'text4', type: 'text', content: 'Mountain climbing requires preparation and determination', tags: ['nature', 'mountain', 'adventure'], emoji: '📝' },
    { id: 'text5', type: 'text', content: 'City life offers convenience but can be overwhelming', tags: ['urban', 'city', 'lifestyle'], emoji: '📝' },
    { id: 'text6', type: 'text', content: 'Pizza is a popular Italian dish loved worldwide', tags: ['food', 'pizza', 'italian'], emoji: '📝' },
  ];

  // Simulate CLIP-style cross-modal search
  const performSearch = () => {
    setIsSearching(true);

    setTimeout(() => {
      let results: SearchResult[] = [];

      if (searchMode === 'text-to-image') {
        // Text query to find similar images
        const queryLower = queryText.toLowerCase();
        results = mediaDatabase
          .filter(item => filterModality === 'all' || item.type === filterModality)
          .map(item => {
            let score = 0;

            // Simple keyword matching simulation
            const keywords = queryLower.split(' ');
            keywords.forEach(keyword => {
              if (item.caption?.toLowerCase().includes(keyword)) score += 0.3;
              if (item.tags.some(tag => tag.includes(keyword))) score += 0.2;
              if (item.content.toLowerCase().includes(keyword)) score += 0.1;
            });

            // Add some randomness to simulate semantic similarity
            score += Math.random() * 0.3;
            score = Math.min(1, score);

            return {
              ...item,
              score,
              matchedQuery: queryText
            };
          })
          .filter(result => result.score >= minScore)
          .sort((a, b) => b.score - a.score)
          .slice(0, 10);
      } else {
        // Image query to find similar texts/images
        const selectedItem = mediaDatabase.find(item => item.id === selectedImage);
        if (selectedItem) {
          results = mediaDatabase
            .filter(item => item.id !== selectedImage)
            .filter(item => filterModality === 'all' || item.type === filterModality)
            .map(item => {
              let score = 0;

              // Tag similarity
              const commonTags = selectedItem.tags.filter(tag =>
                item.tags.includes(tag)
              ).length;
              score += commonTags * 0.3;

              // Semantic similarity simulation
              if (selectedItem.caption && item.caption) {
                const words1 = selectedItem.caption.toLowerCase().split(' ');
                const words2 = item.caption.toLowerCase().split(' ');
                const commonWords = words1.filter(w => words2.includes(w)).length;
                score += (commonWords / words1.length) * 0.4;
              }

              // Add randomness
              score += Math.random() * 0.3;
              score = Math.min(1, score);

              return {
                ...item,
                score,
                matchedQuery: selectedItem.content
              };
            })
            .filter(result => result.score >= minScore)
            .sort((a, b) => b.score - a.score)
            .slice(0, 10);
        }
      }

      setSearchResults(results);
      setIsSearching(false);
    }, 800);
  };

  const handleSearch = () => {
    if (searchMode === 'text-to-image' && !queryText.trim()) {
      alert('검색 텍스트를 입력하세요.');
      return;
    }
    if (searchMode === 'image-to-text' && !selectedImage) {
      alert('이미지를 선택하세요.');
      return;
    }
    performSearch();
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-violet-500 to-purple-600 rounded-xl p-6 text-white">
        <div className="flex items-center gap-3 mb-2">
          <Search size={32} />
          <h2 className="text-2xl font-bold">크로스모달 검색 엔진</h2>
        </div>
        <p className="text-violet-100">
          텍스트로 이미지를 찾거나, 이미지로 관련 텍스트를 검색하세요
        </p>
      </div>

      {/* Search Mode Selection */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-bold mb-4">검색 모드 선택</h3>
        <div className="grid md:grid-cols-2 gap-4">
          <button
            onClick={() => {
              setSearchMode('text-to-image');
              setSearchResults([]);
            }}
            className={`p-6 rounded-lg border-2 transition-all ${
              searchMode === 'text-to-image'
                ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/30'
                : 'border-gray-200 dark:border-gray-700 hover:border-purple-300'
            }`}
          >
            <div className="flex items-center justify-center gap-3 mb-3">
              <Type className="text-green-600" size={32} />
              <ArrowRight size={24} className="text-gray-400" />
              <ImageIcon className="text-blue-600" size={32} />
            </div>
            <h4 className="font-bold mb-2">Text → Image</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              텍스트 쿼리로 관련 이미지 검색
            </p>
          </button>

          <button
            onClick={() => {
              setSearchMode('image-to-text');
              setSearchResults([]);
            }}
            className={`p-6 rounded-lg border-2 transition-all ${
              searchMode === 'image-to-text'
                ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/30'
                : 'border-gray-200 dark:border-gray-700 hover:border-purple-300'
            }`}
          >
            <div className="flex items-center justify-center gap-3 mb-3">
              <ImageIcon className="text-blue-600" size={32} />
              <ArrowRight size={24} className="text-gray-400" />
              <Type className="text-green-600" size={32} />
            </div>
            <h4 className="font-bold mb-2">Image → Text</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              이미지로 유사한 텍스트/이미지 검색
            </p>
          </button>
        </div>
      </div>

      {/* Search Input */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
          <Search className="text-purple-600" />
          {searchMode === 'text-to-image' ? '텍스트 쿼리 입력' : '이미지 선택'}
        </h3>

        {searchMode === 'text-to-image' ? (
          <div>
            <div className="flex gap-3">
              <input
                type="text"
                value={queryText}
                onChange={(e) => setQueryText(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                placeholder="예: a cat sleeping on a couch..."
                className="flex-1 px-4 py-3 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-purple-500"
              />
              <button
                onClick={handleSearch}
                disabled={isSearching}
                className="px-6 py-3 bg-purple-500 hover:bg-purple-600 disabled:bg-gray-400 text-white rounded-lg font-semibold transition-colors"
              >
                {isSearching ? '검색중...' : '검색'}
              </button>
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              <span className="text-sm text-gray-500">예시:</span>
              {['cute cat', 'beautiful sunset', 'mountain landscape', 'delicious food'].map(example => (
                <button
                  key={example}
                  onClick={() => setQueryText(example)}
                  className="px-3 py-1 bg-purple-50 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 text-sm rounded-full hover:bg-purple-100 dark:hover:bg-purple-900/50 transition-colors"
                >
                  {example}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div>
            <div className="grid grid-cols-4 md:grid-cols-6 gap-3 mb-4">
              {mediaDatabase.filter(item => item.type === 'image').map(item => (
                <button
                  key={item.id}
                  onClick={() => setSelectedImage(item.id)}
                  className={`p-4 rounded-lg border-2 transition-all text-center ${
                    selectedImage === item.id
                      ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/30'
                      : 'border-gray-200 dark:border-gray-700 hover:border-purple-300'
                  }`}
                >
                  <div className="text-3xl mb-1">{item.emoji}</div>
                  <div className="text-xs truncate">{item.content}</div>
                </button>
              ))}
            </div>
            <button
              onClick={handleSearch}
              disabled={isSearching || !selectedImage}
              className="w-full px-6 py-3 bg-purple-500 hover:bg-purple-600 disabled:bg-gray-400 text-white rounded-lg font-semibold transition-colors"
            >
              {isSearching ? '검색중...' : '유사한 콘텐츠 찾기'}
            </button>
          </div>
        )}
      </div>

      {/* Filters */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
          <SlidersHorizontal className="text-purple-600" />
          필터 설정
        </h3>
        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-semibold mb-2">결과 타입</label>
            <select
              value={filterModality}
              onChange={(e) => setFilterModality(e.target.value as any)}
              className="w-full px-3 py-2 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg"
            >
              <option value="all">모두 보기</option>
              <option value="image">이미지만</option>
              <option value="text">텍스트만</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-semibold mb-2">
              최소 유사도: {(minScore * 100).toFixed(0)}%
            </label>
            <input
              type="range"
              value={minScore}
              onChange={(e) => setMinScore(parseFloat(e.target.value))}
              min={0}
              max={1}
              step={0.05}
              className="w-full"
            />
          </div>
        </div>
      </div>

      {/* Search Results */}
      {searchResults.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
          <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
            <TrendingUp className="text-purple-600" />
            검색 결과 ({searchResults.length}개)
          </h3>
          <div className="space-y-3">
            {searchResults.map((result, idx) => (
              <div
                key={result.id}
                className="flex items-center gap-4 p-4 bg-gray-50 dark:bg-gray-900 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
              >
                <div className="text-2xl font-bold text-gray-400 w-8">{idx + 1}</div>
                <div className="text-4xl">{result.emoji}</div>
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-semibold">{result.content}</span>
                    <span className={`px-2 py-0.5 text-xs rounded-full ${
                      result.type === 'image'
                        ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300'
                        : 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300'
                    }`}>
                      {result.type.toUpperCase()}
                    </span>
                  </div>
                  {result.caption && (
                    <div className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                      {result.caption}
                    </div>
                  )}
                  <div className="flex flex-wrap gap-1">
                    {result.tags.map(tag => (
                      <span
                        key={tag}
                        className="px-2 py-0.5 bg-purple-50 dark:bg-purple-900/30 text-purple-600 dark:text-purple-300 text-xs rounded-full"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-2xl font-bold text-purple-600">
                    {(result.score * 100).toFixed(0)}%
                  </div>
                  <div className="text-xs text-gray-500">유사도</div>
                </div>
                <div className="w-24">
                  <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-purple-500 to-pink-600 transition-all duration-500"
                      style={{ width: `${result.score * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {searchResults.length === 0 && (searchMode === 'text-to-image' ? queryText : selectedImage) && !isSearching && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-12 shadow-lg text-center">
          <div className="text-gray-400 mb-2">
            <Search size={48} className="mx-auto" />
          </div>
          <p className="text-gray-500">검색 결과가 없습니다. 다른 쿼리를 시도해보세요.</p>
        </div>
      )}

      {/* How It Works */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
          <Sparkles className="text-purple-600" />
          크로스모달 검색 작동 원리
        </h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-semibold mb-2 text-purple-600">1. 공통 임베딩 공간</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              CLIP과 같은 멀티모달 모델은 텍스트와 이미지를 동일한 512차원 공간에 매핑합니다.
            </p>
          </div>
          <div>
            <h4 className="font-semibold mb-2 text-purple-600">2. 의미적 유사도</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              코사인 유사도로 쿼리와 데이터베이스 항목 간 의미적 관련성을 측정합니다.
            </p>
          </div>
          <div>
            <h4 className="font-semibold mb-2 text-purple-600">3. 양방향 검색</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              텍스트 → 이미지, 이미지 → 텍스트 양방향 검색이 동일한 메커니즘으로 작동합니다.
            </p>
          </div>
          <div>
            <h4 className="font-semibold mb-2 text-purple-600">4. Zero-Shot 검색</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              사전 학습된 모델로 새로운 쿼리에도 즉시 대응 가능합니다.
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
import numpy as np

# Load model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Text-to-Image search
text_query = "${queryText || 'a cute cat'}"
images = [Image.open(f) for f in image_paths]

inputs = processor(text=[text_query], images=images, return_tensors="pt", padding=True)
outputs = model(**inputs)

# Compute similarities
text_embeds = outputs.text_embeds
image_embeds = outputs.image_embeds

similarities = torch.nn.functional.cosine_similarity(
    text_embeds.unsqueeze(1),
    image_embeds.unsqueeze(0),
    dim=-1
)

# Get top-k results
top_k = 5
scores, indices = similarities[0].topk(top_k)
for idx, score in zip(indices, scores):
    print(f"Image {idx}: {score.item():.3f}")

# Image-to-Image search (similar process)
query_image = Image.open("query.jpg")
# ... compute embeddings and similarities`}
        </pre>
      </div>

      {/* Stats */}
      <div className="grid md:grid-cols-4 gap-4">
        <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg p-4 text-white">
          <div className="text-3xl font-bold">{mediaDatabase.filter(m => m.type === 'image').length}</div>
          <div className="text-sm text-blue-100">이미지 데이터</div>
        </div>
        <div className="bg-gradient-to-br from-green-500 to-green-600 rounded-lg p-4 text-white">
          <div className="text-3xl font-bold">{mediaDatabase.filter(m => m.type === 'text').length}</div>
          <div className="text-sm text-green-100">텍스트 데이터</div>
        </div>
        <div className="bg-gradient-to-br from-purple-500 to-purple-600 rounded-lg p-4 text-white">
          <div className="text-3xl font-bold">{searchResults.length}</div>
          <div className="text-sm text-purple-100">검색 결과</div>
        </div>
        <div className="bg-gradient-to-br from-orange-500 to-orange-600 rounded-lg p-4 text-white">
          <div className="text-3xl font-bold">
            {searchResults.length > 0 ? (searchResults[0].score * 100).toFixed(0) : 0}%
          </div>
          <div className="text-sm text-orange-100">최고 유사도</div>
        </div>
      </div>
    </div>
  );
}
