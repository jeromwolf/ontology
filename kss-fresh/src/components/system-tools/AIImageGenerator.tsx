'use client'

import { useState } from 'react'
import { Sparkles, Download, Copy, Loader2, Wand2, Image as ImageIcon, Search, ExternalLink } from 'lucide-react'

interface GeneratedImage {
  url: string
  prompt: string
  localPath?: string
  timestamp: number
  source?: 'ai' | 'unsplash'
  author?: {
    name: string
    username: string
    profile_url: string
  }
}

interface UnsplashImage {
  id: string
  description: string
  urls: {
    raw: string
    full: string
    regular: string
    small: string
    thumb: string
  }
  user: {
    name: string
    username: string
    profile_url: string
  }
  download_url: string
  html_url: string
  width: number
  height: number
  color: string
  tags: string[]
}

// 대안 이미지 소스 (API 한도 도달 시 사용)
const fallbackImageSources = [
  {
    name: "Unsplash",
    url: "https://unsplash.com/ko/s/사진/",
    description: "고품질 무료 이미지"
  },
  {
    name: "Pexels", 
    url: "https://www.pexels.com/ko-kr/",
    description: "무료 스톡 사진"
  },
  {
    name: "Pixabay",
    url: "https://pixabay.com/ko/",
    description: "무료 이미지, 벡터, 일러스트"
  }
]

const presetPrompts = [
  {
    name: "Transformer 아키텍처",
    prompt: "Clean technical diagram of Transformer architecture showing encoder and decoder stacks, multi-head attention, feed-forward networks, with clear labels and arrows. Educational style, white background.",
    category: "Architecture"
  },
  {
    name: "Neural Network",
    prompt: "Simple neural network diagram with input layer, hidden layers, and output layer. Show nodes and connections clearly. Clean, educational style with labels.",
    category: "AI/ML"
  },
  {
    name: "Data Flow",
    prompt: "Clean flowchart showing data processing pipeline with boxes, arrows, and decision points. Professional technical diagram style.",
    category: "System"
  },
  {
    name: "Attention Mechanism",
    prompt: "Visualization of attention mechanism in AI, showing how different parts of input connect to output with colored lines and weights. Clear, educational diagram.",
    category: "AI/ML"
  },
  {
    name: "API Architecture",
    prompt: "Clean system architecture diagram showing API endpoints, databases, microservices with clear connections and labels. Technical but easy to understand.",
    category: "Architecture"
  },
  {
    name: "Database Schema",
    prompt: "Entity relationship diagram showing database tables, relationships, and keys. Clean, professional database design diagram.",
    category: "Database"
  }
]

export default function AIImageGenerator() {
  // AI 생성 상태
  const [prompt, setPrompt] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [selectedSize, setSelectedSize] = useState<'1024x1024' | '1024x1792' | '1792x1024'>('1024x1024')
  const [selectedQuality, setSelectedQuality] = useState<'standard' | 'hd'>('standard')
  
  // 통합 이미지 리스트
  const [generatedImages, setGeneratedImages] = useState<GeneratedImage[]>([])
  
  // Unsplash 검색 상태
  const [searchQuery, setSearchQuery] = useState('')
  const [isSearching, setIsSearching] = useState(false)
  const [searchResults, setSearchResults] = useState<UnsplashImage[]>([])
  const [currentTab, setCurrentTab] = useState<'ai' | 'search'>('ai')

  const generateImage = async () => {
    if (!prompt.trim()) return

    setIsGenerating(true)
    try {
      const response = await fetch('/api/generate-image', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt: prompt.trim(),
          size: selectedSize,
          quality: selectedQuality
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const result = await response.json()
      
      if (result.success && result.imageUrl) {
        const newImage: GeneratedImage = {
          url: result.imageUrl,
          prompt: prompt.trim(),
          localPath: result.localPath,
          timestamp: Date.now(),
          source: 'ai'
        }
        setGeneratedImages(prev => [newImage, ...prev])
      } else {
        // API에서 반환된 구체적인 에러 메시지 사용
        const errorMessage = result.error || '이미지 생성에 실패했습니다.'
        
        if (result.errorType === 'billing_limit') {
          alert(`⚠️ 결제 한도 도달\n\n${errorMessage}\n\n💡 팁: 상단의 "Unsplash 검색" 탭을 사용하여 고품질 무료 이미지를 찾아보세요!`)
          // 자동으로 Unsplash 탭으로 전환
          setCurrentTab('search')
        } else if (result.errorType === 'quota_exceeded' || result.errorType === 'rate_limit') {
          alert(`⏱️ 사용량 제한\n\n${errorMessage}\n\n잠시 후 다시 시도해주세요.`)
        } else {
          alert(`❌ 생성 실패\n\n${errorMessage}`)
        }
      }

    } catch (error) {
      console.error('Failed to generate image:', error)
      alert('네트워크 오류가 발생했습니다. 인터넷 연결을 확인하고 다시 시도해주세요.')
    } finally {
      setIsGenerating(false)
    }
  }

  // Unsplash 이미지 검색
  const searchImages = async () => {
    if (!searchQuery.trim()) return

    setIsSearching(true)
    try {
      const response = await fetch(`/api/search-images?query=${encodeURIComponent(searchQuery.trim())}&per_page=12`)
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const result = await response.json()
      setSearchResults(result.results || [])

    } catch (error) {
      console.error('Failed to search images:', error)
      alert('이미지 검색에 실패했습니다. 다시 시도해주세요.')
    } finally {
      setIsSearching(false)
    }
  }

  // Unsplash 이미지 다운로드 및 프로젝트에 추가
  const downloadAndAddImage = async (image: UnsplashImage) => {
    try {
      const filename = `unsplash-${image.id}.jpg`
      
      // 1. 프로젝트에 저장
      const response = await fetch('/api/search-images', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          imageUrl: image.urls.regular,
          filename: filename
        }),
      })

      if (!response.ok) {
        throw new Error('Project save failed')
      }

      const result = await response.json()
      
      if (result.success) {
        // 2. 브라우저에서도 다운로드 (개인 다운로드 폴더)
        await downloadImage(image.urls.regular, filename)
        
        // 3. 히스토리에 추가
        const newImage: GeneratedImage = {
          url: image.urls.regular,
          prompt: searchQuery,
          localPath: result.localPath,
          timestamp: Date.now(),
          source: 'unsplash',
          author: {
            name: image.user.name,
            username: image.user.username,
            profile_url: image.user.profile_url
          }
        }
        setGeneratedImages(prev => [newImage, ...prev])
        alert('이미지가 프로젝트에 저장되고 다운로드 폴더에도 다운로드되었습니다!')
      }

    } catch (error) {
      console.error('Failed to download image:', error)
      alert('이미지 다운로드에 실패했습니다.')
    }
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
    alert('클립보드에 복사되었습니다!')
  }

  const downloadImage = async (imageUrl: string, filename: string) => {
    try {
      // OpenAI 이미지는 CORS 문제로 직접 다운로드가 안 될 수 있으므로
      // 서버를 통해 프록시 다운로드 시도
      const response = await fetch('/api/download-image', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          imageUrl: imageUrl,
          filename: filename
        })
      })

      if (!response.ok) {
        throw new Error('Server download failed')
      }

      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      
      const a = document.createElement('a')
      a.href = url
      a.download = filename
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Download failed:', error)
      // 대안: 새 탭에서 이미지 열기
      try {
        window.open(imageUrl, '_blank')
        alert('다운로드에 실패했습니다. 새 탭에서 이미지를 열었습니다. 우클릭으로 저장하세요.')
      } catch (openError) {
        alert('다운로드에 실패했습니다. 이미지 URL을 클립보드에 복사하겠습니다.')
        copyToClipboard(imageUrl)
      }
    }
  }

  return (
    <div className="p-6 space-y-6">
      {/* Tab Navigation */}
      <div className="flex space-x-1 bg-gray-100 dark:bg-gray-700 p-1 rounded-lg">
        <button
          onClick={() => setCurrentTab('ai')}
          className={`flex-1 py-2 px-4 rounded-lg text-sm font-medium transition-all ${
            currentTab === 'ai' 
              ? 'bg-white dark:bg-gray-600 text-blue-600 dark:text-blue-400 shadow-sm' 
              : 'text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200'
          }`}
        >
          <Sparkles className="w-4 h-4 inline mr-2" />
          AI 이미지 생성
        </button>
        <button
          onClick={() => setCurrentTab('search')}
          className={`flex-1 py-2 px-4 rounded-lg text-sm font-medium transition-all ${
            currentTab === 'search' 
              ? 'bg-white dark:bg-gray-600 text-green-600 dark:text-green-400 shadow-sm' 
              : 'text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200'
          }`}
        >
          <Search className="w-4 h-4 inline mr-2" />
          Unsplash 검색
        </button>
      </div>

      {/* AI Generation Tab */}
      {currentTab === 'ai' && (
        <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            이미지 설명 (프롬프트)
          </label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="생성하고 싶은 이미지를 자세히 설명해주세요..."
            className="w-full h-24 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:text-white resize-none"
          />
        </div>

        {/* Options */}
        <div className="flex flex-wrap gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              크기
            </label>
            <select
              value={selectedSize}
              onChange={(e) => setSelectedSize(e.target.value as any)}
              className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg dark:bg-gray-700 dark:text-white"
            >
              <option value="1024x1024">정사각형 (1024×1024)</option>
              <option value="1024x1792">세로 (1024×1792)</option>
              <option value="1792x1024">가로 (1792×1024)</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              품질
            </label>
            <select
              value={selectedQuality}
              onChange={(e) => setSelectedQuality(e.target.value as any)}
              className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg dark:bg-gray-700 dark:text-white"
            >
              <option value="standard">표준</option>
              <option value="hd">고화질 (HD)</option>
            </select>
          </div>
        </div>

        <button
          onClick={generateImage}
          disabled={isGenerating || !prompt.trim()}
          className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
        >
          {isGenerating ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              생성 중...
            </>
          ) : (
            <>
              <Sparkles className="w-4 h-4" />
              이미지 생성
            </>
          )}
        </button>
      </div>
      )}

      {/* Unsplash Search Tab */}
      {currentTab === 'search' && (
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              이미지 검색
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="검색할 키워드를 입력하세요... (예: laptop, nature, business)"
                className="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent dark:bg-gray-700 dark:text-white"
                onKeyPress={(e) => e.key === 'Enter' && searchImages()}
              />
              <button
                onClick={searchImages}
                disabled={isSearching || !searchQuery.trim()}
                className="px-6 py-2 bg-gradient-to-r from-green-600 to-teal-600 text-white rounded-lg hover:from-green-700 hover:to-teal-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2"
              >
                {isSearching ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    검색 중...
                  </>
                ) : (
                  <>
                    <Search className="w-4 h-4" />
                    검색
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Search Results */}
          {searchResults.length > 0 && (
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                <ImageIcon className="w-5 h-5" />
                검색 결과 ({searchResults.length}개)
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                {searchResults.map((image) => (
                  <div key={image.id} className="group relative bg-white dark:bg-gray-800 rounded-lg overflow-hidden shadow-md hover:shadow-lg transition-all">
                    <div className="aspect-square overflow-hidden">
                      <img
                        src={image.urls.small}
                        alt={image.description || 'Unsplash image'}
                        className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200"
                      />
                    </div>
                    
                    <div className="p-3">
                      <div className="flex items-center gap-2 mb-2">
                        <img
                          src={`https://images.unsplash.com/profile-${image.user.username}?w=32&h=32&fit=crop&crop=face&dpr=1`}
                          alt={image.user.name}
                          className="w-6 h-6 rounded-full"
                          onError={(e) => {
                            const target = e.target as HTMLImageElement;
                            target.src = `https://ui-avatars.com/api/?name=${encodeURIComponent(image.user.name)}&size=32`;
                          }}
                        />
                        <div className="min-w-0 flex-1">
                          <p className="text-xs text-gray-600 dark:text-gray-400 truncate">
                            {image.user.name}
                          </p>
                        </div>
                      </div>
                      
                      {image.description && (
                        <p className="text-sm text-gray-700 dark:text-gray-300 mb-2 line-clamp-2">
                          {image.description}
                        </p>
                      )}
                      
                      <div className="flex gap-2">
                        <button
                          onClick={() => downloadAndAddImage(image)}
                          className="flex-1 px-3 py-2 bg-green-600 text-white rounded text-xs hover:bg-green-700 transition-colors flex items-center justify-center gap-1"
                        >
                          <Download className="w-3 h-3" />
                          다운로드
                        </button>
                        <a
                          href={image.html_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="px-3 py-2 bg-gray-600 text-white rounded text-xs hover:bg-gray-700 transition-colors flex items-center justify-center"
                        >
                          <ExternalLink className="w-3 h-3" />
                        </a>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Preset Prompts */}
      {currentTab === 'ai' && (
      <div>
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
          <Wand2 className="w-5 h-5 inline mr-2" />
          프리셋 프롬프트
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {presetPrompts.map((preset, index) => (
            <div
              key={index}
              onClick={() => setPrompt(preset.prompt)}
              className="p-3 border border-gray-200 dark:border-gray-600 rounded-lg cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
            >
              <div className="flex items-center justify-between mb-1">
                <h4 className="font-medium text-sm text-gray-900 dark:text-white">
                  {preset.name}
                </h4>
                <span className="text-xs text-blue-600 dark:text-blue-400 bg-blue-100 dark:bg-blue-900/30 px-2 py-1 rounded">
                  {preset.category}
                </span>
              </div>
              <p className="text-xs text-gray-600 dark:text-gray-400 line-clamp-2">
                {preset.prompt}
              </p>
            </div>
          ))}
        </div>
      </div>
      )}

      {/* Generated Images */}
      {generatedImages.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            <ImageIcon className="w-5 h-5 inline mr-2" />
            생성된 이미지
          </h3>
          <div className="space-y-6">
            {generatedImages.map((image, index) => (
              <div key={index} className="border border-gray-200 dark:border-gray-600 rounded-xl p-4">
                <div className="flex flex-col lg:flex-row gap-4">
                  <div className="lg:w-1/3">
                    <img
                      src={image.url}
                      alt={image.prompt}
                      className="w-full rounded-lg shadow-md"
                    />
                  </div>
                  <div className="lg:w-2/3 space-y-3">
                    <div>
                      <h4 className="font-medium text-gray-900 dark:text-white mb-1">
                        {image.source === 'ai' ? '프롬프트' : '검색어'}
                      </h4>
                      <p className="text-sm text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-700 p-2 rounded">
                        {image.prompt}
                      </p>
                    </div>
                    
                    {image.author && (
                      <div>
                        <h4 className="font-medium text-gray-900 dark:text-white mb-1">작가 정보</h4>
                        <p className="text-sm text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-700 p-2 rounded">
                          Photo by <a href={image.author.profile_url} target="_blank" rel="noopener noreferrer" className="text-blue-600 dark:text-blue-400 hover:underline">
                            {image.author.name}
                          </a> on <a href="https://unsplash.com" target="_blank" rel="noopener noreferrer" className="text-blue-600 dark:text-blue-400 hover:underline">
                            Unsplash
                          </a>
                        </p>
                      </div>
                    )}
                    
                    {image.localPath && (
                      <div>
                        <h4 className="font-medium text-gray-900 dark:text-white mb-1">사용 코드</h4>
                        <div className="text-sm bg-gray-900 text-gray-100 p-2 rounded font-mono">
                          {`<img src="${image.localPath}" alt="Generated image" className="w-full" />`}
                        </div>
                      </div>
                    )}
                    
                    <div className="flex flex-wrap gap-2">
                      <button
                        onClick={() => downloadImage(image.url, `generated-${image.timestamp}.png`)}
                        className="inline-flex items-center gap-1 px-3 py-1 text-sm bg-green-100 text-green-700 rounded hover:bg-green-200 transition-colors"
                      >
                        <Download className="w-4 h-4" />
                        다운로드
                      </button>
                      <button
                        onClick={() => copyToClipboard(image.url)}
                        className="inline-flex items-center gap-1 px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded hover:bg-blue-200 transition-colors"
                      >
                        <Copy className="w-4 h-4" />
                        URL 복사
                      </button>
                      {image.localPath && (
                        <button
                          onClick={() => copyToClipboard(`<img src="${image.localPath}" alt="Generated image" className="w-full" />`)}
                          className="inline-flex items-center gap-1 px-3 py-1 text-sm bg-purple-100 text-purple-700 rounded hover:bg-purple-200 transition-colors"
                        >
                          <Copy className="w-4 h-4" />
                          코드 복사
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 대안 이미지 소스 */}
      <div className="mt-8 border-t border-gray-200 dark:border-gray-600 pt-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          💡 대안 이미지 소스
        </h3>
        <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
          AI 생성이 어려운 경우, 다음 무료 이미지 사이트를 이용하세요:
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {fallbackImageSources.map((source, index) => (
            <a
              key={index}
              href={source.url}
              target="_blank"
              rel="noopener noreferrer"
              className="p-4 border border-gray-200 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
            >
              <h4 className="font-medium text-gray-900 dark:text-white mb-1">
                {source.name}
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                {source.description}
              </p>
            </a>
          ))}
        </div>
        <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
          <h4 className="font-medium text-blue-900 dark:text-blue-200 mb-2">
            📋 사용 팁
          </h4>
          <ul className="text-sm text-blue-800 dark:text-blue-300 space-y-1">
            <li>• 교육용 이미지는 "educational", "diagram", "illustration" 키워드 사용</li>
            <li>• 라이선스를 확인하고 출처를 명시하세요</li>
            <li>• SVG 아이콘은 Heroicons, Lucide 등의 아이콘 라이브러리 활용</li>
          </ul>
        </div>
      </div>
    </div>
  )
}