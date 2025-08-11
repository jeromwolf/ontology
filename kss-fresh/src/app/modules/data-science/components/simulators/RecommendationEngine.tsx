'use client'

import { useState, useEffect, useRef } from 'react'
import { Users, Heart, Star, TrendingUp, Filter, Sparkles, ShoppingBag, Music, Film, Book } from 'lucide-react'

interface User {
  id: string
  name: string
  preferences: { [key: string]: number }
  history: string[]
  demographics: {
    age: number
    gender: string
    location: string
  }
}

interface Item {
  id: string
  title: string
  category: string
  features: { [key: string]: number }
  popularity: number
  rating: number
  tags: string[]
}

interface Recommendation {
  itemId: string
  score: number
  reason: string
  method: string
}

export default function RecommendationEngine() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const matrixRef = useRef<HTMLCanvasElement>(null)
  const [recommendationType, setRecommendationType] = useState<'movie' | 'music' | 'book' | 'product'>('movie')
  const [algorithm, setAlgorithm] = useState<'collaborative' | 'content' | 'hybrid' | 'deeplearning'>('hybrid')
  const [selectedUser, setSelectedUser] = useState<User | null>(null)
  const [recommendations, setRecommendations] = useState<Recommendation[]>([])
  const [showMatrix, setShowMatrix] = useState(false)
  const [filterCategory, setFilterCategory] = useState('all')
  
  // 샘플 사용자 생성
  const generateUsers = (): User[] => {
    const names = ['김민수', '이영희', '박철수', '최지은', '정대한']
    const preferences = ['액션', '로맨스', '코미디', '스릴러', '다큐멘터리']
    
    return names.map((name, i) => ({
      id: `user-${i}`,
      name,
      preferences: preferences.reduce((acc, pref, j) => {
        acc[pref] = Math.random() > 0.5 ? Math.random() : 0
        return acc
      }, {} as { [key: string]: number }),
      history: Array.from({ length: 5 + Math.floor(Math.random() * 10) }, 
        (_, j) => `item-${Math.floor(Math.random() * 50)}`),
      demographics: {
        age: 20 + Math.floor(Math.random() * 40),
        gender: Math.random() > 0.5 ? '남성' : '여성',
        location: ['서울', '부산', '대구', '인천', '광주'][Math.floor(Math.random() * 5)]
      }
    }))
  }
  
  // 샘플 아이템 생성
  const generateItems = (): Item[] => {
    const categories = {
      movie: {
        titles: ['어벤져스', '타이타닉', '기생충', '인터스텔라', '라라랜드', '매드맥스', '조커', '인셉션'],
        tags: ['액션', '로맨스', '스릴러', 'SF', '코미디', '드라마', '판타지']
      },
      music: {
        titles: ['Dynamite', 'Savage', 'Eight', 'Lovesick Girls', 'Life Goes On', 'Panorama'],
        tags: ['팝', '힙합', 'R&B', '댄스', '발라드', '록', '인디']
      },
      book: {
        titles: ['해리포터', '1984', '노르웨이의 숲', '데미안', '어린왕자', '호밀밭의 파수꾼'],
        tags: ['판타지', 'SF', '로맨스', '자기계발', '에세이', '소설']
      },
      product: {
        titles: ['노트북', '헤드폰', '운동화', '가방', '시계', '카메라'],
        tags: ['전자제품', '패션', '스포츠', '액세서리', '가전', '뷰티']
      }
    }
    
    const items: Item[] = []
    const categoryData = categories[recommendationType]
    
    for (let i = 0; i < 50; i++) {
      const title = categoryData.titles[i % categoryData.titles.length] + ` ${Math.floor(i / categoryData.titles.length) + 1}`
      const mainTag = categoryData.tags[Math.floor(Math.random() * categoryData.tags.length)]
      const secondaryTag = categoryData.tags[Math.floor(Math.random() * categoryData.tags.length)]
      
      items.push({
        id: `item-${i}`,
        title,
        category: mainTag,
        features: categoryData.tags.reduce((acc, tag) => {
          acc[tag] = tag === mainTag ? 0.8 + Math.random() * 0.2 : Math.random() * 0.5
          return acc
        }, {} as { [key: string]: number }),
        popularity: Math.random(),
        rating: 3 + Math.random() * 2,
        tags: [mainTag, secondaryTag].filter((tag, index, self) => self.indexOf(tag) === index)
      })
    }
    
    return items
  }
  
  const [users] = useState<User[]>(generateUsers())
  const [items] = useState<Item[]>(generateItems())
  
  // 협업 필터링
  const collaborativeFiltering = (user: User): Recommendation[] => {
    // 사용자 간 유사도 계산
    const similarities = users
      .filter(u => u.id !== user.id)
      .map(otherUser => {
        const commonItems = user.history.filter(item => otherUser.history.includes(item))
        const similarity = commonItems.length / Math.sqrt(user.history.length * otherUser.history.length)
        return { user: otherUser, similarity }
      })
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, 3)
    
    // 유사한 사용자들이 좋아한 아이템 추천
    const recommendedItems = new Map<string, number>()
    
    similarities.forEach(({ user: similarUser, similarity }) => {
      similarUser.history.forEach(itemId => {
        if (!user.history.includes(itemId)) {
          const current = recommendedItems.get(itemId) || 0
          recommendedItems.set(itemId, current + similarity)
        }
      })
    })
    
    return Array.from(recommendedItems.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(([itemId, score]) => ({
        itemId,
        score,
        reason: '비슷한 취향의 사용자들이 좋아한 콘텐츠',
        method: 'Collaborative Filtering'
      }))
  }
  
  // 콘텐츠 기반 필터링
  const contentBasedFiltering = (user: User): Recommendation[] => {
    // 사용자가 좋아한 아이템들의 특성 분석
    const userProfile: { [key: string]: number } = {}
    
    user.history.forEach(itemId => {
      const item = items.find(i => i.id === itemId)
      if (item) {
        Object.entries(item.features).forEach(([feature, value]) => {
          userProfile[feature] = (userProfile[feature] || 0) + value
        })
      }
    })
    
    // 정규화
    const totalWeight = Object.values(userProfile).reduce((sum, val) => sum + val, 0)
    Object.keys(userProfile).forEach(key => {
      userProfile[key] /= totalWeight
    })
    
    // 유사한 아이템 추천
    const recommendations = items
      .filter(item => !user.history.includes(item.id))
      .map(item => {
        let similarity = 0
        Object.entries(item.features).forEach(([feature, value]) => {
          similarity += (userProfile[feature] || 0) * value
        })
        return {
          itemId: item.id,
          score: similarity,
          reason: `선호하는 ${Object.keys(userProfile).sort((a, b) => userProfile[b] - userProfile[a])[0]} 장르`,
          method: 'Content-Based Filtering'
        }
      })
      .sort((a, b) => b.score - a.score)
      .slice(0, 10)
    
    return recommendations
  }
  
  // 하이브리드 추천
  const hybridRecommendation = (user: User): Recommendation[] => {
    const collaborative = collaborativeFiltering(user)
    const contentBased = contentBasedFiltering(user)
    
    // 두 방법의 결과를 결합
    const combined = new Map<string, Recommendation>()
    
    collaborative.forEach(rec => {
      combined.set(rec.itemId, {
        ...rec,
        score: rec.score * 0.5,
        reason: rec.reason,
        method: 'Hybrid (Collaborative)'
      })
    })
    
    contentBased.forEach(rec => {
      const existing = combined.get(rec.itemId)
      if (existing) {
        existing.score += rec.score * 0.5
        existing.reason = '협업 + 콘텐츠 기반 추천'
        existing.method = 'Hybrid'
      } else {
        combined.set(rec.itemId, {
          ...rec,
          score: rec.score * 0.5,
          method: 'Hybrid (Content-Based)'
        })
      }
    })
    
    // 인기도 보정
    combined.forEach((rec, itemId) => {
      const item = items.find(i => i.id === itemId)
      if (item) {
        rec.score += item.popularity * 0.1
      }
    })
    
    return Array.from(combined.values())
      .sort((a, b) => b.score - a.score)
      .slice(0, 10)
  }
  
  // 딥러닝 기반 추천 (시뮬레이션)
  const deepLearningRecommendation = (user: User): Recommendation[] => {
    // 임베딩 시뮬레이션
    const userEmbedding = [
      user.demographics.age / 60,
      user.demographics.gender === '남성' ? 1 : 0,
      Object.values(user.preferences).reduce((a, b) => a + b, 0) / 5,
      user.history.length / 20
    ]
    
    return items
      .filter(item => !user.history.includes(item.id))
      .map(item => {
        // 아이템 임베딩 시뮬레이션
        const itemEmbedding = [
          item.popularity,
          item.rating / 5,
          Object.values(item.features).reduce((a, b) => a + b, 0) / 5,
          item.tags.length / 3
        ]
        
        // 내적으로 유사도 계산
        const score = userEmbedding.reduce((sum, val, i) => sum + val * itemEmbedding[i], 0)
        
        return {
          itemId: item.id,
          score: score + Math.random() * 0.2,
          reason: 'AI가 분석한 취향 패턴',
          method: 'Deep Learning (Neural CF)'
        }
      })
      .sort((a, b) => b.score - a.score)
      .slice(0, 10)
  }
  
  // 추천 실행
  const generateRecommendations = () => {
    if (!selectedUser) return
    
    let recs: Recommendation[] = []
    
    switch (algorithm) {
      case 'collaborative':
        recs = collaborativeFiltering(selectedUser)
        break
      case 'content':
        recs = contentBasedFiltering(selectedUser)
        break
      case 'hybrid':
        recs = hybridRecommendation(selectedUser)
        break
      case 'deeplearning':
        recs = deepLearningRecommendation(selectedUser)
        break
    }
    
    setRecommendations(recs)
  }
  
  // 사용자-아이템 매트릭스 시각화
  const drawUserItemMatrix = () => {
    const canvas = matrixRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    const cellSize = 15
    const width = Math.min(items.length * cellSize, 600)
    const height = users.length * cellSize + 50
    
    canvas.width = width
    canvas.height = height
    
    ctx.clearRect(0, 0, width, height)
    
    // 매트릭스 그리기
    users.forEach((user, i) => {
      items.slice(0, width / cellSize).forEach((item, j) => {
        const interacted = user.history.includes(item.id)
        const recommended = recommendations.some(r => r.itemId === item.id && selectedUser?.id === user.id)
        
        if (interacted) {
          ctx.fillStyle = '#3b82f6'
        } else if (recommended) {
          ctx.fillStyle = '#10b981'
        } else {
          ctx.fillStyle = '#f3f4f6'
        }
        
        ctx.fillRect(j * cellSize, i * cellSize + 50, cellSize - 1, cellSize - 1)
      })
      
      // 사용자 이름
      ctx.fillStyle = selectedUser?.id === user.id ? '#3b82f6' : '#374151'
      ctx.font = selectedUser?.id === user.id ? 'bold 10px sans-serif' : '10px sans-serif'
      ctx.textAlign = 'right'
      ctx.textBaseline = 'middle'
      ctx.fillText(user.name, width + 60, i * cellSize + cellSize / 2 + 50)
    })
    
    // 범례
    ctx.fillStyle = '#374151'
    ctx.font = '12px sans-serif'
    ctx.textAlign = 'left'
    ctx.fillText('사용자-아이템 상호작용 매트릭스', 10, 20)
    
    // 범례 박스
    const legendY = 30
    ctx.fillStyle = '#3b82f6'
    ctx.fillRect(10, legendY, 10, 10)
    ctx.fillStyle = '#374151'
    ctx.fillText('이용 기록', 25, legendY + 8)
    
    ctx.fillStyle = '#10b981'
    ctx.fillRect(100, legendY, 10, 10)
    ctx.fillText('추천', 115, legendY + 8)
    
    ctx.fillStyle = '#f3f4f6'
    ctx.fillRect(180, legendY, 10, 10)
    ctx.fillText('미이용', 195, legendY + 8)
  }
  
  // 추천 시각화
  const drawRecommendationFlow = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    ctx.clearRect(0, 0, 600, 400)
    
    if (!selectedUser || recommendations.length === 0) return
    
    // 사용자 노드
    ctx.fillStyle = '#3b82f6'
    ctx.beginPath()
    ctx.arc(100, 200, 30, 0, Math.PI * 2)
    ctx.fill()
    
    ctx.fillStyle = '#fff'
    ctx.font = 'bold 12px sans-serif'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText(selectedUser.name, 100, 200)
    
    // 추천 아이템들
    recommendations.slice(0, 5).forEach((rec, index) => {
      const item = items.find(i => i.id === rec.itemId)
      if (!item) return
      
      const angle = (index - 2) * 0.3
      const x = 400 + Math.cos(angle) * 100
      const y = 200 + Math.sin(angle) * 100
      
      // 연결선
      ctx.strokeStyle = `rgba(59, 130, 246, ${rec.score})`
      ctx.lineWidth = 2 + rec.score * 3
      ctx.beginPath()
      ctx.moveTo(130, 200)
      ctx.lineTo(x - 30, y)
      ctx.stroke()
      
      // 아이템 노드
      ctx.fillStyle = '#10b981'
      ctx.beginPath()
      ctx.arc(x, y, 25, 0, Math.PI * 2)
      ctx.fill()
      
      // 아이템 아이콘
      ctx.fillStyle = '#fff'
      ctx.font = '16px sans-serif'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      
      const icon = recommendationType === 'movie' ? '🎬' :
                   recommendationType === 'music' ? '🎵' :
                   recommendationType === 'book' ? '📚' : '🛍️'
      ctx.fillText(icon, x, y)
      
      // 점수
      ctx.fillStyle = '#374151'
      ctx.font = '10px sans-serif'
      ctx.fillText(`${(rec.score * 100).toFixed(0)}%`, x, y + 35)
    })
    
    // 알고리즘 이름
    ctx.fillStyle = '#6b7280'
    ctx.font = '14px sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText(
      algorithm === 'collaborative' ? '협업 필터링' :
      algorithm === 'content' ? '콘텐츠 기반' :
      algorithm === 'hybrid' ? '하이브리드' : '딥러닝',
      300, 50
    )
  }
  
  useEffect(() => {
    if (selectedUser) {
      generateRecommendations()
    }
  }, [selectedUser, algorithm, recommendationType])
  
  useEffect(() => {
    drawRecommendationFlow()
    if (showMatrix) {
      drawUserItemMatrix()
    }
  }, [recommendations, showMatrix])
  
  return (
    <div className="w-full max-w-7xl mx-auto">
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
        <h2 className="text-2xl font-bold mb-6">추천 시스템 엔진</h2>
        
        <div className="grid lg:grid-cols-3 gap-6">
          {/* 메인 영역 */}
          <div className="lg:col-span-2 space-y-6">
            {/* 추천 타입 및 알고리즘 선택 */}
            <div className="flex gap-4 flex-wrap">
              <div className="flex gap-2">
                {[
                  { id: 'movie', icon: <Film className="w-4 h-4" />, label: '영화' },
                  { id: 'music', icon: <Music className="w-4 h-4" />, label: '음악' },
                  { id: 'book', icon: <Book className="w-4 h-4" />, label: '도서' },
                  { id: 'product', icon: <ShoppingBag className="w-4 h-4" />, label: '상품' }
                ].map(type => (
                  <button
                    key={type.id}
                    onClick={() => setRecommendationType(type.id as any)}
                    className={`flex items-center gap-2 px-3 py-2 rounded-lg font-medium transition-colors ${
                      recommendationType === type.id
                        ? 'bg-blue-500 text-white'
                        : 'bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600'
                    }`}
                  >
                    {type.icon}
                    {type.label}
                  </button>
                ))}
              </div>
            </div>
            
            {/* 사용자 선택 */}
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
              <h3 className="font-semibold mb-3">사용자 선택</h3>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {users.map(user => (
                  <button
                    key={user.id}
                    onClick={() => setSelectedUser(user)}
                    className={`p-3 rounded-lg border-2 transition-colors ${
                      selectedUser?.id === user.id
                        ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                        : 'border-gray-300 dark:border-gray-600 hover:border-gray-400'
                    }`}
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <Users className="w-4 h-4" />
                      <span className="font-medium">{user.name}</span>
                    </div>
                    <div className="text-xs text-gray-600 dark:text-gray-400">
                      {user.demographics.age}세, {user.demographics.location}
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                      {user.history.length}개 이용
                    </div>
                  </button>
                ))}
              </div>
            </div>
            
            {/* 시각화 */}
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
              <h3 className="font-semibold mb-3">추천 플로우</h3>
              <canvas
                ref={canvasRef}
                width={600}
                height={400}
                className="w-full"
              />
            </div>
            
            {/* 추천 결과 */}
            {selectedUser && recommendations.length > 0 && (
              <div>
                <h3 className="font-semibold mb-3 flex items-center gap-2">
                  <Sparkles className="w-5 h-5" />
                  추천 결과
                </h3>
                <div className="grid gap-3">
                  {recommendations.map((rec, index) => {
                    const item = items.find(i => i.id === rec.itemId)
                    if (!item) return null
                    
                    return (
                      <div
                        key={rec.itemId}
                        className="bg-white dark:bg-gray-700 rounded-lg p-4 border border-gray-200 dark:border-gray-600"
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <div className="flex items-center gap-3">
                              <span className="text-lg font-bold text-gray-400">
                                #{index + 1}
                              </span>
                              <h4 className="font-semibold">{item.title}</h4>
                              <span className="px-2 py-1 bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 rounded text-xs">
                                {item.category}
                              </span>
                            </div>
                            <div className="flex items-center gap-4 mt-2 text-sm">
                              <div className="flex items-center gap-1">
                                <Star className="w-4 h-4 text-yellow-500" />
                                <span>{item.rating.toFixed(1)}</span>
                              </div>
                              <div className="flex items-center gap-1">
                                <TrendingUp className="w-4 h-4 text-green-500" />
                                <span>{(item.popularity * 100).toFixed(0)}% 인기</span>
                              </div>
                            </div>
                            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                              {rec.reason}
                            </p>
                          </div>
                          <div className="text-right">
                            <div className="text-2xl font-bold text-blue-600">
                              {(rec.score * 100).toFixed(0)}%
                            </div>
                            <div className="text-xs text-gray-500">
                              매치율
                            </div>
                          </div>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}
          </div>
          
          {/* 사이드바 */}
          <div className="space-y-6">
            {/* 알고리즘 선택 */}
            <div>
              <h3 className="text-lg font-semibold mb-3">추천 알고리즘</h3>
              <div className="space-y-2">
                {[
                  { id: 'collaborative', name: '협업 필터링', desc: '비슷한 사용자의 선호도 기반' },
                  { id: 'content', name: '콘텐츠 기반', desc: '아이템의 특성 분석' },
                  { id: 'hybrid', name: '하이브리드', desc: '여러 방법을 조합' },
                  { id: 'deeplearning', name: '딥러닝', desc: '신경망 기반 추천' }
                ].map(algo => (
                  <button
                    key={algo.id}
                    onClick={() => setAlgorithm(algo.id as any)}
                    className={`w-full p-3 rounded-lg border-2 text-left transition-colors ${
                      algorithm === algo.id
                        ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                        : 'border-gray-300 dark:border-gray-600 hover:border-gray-400'
                    }`}
                  >
                    <div className="font-medium">{algo.name}</div>
                    <div className="text-xs text-gray-600 dark:text-gray-400">{algo.desc}</div>
                  </button>
                ))}
              </div>
            </div>
            
            {/* 필터 */}
            <div>
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <Filter className="w-5 h-5" />
                필터 옵션
              </h3>
              <div className="space-y-3">
                <div>
                  <label className="text-sm font-medium">카테고리</label>
                  <select
                    value={filterCategory}
                    onChange={(e) => setFilterCategory(e.target.value)}
                    className="w-full mt-1 px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700"
                  >
                    <option value="all">전체</option>
                    <option value="popular">인기</option>
                    <option value="new">신규</option>
                    <option value="trending">트렌딩</option>
                  </select>
                </div>
                
                <button
                  onClick={() => setShowMatrix(!showMatrix)}
                  className="w-full px-4 py-2 bg-purple-500 text-white rounded-lg font-medium hover:bg-purple-600 transition-colors"
                >
                  {showMatrix ? '매트릭스 숨기기' : '사용자-아이템 매트릭스 보기'}
                </button>
              </div>
            </div>
            
            {/* 통계 */}
            {selectedUser && (
              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                <h4 className="font-semibold mb-2">사용자 프로필</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">이용 기록:</span>
                    <span className="font-medium">{selectedUser.history.length}개</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">선호 장르:</span>
                    <span className="font-medium">
                      {Object.entries(selectedUser.preferences)
                        .sort((a, b) => b[1] - a[1])[0]?.[0] || '없음'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">위치:</span>
                    <span className="font-medium">{selectedUser.demographics.location}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
        
        {/* 사용자-아이템 매트릭스 */}
        {showMatrix && (
          <div className="mt-6 bg-gray-50 dark:bg-gray-900 rounded-lg p-4 overflow-x-auto">
            <canvas
              ref={matrixRef}
              className="max-w-full"
            />
          </div>
        )}
      </div>
    </div>
  )
}