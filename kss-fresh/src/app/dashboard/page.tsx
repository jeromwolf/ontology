import { auth } from "@/lib/auth"
import { redirect } from "next/navigation"
import { prisma } from "@/lib/prisma"
import Link from "next/link"
import { 
  User, 
  BookOpen, 
  Trophy, 
  Clock, 
  TrendingUp,
  ChevronRight,
  Calendar,
  Target,
  Award
} from "lucide-react"

export default async function DashboardPage() {
  const session = await auth()

  if (!session?.user) {
    redirect("/auth/signin")
  }

  // Fetch user's progress data
  const userProgress = await prisma.progress.findMany({
    where: { userId: session.user.id },
    orderBy: { lastAccess: "desc" },
    take: 5,
  })

  const enrollments = await prisma.enrollment.findMany({
    where: { userId: session.user.id },
    orderBy: { enrolledAt: "desc" },
  })

  const totalProgress = userProgress.reduce((acc, p) => acc + p.progress, 0)
  const avgProgress = userProgress.length > 0 ? Math.round(totalProgress / userProgress.length) : 0
  const totalTimeSpent = userProgress.reduce((acc, p) => acc + p.timeSpent, 0)

  const formatTime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    if (hours > 0) return `${hours}시간 ${minutes}분`
    return `${minutes}분`
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="bg-gray-800 rounded-2xl p-8 mb-8 border border-gray-700">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              {session.user.image ? (
                <img
                  src={session.user.image}
                  alt={session.user.name || "User"}
                  className="w-20 h-20 rounded-full border-2 border-blue-500"
                />
              ) : (
                <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                  <User size={40} className="text-white" />
                </div>
              )}
              <div>
                <h1 className="text-3xl font-bold text-white">
                  안녕하세요, {session.user.name || session.user.email}님!
                </h1>
                <p className="text-gray-400 mt-1">
                  {session.user.role === "PREMIUM_STUDENT" ? "프리미엄 " : ""}
                  {session.user.role === "INSTRUCTOR" ? "강사" : "학생"} 계정
                </p>
              </div>
            </div>
            <Link
              href="/dashboard/profile"
              className="px-6 py-3 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors flex items-center gap-2"
            >
              프로필 편집
              <ChevronRight size={20} />
            </Link>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <BookOpen className="text-blue-400" size={24} />
              <span className="text-2xl font-bold text-white">{enrollments.length}</span>
            </div>
            <p className="text-gray-400">수강중인 코스</p>
          </div>

          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <TrendingUp className="text-green-400" size={24} />
              <span className="text-2xl font-bold text-white">{avgProgress}%</span>
            </div>
            <p className="text-gray-400">평균 진도율</p>
          </div>

          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <Clock className="text-yellow-400" size={24} />
              <span className="text-2xl font-bold text-white">{formatTime(totalTimeSpent)}</span>
            </div>
            <p className="text-gray-400">총 학습 시간</p>
          </div>

          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <Trophy className="text-purple-400" size={24} />
              <span className="text-2xl font-bold text-white">
                {userProgress.filter(p => p.completed).length}
              </span>
            </div>
            <p className="text-gray-400">완료한 챕터</p>
          </div>
        </div>

        {/* Recent Activity */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Recent Progress */}
          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
              <Calendar size={20} />
              최근 학습 내역
            </h2>
            <div className="space-y-4">
              {userProgress.length > 0 ? (
                userProgress.map((progress) => (
                  <div
                    key={progress.id}
                    className="bg-gray-700 rounded-lg p-4 hover:bg-gray-600 transition-colors"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-white font-medium">
                        {progress.moduleId} - Chapter {progress.chapterId}
                      </span>
                      <span className="text-xs text-gray-400">
                        {new Date(progress.lastAccess).toLocaleDateString("ko-KR")}
                      </span>
                    </div>
                    <div className="w-full bg-gray-600 rounded-full h-2">
                      <div
                        className="bg-gradient-to-r from-blue-500 to-purple-600 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${progress.progress}%` }}
                      />
                    </div>
                    <div className="mt-2 flex items-center justify-between text-xs text-gray-400">
                      <span>{progress.progress}% 완료</span>
                      <span>{formatTime(progress.timeSpent)} 학습</span>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center py-8 text-gray-400">
                  <BookOpen size={48} className="mx-auto mb-4 opacity-50" />
                  <p>아직 학습 기록이 없습니다</p>
                  <Link
                    href="/modules/ontology"
                    className="mt-4 inline-block text-blue-400 hover:text-blue-300"
                  >
                    학습 시작하기 →
                  </Link>
                </div>
              )}
            </div>
          </div>

          {/* Quick Actions */}
          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
              <Target size={20} />
              빠른 시작
            </h2>
            <div className="space-y-3">
              <Link
                href="/modules/ontology"
                className="block bg-gradient-to-r from-blue-500/10 to-purple-600/10 border border-blue-500/50 rounded-lg p-4 hover:from-blue-500/20 hover:to-purple-600/20 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-white font-medium">온톨로지 기초</h3>
                    <p className="text-gray-400 text-sm mt-1">지식 표현과 추론의 기초를 학습합니다</p>
                  </div>
                  <ChevronRight className="text-blue-400" size={20} />
                </div>
              </Link>

              <Link
                href="/modules/llm"
                className="block bg-gradient-to-r from-green-500/10 to-emerald-600/10 border border-green-500/50 rounded-lg p-4 hover:from-green-500/20 hover:to-emerald-600/20 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-white font-medium">LLM 이해하기</h3>
                    <p className="text-gray-400 text-sm mt-1">대규모 언어 모델의 원리를 탐구합니다</p>
                  </div>
                  <ChevronRight className="text-green-400" size={20} />
                </div>
              </Link>

              <Link
                href="/modules/rag"
                className="block bg-gradient-to-r from-purple-500/10 to-pink-600/10 border border-purple-500/50 rounded-lg p-4 hover:from-purple-500/20 hover:to-pink-600/20 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-white font-medium">RAG 시스템</h3>
                    <p className="text-gray-400 text-sm mt-1">검색 증강 생성 기술을 마스터합니다</p>
                  </div>
                  <ChevronRight className="text-purple-400" size={20} />
                </div>
              </Link>

              {session.user.role === "PREMIUM_STUDENT" && (
                <Link
                  href="/modules/multi-agent"
                  className="block bg-gradient-to-r from-yellow-500/10 to-orange-600/10 border border-yellow-500/50 rounded-lg p-4 hover:from-yellow-500/20 hover:to-orange-600/20 transition-colors"
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="flex items-center gap-2">
                        <h3 className="text-white font-medium">멀티 에이전트</h3>
                        <Award className="text-yellow-400" size={16} />
                      </div>
                      <p className="text-gray-400 text-sm mt-1">협업 AI 시스템을 구축합니다</p>
                    </div>
                    <ChevronRight className="text-yellow-400" size={20} />
                  </div>
                </Link>
              )}
            </div>

            {session.user.role === "STUDENT" && (
              <div className="mt-6 p-4 bg-gradient-to-r from-purple-500/10 to-pink-500/10 rounded-lg border border-purple-500/30">
                <h3 className="text-white font-medium mb-2">프리미엄으로 업그레이드</h3>
                <p className="text-gray-400 text-sm mb-3">
                  고급 시뮬레이터와 AI 멘토 기능을 무제한으로 이용하세요
                </p>
                <Link
                  href="/upgrade"
                  className="inline-block px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white text-sm rounded-lg hover:from-purple-600 hover:to-pink-600 transition-colors"
                >
                  프리미엄 시작하기
                </Link>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}