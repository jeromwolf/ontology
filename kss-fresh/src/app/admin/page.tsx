import { auth } from "@/lib/auth"
import { redirect } from "next/navigation"
import { prisma } from "@/lib/prisma"
import Link from "next/link"
import { 
  Users, 
  BookOpen, 
  BarChart3, 
  Settings,
  Shield,
  Activity,
  TrendingUp,
  UserCheck,
  ChevronRight
} from "lucide-react"

export default async function AdminDashboardPage() {
  const session = await auth()

  if (!session?.user || session.user.role !== "ADMIN") {
    redirect("/unauthorized")
  }

  // Fetch statistics
  const [userCount, studentCount, premiumCount, progressCount] = await Promise.all([
    prisma.user.count(),
    prisma.user.count({ where: { role: "STUDENT" } }),
    prisma.user.count({ where: { role: "PREMIUM_STUDENT" } }),
    prisma.progress.count(),
  ])

  const recentUsers = await prisma.user.findMany({
    take: 5,
    orderBy: { createdAt: "desc" },
    select: {
      id: true,
      email: true,
      name: true,
      role: true,
      createdAt: true,
    },
  })

  const recentProgress = await prisma.progress.findMany({
    take: 5,
    orderBy: { updatedAt: "desc" },
    include: {
      user: {
        select: {
          name: true,
          email: true,
        },
      },
    },
  })

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="bg-gray-800 rounded-2xl p-8 mb-8 border border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-white flex items-center gap-3">
                <Shield className="text-red-400" />
                관리자 대시보드
              </h1>
              <p className="text-gray-400 mt-2">KSS 플랫폼 관리 센터</p>
            </div>
            <div className="text-right">
              <p className="text-gray-400 text-sm">관리자</p>
              <p className="text-white font-medium">{session.user.email}</p>
            </div>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <Users className="text-blue-400" size={24} />
              <span className="text-2xl font-bold text-white">{userCount}</span>
            </div>
            <p className="text-gray-400">전체 사용자</p>
            <div className="mt-2 text-xs text-gray-500">
              학생: {studentCount} | 프리미엄: {premiumCount}
            </div>
          </div>

          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <UserCheck className="text-green-400" size={24} />
              <span className="text-2xl font-bold text-white">{premiumCount}</span>
            </div>
            <p className="text-gray-400">프리미엄 구독자</p>
            <div className="mt-2 text-xs text-green-400">
              {userCount > 0 ? `${Math.round((premiumCount / userCount) * 100)}%` : "0%"} 전환율
            </div>
          </div>

          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <Activity className="text-yellow-400" size={24} />
              <span className="text-2xl font-bold text-white">{progressCount}</span>
            </div>
            <p className="text-gray-400">학습 기록</p>
            <div className="mt-2 text-xs text-gray-500">
              진행 중인 학습
            </div>
          </div>

          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <TrendingUp className="text-purple-400" size={24} />
              <span className="text-2xl font-bold text-white">15+</span>
            </div>
            <p className="text-gray-400">활성 모듈</p>
            <div className="mt-2 text-xs text-gray-500">
              100+ 챕터
            </div>
          </div>
        </div>

        {/* Management Sections */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Recent Users */}
          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-white flex items-center gap-2">
                <Users size={20} />
                최근 가입 사용자
              </h2>
              <Link
                href="/admin/users"
                className="text-blue-400 hover:text-blue-300 text-sm flex items-center gap-1"
              >
                전체 보기
                <ChevronRight size={16} />
              </Link>
            </div>
            <div className="space-y-3">
              {recentUsers.map((user) => (
                <div
                  key={user.id}
                  className="bg-gray-700 rounded-lg p-3 hover:bg-gray-600 transition-colors"
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-white font-medium">
                        {user.name || "이름 없음"}
                      </p>
                      <p className="text-gray-400 text-sm">{user.email}</p>
                    </div>
                    <div className="text-right">
                      <span className={`inline-block px-2 py-1 text-xs rounded-full ${
                        user.role === "PREMIUM_STUDENT"
                          ? "bg-yellow-500/20 text-yellow-400"
                          : "bg-green-500/20 text-green-400"
                      }`}>
                        {user.role === "PREMIUM_STUDENT" ? "프리미엄" : "학생"}
                      </span>
                      <p className="text-gray-500 text-xs mt-1">
                        {new Date(user.createdAt).toLocaleDateString("ko-KR")}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Recent Learning Activity */}
          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-white flex items-center gap-2">
                <Activity size={20} />
                최근 학습 활동
              </h2>
              <Link
                href="/admin/analytics"
                className="text-blue-400 hover:text-blue-300 text-sm flex items-center gap-1"
              >
                분석 보기
                <ChevronRight size={16} />
              </Link>
            </div>
            <div className="space-y-3">
              {recentProgress.map((progress) => (
                <div
                  key={progress.id}
                  className="bg-gray-700 rounded-lg p-3"
                >
                  <div className="flex items-center justify-between mb-2">
                    <p className="text-white text-sm">
                      {progress.user.name || progress.user.email}
                    </p>
                    <span className="text-xs text-gray-400">
                      {new Date(progress.lastAccess).toLocaleDateString("ko-KR")}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <p className="text-gray-400 text-xs">
                      {progress.moduleId} - Chapter {progress.chapterId}
                    </p>
                    <div className="flex items-center gap-2">
                      <div className="w-24 bg-gray-600 rounded-full h-1.5">
                        <div
                          className="bg-blue-500 h-1.5 rounded-full"
                          style={{ width: `${progress.progress}%` }}
                        />
                      </div>
                      <span className="text-xs text-gray-400">{progress.progress}%</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h2 className="text-xl font-bold text-white mb-4">빠른 작업</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Link
              href="/admin/users"
              className="bg-gray-700 hover:bg-gray-600 rounded-lg p-4 transition-colors"
            >
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-white font-medium">사용자 관리</h3>
                  <p className="text-gray-400 text-sm mt-1">
                    사용자 목록, 역할 변경, 계정 관리
                  </p>
                </div>
                <Users className="text-blue-400" size={24} />
              </div>
            </Link>

            <Link
              href="/modules/content-manager"
              className="bg-gray-700 hover:bg-gray-600 rounded-lg p-4 transition-colors"
            >
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-white font-medium">콘텐츠 관리</h3>
                  <p className="text-gray-400 text-sm mt-1">
                    모듈 업데이트, 콘텐츠 검토
                  </p>
                </div>
                <BookOpen className="text-green-400" size={24} />
              </div>
            </Link>

            <Link
              href="/admin/analytics"
              className="bg-gray-700 hover:bg-gray-600 rounded-lg p-4 transition-colors"
            >
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-white font-medium">분석 & 통계</h3>
                  <p className="text-gray-400 text-sm mt-1">
                    사용자 행동, 학습 패턴 분석
                  </p>
                </div>
                <BarChart3 className="text-purple-400" size={24} />
              </div>
            </Link>
          </div>
        </div>
      </div>
    </div>
  )
}