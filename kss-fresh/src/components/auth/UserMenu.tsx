"use client"

import { useState } from "react"
import { useSession, signOut } from "next-auth/react"
import Link from "next/link"
import { 
  User, 
  LogOut, 
  Settings, 
  BookOpen, 
  Shield,
  ChevronDown,
  Loader2,
  Crown
} from "lucide-react"

export default function UserMenu() {
  const { data: session, status } = useSession()
  const [isOpen, setIsOpen] = useState(false)

  if (status === "loading") {
    return (
      <div className="flex items-center space-x-2">
        <Loader2 className="animate-spin text-gray-400" size={20} />
      </div>
    )
  }

  if (!session?.user) {
    return (
      <div className="flex items-center space-x-3">
        <Link
          href="/auth/signin"
          className="px-4 py-2 text-gray-300 hover:text-white transition-colors"
        >
          로그인
        </Link>
        <Link
          href="/auth/signup"
          className="px-4 py-2 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-lg hover:from-blue-600 hover:to-purple-700 transition-all"
        >
          회원가입
        </Link>
      </div>
    )
  }

  const handleSignOut = async () => {
    await signOut({ callbackUrl: "/" })
  }

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center space-x-3 px-4 py-2 rounded-lg hover:bg-gray-700 transition-colors"
      >
        {session.user.image ? (
          <img
            src={session.user.image}
            alt={session.user.name || "User"}
            className="w-8 h-8 rounded-full border border-gray-600"
          />
        ) : (
          <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
            <User size={16} className="text-white" />
          </div>
        )}
        <span className="text-gray-300 hidden md:block">
          {session.user.name || session.user.email}
        </span>
        {session.user.role === "PREMIUM_STUDENT" && (
          <Crown className="text-yellow-400" size={16} />
        )}
        <ChevronDown
          className={`text-gray-400 transition-transform ${isOpen ? "rotate-180" : ""}`}
          size={16}
        />
      </button>

      {isOpen && (
        <>
          <div
            className="fixed inset-0 z-10"
            onClick={() => setIsOpen(false)}
          />
          <div className="absolute right-0 mt-2 w-64 bg-gray-800 rounded-lg shadow-xl border border-gray-700 z-20">
            <div className="px-4 py-3 border-b border-gray-700">
              <p className="text-white font-medium">
                {session.user.name || "사용자"}
              </p>
              <p className="text-gray-400 text-sm">{session.user.email}</p>
              <div className="mt-2">
                <span className={`inline-block px-2 py-1 text-xs rounded-full ${
                  session.user.role === "ADMIN" 
                    ? "bg-red-500/20 text-red-400"
                    : session.user.role === "INSTRUCTOR"
                    ? "bg-blue-500/20 text-blue-400"
                    : session.user.role === "PREMIUM_STUDENT"
                    ? "bg-yellow-500/20 text-yellow-400"
                    : "bg-green-500/20 text-green-400"
                }`}>
                  {session.user.role === "ADMIN" && "관리자"}
                  {session.user.role === "INSTRUCTOR" && "강사"}
                  {session.user.role === "PREMIUM_STUDENT" && "프리미엄"}
                  {session.user.role === "STUDENT" && "학생"}
                </span>
              </div>
            </div>

            <div className="py-2">
              <Link
                href="/dashboard"
                className="flex items-center px-4 py-2 text-gray-300 hover:bg-gray-700 hover:text-white transition-colors"
                onClick={() => setIsOpen(false)}
              >
                <BookOpen className="mr-3" size={16} />
                대시보드
              </Link>
              
              <Link
                href="/dashboard/profile"
                className="flex items-center px-4 py-2 text-gray-300 hover:bg-gray-700 hover:text-white transition-colors"
                onClick={() => setIsOpen(false)}
              >
                <Settings className="mr-3" size={16} />
                프로필 설정
              </Link>

              {(session.user.role === "ADMIN" || session.user.role === "INSTRUCTOR") && (
                <>
                  <div className="my-2 border-t border-gray-700" />
                  {session.user.role === "ADMIN" && (
                    <Link
                      href="/admin"
                      className="flex items-center px-4 py-2 text-gray-300 hover:bg-gray-700 hover:text-white transition-colors"
                      onClick={() => setIsOpen(false)}
                    >
                      <Shield className="mr-3" size={16} />
                      관리자 패널
                    </Link>
                  )}
                  <Link
                    href="/modules/content-manager"
                    className="flex items-center px-4 py-2 text-gray-300 hover:bg-gray-700 hover:text-white transition-colors"
                    onClick={() => setIsOpen(false)}
                  >
                    <Settings className="mr-3" size={16} />
                    콘텐츠 관리
                  </Link>
                </>
              )}

              {session.user.role === "STUDENT" && (
                <>
                  <div className="my-2 border-t border-gray-700" />
                  <Link
                    href="/upgrade"
                    className="flex items-center px-4 py-2 text-yellow-400 hover:bg-gray-700 transition-colors"
                    onClick={() => setIsOpen(false)}
                  >
                    <Crown className="mr-3" size={16} />
                    프리미엄 업그레이드
                  </Link>
                </>
              )}
            </div>

            <div className="border-t border-gray-700 py-2">
              <button
                onClick={handleSignOut}
                className="flex items-center w-full px-4 py-2 text-gray-300 hover:bg-gray-700 hover:text-white transition-colors"
              >
                <LogOut className="mr-3" size={16} />
                로그아웃
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  )
}