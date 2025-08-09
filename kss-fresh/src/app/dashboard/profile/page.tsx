"use client"

import { useState, useEffect } from "react"
import { useSession } from "next-auth/react"
import { useRouter } from "next/navigation"
import { 
  User, 
  Mail, 
  Phone, 
  Building, 
  Target, 
  Globe, 
  Bell, 
  Loader2,
  Save,
  Camera,
  Shield
} from "lucide-react"

interface ProfileData {
  name: string
  email: string
  bio: string
  phone: string
  organization: string
  learningGoals: string
  preferredLang: string
  timezone: string
  notifications: boolean
}

export default function ProfilePage() {
  const { data: session, update } = useSession()
  const router = useRouter()
  
  const [isLoading, setIsLoading] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [message, setMessage] = useState("")
  const [profileData, setProfileData] = useState<ProfileData>({
    name: "",
    email: "",
    bio: "",
    phone: "",
    organization: "",
    learningGoals: "",
    preferredLang: "ko",
    timezone: "Asia/Seoul",
    notifications: true,
  })

  useEffect(() => {
    fetchProfile()
  }, [])

  const fetchProfile = async () => {
    setIsLoading(true)
    try {
      const response = await fetch("/api/user/profile")
      if (response.ok) {
        const data = await response.json()
        setProfileData(data)
      }
    } catch (error) {
      console.error("Failed to fetch profile:", error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target
    setProfileData(prev => ({
      ...prev,
      [name]: type === "checkbox" ? (e.target as HTMLInputElement).checked : value
    }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsSaving(true)
    setMessage("")

    try {
      const response = await fetch("/api/user/profile", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(profileData),
      })

      if (response.ok) {
        setMessage("프로필이 성공적으로 업데이트되었습니다.")
        // Update session if name changed
        if (profileData.name !== session?.user?.name) {
          await update({ name: profileData.name })
        }
      } else {
        throw new Error("Failed to update profile")
      }
    } catch (error) {
      setMessage("프로필 업데이트에 실패했습니다.")
    } finally {
      setIsSaving(false)
    }
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 flex items-center justify-center">
        <Loader2 className="animate-spin text-white" size={40} />
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        {/* Header */}
        <div className="bg-gray-800 rounded-2xl p-8 mb-8 border border-gray-700">
          <h1 className="text-3xl font-bold text-white mb-2">프로필 설정</h1>
          <p className="text-gray-400">계정 정보와 학습 설정을 관리하세요</p>
        </div>

        {/* Message */}
        {message && (
          <div className={`mb-6 p-4 rounded-lg ${
            message.includes("성공") 
              ? "bg-green-500/10 border border-green-500/50 text-green-400" 
              : "bg-red-500/10 border border-red-500/50 text-red-400"
          }`}>
            {message}
          </div>
        )}

        {/* Profile Form */}
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Avatar Section */}
          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
              <User size={20} />
              프로필 사진
            </h2>
            <div className="flex items-center space-x-6">
              {session?.user?.image ? (
                <img
                  src={session.user.image}
                  alt={session.user.name || "User"}
                  className="w-24 h-24 rounded-full border-2 border-blue-500"
                />
              ) : (
                <div className="w-24 h-24 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                  <User size={40} className="text-white" />
                </div>
              )}
              <div>
                <button
                  type="button"
                  className="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors flex items-center gap-2"
                  disabled
                >
                  <Camera size={16} />
                  사진 변경
                </button>
                <p className="text-gray-400 text-sm mt-2">JPG, PNG, GIF (최대 2MB)</p>
              </div>
            </div>
          </div>

          {/* Basic Information */}
          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <h2 className="text-xl font-bold text-white mb-4">기본 정보</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label htmlFor="name" className="block text-sm font-medium text-gray-300 mb-2">
                  이름
                </label>
                <input
                  id="name"
                  name="name"
                  type="text"
                  value={profileData.name}
                  onChange={handleChange}
                  className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="홍길동"
                />
              </div>

              <div>
                <label htmlFor="email" className="block text-sm font-medium text-gray-300 mb-2">
                  <Mail className="inline mr-1" size={16} />
                  이메일
                </label>
                <input
                  id="email"
                  name="email"
                  type="email"
                  value={profileData.email}
                  disabled
                  className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-gray-400 cursor-not-allowed"
                />
              </div>

              <div>
                <label htmlFor="phone" className="block text-sm font-medium text-gray-300 mb-2">
                  <Phone className="inline mr-1" size={16} />
                  전화번호
                </label>
                <input
                  id="phone"
                  name="phone"
                  type="tel"
                  value={profileData.phone}
                  onChange={handleChange}
                  className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="010-1234-5678"
                />
              </div>

              <div>
                <label htmlFor="organization" className="block text-sm font-medium text-gray-300 mb-2">
                  <Building className="inline mr-1" size={16} />
                  소속
                </label>
                <input
                  id="organization"
                  name="organization"
                  type="text"
                  value={profileData.organization}
                  onChange={handleChange}
                  className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="회사/학교"
                />
              </div>
            </div>

            <div className="mt-6">
              <label htmlFor="bio" className="block text-sm font-medium text-gray-300 mb-2">
                자기소개
              </label>
              <textarea
                id="bio"
                name="bio"
                rows={4}
                value={profileData.bio}
                onChange={handleChange}
                className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                placeholder="간단한 자기소개를 작성해주세요"
              />
            </div>
          </div>

          {/* Learning Preferences */}
          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <h2 className="text-xl font-bold text-white mb-4">학습 설정</h2>
            
            <div className="space-y-6">
              <div>
                <label htmlFor="learningGoals" className="block text-sm font-medium text-gray-300 mb-2">
                  <Target className="inline mr-1" size={16} />
                  학습 목표
                </label>
                <textarea
                  id="learningGoals"
                  name="learningGoals"
                  rows={3}
                  value={profileData.learningGoals}
                  onChange={handleChange}
                  className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                  placeholder="어떤 것을 배우고 싶으신가요?"
                />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label htmlFor="preferredLang" className="block text-sm font-medium text-gray-300 mb-2">
                    <Globe className="inline mr-1" size={16} />
                    선호 언어
                  </label>
                  <select
                    id="preferredLang"
                    name="preferredLang"
                    value={profileData.preferredLang}
                    onChange={handleChange}
                    className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="ko">한국어</option>
                    <option value="en">English</option>
                  </select>
                </div>

                <div>
                  <label htmlFor="timezone" className="block text-sm font-medium text-gray-300 mb-2">
                    시간대
                  </label>
                  <select
                    id="timezone"
                    name="timezone"
                    value={profileData.timezone}
                    onChange={handleChange}
                    className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="Asia/Seoul">서울 (GMT+9)</option>
                    <option value="America/New_York">뉴욕 (GMT-5)</option>
                    <option value="Europe/London">런던 (GMT+0)</option>
                    <option value="Asia/Tokyo">도쿄 (GMT+9)</option>
                  </select>
                </div>
              </div>

              <div className="flex items-center">
                <input
                  id="notifications"
                  name="notifications"
                  type="checkbox"
                  checked={profileData.notifications}
                  onChange={handleChange}
                  className="w-4 h-4 bg-gray-700 border-gray-600 rounded text-blue-500 focus:ring-blue-500"
                />
                <label htmlFor="notifications" className="ml-2 text-gray-300 flex items-center">
                  <Bell className="mr-1" size={16} />
                  학습 알림 받기
                </label>
              </div>
            </div>
          </div>

          {/* Account Security */}
          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
              <Shield size={20} />
              계정 보안
            </h2>
            <div className="space-y-4">
              <button
                type="button"
                className="w-full md:w-auto px-6 py-3 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors"
                onClick={() => router.push("/dashboard/security")}
              >
                비밀번호 변경
              </button>
              <p className="text-gray-400 text-sm">
                마지막 비밀번호 변경: {new Date().toLocaleDateString("ko-KR")}
              </p>
            </div>
          </div>

          {/* Submit Button */}
          <div className="flex justify-end">
            <button
              type="submit"
              disabled={isSaving}
              className="px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white font-medium rounded-lg hover:from-blue-600 hover:to-purple-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-800 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center gap-2"
            >
              {isSaving ? (
                <>
                  <Loader2 className="animate-spin" size={20} />
                  저장 중...
                </>
              ) : (
                <>
                  <Save size={20} />
                  변경사항 저장
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}