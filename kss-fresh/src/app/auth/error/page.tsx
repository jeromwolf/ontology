import Link from "next/link"

export default function AuthErrorPage() {
  return (
    <div className="min-h-screen bg-gray-900 flex items-center justify-center">
      <div className="text-center">
        <h1 className="text-4xl font-bold text-white mb-4">인증 오류</h1>
        <p className="text-gray-400 mb-8">
          현재 인증 시스템이 일시적으로 비활성화되어 있습니다.
        </p>
        <Link
          href="/"
          className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
        >
          홈으로 돌아가기
        </Link>
      </div>
    </div>
  )
}