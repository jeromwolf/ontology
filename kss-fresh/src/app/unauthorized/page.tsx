import Link from "next/link"
import { ShieldX, Home, ArrowLeft } from "lucide-react"

export default function UnauthorizedPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      <div className="w-full max-w-md">
        <div className="bg-gray-800 shadow-2xl rounded-2xl p-8 border border-gray-700 text-center">
          {/* Icon */}
          <div className="inline-flex items-center justify-center w-20 h-20 bg-orange-500/10 rounded-full mb-6">
            <ShieldX className="text-orange-400" size={40} />
          </div>

          {/* Title */}
          <h1 className="text-2xl font-bold text-white mb-2">접근 권한 없음</h1>
          
          {/* Message */}
          <p className="text-gray-400 mb-8">
            이 페이지에 접근할 권한이 없습니다. 
            더 높은 권한이 필요하거나 관리자의 승인이 필요할 수 있습니다.
          </p>

          {/* Actions */}
          <div className="space-y-3">
            <Link
              href="/dashboard"
              className="w-full px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white font-medium rounded-lg hover:from-blue-600 hover:to-purple-700 transition-all duration-200 flex items-center justify-center"
            >
              <ArrowLeft className="mr-2" size={20} />
              대시보드로 돌아가기
            </Link>

            <Link
              href="/"
              className="w-full px-6 py-3 bg-gray-700 hover:bg-gray-600 text-white font-medium rounded-lg transition-colors flex items-center justify-center"
            >
              <Home className="mr-2" size={20} />
              홈으로 돌아가기
            </Link>
          </div>

          {/* Help Text */}
          <p className="text-gray-400 text-sm mt-8">
            이 페이지에 접근해야 한다고 생각하시면{" "}
            <a href="mailto:admin@kss-platform.com" className="text-blue-400 hover:text-blue-300">
              관리자에게 문의
            </a>
            해주세요.
          </p>
        </div>
      </div>
    </div>
  )
}