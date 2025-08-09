import Link from "next/link"
import { 
  Crown, 
  Check, 
  Zap, 
  Users, 
  BookOpen, 
  Cpu,
  Brain,
  Sparkles,
  ArrowRight
} from "lucide-react"

const features = [
  { icon: Brain, text: "AI 멘토 무제한 사용" },
  { icon: Cpu, text: "고급 시뮬레이터 접근" },
  { icon: Users, text: "1:1 튜터링 예약" },
  { icon: BookOpen, text: "모든 프리미엄 콘텐츠" },
  { icon: Zap, text: "우선 지원 서비스" },
  { icon: Sparkles, text: "수료증 발급" },
]

const plans = [
  {
    name: "월간 구독",
    price: "29,900",
    period: "월",
    description: "부담 없이 시작하세요",
    popular: false,
  },
  {
    name: "연간 구독",
    price: "299,000",
    period: "년",
    description: "2개월 무료 혜택",
    popular: true,
    savings: "59,800원 절약",
  },
]

export default function UpgradePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      <div className="container mx-auto px-4 py-12">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-yellow-400 to-orange-500 rounded-full mb-6">
            <Crown size={40} className="text-white" />
          </div>
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-4">
            프리미엄으로 업그레이드
          </h1>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            KSS 플랫폼의 모든 기능을 제한 없이 사용하고
            더 빠르게 성장하세요
          </p>
        </div>

        {/* Features */}
        <div className="max-w-4xl mx-auto mb-12">
          <div className="bg-gray-800 rounded-2xl p-8 border border-gray-700">
            <h2 className="text-2xl font-bold text-white mb-6 text-center">
              프리미엄 혜택
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {features.map((feature, index) => (
                <div key={index} className="flex items-start space-x-3">
                  <div className="flex-shrink-0 w-10 h-10 bg-gradient-to-br from-yellow-400/20 to-orange-500/20 rounded-lg flex items-center justify-center">
                    <feature.icon className="text-yellow-400" size={20} />
                  </div>
                  <div>
                    <p className="text-white font-medium">{feature.text}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Pricing Plans */}
        <div className="max-w-4xl mx-auto mb-12">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {plans.map((plan, index) => (
              <div
                key={index}
                className={`relative bg-gray-800 rounded-2xl p-8 border ${
                  plan.popular
                    ? "border-yellow-500 shadow-xl shadow-yellow-500/20"
                    : "border-gray-700"
                }`}
              >
                {plan.popular && (
                  <div className="absolute -top-4 left-1/2 -translate-x-1/2">
                    <span className="px-4 py-1 bg-gradient-to-r from-yellow-400 to-orange-500 text-white text-sm font-bold rounded-full">
                      가장 인기
                    </span>
                  </div>
                )}
                
                <div className="text-center mb-6">
                  <h3 className="text-2xl font-bold text-white mb-2">{plan.name}</h3>
                  <p className="text-gray-400 text-sm">{plan.description}</p>
                  {plan.savings && (
                    <p className="text-green-400 text-sm mt-1">{plan.savings}</p>
                  )}
                </div>

                <div className="text-center mb-6">
                  <span className="text-4xl font-bold text-white">₩{plan.price}</span>
                  <span className="text-gray-400">/{plan.period}</span>
                </div>

                <button
                  className={`w-full py-3 px-4 font-medium rounded-lg transition-all duration-200 ${
                    plan.popular
                      ? "bg-gradient-to-r from-yellow-400 to-orange-500 text-white hover:from-yellow-500 hover:to-orange-600"
                      : "bg-gray-700 text-white hover:bg-gray-600"
                  }`}
                >
                  시작하기
                  <ArrowRight className="inline ml-2" size={16} />
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* Comparison Table */}
        <div className="max-w-4xl mx-auto mb-12">
          <div className="bg-gray-800 rounded-2xl p-8 border border-gray-700">
            <h2 className="text-2xl font-bold text-white mb-6 text-center">
              플랜 비교
            </h2>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-700">
                    <th className="text-left py-4 px-4 text-gray-400">기능</th>
                    <th className="text-center py-4 px-4 text-gray-400">무료</th>
                    <th className="text-center py-4 px-4 text-yellow-400">프리미엄</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-gray-700/50">
                    <td className="py-4 px-4 text-white">기본 콘텐츠</td>
                    <td className="text-center py-4 px-4">
                      <Check className="inline text-green-400" size={20} />
                    </td>
                    <td className="text-center py-4 px-4">
                      <Check className="inline text-green-400" size={20} />
                    </td>
                  </tr>
                  <tr className="border-b border-gray-700/50">
                    <td className="py-4 px-4 text-white">기본 시뮬레이터</td>
                    <td className="text-center py-4 px-4">
                      <Check className="inline text-green-400" size={20} />
                    </td>
                    <td className="text-center py-4 px-4">
                      <Check className="inline text-green-400" size={20} />
                    </td>
                  </tr>
                  <tr className="border-b border-gray-700/50">
                    <td className="py-4 px-4 text-white">고급 시뮬레이터</td>
                    <td className="text-center py-4 px-4 text-gray-500">-</td>
                    <td className="text-center py-4 px-4">
                      <Check className="inline text-green-400" size={20} />
                    </td>
                  </tr>
                  <tr className="border-b border-gray-700/50">
                    <td className="py-4 px-4 text-white">AI 멘토</td>
                    <td className="text-center py-4 px-4 text-gray-400">일일 5회</td>
                    <td className="text-center py-4 px-4 text-yellow-400">무제한</td>
                  </tr>
                  <tr className="border-b border-gray-700/50">
                    <td className="py-4 px-4 text-white">1:1 튜터링</td>
                    <td className="text-center py-4 px-4 text-gray-500">-</td>
                    <td className="text-center py-4 px-4">
                      <Check className="inline text-green-400" size={20} />
                    </td>
                  </tr>
                  <tr className="border-b border-gray-700/50">
                    <td className="py-4 px-4 text-white">소스코드 다운로드</td>
                    <td className="text-center py-4 px-4 text-gray-500">-</td>
                    <td className="text-center py-4 px-4">
                      <Check className="inline text-green-400" size={20} />
                    </td>
                  </tr>
                  <tr>
                    <td className="py-4 px-4 text-white">수료증 발급</td>
                    <td className="text-center py-4 px-4 text-gray-500">-</td>
                    <td className="text-center py-4 px-4">
                      <Check className="inline text-green-400" size={20} />
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* FAQ */}
        <div className="max-w-4xl mx-auto">
          <div className="bg-gray-800 rounded-2xl p-8 border border-gray-700">
            <h2 className="text-2xl font-bold text-white mb-6 text-center">
              자주 묻는 질문
            </h2>
            <div className="space-y-6">
              <div>
                <h3 className="text-white font-medium mb-2">언제든지 취소할 수 있나요?</h3>
                <p className="text-gray-400">
                  네, 언제든지 구독을 취소할 수 있습니다. 취소 후에도 구독 기간이 끝날 때까지 프리미엄 기능을 이용할 수 있습니다.
                </p>
              </div>
              <div>
                <h3 className="text-white font-medium mb-2">환불 정책은 어떻게 되나요?</h3>
                <p className="text-gray-400">
                  구매 후 7일 이내에는 전액 환불이 가능합니다. 이후에는 남은 기간에 대해 부분 환불을 받을 수 있습니다.
                </p>
              </div>
              <div>
                <h3 className="text-white font-medium mb-2">기업 할인이 있나요?</h3>
                <p className="text-gray-400">
                  5명 이상의 팀 구독 시 특별 할인을 제공합니다. 자세한 내용은 문의해주세요.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Back Link */}
        <div className="text-center mt-12">
          <Link
            href="/dashboard"
            className="text-gray-400 hover:text-white transition-colors"
          >
            ← 대시보드로 돌아가기
          </Link>
        </div>
      </div>
    </div>
  )
}