export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-primary-50 to-primary-100">
      <div className="container mx-auto px-4 py-16">
        <h1 className="text-5xl font-bold text-primary-900 mb-6">
          온톨로지 학습 플랫폼
        </h1>
        <p className="text-xl text-primary-700 mb-8">
          시맨틱 웹과 온톨로지의 세계로 오신 것을 환영합니다
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-semibold text-primary-800 mb-3">
              기초 개념
            </h2>
            <p className="text-gray-600">
              온톨로지의 기본 개념부터 시작하세요
            </p>
          </div>
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-semibold text-primary-800 mb-3">
              실습 프로젝트
            </h2>
            <p className="text-gray-600">
              실제 프로젝트를 통해 배워보세요
            </p>
          </div>
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-semibold text-primary-800 mb-3">
              고급 주제
            </h2>
            <p className="text-gray-600">
              SPARQL, OWL 등 심화 내용 학습
            </p>
          </div>
        </div>
      </div>
    </main>
  )
}