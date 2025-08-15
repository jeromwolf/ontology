'use client';

export default function Chapter15() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6">Chapter 15: 온톨로지 도구와 플랫폼</h1>
        
        <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-xl p-6 mb-8">
          <p className="text-lg">
            온톨로지 개발과 관리를 위한 다양한 도구와 플랫폼을 소개합니다. 
            각 도구의 특징과 사용 시나리오를 이해하고, 프로젝트에 맞는 도구를 선택할 수 있게 됩니다.
          </p>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">온톨로지 편집기</h2>
        
        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-blue-600 dark:text-blue-400 mb-3">Protégé</h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium mb-2">특징</h4>
                <ul className="text-sm space-y-1">
                  <li>• 스탠포드 대학 개발</li>
                  <li>• 가장 널리 사용되는 도구</li>
                  <li>• 풍부한 플러그인 생태계</li>
                  <li>• 데스크톱/웹 버전 제공</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium mb-2">적합한 경우</h4>
                <ul className="text-sm space-y-1">
                  <li>• OWL 온톨로지 개발</li>
                  <li>• 학술 연구 프로젝트</li>
                  <li>• 복잡한 추론 필요</li>
                  <li>• 팀 협업 환경</li>
                </ul>
              </div>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-green-600 dark:text-green-400 mb-3">TopBraid Composer</h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium mb-2">특징</h4>
                <ul className="text-sm space-y-1">
                  <li>• 상용 도구</li>
                  <li>• SHACL 지원</li>
                  <li>• 엔터프라이즈 기능</li>
                  <li>• 데이터 통합 도구</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium mb-2">적합한 경우</h4>
                <ul className="text-sm space-y-1">
                  <li>• 기업 환경</li>
                  <li>• 데이터 거버넌스</li>
                  <li>• 대규모 프로젝트</li>
                  <li>• 전문 지원 필요</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">트리플 스토어</h2>
        
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-lg p-6">
            <h3 className="font-semibold mb-3">Apache Jena Fuseki</h3>
            <ul className="text-sm space-y-2">
              <li>✓ 오픈소스 SPARQL 서버</li>
              <li>✓ TDB 네이티브 스토리지</li>
              <li>✓ 추론 엔진 내장</li>
              <li>✓ REST API 제공</li>
            </ul>
          </div>
          
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
            <h3 className="font-semibold mb-3">GraphDB</h3>
            <ul className="text-sm space-y-2">
              <li>✓ 고성능 상용 솔루션</li>
              <li>✓ 클러스터링 지원</li>
              <li>✓ 시각화 도구 내장</li>
              <li>✓ 엔터프라이즈 기능</li>
            </ul>
          </div>
          
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
            <h3 className="font-semibold mb-3">Virtuoso</h3>
            <ul className="text-sm space-y-2">
              <li>✓ 하이브리드 데이터베이스</li>
              <li>✓ SQL과 SPARQL 지원</li>
              <li>✓ 대용량 처리 최적화</li>
              <li>✓ LOD 클라우드 호스팅</li>
            </ul>
          </div>
          
          <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6">
            <h3 className="font-semibold mb-3">Amazon Neptune</h3>
            <ul className="text-sm space-y-2">
              <li>✓ 완전 관리형 서비스</li>
              <li>✓ 자동 백업/복구</li>
              <li>✓ 높은 가용성</li>
              <li>✓ AWS 생태계 통합</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">개발 프레임워크</h2>
        
        <div className="space-y-4">
          <div className="bg-gray-50 dark:bg-gray-900 rounded-xl p-6">
            <h3 className="font-semibold mb-4">프로그래밍 언어별 라이브러리</h3>
            <div className="grid md:grid-cols-3 gap-4">
              <div>
                <h4 className="font-medium text-blue-600 dark:text-blue-400 mb-2">Java</h4>
                <ul className="text-sm space-y-1">
                  <li>• Apache Jena</li>
                  <li>• OWL API</li>
                  <li>• RDF4J (Sesame)</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium text-green-600 dark:text-green-400 mb-2">Python</h4>
                <ul className="text-sm space-y-1">
                  <li>• RDFLib</li>
                  <li>• Owlready2</li>
                  <li>• PyShacl</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium text-purple-600 dark:text-purple-400 mb-2">JavaScript</h4>
                <ul className="text-sm space-y-1">
                  <li>• rdflib.js</li>
                  <li>• Comunica</li>
                  <li>• LDflex</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">온톨로지 검증 도구</h2>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="font-semibold mb-3">검증 도구 비교</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left py-2">도구</th>
                  <th className="text-left py-2">용도</th>
                  <th className="text-left py-2">특징</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="py-2">OOPS!</td>
                  <td className="py-2">온톨로지 품질 평가</td>
                  <td className="py-2">자동 오류 검출</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="py-2">Pellet</td>
                  <td className="py-2">일관성 검사</td>
                  <td className="py-2">OWL DL 추론</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="py-2">SHACL</td>
                  <td className="py-2">데이터 검증</td>
                  <td className="py-2">제약사항 검사</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-xl p-6">
        <h2 className="text-xl font-bold mb-3 flex items-center gap-2">
          <span className="text-2xl">🛠️</span>
          도구 선택 가이드
        </h2>
        <div className="space-y-4">
          <div>
            <h3 className="font-medium mb-2">소규모 프로젝트</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              Protégé + Apache Jena + RDFLib
            </p>
          </div>
          <div>
            <h3 className="font-medium mb-2">중규모 프로젝트</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              TopBraid + GraphDB + OWL API
            </p>
          </div>
          <div>
            <h3 className="font-medium mb-2">엔터프라이즈</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              Enterprise 플랫폼 + Neptune/Virtuoso + 커스텀 개발
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}