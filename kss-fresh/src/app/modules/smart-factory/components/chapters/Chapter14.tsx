'use client'

import { 
  Network, Database, Cloud, Server, Shield, Layers,
  Cpu, HardDrive, Lock, Monitor, Workflow, GitBranch
} from 'lucide-react'

export default function Chapter14() {
  return (
    <div className="space-y-8">
      <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 p-8 rounded-xl border border-purple-200 dark:border-purple-800">
        <h3 className="text-2xl font-bold text-purple-900 dark:text-purple-200 mb-6 flex items-center gap-3">
          <Layers className="w-8 h-8" />
          스마트팩토리 참조 아키텍처
        </h3>
        <div className="grid lg:grid-cols-5 gap-4">
          {[
            { layer: "Level 5", title: "경영층", desc: "ERP, SCM", icon: "🏢", color: "purple" },
            { layer: "Level 4", title: "운영층", desc: "MES, MOM", icon: "📊", color: "blue" },
            { layer: "Level 3", title: "감시제어", desc: "SCADA, HMI", icon: "🖥️", color: "green" },
            { layer: "Level 2", title: "제어층", desc: "PLC, DCS", icon: "⚙️", color: "orange" },
            { layer: "Level 1", title: "디바이스", desc: "센서, 액추에이터", icon: "📡", color: "red" }
          ].map((level, idx) => (
            <div key={idx} className="text-center">
              <div className={`w-20 h-20 bg-${level.color}-500 rounded-lg flex items-center justify-center mx-auto mb-3`}>
                <span className="text-3xl">{level.icon}</span>
              </div>
              <h4 className="font-bold text-gray-900 dark:text-white text-sm">{level.layer}</h4>
              <h5 className="font-semibold text-gray-800 dark:text-gray-200 text-sm">{level.title}</h5>
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">{level.desc}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        <div className="bg-white dark:bg-gray-800 p-6 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <Cloud className="w-6 h-6 text-blue-600" />
            클라우드 vs 온프레미스
          </h3>
          <div className="space-y-4">
            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-400 rounded">
              <h4 className="font-semibold text-blue-800 dark:text-blue-300 mb-2">클라우드 아키텍처</h4>
              <ul className="text-sm text-blue-700 dark:text-blue-400 space-y-1">
                <li>• <strong>장점:</strong> 확장성, 유연성, 초기 투자 절감</li>
                <li>• <strong>단점:</strong> 지연시간, 보안 우려, 종속성</li>
                <li>• <strong>적합한 경우:</strong> AI/빅데이터 분석, 글로벌 협업</li>
              </ul>
            </div>
            
            <div className="p-4 bg-green-50 dark:bg-green-900/20 border-l-4 border-green-400 rounded">
              <h4 className="font-semibold text-green-800 dark:text-green-300 mb-2">온프레미스</h4>
              <ul className="text-sm text-green-700 dark:text-green-400 space-y-1">
                <li>• <strong>장점:</strong> 보안 통제, 낮은 지연, 커스터마이징</li>
                <li>• <strong>단점:</strong> 높은 초기 투자, 유지보수 부담</li>
                <li>• <strong>적합한 경우:</strong> 실시간 제어, 민감 데이터</li>
              </ul>
            </div>

            <div className="p-4 bg-purple-50 dark:bg-purple-900/20 border-l-4 border-purple-400 rounded">
              <h4 className="font-semibold text-purple-800 dark:text-purple-300 mb-2">하이브리드</h4>
              <ul className="text-sm text-purple-700 dark:text-purple-400 space-y-1">
                <li>• <strong>최적 조합:</strong> 핵심 제어는 온프레미스</li>
                <li>• <strong>클라우드 활용:</strong> 분석, 백업, 협업</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 p-6 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <Database className="w-6 h-6 text-indigo-600" />
            데이터 아키텍처
          </h3>
          <div className="space-y-4">
            <div className="p-4 bg-gray-50 dark:bg-gray-700 rounded border border-gray-200 dark:border-gray-600">
              <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">데이터 레이크</h4>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <p className="font-medium text-gray-700 dark:text-gray-300">원시 데이터</p>
                  <ul className="text-gray-600 dark:text-gray-400 text-xs mt-1">
                    <li>• 센서 로그</li>
                    <li>• 이미지/비디오</li>
                    <li>• 비정형 데이터</li>
                  </ul>
                </div>
                <div>
                  <p className="font-medium text-gray-700 dark:text-gray-300">기술 스택</p>
                  <ul className="text-gray-600 dark:text-gray-400 text-xs mt-1">
                    <li>• HDFS/S3</li>
                    <li>• Apache Spark</li>
                    <li>• Kafka Streams</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="p-4 bg-gray-50 dark:bg-gray-700 rounded border border-gray-200 dark:border-gray-600">
              <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">데이터 웨어하우스</h4>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <p className="font-medium text-gray-700 dark:text-gray-300">정제된 데이터</p>
                  <ul className="text-gray-600 dark:text-gray-400 text-xs mt-1">
                    <li>• KPI 지표</li>
                    <li>• 생산 리포트</li>
                    <li>• 분석 대시보드</li>
                  </ul>
                </div>
                <div>
                  <p className="font-medium text-gray-700 dark:text-gray-300">기술 스택</p>
                  <ul className="text-gray-600 dark:text-gray-400 text-xs mt-1">
                    <li>• PostgreSQL</li>
                    <li>• ClickHouse</li>
                    <li>• TimescaleDB</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 p-8 rounded-xl border border-indigo-200 dark:border-indigo-800">
        <h3 className="text-2xl font-bold text-indigo-900 dark:text-indigo-200 mb-6 flex items-center gap-3">
          <Network className="w-8 h-8" />
          마이크로서비스 아키텍처
        </h3>
        <div className="grid md:grid-cols-3 gap-6">
          <div className="bg-white dark:bg-indigo-800/30 p-6 rounded-lg border border-indigo-200 dark:border-indigo-600">
            <h4 className="font-bold text-indigo-800 dark:text-indigo-200 mb-4 flex items-center gap-2">
              <Server className="w-5 h-5" />
              서비스 분해
            </h4>
            <ul className="text-sm text-indigo-700 dark:text-indigo-300 space-y-2">
              <li>• 생산관리 서비스</li>
              <li>• 품질관리 서비스</li>
              <li>• 설비관리 서비스</li>
              <li>• 재고관리 서비스</li>
              <li>• 분석 서비스</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-indigo-800/30 p-6 rounded-lg border border-indigo-200 dark:border-indigo-600">
            <h4 className="font-bold text-indigo-800 dark:text-indigo-200 mb-4 flex items-center gap-2">
              <GitBranch className="w-5 h-5" />
              통신 패턴
            </h4>
            <ul className="text-sm text-indigo-700 dark:text-indigo-300 space-y-2">
              <li>• REST API</li>
              <li>• gRPC</li>
              <li>• Message Queue</li>
              <li>• Event Streaming</li>
              <li>• Service Mesh</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-indigo-800/30 p-6 rounded-lg border border-indigo-200 dark:border-indigo-600">
            <h4 className="font-bold text-indigo-800 dark:text-indigo-200 mb-4 flex items-center gap-2">
              <Shield className="w-5 h-5" />
              컨테이너화
            </h4>
            <ul className="text-sm text-indigo-700 dark:text-indigo-300 space-y-2">
              <li>• Docker 컨테이너</li>
              <li>• Kubernetes 오케스트레이션</li>
              <li>• CI/CD 파이프라인</li>
              <li>• 자동 스케일링</li>
              <li>• 무중단 배포</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-800 p-8 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">
        <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Monitor className="w-8 h-8 text-green-600" />
          실시간 모니터링 아키텍처
        </h3>
        <div className="grid md:grid-cols-4 gap-6">
          <div>
            <h4 className="font-bold text-gray-800 dark:text-gray-200 mb-3">수집 계층</h4>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <li>• Telegraf</li>
              <li>• Prometheus</li>
              <li>• Fluentd</li>
              <li>• Beats</li>
            </ul>
          </div>
          <div>
            <h4 className="font-bold text-gray-800 dark:text-gray-200 mb-3">저장 계층</h4>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <li>• InfluxDB</li>
              <li>• Elasticsearch</li>
              <li>• Redis</li>
              <li>• Cassandra</li>
            </ul>
          </div>
          <div>
            <h4 className="font-bold text-gray-800 dark:text-gray-200 mb-3">분석 계층</h4>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <li>• Apache Flink</li>
              <li>• Spark Streaming</li>
              <li>• Storm</li>
              <li>• Kapacitor</li>
            </ul>
          </div>
          <div>
            <h4 className="font-bold text-gray-800 dark:text-gray-200 mb-3">시각화 계층</h4>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <li>• Grafana</li>
              <li>• Kibana</li>
              <li>• Tableau</li>
              <li>• Custom UI</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}