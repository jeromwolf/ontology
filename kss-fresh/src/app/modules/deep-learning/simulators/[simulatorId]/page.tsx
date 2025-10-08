'use client'

import { useParams } from 'next/navigation'
import dynamic from 'next/dynamic'
import Link from 'next/link'
import { ArrowLeft, Wrench } from 'lucide-react'
import { deepLearningModule } from '../../metadata'

// Dynamic imports for simulators
const NeuralNetworkPlayground = dynamic(
  () => import('@/components/deep-learning-simulators/NeuralNetworkPlayground'),
  { ssr: false }
)

const OptimizerComparison = dynamic(
  () => import('@/components/deep-learning-simulators/OptimizerComparison'),
  { ssr: false }
)

const AttentionVisualizer = dynamic(
  () => import('@/components/deep-learning-simulators/AttentionVisualizer'),
  { ssr: false }
)

const CNNVisualizer = dynamic(
  () => import('@/components/deep-learning-simulators/CNNVisualizer'),
  { ssr: false }
)

const GANGenerator = dynamic(
  () => import('@/components/deep-learning-simulators/GANGenerator'),
  { ssr: false }
)

const TrainingDashboard = dynamic(
  () => import('@/components/deep-learning-simulators/TrainingDashboard'),
  { ssr: false }
)

export default function DeepLearningSimulatorPage() {
  const params = useParams()
  const simulatorId = params.simulatorId as string

  const simulator = deepLearningModule.simulators.find(sim => sim.id === simulatorId)

  if (!simulator) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white dark:from-gray-900 dark:to-gray-800 flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
            시뮬레이터를 찾을 수 없습니다
          </h1>
          <Link
            href="/modules/deep-learning"
            className="inline-flex items-center text-violet-600 hover:text-violet-700 dark:text-violet-400 transition-colors"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            학습 모듈로 돌아가기
          </Link>
        </div>
      </div>
    )
  }

  // Render the appropriate simulator component
  const renderSimulator = () => {
    switch (simulatorId) {
      case 'neural-network-playground':
        return <NeuralNetworkPlayground />
      case 'optimizer-comparison':
        return <OptimizerComparison />
      case 'attention-visualizer':
        return <AttentionVisualizer />
      case 'cnn-visualizer':
        return <CNNVisualizer />
      case 'gan-generator':
        return <GANGenerator />
      case 'training-dashboard':
        return <TrainingDashboard />
      default:
        return null
    }
  }

  const simulatorComponent = renderSimulator()

  // If simulator component is available, render it
  if (simulatorComponent) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white dark:from-gray-900 dark:to-gray-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="mb-8">
            <Link
              href="/modules/deep-learning"
              className="inline-flex items-center text-violet-600 hover:text-violet-700 dark:text-violet-400 transition-colors"
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              학습 모듈로 돌아가기
            </Link>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-8 mb-8">
            <div className="flex items-start gap-4 mb-6">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center text-white text-3xl">
                🧠
              </div>
              <div>
                <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
                  {simulator.name}
                </h1>
                <p className="text-gray-600 dark:text-gray-300 text-lg">
                  {simulator.description}
                </p>
              </div>
            </div>

            {/* Render the simulator component */}
            {simulatorComponent}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white dark:from-gray-900 dark:to-gray-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <Link
            href="/modules/deep-learning"
            className="inline-flex items-center text-violet-600 hover:text-violet-700 dark:text-violet-400 transition-colors"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            학습 모듈로 돌아가기
          </Link>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-8 mb-8">
          <div className="flex items-start gap-4 mb-6">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center text-white text-3xl">
              🧠
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
                {simulator.name}
              </h1>
              <p className="text-gray-600 dark:text-gray-300 text-lg">
                {simulator.description}
              </p>
            </div>
          </div>

          {/* Coming Soon Message */}
          <div className="mt-12 bg-gradient-to-br from-violet-50 to-purple-50 dark:from-violet-900/20 dark:to-purple-900/20 rounded-2xl p-12 text-center border border-violet-200 dark:border-violet-800">
            <Wrench className="w-20 h-20 mx-auto text-violet-500 mb-6" />
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
              시뮬레이터 준비 중
            </h2>
            <p className="text-gray-600 dark:text-gray-300 text-lg mb-6 max-w-2xl mx-auto">
              이 시뮬레이터는 현재 개발 중입니다. 곧 완성된 인터랙티브 학습 도구를 만나보실 수 있습니다.
            </p>
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 max-w-xl mx-auto text-left">
              <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
                개발 예정 기능:
              </h3>
              <ul className="space-y-2 text-gray-600 dark:text-gray-400">
                {simulatorId === 'neural-network-playground' && (
                  <>
                    <li className="flex items-start gap-2">
                      <span className="text-violet-500">•</span>
                      신경망 레이어 구조 직접 설계
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-violet-500">•</span>
                      실시간 학습 과정 시각화
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-violet-500">•</span>
                      활성화 함수 비교 실험
                    </li>
                  </>
                )}
                {simulatorId === 'cnn-visualizer' && (
                  <>
                    <li className="flex items-start gap-2">
                      <span className="text-violet-500">•</span>
                      CNN 필터 시각화
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-violet-500">•</span>
                      Feature Map 레이어별 분석
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-violet-500">•</span>
                      이미지 업로드 및 실시간 처리
                    </li>
                  </>
                )}
                {simulatorId === 'attention-visualizer' && (
                  <>
                    <li className="flex items-start gap-2">
                      <span className="text-violet-500">•</span>
                      Self-Attention 메커니즘 시각화
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-violet-500">•</span>
                      Multi-Head Attention 분석
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-violet-500">•</span>
                      문장 입력 및 Attention Score 확인
                    </li>
                  </>
                )}
                {simulatorId === 'gan-generator' && (
                  <>
                    <li className="flex items-start gap-2">
                      <span className="text-violet-500">•</span>
                      GAN 기반 이미지 생성
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-violet-500">•</span>
                      Latent Space 탐색
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-violet-500">•</span>
                      Generator-Discriminator 대립 과정 시각화
                    </li>
                  </>
                )}
                {simulatorId === 'training-dashboard' && (
                  <>
                    <li className="flex items-start gap-2">
                      <span className="text-violet-500">•</span>
                      Loss, Accuracy 실시간 모니터링
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-violet-500">•</span>
                      Gradient 흐름 시각화
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-violet-500">•</span>
                      학습 중단 및 재개 기능
                    </li>
                  </>
                )}
                {simulatorId === 'optimizer-comparison' && (
                  <>
                    <li className="flex items-start gap-2">
                      <span className="text-violet-500">•</span>
                      SGD, Adam, RMSprop 등 비교 실험
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-violet-500">•</span>
                      학습률 변화에 따른 성능 분석
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-violet-500">•</span>
                      최적화 경로 2D/3D 시각화
                    </li>
                  </>
                )}
              </ul>
            </div>
          </div>
        </div>

        {/* Related Chapters */}
        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-8">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            관련 챕터
          </h2>
          <div className="grid md:grid-cols-2 gap-4">
            {deepLearningModule.chapters.slice(0, 4).map((chapter) => (
              <Link
                key={chapter.id}
                href={`/modules/deep-learning/${chapter.id}`}
                className="p-4 rounded-xl border border-gray-200 dark:border-gray-700 hover:border-violet-300 dark:hover:border-violet-600 hover:shadow-md transition-all duration-200"
              >
                <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
                  {chapter.title}
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {chapter.description}
                </p>
              </Link>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
