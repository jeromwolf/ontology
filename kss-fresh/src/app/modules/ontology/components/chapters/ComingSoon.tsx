'use client';

export default function ComingSoon() {
  return (
    <div className="text-center py-16">
      <div className="text-6xl mb-4">🚧</div>
      <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
        콘텐츠 준비 중
      </h2>
      <p className="text-gray-600 dark:text-gray-400">
        이 챕터의 콘텐츠는 곧 업데이트될 예정입니다.
      </p>
      <p className="text-sm text-gray-500 dark:text-gray-500 mt-4">
        기존 시뮬레이터들을 React 컴포넌트로 통합하는 작업이 진행 중입니다.
      </p>
    </div>
  )
}