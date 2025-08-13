'use client';

export default function Chapter3() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4">특징점 검출 알고리즘</h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          특징점(feature points)은 이미지에서 독특하고 식별 가능한 지점들입니다. 
          이러한 특징점들은 이미지 매칭, 객체 인식, 3D 재구성 등에 활용됩니다.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4">
            <h3 className="font-semibold text-teal-600 dark:text-teal-400 mb-2">SIFT</h3>
            <p className="text-sm text-gray-600 dark:text-gray-300">
              Scale-Invariant Feature Transform. 크기와 회전에 불변한 특징점 검출
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4">
            <h3 className="font-semibold text-teal-600 dark:text-teal-400 mb-2">SURF</h3>
            <p className="text-sm text-gray-600 dark:text-gray-300">
              Speeded-Up Robust Features. SIFT보다 빠른 특징점 검출
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4">
            <h3 className="font-semibold text-teal-600 dark:text-teal-400 mb-2">ORB</h3>
            <p className="text-sm text-gray-600 dark:text-gray-300">
              Oriented FAST and Rotated BRIEF. 실시간 처리에 적합한 빠른 알고리즘
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}