'use client';

export default function Chapter2() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4">AWS Core Services</h2>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          Amazon Web Services provides 200+ services for computing, storage, databases, and more.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">Compute Services</h2>
        <div className="grid gap-4">
          <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-orange-800 dark:text-orange-200 mb-2">
              EC2 (Elastic Compute Cloud)
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              Scalable virtual servers in the cloud
            </p>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <li>• Multiple instance types (t2, m5, c5, r5)</li>
              <li>• Auto Scaling</li>
              <li>• Load Balancing</li>
            </ul>
          </div>

          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-2">
              Lambda (Serverless)
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              Run code without managing servers
            </p>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <li>• Event-driven execution</li>
              <li>• Pay per request</li>
              <li>• Auto scaling</li>
            </ul>
          </div>
        </div>
      </section>

      <section className="bg-gradient-to-r from-orange-50 to-amber-50 dark:from-orange-900/20 dark:to-amber-900/20 rounded-xl p-6 mt-8">
        <h2 className="text-xl font-bold mb-4 text-orange-800 dark:text-orange-200">Key Takeaways</h2>
        <ul className="space-y-2">
          <li className="flex items-start gap-2">
            <span className="text-orange-600 dark:text-orange-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">Compute: EC2, Lambda</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-orange-600 dark:text-orange-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">Storage: S3, EBS</span>
          </li>
        </ul>
      </section>
    </div>
  )
}
