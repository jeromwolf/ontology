'use client';

export default function Chapter1() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4">Cloud Computing Fundamentals</h2>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          Cloud computing delivers computing services over the internet.
          This includes servers, storage, databases, networking, and software.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">Service Models</h2>
        <div className="grid gap-4">
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-2">
              IaaS (Infrastructure as a Service)
            </h3>
            <p className="text-gray-700 dark:text-gray-300">
              Virtual machines, storage, and networks
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
              Examples: AWS EC2, Azure VM, Google Compute Engine
            </p>
          </div>

          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-2">
              PaaS (Platform as a Service)
            </h3>
            <p className="text-gray-700 dark:text-gray-300">
              Platform for application development and deployment
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
              Examples: AWS Elastic Beanstalk, Azure App Service
            </p>
          </div>

          <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-emerald-800 dark:text-emerald-200 mb-2">
              SaaS (Software as a Service)
            </h3>
            <p className="text-gray-700 dark:text-gray-300">
              Ready-to-use software applications
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
              Examples: Gmail, Salesforce, Microsoft 365
            </p>
          </div>
        </div>
      </section>

      <section className="bg-gradient-to-r from-sky-50 to-blue-50 dark:from-sky-900/20 dark:to-blue-900/20 rounded-xl p-6 mt-8">
        <h2 className="text-xl font-bold mb-4 text-sky-800 dark:text-sky-200">Key Takeaways</h2>
        <ul className="space-y-2">
          <li className="flex items-start gap-2">
            <span className="text-sky-600 dark:text-sky-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">Service Models: IaaS, PaaS, SaaS</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-sky-600 dark:text-sky-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">Deployment: Public, Private, Hybrid, Multi-Cloud</span>
          </li>
        </ul>
      </section>
    </div>
  )
}
