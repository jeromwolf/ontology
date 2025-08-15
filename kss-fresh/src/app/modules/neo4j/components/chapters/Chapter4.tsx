'use client';

export default function Chapter4() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6 text-center">Cypher Advanced Features ⚡</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6 text-center">
          Unleash the true power of Cypher with APOC, dynamic queries, and performance optimization!
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🔧 APOC: Awesome Procedures on Cypher</h2>
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-6">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            APOC is Neo4j's most powerful extension library. With over 500 procedures and functions,
            it enables you to solve complex graph operations with ease.
          </p>
          
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h3 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">Main Categories</h3>
              <ul className="space-y-2 text-sm">
                <li>• 📊 <strong>apoc.algo</strong>: Graph algorithms</li>
                <li>• 🔄 <strong>apoc.refactor</strong>: Graph refactoring</li>
                <li>• 📥 <strong>apoc.load</strong>: Data import/export</li>
                <li>• 🛠️ <strong>apoc.create</strong>: Dynamic node/relationship creation</li>
                <li>• 📈 <strong>apoc.stats</strong>: Database statistics</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">💡 Dynamic Queries</h2>
        <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-xl p-6">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            Dynamic queries allow you to build and execute Cypher queries programmatically.
          </p>
          
          <div className="bg-gray-900 rounded-lg p-4 mb-4">
            <code className="text-green-400 text-sm">
              {`// Execute dynamic Cypher
CALL apoc.cypher.run(
  'MATCH (n:' + $nodeLabel + ') RETURN n LIMIT 10',
  {nodeLabel: 'Person'}
) YIELD value
RETURN value.n AS node`}
            </code>
          </div>
          
          <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
            <p className="text-sm">
              <strong>💡 Use Cases:</strong> Multi-tenant applications, dynamic schema queries,
              conditional logic execution
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🚀 Performance Optimization</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-6">
            <h3 className="font-bold text-purple-700 dark:text-purple-300 mb-3">Query Optimization</h3>
            <ul className="space-y-2 text-sm">
              <li>✅ Use indexes appropriately</li>
              <li>✅ PROFILE queries before production</li>
              <li>✅ Avoid Cartesian products</li>
              <li>✅ Use LIMIT early in queries</li>
            </ul>
          </div>
          
          <div className="bg-gradient-to-br from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-xl p-6">
            <h3 className="font-bold text-orange-700 dark:text-orange-300 mb-3">Memory Management</h3>
            <ul className="space-y-2 text-sm">
              <li>📦 Configure heap size properly</li>
              <li>📦 Use page cache effectively</li>
              <li>📦 Monitor query memory usage</li>
              <li>📦 Implement query timeouts</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎯 Practice Exercise</h2>
        <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-xl p-6">
          <h3 className="font-bold mb-3">Build a Recommendation Engine</h3>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            Use APOC and advanced Cypher to create a product recommendation system:
          </p>
          <ol className="space-y-2 text-sm">
            <li>1. Import product and user data using APOC</li>
            <li>2. Create similarity relationships between products</li>
            <li>3. Implement collaborative filtering algorithm</li>
            <li>4. Optimize queries for real-time recommendations</li>
          </ol>
        </div>
      </section>
    </div>
  );
}