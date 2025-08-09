import neo4j, { Driver, Session } from 'neo4j-driver'

let driver: Driver | null = null

export const getNeo4jDriver = (): Driver => {
  if (!driver) {
    const uri = process.env.NEO4J_URI || 'bolt://localhost:7687'
    const user = process.env.NEO4J_USER || 'neo4j'
    const password = process.env.NEO4J_PASSWORD || 'password'

    driver = neo4j.driver(uri, neo4j.auth.basic(user, password), {
      maxConnectionLifetime: 3 * 60 * 60 * 1000, // 3 hours
      maxConnectionPoolSize: 50,
      connectionAcquisitionTimeout: 2 * 60 * 1000, // 120 seconds
    })
  }

  return driver
}

export const getNeo4jSession = (database?: string): Session => {
  const driver = getNeo4jDriver()
  return driver.session({
    database: database || process.env.NEO4J_DATABASE || 'neo4j',
    defaultAccessMode: neo4j.session.WRITE,
  })
}

export const closeNeo4j = async (): Promise<void> => {
  if (driver) {
    await driver.close()
    driver = null
  }
}

// Helper functions for common operations
export const runQuery = async <T = any>(
  query: string,
  params: Record<string, any> = {},
  database?: string
): Promise<T[]> => {
  const session = getNeo4jSession(database)
  try {
    const result = await session.run(query, params)
    return result.records.map(record => record.toObject() as T)
  } finally {
    await session.close()
  }
}

export const runTransaction = async <T = any>(
  work: (tx: any) => Promise<T>,
  database?: string
): Promise<T> => {
  const session = getNeo4jSession(database)
  try {
    return await session.writeTransaction(work)
  } finally {
    await session.close()
  }
}