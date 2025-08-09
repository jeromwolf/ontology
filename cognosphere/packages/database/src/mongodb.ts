import { MongoClient, Db, Collection } from 'mongodb'

let client: MongoClient | null = null
let database: Db | null = null

export const getMongoClient = async (): Promise<MongoClient> => {
  if (!client) {
    const uri = process.env.MONGODB_URI || 'mongodb://localhost:27017'
    client = new MongoClient(uri, {
      maxPoolSize: 10,
      minPoolSize: 2,
    })
    await client.connect()
  }
  return client
}

export const getMongoDB = async (dbName?: string): Promise<Db> => {
  if (!database) {
    const client = await getMongoClient()
    database = client.db(dbName || process.env.MONGODB_DATABASE || 'cognosphere')
  }
  return database
}

export const getCollection = async <T = any>(
  collectionName: string,
  dbName?: string
): Promise<Collection<T>> => {
  const db = await getMongoDB(dbName)
  return db.collection<T>(collectionName)
}

export const closeMongoDB = async (): Promise<void> => {
  if (client) {
    await client.close()
    client = null
    database = null
  }
}

// Collections
export const collections = {
  content: async () => getCollection('content'),
  simulations: async () => getCollection('simulations'),
  userSimulationStates: async () => getCollection('user_simulation_states'),
  analytics: async () => getCollection('analytics'),
}

// Helper types
export interface ContentDocument {
  _id?: string
  chapter_id: string
  type: 'theory' | 'exercise' | 'simulation'
  title: string
  content: {
    markdown: string
    html: string
    components: Array<{
      type: string
      content?: string
      config?: any
    }>
  }
  media: Array<{
    type: string
    url: string
    alt?: string
  }>
  metadata: {
    author: string
    created_at: Date
    updated_at: Date
    version: number
    tags: string[]
  }
}

export interface SimulationDocument {
  _id?: string
  simulation_id: string
  name: string
  description: string
  type: string
  category: string
  difficulty: string
  config: any
  instructions: {
    steps: any[]
    hints: any[]
    solution: any
  }
  scoring: {
    criteria: any[]
    max_points: number
  }
  created_at: Date
  updated_at: Date
}