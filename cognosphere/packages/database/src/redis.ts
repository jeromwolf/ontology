import Redis from 'ioredis'

let redis: Redis | null = null

export const getRedis = (): Redis => {
  if (!redis) {
    const redisUrl = process.env.REDIS_URL || 'redis://localhost:6379'
    
    redis = new Redis(redisUrl, {
      maxRetriesPerRequest: 3,
      enableReadyCheck: true,
      enableOfflineQueue: true,
      retryStrategy: (times) => {
        const delay = Math.min(times * 50, 2000)
        return delay
      },
    })

    redis.on('error', (err) => {
      console.error('Redis Client Error:', err)
    })

    redis.on('connect', () => {
      console.log('Redis Client Connected')
    })
  }

  return redis
}

export const closeRedis = async (): Promise<void> => {
  if (redis) {
    await redis.quit()
    redis = null
  }
}

// Cache helpers
export const cache = {
  get: async (key: string): Promise<any> => {
    const redis = getRedis()
    const value = await redis.get(key)
    return value ? JSON.parse(value) : null
  },

  set: async (key: string, value: any, ttl?: number): Promise<void> => {
    const redis = getRedis()
    const serialized = JSON.stringify(value)
    if (ttl) {
      await redis.setex(key, ttl, serialized)
    } else {
      await redis.set(key, serialized)
    }
  },

  delete: async (key: string): Promise<void> => {
    const redis = getRedis()
    await redis.del(key)
  },

  exists: async (key: string): Promise<boolean> => {
    const redis = getRedis()
    const exists = await redis.exists(key)
    return exists === 1
  },
}

// Session helpers
export const session = {
  create: async (sessionId: string, userId: string, data: any, ttl: number): Promise<void> => {
    const key = `session:${sessionId}`
    await cache.set(key, { userId, ...data }, ttl)
    
    // Add to user's session set
    const userKey = `user:sessions:${userId}`
    await getRedis().sadd(userKey, sessionId)
    await getRedis().expire(userKey, ttl)
  },

  get: async (sessionId: string): Promise<any> => {
    return cache.get(`session:${sessionId}`)
  },

  destroy: async (sessionId: string): Promise<void> => {
    const session = await cache.get(`session:${sessionId}`)
    if (session?.userId) {
      await getRedis().srem(`user:sessions:${session.userId}`, sessionId)
    }
    await cache.delete(`session:${sessionId}`)
  },

  getUserSessions: async (userId: string): Promise<string[]> => {
    const redis = getRedis()
    return redis.smembers(`user:sessions:${userId}`)
  },
}

// Rate limiting
export const rateLimiter = {
  check: async (key: string, limit: number, window: number): Promise<boolean> => {
    const redis = getRedis()
    const current = await redis.incr(key)
    
    if (current === 1) {
      await redis.expire(key, window)
    }
    
    return current <= limit
  },

  reset: async (key: string): Promise<void> => {
    await cache.delete(key)
  },
}