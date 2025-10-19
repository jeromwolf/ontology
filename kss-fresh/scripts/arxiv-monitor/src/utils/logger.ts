/**
 * Logger utility using Winston
 * Provides structured logging for the ArXiv Monitor system
 */

import winston from 'winston'
import path from 'path'
import fs from 'fs'

// Ensure logs directory exists
const logsDir = path.join(process.cwd(), 'logs')
if (!fs.existsSync(logsDir)) {
  fs.mkdirSync(logsDir, { recursive: true })
}

// Define log format
const logFormat = winston.format.combine(
  winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
  winston.format.errors({ stack: true }),
  winston.format.splat(),
  winston.format.json()
)

// Console format (more readable)
const consoleFormat = winston.format.combine(
  winston.format.colorize(),
  winston.format.timestamp({ format: 'HH:mm:ss' }),
  winston.format.printf(({ timestamp, level, message, ...meta }) => {
    let msg = `${timestamp} [${level}]: ${message}`
    if (Object.keys(meta).length > 0) {
      msg += ` ${JSON.stringify(meta)}`
    }
    return msg
  })
)

// Create logger instance
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: logFormat,
  transports: [
    // File transport - all logs
    new winston.transports.File({
      filename: path.join(logsDir, 'arxiv-monitor.log'),
      maxsize: 10 * 1024 * 1024, // 10MB
      maxFiles: 5,
    }),
    // File transport - errors only
    new winston.transports.File({
      filename: path.join(logsDir, 'error.log'),
      level: 'error',
      maxsize: 10 * 1024 * 1024,
      maxFiles: 5,
    }),
    // Console transport
    new winston.transports.Console({
      format: consoleFormat,
    }),
  ],
})

// Export logger with typed methods
export default {
  info: (message: string, meta?: Record<string, any>) => logger.info(message, meta),
  error: (message: string, meta?: Record<string, any>) => logger.error(message, meta),
  warn: (message: string, meta?: Record<string, any>) => logger.warn(message, meta),
  debug: (message: string, meta?: Record<string, any>) => logger.debug(message, meta),
}
