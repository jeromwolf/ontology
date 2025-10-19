/**
 * 데이터베이스 초기화 스크립트
 * ArXiv 관련 데이터만 삭제합니다
 */
import { PrismaClient } from '@prisma/client'

const prisma = new PrismaClient()

async function resetDatabase() {
  console.log('🗑️  Deleting ArXiv data...')
  
  // 처리 로그 먼저 삭제 (Foreign Key 때문에)
  await prisma.arXiv_ProcessingLog.deleteMany({})
  console.log('✅ Deleted all processing logs')
  
  // 논문 삭제
  await prisma.arXiv_Paper.deleteMany({})
  console.log('✅ Deleted all papers')
  
  console.log('🎉 Database reset complete!')
  
  await prisma.$disconnect()
}

resetDatabase()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error('❌ Error:', error)
    process.exit(1)
  })
