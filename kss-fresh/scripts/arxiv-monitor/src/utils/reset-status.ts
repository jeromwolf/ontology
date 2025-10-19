/**
 * Reset failed papers to CRAWLED status
 */

import { PrismaClient } from '../../../../node_modules/@prisma/client'

const prisma = new PrismaClient()

async function resetStatus() {
  try {
    const result = await prisma.arXiv_Paper.updateMany({
      where: {
        status: 'FAILED',
      },
      data: {
        status: 'CRAWLED',
        errorMessage: null,
      },
    })

    console.log(`✅ Reset ${result.count} papers to CRAWLED status`)
  } catch (error) {
    console.error('❌ Failed to reset status:', error)
  } finally {
    await prisma.$disconnect()
  }
}

resetStatus()
