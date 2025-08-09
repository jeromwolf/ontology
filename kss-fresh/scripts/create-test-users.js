const { PrismaClient } = require('@prisma/client')
const bcrypt = require('bcryptjs')

const prisma = new PrismaClient()

async function createTestUsers() {
  try {
    const password = 'Test123!@#'
    const hashedPassword = await bcrypt.hash(password, 10)
    
    const users = [
      {
        email: 'student@kss.com',
        name: '김학생',
        role: 'STUDENT',
        profile: {
          bio: 'KSS 플랫폼에서 열심히 공부하는 학생입니다.',
          organization: '서울대학교',
          preferredLang: 'ko'
        }
      },
      {
        email: 'premium@kss.com',
        name: '이프리미엄',
        role: 'PREMIUM_STUDENT',
        profile: {
          bio: '프리미엄 구독자입니다.',
          organization: '삼성전자',
          preferredLang: 'ko'
        }
      },
      {
        email: 'instructor@kss.com',
        name: '박강사',
        role: 'INSTRUCTOR',
        profile: {
          bio: 'AI와 시스템 설계를 가르치는 강사입니다.',
          organization: 'KSS Academy',
          preferredLang: 'ko'
        }
      }
    ]
    
    for (const userData of users) {
      // Check if user already exists
      const existing = await prisma.user.findUnique({
        where: { email: userData.email }
      })
      
      if (existing) {
        console.log(`⚠️  User ${userData.email} already exists, skipping...`)
        continue
      }
      
      const user = await prisma.user.create({
        data: {
          email: userData.email,
          password: hashedPassword,
          name: userData.name,
          role: userData.role,
          emailVerified: new Date(),
          profile: {
            create: userData.profile
          }
        }
      })
      
      console.log(`✅ Created ${userData.role}: ${userData.email}`)
    }
    
    console.log('\n📝 Test Users Summary:')
    console.log('========================')
    console.log('🔑 Password for all users: Test123!@#')
    console.log('\n👥 Users:')
    console.log('1. admin@kss.com (ADMIN)')
    console.log('2. student@kss.com (STUDENT)')
    console.log('3. premium@kss.com (PREMIUM_STUDENT)')
    console.log('4. instructor@kss.com (INSTRUCTOR)')
    console.log('\n⚠️  Please change passwords after first login!')
    
  } catch (error) {
    console.error('Error creating test users:', error)
  } finally {
    await prisma.$disconnect()
  }
}

createTestUsers()