import { NextResponse } from 'next/server'
import bcrypt from 'bcryptjs'
import { prisma } from '@/lib/prisma'
import { z } from 'zod'

// Input validation schema
const signupSchema = z.object({
  email: z.string().email('올바른 이메일 주소를 입력해주세요.'),
  password: z
    .string()
    .min(8, '비밀번호는 최소 8자 이상이어야 합니다.')
    .regex(
      /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]/,
      '비밀번호는 대소문자, 숫자, 특수문자를 포함해야 합니다.'
    ),
  name: z
    .string()
    .min(2, '이름은 최소 2자 이상이어야 합니다.')
    .max(50, '이름은 50자를 초과할 수 없습니다.')
})

export async function POST(request: Request) {
  try {
    const body = await request.json()

    // Validate input
    const validation = signupSchema.safeParse(body)
    if (!validation.success) {
      return NextResponse.json(
        { 
          error: validation.error.errors[0].message,
          field: validation.error.errors[0].path[0]
        },
        { status: 400 }
      )
    }

    const { email, password, name } = validation.data

    // Check if user already exists
    const existingUser = await prisma.user.findUnique({
      where: { email }
    })

    if (existingUser) {
      return NextResponse.json(
        { error: '이미 등록된 이메일 주소입니다.' },
        { status: 409 }
      )
    }

    // Hash password
    const saltRounds = 12
    const hashedPassword = await bcrypt.hash(password, saltRounds)

    // Create user with transaction for data consistency
    const result = await prisma.$transaction(async (tx) => {
      // Create user
      const user = await tx.user.create({
        data: {
          email,
          password: hashedPassword,
          name,
          role: 'STUDENT', // Default role
          emailVerified: new Date(), // Auto-verify for now
        },
        select: {
          id: true,
          email: true,
          name: true,
          role: true,
          createdAt: true
        }
      })

      // Create user profile with default preferences
      await tx.profile.create({
        data: {
          userId: user.id,
          bio: null,
          location: null,
          timezone: 'Asia/Seoul', // Default to Korean timezone
          language: 'ko', // Default to Korean
          theme: 'light',
          notifications: true, // Enable notifications by default
          emailNotifications: true,
          pushNotifications: false, // Disabled by default until user grants permission
          studyReminders: false,
          weeklyReports: true
        }
      })

      return user
    })

    // Log successful signup (for analytics)
    console.log(`New user registered: ${result.email} (ID: ${result.id})`)

    return NextResponse.json(
      {
        message: '회원가입이 완료되었습니다.',
        user: result
      },
      { status: 201 }
    )

  } catch (error) {
    console.error('Signup error:', error)

    // Handle specific Prisma errors
    if (error instanceof Error) {
      if (error.message.includes('Unique constraint failed')) {
        return NextResponse.json(
          { error: '이미 등록된 이메일 주소입니다.' },
          { status: 409 }
        )
      }
    }

    return NextResponse.json(
      { error: '서버 오류가 발생했습니다. 잠시 후 다시 시도해주세요.' },
      { status: 500 }
    )
  }
}

// Handle unsupported methods
export async function GET() {
  return NextResponse.json(
    { error: 'Method not allowed' },
    { status: 405 }
  )
}