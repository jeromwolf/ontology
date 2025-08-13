import { NextRequest, NextResponse } from "next/server"
import { auth } from "@/lib/auth"
import { prisma } from "@/lib/prisma"

export async function GET(request: NextRequest) {
  try {
    const session = await auth()
    
    if (!session?.user?.id) {
      return NextResponse.json(
        { error: "Unauthorized" },
        { status: 401 }
      )
    }

    const user = await prisma.user.findUnique({
      where: { id: session.user.id },
      include: { profile: true },
    })

    if (!user) {
      return NextResponse.json(
        { error: "User not found" },
        { status: 404 }
      )
    }

    return NextResponse.json({
      name: user.name || "",
      email: user.email,
      bio: user.profile?.bio || "",
      phone: user.profile?.phone || "",
      organization: user.profile?.organization || "",
      learningGoals: user.profile?.learningGoals || "",
      preferredLang: user.profile?.language || "ko",
      timezone: user.profile?.timezone || "Asia/Seoul",
      notifications: user.profile?.notifications ?? true,
    })
  } catch (error) {
    console.error("Profile fetch error:", error)
    return NextResponse.json(
      { error: "Failed to fetch profile" },
      { status: 500 }
    )
  }
}

export async function PUT(request: NextRequest) {
  try {
    const session = await auth()
    
    if (!session?.user?.id) {
      return NextResponse.json(
        { error: "Unauthorized" },
        { status: 401 }
      )
    }

    const body = await request.json()
    const {
      name,
      bio,
      phone,
      organization,
      learningGoals,
      preferredLang,
      timezone,
      notifications,
    } = body

    // Update user name
    if (name !== undefined) {
      await prisma.user.update({
        where: { id: session.user.id },
        data: { name },
      })
    }

    // Update or create profile
    await prisma.profile.upsert({
      where: { userId: session.user.id },
      update: {
        bio,
        phone,
        organization,
        learningGoals,
        language: preferredLang,
        timezone,
        notifications,
      },
      create: {
        userId: session.user.id,
        bio,
        phone,
        organization,
        learningGoals,
        language: preferredLang,
        timezone,
        notifications,
      },
    })

    return NextResponse.json({
      message: "Profile updated successfully",
    })
  } catch (error) {
    console.error("Profile update error:", error)
    return NextResponse.json(
      { error: "Failed to update profile" },
      { status: 500 }
    )
  }
}