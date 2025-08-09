// Common type definitions for Cognosphere ecosystem

export interface User {
  id: string
  email: string
  name: string
  createdAt: Date
  updatedAt: Date
}

export interface Course {
  id: string
  title: string
  description: string
  chapters: Chapter[]
  createdAt: Date
  updatedAt: Date
}

export interface Chapter {
  id: string
  title: string
  content: string
  order: number
  courseId: string
}