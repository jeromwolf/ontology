import Link from 'next/link'
import { Button } from '@cognosphere/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@cognosphere/ui/card'
import { Brain, Network, Sparkles, Zap } from 'lucide-react'

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-neutral-50 to-neutral-100 dark:from-neutral-950 dark:to-neutral-900">
      {/* Hero Section */}
      <section className="relative px-6 py-24 lg:px-8">
        <div className="mx-auto max-w-7xl">
          <div className="text-center">
            <h1 className="text-4xl font-bold tracking-tight text-neutral-900 dark:text-neutral-100 sm:text-6xl">
              Cognosphere
            </h1>
            <p className="mt-6 text-lg leading-8 text-neutral-600 dark:text-neutral-400">
              지식 시뮬레이터 플랫폼 - 온톨로지와 시맨틱 웹 기술의 대화형 학습 환경
            </p>
            <div className="mt-10 flex items-center justify-center gap-x-6">
              <Link href="/learn">
                <Button size="lg" className="gap-2">
                  <Sparkles className="h-4 w-4" />
                  학습 시작하기
                </Button>
              </Link>
              <Link href="/playground">
                <Button variant="outline" size="lg">
                  플레이그라운드 체험
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="px-6 py-24 lg:px-8">
        <div className="mx-auto max-w-7xl">
          <div className="grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-4">
            <Card>
              <CardHeader>
                <Brain className="h-10 w-10 text-primary mb-4" />
                <CardTitle>대화형 학습</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  실시간 시뮬레이션과 시각화를 통해 복잡한 개념을 쉽게 이해할 수 있습니다
                </CardDescription>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <Network className="h-10 w-10 text-primary mb-4" />
                <CardTitle>지식 그래프</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  개념 간의 관계를 시각적으로 탐색하고 연결된 지식을 구축합니다
                </CardDescription>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <Zap className="h-10 w-10 text-primary mb-4" />
                <CardTitle>실시간 피드백</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  즉각적인 검증과 제안으로 학습 효과를 극대화합니다
                </CardDescription>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <Sparkles className="h-10 w-10 text-primary mb-4" />
                <CardTitle>맞춤형 경로</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  개인의 학습 스타일과 진도에 맞춘 최적화된 학습 경로를 제공합니다
                </CardDescription>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Learning Paths Preview */}
      <section className="px-6 py-24 lg:px-8 bg-neutral-100 dark:bg-neutral-900">
        <div className="mx-auto max-w-7xl">
          <h2 className="text-3xl font-bold text-center mb-12">학습 경로</h2>
          <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
            <Card>
              <CardHeader>
                <CardTitle>온톨로지 기초</CardTitle>
                <CardDescription>개념과 원리 이해하기</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-neutral-600 dark:text-neutral-400">
                  온톨로지의 기본 개념부터 철학적 배경까지 체계적으로 학습합니다
                </p>
                <Link href="/learn/ontology-basics">
                  <Button variant="link" className="mt-4 p-0">
                    시작하기 →
                  </Button>
                </Link>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>시맨틱 웹 실습</CardTitle>
                <CardDescription>RDF, OWL, SPARQL 마스터하기</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-neutral-600 dark:text-neutral-400">
                  실제 데이터로 시맨틱 웹 기술을 실습하고 응용합니다
                </p>
                <Link href="/learn/semantic-web">
                  <Button variant="link" className="mt-4 p-0">
                    시작하기 →
                  </Button>
                </Link>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>프로젝트 기반 학습</CardTitle>
                <CardDescription>실제 온톨로지 구축하기</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-neutral-600 dark:text-neutral-400">
                  도메인별 온톨로지를 직접 설계하고 구현해봅니다
                </p>
                <Link href="/learn/projects">
                  <Button variant="link" className="mt-4 p-0">
                    시작하기 →
                  </Button>
                </Link>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>
    </div>
  )
}