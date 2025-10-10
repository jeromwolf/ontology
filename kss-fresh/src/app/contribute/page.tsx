'use client'

import React, { useState } from 'react';
import Link from 'next/link';
import {
  BookOpen,
  Code,
  Users,
  Award,
  GitPullRequest,
  MessageSquare,
  CheckCircle,
  ExternalLink,
  Github,
  Heart,
  Zap,
  Target,
  TrendingUp
} from 'lucide-react';

export default function ContributePage() {
  const [selectedType, setSelectedType] = useState<string | null>(null);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-gray-900 dark:to-gray-800">
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-r from-indigo-600 to-purple-600 text-white">
        <div className="absolute inset-0 bg-black/10"></div>

        {/* Animated background pattern */}
        <div className="absolute inset-0 opacity-10">
          <div className="absolute inset-0" style={{
            backgroundImage: 'radial-gradient(circle at 2px 2px, white 1px, transparent 0)',
            backgroundSize: '40px 40px'
          }}></div>
        </div>

        <div className="relative max-w-7xl mx-auto px-6 py-24">
          <div className="text-center">
            <h1 className="text-5xl md:text-7xl font-bold mb-6 animate-fade-in">
              함께 만드는 지식 🚀
            </h1>
            <p className="text-xl md:text-2xl mb-8 text-indigo-100 max-w-3xl mx-auto leading-relaxed">
              KSS는 여러분과 함께 성장합니다. <br />
              AI 시대의 지식을 확장하는 여정에 동참하세요.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
              <Link
                href="#quick-start"
                className="px-8 py-4 bg-white text-indigo-600 rounded-lg font-semibold text-lg hover:bg-indigo-50 transition-all transform hover:scale-105 shadow-lg"
              >
                5분 안에 시작하기
              </Link>
              <Link
                href="https://github.com/jeromwolf/ontology"
                target="_blank"
                className="px-8 py-4 bg-indigo-700 text-white rounded-lg font-semibold text-lg hover:bg-indigo-800 transition-all transform hover:scale-105 shadow-lg flex items-center gap-2"
              >
                <Github className="w-5 h-5" />
                GitHub 보기
              </Link>
            </div>
          </div>

          {/* Stats */}
          <div className="mt-16 grid grid-cols-2 md:grid-cols-4 gap-6">
            {[
              { label: '전체 모듈', value: '27+', icon: BookOpen },
              { label: '시뮬레이터', value: '173+', icon: Code },
              { label: '기여자', value: '10+', icon: Users },
              { label: 'GitHub Stars', value: '50+', icon: Award }
            ].map((stat, i) => (
              <div
                key={i}
                className="bg-white/10 backdrop-blur-sm rounded-lg p-6 text-center transform hover:scale-105 transition-all"
              >
                <stat.icon className="w-8 h-8 mx-auto mb-2 opacity-80" />
                <div className="text-3xl font-bold mb-1">{stat.value}</div>
                <div className="text-indigo-200 text-sm">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Why Contribute */}
      <section className="py-20 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
              왜 기여해야 하나요?
            </h2>
            <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
              KSS에 기여하면 개인적 성장과 함께 수만 명의 학습자에게 영향을 줍니다
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                icon: TrendingUp,
                title: '실력 향상',
                description: '최신 AI 기술을 깊이 있게 학습하고, 전문가 리뷰를 받으며 성장합니다.',
                color: 'from-blue-500 to-cyan-500'
              },
              {
                icon: Users,
                title: '커뮤니티',
                description: '전 세계 AI 전문가들과 네트워킹하고 협업 경험을 쌓습니다.',
                color: 'from-purple-500 to-pink-500'
              },
              {
                icon: Heart,
                title: '임팩트',
                description: '여러분의 기여가 수만 명의 AI 학습 여정에 영향을 미칩니다.',
                color: 'from-orange-500 to-red-500'
              }
            ].map((benefit, i) => (
              <div
                key={i}
                className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg hover:shadow-xl transition-all transform hover:-translate-y-2"
              >
                <div className={`w-16 h-16 rounded-xl bg-gradient-to-br ${benefit.color} flex items-center justify-center mb-6`}>
                  <benefit.icon className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                  {benefit.title}
                </h3>
                <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
                  {benefit.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Contribution Types */}
      <section className="py-20 px-6 bg-white dark:bg-gray-900">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
              기여 방법 선택하기
            </h2>
            <p className="text-xl text-gray-600 dark:text-gray-300">
              여러분의 강점에 맞는 방법으로 시작하세요
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[
              {
                id: 'chapter',
                icon: BookOpen,
                title: '챕터 작성',
                description: 'AI 개념을 명확하게 설명하는 교육 콘텐츠를 작성합니다.',
                difficulty: '중급',
                time: '3-5시간',
                color: 'blue',
                skills: ['글쓰기', 'AI 지식', 'MDX']
              },
              {
                id: 'simulator',
                icon: Code,
                title: '시뮬레이터 개발',
                description: 'React로 인터랙티브한 교육용 시뮬레이터를 만듭니다.',
                difficulty: '고급',
                time: '5-10시간',
                color: 'purple',
                skills: ['React', 'TypeScript', 'Canvas/SVG']
              },
              {
                id: 'improve',
                icon: Zap,
                title: '콘텐츠 개선',
                description: '오타 수정, 예제 추가, 설명 보완 등 기존 콘텐츠를 개선합니다.',
                difficulty: '초급',
                time: '30분-1시간',
                color: 'green',
                skills: ['관찰력', '디테일']
              },
              {
                id: 'review',
                icon: MessageSquare,
                title: '리뷰 & 피드백',
                description: '다른 기여자의 PR을 리뷰하고 건설적 피드백을 제공합니다.',
                difficulty: '중급',
                time: '1-2시간',
                color: 'orange',
                skills: ['비판적 사고', '커뮤니케이션']
              },
              {
                id: 'translate',
                icon: Users,
                title: '번역',
                description: '영어 콘텐츠를 한국어로, 또는 그 반대로 번역합니다.',
                difficulty: '초급',
                time: '2-4시간',
                color: 'pink',
                skills: ['이중 언어', '문서화']
              },
              {
                id: 'references',
                icon: Target,
                title: 'References 추가',
                description: '신뢰할 수 있는 논문, 문서, GitHub 링크를 큐레이션합니다.',
                difficulty: '초급',
                time: '1-2시간',
                color: 'indigo',
                skills: ['리서치', '큐레이션']
              }
            ].map((type) => (
              <div
                key={type.id}
                onClick={() => setSelectedType(type.id)}
                className={`
                  bg-gradient-to-br ${
                    selectedType === type.id
                      ? 'from-indigo-50 to-purple-50 dark:from-indigo-900/30 dark:to-purple-900/30 ring-2 ring-indigo-500'
                      : 'from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-700'
                  }
                  rounded-2xl p-6 cursor-pointer transition-all transform hover:scale-105 hover:shadow-xl
                `}
              >
                <div className={`w-14 h-14 rounded-xl bg-gradient-to-br from-${type.color}-500 to-${type.color}-600 flex items-center justify-center mb-4`}>
                  <type.icon className="w-7 h-7 text-white" />
                </div>

                <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                  {type.title}
                </h3>

                <p className="text-gray-600 dark:text-gray-300 mb-4 text-sm leading-relaxed">
                  {type.description}
                </p>

                <div className="flex items-center gap-4 mb-4 text-sm">
                  <span className={`px-3 py-1 rounded-full ${
                    type.difficulty === '초급' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300' :
                    type.difficulty === '중급' ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-300' :
                    'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300'
                  }`}>
                    {type.difficulty}
                  </span>
                  <span className="text-gray-500 dark:text-gray-400">⏱️ {type.time}</span>
                </div>

                <div className="flex flex-wrap gap-2">
                  {type.skills.map((skill, i) => (
                    <span
                      key={i}
                      className="px-2 py-1 bg-white dark:bg-gray-800 rounded-md text-xs text-gray-600 dark:text-gray-300"
                    >
                      {skill}
                    </span>
                  ))}
                </div>

                {selectedType === type.id && (
                  <div className="mt-4 pt-4 border-t border-indigo-200 dark:border-indigo-800">
                    <Link
                      href={`#quick-start-${type.id}`}
                      className="flex items-center gap-2 text-indigo-600 dark:text-indigo-400 font-semibold hover:gap-3 transition-all"
                    >
                      시작하기 →
                    </Link>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Quick Start Guide */}
      <section id="quick-start" className="py-20 px-6">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
              5분 Quick Start 🚀
            </h2>
            <p className="text-xl text-gray-600 dark:text-gray-300">
              처음이신가요? 이 4단계만 따라하세요!
            </p>
          </div>

          <div className="space-y-6">
            {[
              {
                step: 1,
                title: 'GitHub 계정 & Repository Fork',
                description: 'GitHub 계정을 만들고 KSS 저장소를 Fork합니다.',
                action: 'Fork하러 가기',
                link: 'https://github.com/jeromwolf/ontology/fork',
                time: '1분',
                code: null
              },
              {
                step: 2,
                title: '로컬에 Clone & 설치',
                description: '개발 환경을 설정합니다.',
                action: null,
                link: null,
                time: '2분',
                code: `# Clone your fork
git clone https://github.com/YOUR-USERNAME/ontology.git
cd ontology/kss-fresh

# Install dependencies
npm install

# Run dev server
npm run dev`
              },
              {
                step: 3,
                title: '변경사항 작성 & 커밋',
                description: '코드를 수정하고 커밋합니다.',
                action: null,
                link: null,
                time: '1-5시간 (내용에 따라)',
                code: `# Create a new branch
git checkout -b feature/my-contribution

# Make your changes...

# Commit
git add .
git commit -m "feat: Add awesome feature

- Description of changes
- Why this is useful"

# Push
git push origin feature/my-contribution`
              },
              {
                step: 4,
                title: 'Pull Request 생성',
                description: 'GitHub에서 PR을 열고 리뷰를 기다립니다.',
                action: 'PR 템플릿 보기',
                link: '/CONTRIBUTING.md',
                time: '1분',
                code: null
              }
            ].map((step) => (
              <div
                key={step.step}
                className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg hover:shadow-xl transition-all"
              >
                <div className="flex items-start gap-6">
                  <div className="flex-shrink-0">
                    <div className="w-12 h-12 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white font-bold text-xl">
                      {step.step}
                    </div>
                  </div>

                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="text-2xl font-bold text-gray-900 dark:text-white">
                        {step.title}
                      </h3>
                      <span className="text-sm text-gray-500 dark:text-gray-400">
                        ⏱️ {step.time}
                      </span>
                    </div>

                    <p className="text-gray-600 dark:text-gray-300 mb-4">
                      {step.description}
                    </p>

                    {step.code && (
                      <pre className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto text-sm mb-4">
                        <code>{step.code}</code>
                      </pre>
                    )}

                    {step.action && (
                      <Link
                        href={step.link || '#'}
                        target={step.link?.startsWith('http') ? '_blank' : undefined}
                        className="inline-flex items-center gap-2 px-6 py-3 bg-indigo-600 text-white rounded-lg font-semibold hover:bg-indigo-700 transition-all"
                      >
                        {step.action}
                        {step.link?.startsWith('http') && <ExternalLink className="w-4 h-4" />}
                      </Link>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-12 bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-2xl p-8 border border-green-200 dark:border-green-800">
            <div className="flex items-start gap-4">
              <CheckCircle className="w-8 h-8 text-green-600 dark:text-green-400 flex-shrink-0" />
              <div>
                <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                  축하합니다! 🎉
                </h3>
                <p className="text-gray-700 dark:text-gray-300 mb-4">
                  PR을 생성하셨다면 첫 번째 기여를 완료하신 겁니다!<br />
                  리뷰어가 피드백을 드릴 예정이며, 승인되면 🌱 Contributor 배지를 받게 됩니다.
                </p>
                <Link
                  href="https://github.com/jeromwolf/ontology/pulls"
                  target="_blank"
                  className="inline-flex items-center gap-2 text-green-600 dark:text-green-400 font-semibold hover:gap-3 transition-all"
                >
                  내 PR 상태 확인하기 <ExternalLink className="w-4 h-4" />
                </Link>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Resources */}
      <section className="py-20 px-6 bg-gray-50 dark:bg-gray-900">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
              유용한 리소스
            </h2>
            <p className="text-xl text-gray-600 dark:text-gray-300">
              기여에 도움이 되는 문서와 도구들
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[
              {
                title: 'Contributing Guide',
                description: '전체 기여 가이드라인과 코드 스타일',
                link: '/CONTRIBUTING.md',
                icon: BookOpen,
                color: 'blue'
              },
              {
                title: 'Code of Conduct',
                description: '커뮤니티 행동 강령',
                link: '/CODE_OF_CONDUCT.md',
                icon: Heart,
                color: 'red'
              },
              {
                title: 'GitHub Discussions',
                description: '질문하고 토론하는 공간',
                link: 'https://github.com/jeromwolf/ontology/discussions',
                icon: MessageSquare,
                color: 'purple'
              },
              {
                title: 'Existing PRs',
                description: '다른 기여자들의 PR 보기',
                link: 'https://github.com/jeromwolf/ontology/pulls',
                icon: GitPullRequest,
                color: 'green'
              },
              {
                title: 'Good First Issues',
                description: '초보자를 위한 쉬운 이슈들',
                link: 'https://github.com/jeromwolf/ontology/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22',
                icon: Target,
                color: 'yellow'
              },
              {
                title: 'Project Roadmap',
                description: 'KSS의 미래 계획',
                link: 'https://github.com/jeromwolf/ontology/projects',
                icon: TrendingUp,
                color: 'indigo'
              }
            ].map((resource, i) => (
              <Link
                key={i}
                href={resource.link}
                target={resource.link.startsWith('http') ? '_blank' : undefined}
                className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg hover:shadow-xl transition-all transform hover:-translate-y-1"
              >
                <div className={`w-12 h-12 rounded-lg bg-gradient-to-br from-${resource.color}-500 to-${resource.color}-600 flex items-center justify-center mb-4`}>
                  <resource.icon className="w-6 h-6 text-white" />
                </div>
                <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2 flex items-center gap-2">
                  {resource.title}
                  {resource.link.startsWith('http') && <ExternalLink className="w-4 h-4" />}
                </h3>
                <p className="text-gray-600 dark:text-gray-300 text-sm">
                  {resource.description}
                </p>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 px-6 bg-gradient-to-r from-indigo-600 to-purple-600 text-white">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-4xl md:text-5xl font-bold mb-6">
            준비되셨나요? 🚀
          </h2>
          <p className="text-xl mb-8 text-indigo-100">
            AI 시대의 지식을 함께 만드는 여정, 지금 시작하세요!
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              href="https://github.com/jeromwolf/ontology/fork"
              target="_blank"
              className="px-8 py-4 bg-white text-indigo-600 rounded-lg font-semibold text-lg hover:bg-indigo-50 transition-all transform hover:scale-105 shadow-lg flex items-center justify-center gap-2"
            >
              <Github className="w-5 h-5" />
              지금 Fork하기
            </Link>
            <Link
              href="https://github.com/jeromwolf/ontology/discussions"
              target="_blank"
              className="px-8 py-4 bg-indigo-700 text-white rounded-lg font-semibold text-lg hover:bg-indigo-800 transition-all transform hover:scale-105 shadow-lg flex items-center justify-center gap-2"
            >
              <MessageSquare className="w-5 h-5" />
              질문하기
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}
