'use client'

import { useState } from 'react'
import Link from 'next/link'
import { bioinformaticsMetadata } from './metadata'
import { 
  Dna, FlaskConical, Brain, Microscope, 
  ChevronRight, PlayCircle, BookOpen, Target,
  Activity, Network, Beaker, Pill
} from 'lucide-react'

const iconMap: { [key: string]: React.ElementType } = {
  'biology-fundamentals': Dna,
  'cell-genetics': Network,
  'genomics-sequencing': Activity,
  'sequence-alignment': FlaskConical,
  'proteomics-structure': Microscope,
  'drug-discovery': Pill,
  'omics-integration': Brain,
  'ml-genomics': Target,
  'single-cell': FlaskConical,
  'clinical-applications': Target
}

export default function BioinformaticsPage() {
  const [hoveredChapter, setHoveredChapter] = useState<string | null>(null)
  const [activeSimulator, setActiveSimulator] = useState<string | null>(null)

  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <section className="bg-gradient-to-r from-emerald-600 to-teal-600 rounded-3xl p-8 text-white shadow-2xl">
        <div className="flex items-center gap-4 mb-6">
          <div className="p-3 bg-white/20 rounded-xl">
            <Dna className="w-8 h-8" />
          </div>
          <div>
            <h2 className="text-3xl font-bold">생명과학과 컴퓨터의 융합</h2>
            <p className="text-emerald-100 mt-1">유전체부터 신약 개발까지, 계산 생물학의 최전선</p>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mt-6">
          <div className="bg-white/10 rounded-xl p-4">
            <div className="text-2xl font-bold">21+</div>
            <div className="text-emerald-100">학습 시간</div>
          </div>
          <div className="bg-white/10 rounded-xl p-4">
            <div className="text-2xl font-bold">10</div>
            <div className="text-emerald-100">챕터</div>
          </div>
          <div className="bg-white/10 rounded-xl p-4">
            <div className="text-2xl font-bold">4</div>
            <div className="text-emerald-100">시뮬레이터</div>
          </div>
          <div className="bg-white/10 rounded-xl p-4">
            <div className="text-2xl font-bold">342</div>
            <div className="text-emerald-100">수강생</div>
          </div>
        </div>
      </section>

      {/* Learning Path */}
      <section className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg">
        <h3 className="text-xl font-bold text-gray-800 dark:text-gray-200 mb-4">
          🧬 학습 경로
        </h3>
        <div className="flex items-center gap-2 overflow-x-auto pb-2">
          <div className="flex-shrink-0 px-4 py-2 bg-red-100 dark:bg-red-900/50 rounded-lg text-sm">
            분자생물학 기초
          </div>
          <ChevronRight className="w-4 h-4 text-gray-400" />
          <div className="flex-shrink-0 px-4 py-2 bg-orange-100 dark:bg-orange-900/50 rounded-lg text-sm">
            세포 유전학
          </div>
          <ChevronRight className="w-4 h-4 text-gray-400" />
          <div className="flex-shrink-0 px-4 py-2 bg-emerald-100 dark:bg-emerald-900/50 rounded-lg text-sm">
            유전체 시퀀싱
          </div>
          <ChevronRight className="w-4 h-4 text-gray-400" />
          <div className="flex-shrink-0 px-4 py-2 bg-teal-100 dark:bg-teal-900/50 rounded-lg text-sm">
            서열 정렬
          </div>
          <ChevronRight className="w-4 h-4 text-gray-400" />
          <div className="flex-shrink-0 px-4 py-2 bg-cyan-100 dark:bg-cyan-900/50 rounded-lg text-sm">
            단백질 구조
          </div>
          <ChevronRight className="w-4 h-4 text-gray-400" />
          <div className="flex-shrink-0 px-4 py-2 bg-blue-100 dark:bg-blue-900/50 rounded-lg text-sm">
            약물 설계
          </div>
          <ChevronRight className="w-4 h-4 text-gray-400" />
          <div className="flex-shrink-0 px-4 py-2 bg-indigo-100 dark:bg-indigo-900/50 rounded-lg text-sm">
            ML 응용
          </div>
          <ChevronRight className="w-4 h-4 text-gray-400" />
          <div className="flex-shrink-0 px-4 py-2 bg-purple-100 dark:bg-purple-900/50 rounded-lg text-sm">
            임상 응용
          </div>
        </div>
      </section>

      {/* Chapters Grid */}
      <section>
        <h3 className="text-2xl font-bold text-gray-800 dark:text-gray-200 mb-6">
          📚 학습 챕터
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {bioinformaticsMetadata.chapters.map((chapter) => {
            const Icon = iconMap[chapter.id] || BookOpen
            return (
              <Link
                key={chapter.id}
                href={`/modules/bioinformatics/${chapter.id}`}
                className="group"
                onMouseEnter={() => setHoveredChapter(chapter.id)}
                onMouseLeave={() => setHoveredChapter(null)}
              >
                <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border-2 border-transparent hover:border-emerald-500 transition-all duration-300 hover:shadow-xl">
                  <div className="flex items-start gap-4">
                    <div className="p-3 bg-gradient-to-br from-emerald-100 to-teal-100 dark:from-emerald-900/50 dark:to-teal-900/50 rounded-xl group-hover:scale-110 transition-transform">
                      <Icon className="w-6 h-6 text-emerald-600 dark:text-emerald-400" />
                    </div>
                    <div className="flex-1">
                      <h4 className="font-bold text-gray-800 dark:text-gray-200 mb-2">
                        {chapter.title}
                      </h4>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                        {chapter.description}
                      </p>
                      {hoveredChapter === chapter.id && (
                        <div className="space-y-1 animate-in slide-in-from-top-2">
                          {chapter.objectives.map((obj, idx) => (
                            <div key={idx} className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
                              <div className="w-1 h-1 bg-emerald-500 rounded-full" />
                              <span>{obj}</span>
                            </div>
                          ))}
                        </div>
                      )}
                      <div className="flex items-center justify-between mt-4">
                        <span className="text-xs text-gray-500 dark:text-gray-400">
                          {chapter.duration}
                        </span>
                        <ChevronRight className="w-4 h-4 text-emerald-500 group-hover:translate-x-1 transition-transform" />
                      </div>
                    </div>
                  </div>
                </div>
              </Link>
            )
          })}
        </div>
      </section>

      {/* Interactive Simulators */}
      <section className="bg-gradient-to-br from-white to-emerald-50 dark:from-gray-800 dark:to-emerald-950/20 rounded-2xl p-6 shadow-lg">
        <h3 className="text-2xl font-bold text-gray-800 dark:text-gray-200 mb-6">
          🧪 인터랙티브 시뮬레이터
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {bioinformaticsMetadata.simulators.map((simulator) => (
            <Link
              key={simulator.id}
              href={`/modules/bioinformatics/simulators/${simulator.id}`}
              className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-md hover:shadow-xl transition-all cursor-pointer block"
              onClick={() => {
                console.log('Card clicked, navigating to:', `/modules/bioinformatics/simulators/${simulator.id}`)
                setActiveSimulator(simulator.id)
              }}
            >
              <div className="flex items-center justify-between mb-3">
                <h4 className="font-bold text-gray-800 dark:text-gray-200">
                  {simulator.name}
                </h4>
                <div className="p-2 bg-emerald-100 dark:bg-emerald-900/50 rounded-lg">
                  <PlayCircle className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
                </div>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                {simulator.description}
              </p>
              {activeSimulator === simulator.id && (
                <div className="mt-4 p-3 bg-emerald-50 dark:bg-emerald-900/20 rounded-lg animate-in slide-in-from-top-2">
                  <p className="text-xs text-emerald-700 dark:text-emerald-300">
                    클릭하여 시뮬레이터 시작 →
                  </p>
                </div>
              )}
            </Link>
          ))}
        </div>
      </section>

      {/* Key Technologies */}
      <section className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg">
        <h3 className="text-xl font-bold text-gray-800 dark:text-gray-200 mb-4">
          🔧 핵심 기술 스택
        </h3>
        <div className="flex flex-wrap gap-2">
          {['Python', 'R', 'Biopython', 'BioConductor', 'BLAST', 'AlphaFold', 
            'PyMOL', 'Clustal', 'GATK', 'Samtools', 'BWA', 'STAR'].map((tech) => (
            <span
              key={tech}
              className="px-3 py-1 bg-gradient-to-r from-emerald-100 to-teal-100 dark:from-emerald-900/50 dark:to-teal-900/50 text-emerald-700 dark:text-emerald-300 rounded-full text-sm"
            >
              {tech}
            </span>
          ))}
        </div>
      </section>

      {/* Prerequisites */}
      <section className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-2xl p-6 border border-yellow-200 dark:border-yellow-800">
        <h3 className="text-lg font-bold text-gray-800 dark:text-gray-200 mb-3">
          📋 선수 지식
        </h3>
        <ul className="space-y-2">
          {bioinformaticsMetadata.prerequisites.map((prereq, idx) => (
            <li key={idx} className="flex items-center gap-2 text-gray-700 dark:text-gray-300">
              <div className="w-2 h-2 bg-yellow-500 rounded-full" />
              <span>{prereq}</span>
            </li>
          ))}
        </ul>
      </section>
    </div>
  )
}