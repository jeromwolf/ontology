export const beginnerCurriculum = {
  title: '초급 과정 - 반도체 기초',
  description: '반도체의 기본 원리와 핵심 개념 학습',
  duration: '12시간',
  level: 'beginner',
  modules: [
    {
      id: 'basics-1',
      chapterId: 'basics',
      title: '1. 반도체란 무엇인가?',
      duration: '2시간',
      topics: [
        '도체, 부도체, 반도체의 차이',
        '실리콘의 결정 구조',
        '원자가 전자와 에너지 밴드',
        '순수 반도체 vs 불순물 반도체'
      ],
      completed: false
    },
    {
      id: 'basics-2',
      chapterId: 'basics',
      title: '2. 도핑과 PN 접합',
      duration: '2시간',
      topics: [
        'N형 반도체 (전자 과잉)',
        'P형 반도체 (정공 과잉)',
        'PN 접합의 형성',
        '공핍 영역과 내장 전위'
      ],
      completed: false
    },
    {
      id: 'basics-3',
      chapterId: 'basics',
      title: '3. 다이오드의 원리',
      duration: '2시간',
      topics: [
        '순방향 바이어스',
        '역방향 바이어스',
        'I-V 특성 곡선',
        '실생활 응용 (정류, LED)'
      ],
      completed: false
    },
    {
      id: 'basics-4',
      chapterId: 'design',
      title: '4. 트랜지스터 기초',
      duration: '3시간',
      topics: [
        'BJT (바이폴라 접합 트랜지스터)',
        'MOSFET (금속-산화물-반도체)',
        '트랜지스터의 스위칭 동작',
        '증폭 작용의 원리'
      ],
      completed: false
    },
    {
      id: 'basics-5',
      chapterId: 'design',
      title: '5. 디지털 논리 회로',
      duration: '3시간',
      topics: [
        'CMOS 인버터',
        '기본 논리 게이트 (AND, OR, NOT)',
        '진리표와 부울 대수',
        '간단한 조합 회로 설계'
      ],
      completed: false
    }
  ]
}
