export const intermediateCurriculum = {
  title: '중급 과정 - 반도체 제조 & 설계',
  description: '반도체 제조 공정과 회로 설계 실무',
  duration: '18시간',
  level: 'intermediate',
  modules: [
    {
      id: 'inter-1',
      chapterId: 'lithography',
      title: '1. 포토리소그래피',
      duration: '4시간',
      topics: [
        '노광 공정의 원리',
        'EUV (극자외선) 기술',
        '마스크와 레티클',
        '다중 패터닝 (LELE, SAQP)'
      ],
      completed: false
    },
    {
      id: 'inter-2',
      chapterId: 'fabrication',
      title: '2. 웨이퍼(Wafer) 제조',
      duration: '3시간',
      topics: [
        'CZ 공정 (단결정 성장)',
        '웨이퍼 슬라이싱과 연마',
        '300mm vs 450mm 웨이퍼',
        '웨이퍼 품질 검사'
      ],
      completed: false
    },
    {
      id: 'inter-3',
      chapterId: 'fabrication',
      title: '3. 박막 증착 (Thin Film Deposition)',
      duration: '3시간',
      topics: [
        'CVD (화학 기상 증착)',
        'ALD (원자층 증착)',
        'PVD (물리 기상 증착)',
        '박막 품질 제어'
      ],
      completed: false
    },
    {
      id: 'inter-4',
      chapterId: 'fabrication',
      title: '4. 에칭 (Etching)',
      duration: '3시간',
      topics: [
        '건식 에칭 (Plasma, RIE)',
        '습식 에칭 (화학 용액)',
        '선택비와 이방성',
        '에칭 프로파일 제어'
      ],
      completed: false
    },
    {
      id: 'inter-5',
      chapterId: 'fabrication',
      title: '5. 이온주입(Ion Implantation)과 CMP',
      duration: '2시간',
      topics: [
        '이온주입 공정',
        '도핑 프로파일 제어',
        'CMP (화학적 기계적 연마)',
        '평탄화 기술'
      ],
      completed: false
    },
    {
      id: 'inter-6',
      chapterId: 'advanced',
      title: '6. 첨단 제조 기술',
      duration: '3시간',
      topics: [
        'FinFET & GAA 기술',
        '3D NAND 적층',
        'TSV (Through-Silicon Via)',
        'Chiplet 아키텍처'
      ],
      completed: false
    }
  ]
}
