export const advancedCurriculum = {
  title: '고급 과정 - 첨단 반도체 기술',
  description: '최신 반도체 기술과 미래 트렌드',
  duration: '24시간',
  level: 'advanced',
  modules: [
    {
      id: 'adv-1',
      chapterId: 'advanced',
      title: '1. FinFET & GAA 기술',
      duration: '4시간',
      topics: [
        'FinFET 3면 게이트 구조',
        'GAA (Gate-All-Around)',
        'Nanosheet & Nanowire',
        '5nm 이하 공정의 과제'
      ],
      completed: false
    },
    {
      id: 'adv-2',
      chapterId: 'advanced',
      title: '2. 3D 적층 기술',
      duration: '3시간',
      topics: [
        '3D NAND (200단 적층)',
        'TSV (Through-Silicon Via)',
        'Chiplet 아키텍처',
        'HBM (고대역폭 메모리)'
      ],
      completed: false
    },
    {
      id: 'adv-3',
      chapterId: 'ai-chips',
      title: '3. AI 반도체 아키텍처',
      duration: '5시간',
      topics: [
        'NPU 아키텍처 (MAC 연산)',
        'TPU Systolic Array',
        'GPU 병렬 처리 (Tensor Core)',
        'AI 가속기 최적화'
      ],
      completed: false
    },
    {
      id: 'adv-4',
      chapterId: 'memory',
      title: '4. 메모리 반도체',
      duration: '3시간',
      topics: [
        'MRAM (자기 저항 메모리)',
        'RRAM (저항 변화 메모리)',
        'PCM (상변화 메모리)',
        'FeRAM (강유전체 메모리)'
      ],
      completed: false
    },
    {
      id: 'adv-5',
      chapterId: 'future',
      title: '5. 미래 반도체 기술',
      duration: '5시간',
      topics: [
        '양자 컴퓨팅 칩 (초전도 큐비트)',
        '뉴로모픽 칩 (Spiking NN)',
        '광자 집적회로 (Silicon Photonics)',
        '탄소 나노튜브 트랜지스터 (CNFET)'
      ],
      completed: false
    },
    {
      id: 'adv-6',
      chapterId: 'image-display',
      title: '6. 이미지센서 & 디스플레이 반도체',
      duration: '4시간',
      topics: [
        'CMOS 이미지센서 (CIS)',
        'BSI/Stacked 구조',
        '디스플레이 드라이버 IC (DDI)',
        'OLED/LCD 기술'
      ],
      completed: false
    }
  ]
}
