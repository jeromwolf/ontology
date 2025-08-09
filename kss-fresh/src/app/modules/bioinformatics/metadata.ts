export const bioinformaticsMetadata = {
  id: 'bioinformatics',
  name: 'Bio-informatics & Computational Biology',
  description: '유전체학, 단백질체학, 약물 설계를 위한 컴퓨터 생물학의 모든 것',
  version: '1.0.0',
  category: '바이오/의료',
  difficulty: 'advanced' as const,
  duration: '10주',
  students: 342,
  rating: 4.8,
  chapters: [
    {
      id: 'biology-fundamentals',
      title: 'Chapter 1: 분자생물학 기초',
      description: 'DNA, RNA, 단백질의 구조와 기능, 유전자 발현 메커니즘',
      duration: '3시간',
      objectives: [
        'DNA 이중나선 구조와 염기서열의 이해',
        'RNA 종류와 기능 (mRNA, tRNA, rRNA, miRNA)',
        '단백질 구조 (1차~4차) 및 폴딩 원리',
        '전사와 번역 과정의 분자적 메커니즘'
      ]
    },
    {
      id: 'cell-genetics',
      title: 'Chapter 2: 세포 유전학과 염색체',
      description: '염색체 구조, 유전자 조절, 후성유전학 기초',
      duration: '2.5시간',
      objectives: [
        '염색체 구조와 히스톤 변형',
        '유전자 조절 네트워크 (전사인자, 인핸서)',
        '후성유전학 마크 (DNA 메틸화, 히스톤 아세틸화)',
        '유전자 발현 조절 메커니즘'
      ]
    },
    {
      id: 'genomics-sequencing',
      title: 'Chapter 3: 유전체학과 시퀀싱 기술',
      description: 'NGS, WGS, RNA-seq 등 최신 시퀀싱 기술과 유전체 분석',
      duration: '2시간',
      objectives: [
        'DNA/RNA 시퀀싱 원리 이해',
        'NGS (Next-Generation Sequencing) 파이프라인 구축',
        'Quality Control 및 데이터 전처리',
        'Genome Assembly와 Annotation'
      ]
    },
    {
      id: 'sequence-alignment',
      title: 'Chapter 4: 서열 정렬과 비교 유전체학',
      description: 'BLAST, 다중 서열 정렬, 계통 발생학적 분석',
      duration: '2시간',
      objectives: [
        'Pairwise & Multiple Sequence Alignment',
        'BLAST와 유사성 검색 알고리즘',
        'Phylogenetic Tree 구축과 분석',
        'Comparative Genomics 응용'
      ]
    },
    {
      id: 'proteomics-structure',
      title: 'Chapter 5: 단백질체학과 구조 예측',
      description: 'AlphaFold, 단백질 폴딩, 3D 구조 모델링',
      duration: '2시간',
      objectives: [
        '단백질 구조 예측 알고리즘',
        'AlphaFold2/3 활용법',
        'Homology Modeling과 Threading',
        'Protein-Protein Interaction 네트워크'
      ]
    },
    {
      id: 'drug-discovery',
      title: 'Chapter 6: 약물 설계와 분자 도킹',
      description: 'Virtual Screening, QSAR, 분자 동역학 시뮬레이션',
      duration: '2시간',
      objectives: [
        'Computer-Aided Drug Design (CADD)',
        'Molecular Docking과 Virtual Screening',
        'QSAR 모델링과 약물 특성 예측',
        'MD Simulation으로 약물-타겟 상호작용 분석'
      ]
    },
    {
      id: 'omics-integration',
      title: 'Chapter 7: 멀티오믹스 통합 분석',
      description: 'Genomics, Transcriptomics, Proteomics, Metabolomics 통합',
      duration: '2시간',
      objectives: [
        'Multi-omics 데이터 통합 전략',
        'Network Biology와 Systems Biology',
        'Pathway Analysis와 Enrichment',
        '질병 바이오마커 발굴'
      ]
    },
    {
      id: 'ml-genomics',
      title: 'Chapter 8: 머신러닝과 유전체 의학',
      description: 'Deep Learning for Genomics, Precision Medicine',
      duration: '2시간',
      objectives: [
        'CNN/RNN for 유전체 서열 분석',
        'Variant Calling과 GWAS',
        'Polygenic Risk Score 계산',
        '정밀의학과 개인 맞춤 치료'
      ]
    },
    {
      id: 'single-cell',
      title: 'Chapter 9: 단일세포 유전체학',
      description: 'scRNA-seq, Cell Type Identification, Trajectory Analysis',
      duration: '2시간',
      objectives: [
        'Single-cell RNA sequencing 분석',
        'Cell Clustering과 Annotation',
        'Pseudotime과 Trajectory Inference',
        'Spatial Transcriptomics'
      ]
    },
    {
      id: 'clinical-applications',
      title: 'Chapter 10: 임상 바이오인포매틱스',
      description: 'Cancer Genomics, Liquid Biopsy, Clinical Decision Support',
      duration: '2시간',
      objectives: [
        'Cancer Genomics와 종양 진화',
        'Liquid Biopsy와 ctDNA 분석',
        'Pharmacogenomics와 약물 반응 예측',
        'Clinical Trial Design과 Real-world Evidence'
      ]
    }
  ],
  simulators: [
    {
      id: 'sequence-analyzer',
      name: 'DNA/RNA 서열 분석기',
      description: 'FASTA 파일 파싱, 서열 정렬, 품질 분석'
    },
    {
      id: 'protein-visualizer',
      name: '3D 단백질 구조 뷰어',
      description: 'PDB 파일 시각화, AlphaFold 예측 결과 비교'
    },
    {
      id: 'drug-docking-sim',
      name: '분자 도킹 시뮬레이터',
      description: '리간드-수용체 도킹, 결합 친화도 계산'
    },
    {
      id: 'gwas-explorer',
      name: 'GWAS 데이터 탐색기',
      description: 'Manhattan Plot, QQ Plot, PRS 계산기'
    }
  ],
  prerequisites: ['생물학 기초', '통계학', '프로그래밍 (Python/R)'],
  learningPath: {
    next: ['medical-ai', 'drug-discovery-advanced'],
    previous: ['molecular-biology-basics']
  }
}