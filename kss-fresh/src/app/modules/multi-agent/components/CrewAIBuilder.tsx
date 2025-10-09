'use client';

import React, { useState, useRef, useEffect } from 'react';
import {
  Users, Brain, Target, Settings, Play, Plus, Trash2,
  ChevronRight, Briefcase, CheckCircle, XCircle, RefreshCw,
  Zap, FileText, Search, Code, MessageSquare, Globe,
  Download, Copy, Network, BarChart3, Pause
} from 'lucide-react';

// Types
interface Agent {
  id: string;
  name: string;
  role: string;
  goal: string;
  backstory: string;
  tools: string[];
  llm?: string;
  temperature?: number;
  status?: 'idle' | 'thinking' | 'working' | 'done';
  progress?: number;
  x?: number;
  y?: number;
}

interface Task {
  id: string;
  description: string;
  expectedOutput: string;
  agent: string;
  tools: string[];
  dependencies: string[];
  status?: 'pending' | 'running' | 'completed';
  output?: string;
}

interface CrewConfig {
  name: string;
  agents: Agent[];
  tasks: Task[];
  process: 'sequential' | 'hierarchical' | 'parallel';
  verbose: boolean;
}

interface TeamTemplate {
  id: string;
  name: string;
  description: string;
  agents: Omit<Agent, 'id' | 'status' | 'progress' | 'x' | 'y'>[];
  tasks: Omit<Task, 'id' | 'agent' | 'status' | 'output'>[];
  process: 'sequential' | 'hierarchical' | 'parallel';
  icon: string;
}

// Team Templates
const TEAM_TEMPLATES: TeamTemplate[] = [
  {
    id: 'content-creation',
    name: '콘텐츠 제작팀',
    description: '블로그, 기사, SNS 콘텐츠를 제작하는 팀',
    icon: '📝',
    process: 'sequential',
    agents: [
      {
        name: 'Content Researcher',
        role: '콘텐츠 리서처',
        goal: '주제에 대한 깊이 있는 리서치 수행',
        backstory: '10년 경력의 콘텐츠 리서치 전문가. 다양한 출처에서 신뢰할 수 있는 정보를 수집합니다.',
        tools: ['search', 'file'],
        llm: 'gpt-4-turbo',
        temperature: 0.5
      },
      {
        name: 'Content Writer',
        role: '콘텐츠 작가',
        goal: '매력적이고 SEO 최적화된 콘텐츠 작성',
        backstory: '전문 작가로 15년간 다양한 매체에서 활동. 독자 친화적인 글쓰기가 특기입니다.',
        tools: ['search', 'file'],
        llm: 'gpt-4-turbo',
        temperature: 0.7
      },
      {
        name: 'Content Editor',
        role: '에디터',
        goal: '콘텐츠의 품질과 일관성 검토',
        backstory: '저널리즘 배경을 가진 편집 전문가. 문법, 스타일, 사실 확인에 뛰어납니다.',
        tools: ['file'],
        llm: 'gpt-4-turbo',
        temperature: 0.3
      },
      {
        name: 'SEO Specialist',
        role: 'SEO 전문가',
        goal: '검색 엔진 최적화 및 키워드 전략 수립',
        backstory: 'SEO 분야 8년 경력. 콘텐츠가 검색 결과 상위에 노출되도록 최적화합니다.',
        tools: ['search', 'file'],
        llm: 'gpt-3.5-turbo',
        temperature: 0.4
      }
    ],
    tasks: [
      {
        description: '주제에 대한 심층 리서치 수행 및 자료 정리',
        expectedOutput: '핵심 정보, 통계, 인용구가 포함된 리서치 보고서',
        tools: ['search', 'file'],
        dependencies: []
      },
      {
        description: '리서치 자료 기반 2000단어 블로그 포스트 작성',
        expectedOutput: '구조화된 블로그 포스트 초안 (제목, 본문, 결론)',
        tools: ['file'],
        dependencies: []
      },
      {
        description: '작성된 콘텐츠 검토 및 수정 제안',
        expectedOutput: '수정된 최종 콘텐츠와 편집 노트',
        tools: ['file'],
        dependencies: []
      },
      {
        description: 'SEO 키워드 분석 및 메타데이터 최적화',
        expectedOutput: 'SEO 최적화 제안과 메타 태그, 키워드 목록',
        tools: ['search'],
        dependencies: []
      }
    ]
  },
  {
    id: 'customer-support',
    name: '고객 지원팀',
    description: '고객 문의를 처리하고 솔루션을 제공하는 팀',
    icon: '💬',
    process: 'parallel',
    agents: [
      {
        name: 'Customer Support Agent',
        role: '1차 상담원',
        goal: '고객 문의를 신속하고 정확하게 처리',
        backstory: '고객 서비스 5년 경력. 공감능력이 뛰어나고 문제 해결 지향적입니다.',
        tools: ['chat', 'search', 'file'],
        llm: 'gpt-3.5-turbo',
        temperature: 0.6
      },
      {
        name: 'Technical Expert',
        role: '기술 전문가',
        goal: '복잡한 기술 문제 해결 및 가이드 제공',
        backstory: '엔지니어 출신 기술 지원 전문가. 복잡한 문제를 쉽게 설명하는 능력이 탁월합니다.',
        tools: ['code', 'search', 'file'],
        llm: 'gpt-4-turbo',
        temperature: 0.4
      },
      {
        name: 'Escalation Manager',
        role: '에스컬레이션 관리자',
        goal: '중요 이슈 처리 및 고객 만족도 관리',
        backstory: '10년 경력의 고객 경험 관리 전문가. 어려운 상황을 긍정적으로 전환합니다.',
        tools: ['chat', 'file'],
        llm: 'gpt-4-turbo',
        temperature: 0.5
      }
    ],
    tasks: [
      {
        description: '고객 문의 분류 및 1차 응답 제공',
        expectedOutput: '문의 분류 결과와 초기 응답 메시지',
        tools: ['chat', 'search'],
        dependencies: []
      },
      {
        description: '기술적 문제에 대한 상세 솔루션 제공',
        expectedOutput: '단계별 해결 가이드와 코드 예시',
        tools: ['code', 'search'],
        dependencies: []
      },
      {
        description: '고객 만족도 확인 및 후속 조치',
        expectedOutput: '만족도 평가와 개선 제안 사항',
        tools: ['chat'],
        dependencies: []
      }
    ]
  },
  {
    id: 'software-dev',
    name: '소프트웨어 개발팀',
    description: '풀스택 소프트웨어 개발 프로젝트를 수행하는 팀',
    icon: '💻',
    process: 'hierarchical',
    agents: [
      {
        name: 'Product Manager',
        role: '프로덕트 매니저',
        goal: '제품 비전 정의 및 요구사항 관리',
        backstory: '스타트업과 대기업 모두 경험한 PM. 사용자 중심 사고와 비즈니스 감각을 겸비했습니다.',
        tools: ['search', 'file'],
        llm: 'gpt-4-turbo',
        temperature: 0.6
      },
      {
        name: 'Backend Developer',
        role: '백엔드 개발자',
        goal: '확장 가능한 서버 아키텍처 설계 및 구현',
        backstory: '분산 시스템 전문가. Node.js, Python, Go 모두 능숙하게 다룹니다.',
        tools: ['code', 'search', 'file'],
        llm: 'gpt-4-turbo',
        temperature: 0.3
      },
      {
        name: 'Frontend Developer',
        role: '프론트엔드 개발자',
        goal: '사용자 친화적인 UI/UX 구현',
        backstory: 'React, Vue, Angular 전문가. 성능과 접근성을 중시합니다.',
        tools: ['code', 'file'],
        llm: 'gpt-4-turbo',
        temperature: 0.4
      },
      {
        name: 'QA Engineer',
        role: '품질 보증 엔지니어',
        goal: '철저한 테스트로 제품 품질 보장',
        backstory: '자동화 테스트 전문가. 버그를 찾아내는 예리한 눈을 가졌습니다.',
        tools: ['code', 'file'],
        llm: 'gpt-3.5-turbo',
        temperature: 0.3
      }
    ],
    tasks: [
      {
        description: '제품 요구사항 정의 및 기술 스펙 문서 작성',
        expectedOutput: 'PRD (Product Requirement Document)와 기술 스펙',
        tools: ['file'],
        dependencies: []
      },
      {
        description: 'REST API 설계 및 데이터베이스 스키마 구현',
        expectedOutput: 'API 엔드포인트 코드와 DB 마이그레이션 스크립트',
        tools: ['code', 'file'],
        dependencies: []
      },
      {
        description: 'React 기반 프론트엔드 컴포넌트 개발',
        expectedOutput: '재사용 가능한 UI 컴포넌트 라이브러리',
        tools: ['code'],
        dependencies: []
      },
      {
        description: '통합 테스트 및 E2E 테스트 작성',
        expectedOutput: '테스트 케이스와 자동화 테스트 스크립트',
        tools: ['code'],
        dependencies: []
      }
    ]
  },
  {
    id: 'research',
    name: '연구팀',
    description: '학술 연구 및 데이터 분석을 수행하는 팀',
    icon: '🔬',
    process: 'sequential',
    agents: [
      {
        name: 'Principal Investigator',
        role: '수석 연구원',
        goal: '연구 방향 설정 및 가설 수립',
        backstory: '20년 경력의 연구자. 여러 편의 논문을 저명한 저널에 게재했습니다.',
        tools: ['search', 'file'],
        llm: 'gpt-4-turbo',
        temperature: 0.5
      },
      {
        name: 'Data Scientist',
        role: '데이터 과학자',
        goal: '데이터 수집, 정제, 분석 수행',
        backstory: '통계학 박사. Python, R, SQL을 활용한 데이터 분석 전문가입니다.',
        tools: ['code', 'search', 'file'],
        llm: 'gpt-4-turbo',
        temperature: 0.3
      },
      {
        name: 'Research Assistant',
        role: '연구 보조원',
        goal: '문헌 조사 및 실험 데이터 정리',
        backstory: '석사 과정 연구원. 꼼꼼한 성격으로 데이터 관리에 능숙합니다.',
        tools: ['search', 'file'],
        llm: 'gpt-3.5-turbo',
        temperature: 0.4
      },
      {
        name: 'Academic Writer',
        role: '학술 작가',
        goal: '연구 결과를 논문 형식으로 작성',
        backstory: '과학 커뮤니케이션 전문가. 복잡한 연구를 명확하게 전달합니다.',
        tools: ['file'],
        llm: 'gpt-4-turbo',
        temperature: 0.5
      }
    ],
    tasks: [
      {
        description: '연구 주제 선정 및 연구 설계',
        expectedOutput: '연구 계획서와 가설',
        tools: ['search', 'file'],
        dependencies: []
      },
      {
        description: '기존 문헌 조사 및 선행 연구 분석',
        expectedOutput: 'Literature Review 문서',
        tools: ['search', 'file'],
        dependencies: []
      },
      {
        description: '데이터 수집 및 통계 분석',
        expectedOutput: '통계 분석 결과와 시각화',
        tools: ['code', 'file'],
        dependencies: []
      },
      {
        description: '연구 논문 작성 및 출판 준비',
        expectedOutput: '저널 제출용 논문 초안',
        tools: ['file'],
        dependencies: []
      }
    ]
  },
  {
    id: 'data-analysis',
    name: '데이터 분석팀',
    description: '비즈니스 인사이트 도출을 위한 데이터 분석팀',
    icon: '📊',
    process: 'parallel',
    agents: [
      {
        name: 'Data Engineer',
        role: '데이터 엔지니어',
        goal: '데이터 파이프라인 구축 및 관리',
        backstory: 'ETL 전문가. Airflow, Spark를 활용한 대규모 데이터 처리 경험이 풍부합니다.',
        tools: ['code', 'search', 'file'],
        llm: 'gpt-4-turbo',
        temperature: 0.3
      },
      {
        name: 'Business Analyst',
        role: '비즈니스 분석가',
        goal: '비즈니스 문제를 데이터 문제로 변환',
        backstory: 'MBA와 데이터 분석 전문성을 겸비. 비즈니스 언어와 데이터 언어를 연결합니다.',
        tools: ['search', 'file'],
        llm: 'gpt-4-turbo',
        temperature: 0.5
      },
      {
        name: 'ML Engineer',
        role: '머신러닝 엔지니어',
        goal: '예측 모델 개발 및 최적화',
        backstory: 'Kaggle 마스터. 다양한 ML 알고리즘을 실무에 적용한 경험이 많습니다.',
        tools: ['code', 'file'],
        llm: 'gpt-4-turbo',
        temperature: 0.4
      },
      {
        name: 'Data Visualization Expert',
        role: '데이터 시각화 전문가',
        goal: '인사이트를 직관적인 비주얼로 전달',
        backstory: 'D3.js, Tableau, Power BI 전문가. 데이터 스토리텔링의 달인입니다.',
        tools: ['code', 'file'],
        llm: 'gpt-3.5-turbo',
        temperature: 0.5
      }
    ],
    tasks: [
      {
        description: '데이터 소스 연결 및 ETL 파이프라인 구축',
        expectedOutput: '자동화된 데이터 파이프라인과 품질 검증 로직',
        tools: ['code'],
        dependencies: []
      },
      {
        description: '비즈니스 KPI 정의 및 분석 방향 설정',
        expectedOutput: 'KPI 대시보드 요구사항 문서',
        tools: ['file'],
        dependencies: []
      },
      {
        description: '예측 모델 개발 및 A/B 테스트',
        expectedOutput: '학습된 ML 모델과 성능 평가 리포트',
        tools: ['code', 'file'],
        dependencies: []
      },
      {
        description: '인터랙티브 대시보드 및 리포트 작성',
        expectedOutput: '경영진 리포트와 실시간 대시보드',
        tools: ['code', 'file'],
        dependencies: []
      }
    ]
  },
  {
    id: 'marketing',
    name: '마케팅팀',
    description: '통합 마케팅 캠페인을 기획하고 실행하는 팀',
    icon: '📢',
    process: 'hierarchical',
    agents: [
      {
        name: 'Marketing Director',
        role: '마케팅 디렉터',
        goal: '통합 마케팅 전략 수립 및 ROI 최대화',
        backstory: '15년 경력의 마케팅 전략가. 데이터 기반 의사결정을 중시합니다.',
        tools: ['search', 'file'],
        llm: 'gpt-4-turbo',
        temperature: 0.6
      },
      {
        name: 'Social Media Manager',
        role: 'SNS 매니저',
        goal: '소셜 미디어 채널 관리 및 커뮤니티 구축',
        backstory: 'Z세대 트렌드에 밝은 SNS 전문가. 바이럴 캠페인 경험이 풍부합니다.',
        tools: ['search', 'chat', 'file'],
        llm: 'gpt-3.5-turbo',
        temperature: 0.7
      },
      {
        name: 'Content Marketer',
        role: '콘텐츠 마케터',
        goal: '브랜드 스토리 전달 및 리드 생성',
        backstory: '스토리텔링과 SEO를 결합한 콘텐츠 마케팅 전문가입니다.',
        tools: ['search', 'file'],
        llm: 'gpt-4-turbo',
        temperature: 0.6
      },
      {
        name: 'Performance Marketer',
        role: '퍼포먼스 마케터',
        goal: 'ROI 중심의 유료 광고 캠페인 운영',
        backstory: 'Google Ads, Meta Ads 전문가. 데이터 기반 최적화에 능숙합니다.',
        tools: ['search', 'file', 'code'],
        llm: 'gpt-3.5-turbo',
        temperature: 0.4
      }
    ],
    tasks: [
      {
        description: '시장 조사 및 타겟 오디언스 분석',
        expectedOutput: '타겟 페르소나와 시장 기회 분석 리포트',
        tools: ['search', 'file'],
        dependencies: []
      },
      {
        description: 'SNS 캠페인 기획 및 콘텐츠 달력 작성',
        expectedOutput: '월간 콘텐츠 달력과 캠페인 아이디어',
        tools: ['file'],
        dependencies: []
      },
      {
        description: '블로그, 백서, 케이스 스터디 작성',
        expectedOutput: '브랜드 콘텐츠 패키지',
        tools: ['search', 'file'],
        dependencies: []
      },
      {
        description: '유료 광고 캠페인 설정 및 최적화',
        expectedOutput: '광고 캠페인 설정과 성과 리포트',
        tools: ['search', 'code'],
        dependencies: []
      }
    ]
  }
];

// Available tools
const AVAILABLE_TOOLS = [
  { id: 'search', name: 'Web Search', icon: Search },
  { id: 'code', name: 'Code Executor', icon: Code },
  { id: 'file', name: 'File Reader', icon: FileText },
  { id: 'api', name: 'API Caller', icon: Globe },
  { id: 'chat', name: 'Chat Interface', icon: MessageSquare }
];

// Available LLMs
const AVAILABLE_LLMS = [
  'gpt-4-turbo',
  'gpt-3.5-turbo',
  'claude-3-opus',
  'claude-3-sonnet',
  'gemini-pro'
];

export default function CrewAIBuilder() {
  const [crew, setCrew] = useState<CrewConfig>({
    name: 'My Custom Crew',
    agents: [],
    tasks: [],
    process: 'sequential',
    verbose: true
  });

  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [executionLog, setExecutionLog] = useState<string[]>([]);
  const [showOrgChart, setShowOrgChart] = useState(true);
  const [showMetrics, setShowMetrics] = useState(true);
  const [showCodeExport, setShowCodeExport] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Metrics
  const [metrics, setMetrics] = useState({
    tasksCompleted: 0,
    successRate: 0,
    avgTaskTime: 0,
    agentUtilization: 0
  });

  // Load template
  const loadTemplate = (templateId: string) => {
    const template = TEAM_TEMPLATES.find(t => t.id === templateId);
    if (!template) return;

    const agents: Agent[] = template.agents.map((a, idx) => ({
      ...a,
      id: `agent-${Date.now()}-${idx}`,
      status: 'idle',
      progress: 0
    }));

    const tasks: Task[] = template.tasks.map((t, idx) => ({
      ...t,
      id: `task-${Date.now()}-${idx}`,
      agent: agents[idx % agents.length].id,
      status: 'pending'
    }));

    setCrew({
      name: template.name,
      agents,
      tasks,
      process: template.process,
      verbose: true
    });

    setExecutionLog([]);
  };

  // Add new agent
  const addAgent = () => {
    const newAgent: Agent = {
      id: `agent-${Date.now()}`,
      name: `Agent ${crew.agents.length + 1}`,
      role: 'Specialist',
      goal: 'Complete assigned tasks',
      backstory: 'Experienced professional',
      tools: [],
      llm: 'gpt-3.5-turbo',
      temperature: 0.7,
      status: 'idle',
      progress: 0
    };
    setCrew({ ...crew, agents: [...crew.agents, newAgent] });
  };

  // Update agent
  const updateAgent = (agentId: string, updates: Partial<Agent>) => {
    setCrew({
      ...crew,
      agents: crew.agents.map(a =>
        a.id === agentId ? { ...a, ...updates } : a
      )
    });
  };

  // Delete agent
  const deleteAgent = (agentId: string) => {
    setCrew({
      ...crew,
      agents: crew.agents.filter(a => a.id !== agentId),
      tasks: crew.tasks.map(t => ({
        ...t,
        agent: t.agent === agentId ? '' : t.agent
      }))
    });
  };

  // Add new task
  const addTask = () => {
    const newTask: Task = {
      id: `task-${Date.now()}`,
      description: 'New task description',
      expectedOutput: 'Expected output',
      agent: crew.agents[0]?.id || '',
      tools: [],
      dependencies: [],
      status: 'pending'
    };
    setCrew({ ...crew, tasks: [...crew.tasks, newTask] });
  };

  // Update task
  const updateTask = (taskId: string, updates: Partial<Task>) => {
    setCrew({
      ...crew,
      tasks: crew.tasks.map(t =>
        t.id === taskId ? { ...t, ...updates } : t
      )
    });
  };

  // Delete task
  const deleteTask = (taskId: string) => {
    setCrew({
      ...crew,
      tasks: crew.tasks.filter(t => t.id !== taskId)
    });
  };

  // Draw org chart
  useEffect(() => {
    if (!showOrgChart || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    canvas.width = canvas.offsetWidth;
    canvas.height = 400;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Calculate positions
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = Math.min(canvas.width, canvas.height) * 0.35;

    const agents = crew.agents.map((agent, index) => {
      const angle = (index / crew.agents.length) * 2 * Math.PI - Math.PI / 2;
      const x = centerX + radius * Math.cos(angle);
      const y = centerY + radius * Math.sin(angle);
      return { ...agent, x, y };
    });

    // Draw connections between agents and tasks
    crew.tasks.forEach(task => {
      const agent = agents.find(a => a.id === task.agent);
      if (agent && agent.x && agent.y) {
        // Draw task execution line
        if (task.status === 'running' || task.status === 'completed') {
          ctx.strokeStyle = task.status === 'completed' ? '#10b981' : '#f59e0b';
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.arc(agent.x, agent.y, 35, 0, 2 * Math.PI);
          ctx.stroke();
        }
      }
    });

    // Draw agent nodes
    agents.forEach(agent => {
      if (!agent.x || !agent.y) return;

      ctx.beginPath();
      ctx.arc(agent.x, agent.y, 30, 0, 2 * Math.PI);

      // Color based on status
      let fillColor = '#e5e7eb';
      if (agent.status === 'thinking') fillColor = '#60a5fa';
      if (agent.status === 'working') fillColor = '#f59e0b';
      if (agent.status === 'done') fillColor = '#10b981';

      ctx.fillStyle = fillColor;
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 3;
      ctx.stroke();

      // Agent initial
      ctx.fillStyle = '#1f2937';
      ctx.font = 'bold 14px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(agent.name.charAt(0).toUpperCase(), agent.x, agent.y);

      // Agent name below
      ctx.fillStyle = '#374151';
      ctx.font = '11px sans-serif';
      ctx.fillText(agent.name, agent.x, agent.y + 45);

      // Progress indicator
      if (agent.progress && agent.progress > 0) {
        ctx.strokeStyle = '#f97316';
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.arc(agent.x, agent.y, 35, -Math.PI / 2, -Math.PI / 2 + (agent.progress / 100) * 2 * Math.PI);
        ctx.stroke();
      }
    });

  }, [crew.agents, crew.tasks, showOrgChart]);

  // Run crew simulation
  const runCrew = async () => {
    setIsRunning(true);
    setIsPaused(false);
    setExecutionLog([]);

    const log = (message: string) => {
      setExecutionLog(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${message}`]);
    };

    log('🚀 Starting CrewAI execution...');
    log(`📋 Crew: ${crew.name}`);
    log(`⚙️  Process: ${crew.process}`);
    log(`👥 Agents: ${crew.agents.length}`);
    log(`📝 Tasks: ${crew.tasks.length}\n`);

    const startTime = Date.now();
    let completedTasks = 0;
    let totalTaskTime = 0;

    // Reset all statuses
    setCrew(prev => ({
      ...prev,
      agents: prev.agents.map(a => ({ ...a, status: 'idle' as const, progress: 0 })),
      tasks: prev.tasks.map(t => ({ ...t, status: 'pending' as const }))
    }));

    // Execute tasks based on process type
    if (crew.process === 'sequential') {
      for (let i = 0; i < crew.tasks.length; i++) {
        if (isPaused) {
          await new Promise(resolve => {
            const interval = setInterval(() => {
              if (!isPaused) {
                clearInterval(interval);
                resolve(null);
              }
            }, 100);
          });
        }

        const task = crew.tasks[i];
        const agent = crew.agents.find(a => a.id === task.agent);
        if (!agent) continue;

        const taskStartTime = Date.now();

        log(`\n━━━━ Task ${i + 1}/${crew.tasks.length} ━━━━`);
        log(`📋 ${task.description}`);
        log(`👤 Assigned to: ${agent.name} (${agent.role})`);

        // Update task status
        setCrew(prev => ({
          ...prev,
          tasks: prev.tasks.map(t => t.id === task.id ? { ...t, status: 'running' as const } : t),
          agents: prev.agents.map(a => a.id === agent.id ? { ...a, status: 'thinking' as const, progress: 0 } : a)
        }));

        await new Promise(resolve => setTimeout(resolve, 800));
        log(`🤔 ${agent.name} is analyzing the task...`);

        // Working phase
        for (let progress = 0; progress <= 100; progress += 20) {
          setCrew(prev => ({
            ...prev,
            agents: prev.agents.map(a =>
              a.id === agent.id ? { ...a, status: 'working' as const, progress } : a
            )
          }));
          await new Promise(resolve => setTimeout(resolve, 300));
        }

        if (task.tools.length > 0) {
          const toolNames = task.tools.map(id => AVAILABLE_TOOLS.find(t => t.id === id)?.name || id);
          log(`🔧 Using tools: ${toolNames.join(', ')}`);
          await new Promise(resolve => setTimeout(resolve, 1000));
        }

        log(`✅ Task completed: ${task.expectedOutput}`);

        // Update to done
        setCrew(prev => ({
          ...prev,
          tasks: prev.tasks.map(t => t.id === task.id ? { ...t, status: 'completed' as const, output: task.expectedOutput } : t),
          agents: prev.agents.map(a => a.id === agent.id ? { ...a, status: 'done' as const, progress: 100 } : a)
        }));

        const taskTime = Date.now() - taskStartTime;
        totalTaskTime += taskTime;
        completedTasks++;
      }
    } else if (crew.process === 'parallel') {
      // Parallel execution
      const taskPromises = crew.tasks.map(async (task, i) => {
        const agent = crew.agents.find(a => a.id === task.agent);
        if (!agent) return;

        const taskStartTime = Date.now();

        log(`\n📋 Task ${i + 1}: ${task.description}`);
        log(`👤 ${agent.name} starting work...`);

        setCrew(prev => ({
          ...prev,
          tasks: prev.tasks.map(t => t.id === task.id ? { ...t, status: 'running' as const } : t),
          agents: prev.agents.map(a => a.id === agent.id ? { ...a, status: 'working' as const, progress: 50 } : a)
        }));

        await new Promise(resolve => setTimeout(resolve, 2000 + Math.random() * 2000));

        log(`✅ ${agent.name} completed: ${task.expectedOutput}`);

        setCrew(prev => ({
          ...prev,
          tasks: prev.tasks.map(t => t.id === task.id ? { ...t, status: 'completed' as const, output: task.expectedOutput } : t),
          agents: prev.agents.map(a => a.id === agent.id ? { ...a, status: 'done' as const, progress: 100 } : a)
        }));

        const taskTime = Date.now() - taskStartTime;
        totalTaskTime += taskTime;
        completedTasks++;
      });

      await Promise.all(taskPromises);
    }

    const totalTime = Date.now() - startTime;
    const avgTime = completedTasks > 0 ? Math.round(totalTaskTime / completedTasks) : 0;
    const utilization = Math.round((completedTasks / crew.agents.length) * 100);

    setMetrics({
      tasksCompleted: completedTasks,
      successRate: 100,
      avgTaskTime: avgTime,
      agentUtilization: utilization
    });

    log(`\n✨ All tasks completed successfully!`);
    log(`⏱️  Total time: ${(totalTime / 1000).toFixed(2)}s`);
    log(`📊 Average task time: ${(avgTime / 1000).toFixed(2)}s`);
    setIsRunning(false);
  };

  // Generate Python code
  const generateCrewAICode = () => {
    let code = `from crewai import Agent, Task, Crew, Process\n\n`;
    code += `# Agents\n`;

    crew.agents.forEach(agent => {
      code += `${agent.name.toLowerCase().replace(/\s+/g, '_')} = Agent(\n`;
      code += `    role="${agent.role}",\n`;
      code += `    goal="${agent.goal}",\n`;
      code += `    backstory="${agent.backstory}",\n`;
      code += `    tools=[${agent.tools.map(t => `"${t}"`).join(', ')}],\n`;
      code += `    llm="${agent.llm}",\n`;
      code += `    temperature=${agent.temperature}\n`;
      code += `)\n\n`;
    });

    code += `# Tasks\n`;
    crew.tasks.forEach((task, idx) => {
      const agent = crew.agents.find(a => a.id === task.agent);
      code += `task_${idx + 1} = Task(\n`;
      code += `    description="${task.description}",\n`;
      code += `    expected_output="${task.expectedOutput}",\n`;
      code += `    agent=${agent?.name.toLowerCase().replace(/\s+/g, '_')}\n`;
      code += `)\n\n`;
    });

    code += `# Crew\n`;
    code += `crew = Crew(\n`;
    code += `    agents=[${crew.agents.map(a => a.name.toLowerCase().replace(/\s+/g, '_')).join(', ')}],\n`;
    code += `    tasks=[${crew.tasks.map((_, idx) => `task_${idx + 1}`).join(', ')}],\n`;
    code += `    process=Process.${crew.process.toUpperCase()},\n`;
    code += `    verbose=${crew.verbose ? 'True' : 'False'}\n`;
    code += `)\n\n`;
    code += `# Execute\n`;
    code += `result = crew.kickoff()\n`;
    code += `print(result)\n`;

    return code;
  };

  const copyCode = () => {
    const code = generateCrewAICode();
    navigator.clipboard.writeText(code);
  };

  const downloadCode = () => {
    const code = generateCrewAICode();
    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${crew.name.toLowerCase().replace(/\s+/g, '_')}_crew.py`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="bg-gray-50 dark:bg-gray-900 rounded-xl p-6">
      <div className="mb-6">
        <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
          🤖 CrewAI Team Builder
        </h3>
        <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
          역할 기반 AI 에이전트 팀을 구성하고 복잡한 작업을 자동화하세요
        </p>

        {/* Team Templates */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3 flex items-center gap-2">
            <Zap className="w-4 h-4 text-orange-600" />
            템플릿 시작하기
          </h4>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
            {TEAM_TEMPLATES.map(template => (
              <button
                key={template.id}
                onClick={() => loadTemplate(template.id)}
                className="p-3 bg-gradient-to-br from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 hover:from-orange-100 hover:to-red-100 dark:hover:from-orange-900/30 dark:hover:to-red-900/30 rounded-lg transition-all text-left"
              >
                <div className="text-2xl mb-1">{template.icon}</div>
                <div className="font-medium text-sm text-gray-900 dark:text-white">{template.name}</div>
                <div className="text-xs text-gray-600 dark:text-gray-400">{template.description}</div>
                <div className="text-xs text-orange-600 dark:text-orange-400 mt-1">
                  {template.agents.length} agents · {template.tasks.length} tasks
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Toggle Controls */}
        <div className="flex flex-wrap gap-2 mb-4">
          <button
            onClick={() => setShowOrgChart(!showOrgChart)}
            className={`px-3 py-1.5 rounded-lg transition-colors text-sm ${
              showOrgChart
                ? 'bg-orange-600 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
            }`}
          >
            <Network className="w-4 h-4 inline mr-1" />
            팀 구조도
          </button>
          <button
            onClick={() => setShowMetrics(!showMetrics)}
            className={`px-3 py-1.5 rounded-lg transition-colors text-sm ${
              showMetrics
                ? 'bg-orange-600 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
            }`}
          >
            <BarChart3 className="w-4 h-4 inline mr-1" />
            성과 지표
          </button>
          <button
            onClick={() => setShowCodeExport(!showCodeExport)}
            className={`px-3 py-1.5 rounded-lg transition-colors text-sm ${
              showCodeExport
                ? 'bg-orange-600 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
            }`}
          >
            <Code className="w-4 h-4 inline mr-1" />
            코드 생성
          </button>
        </div>
      </div>

      {/* Metrics Dashboard */}
      {showMetrics && metrics.tasksCompleted > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
          <div className="bg-gradient-to-br from-blue-500 to-blue-600 text-white rounded-lg p-4">
            <div className="text-sm opacity-90 mb-1">완료 작업</div>
            <div className="text-3xl font-bold">{metrics.tasksCompleted}</div>
          </div>
          <div className="bg-gradient-to-br from-green-500 to-green-600 text-white rounded-lg p-4">
            <div className="text-sm opacity-90 mb-1">성공률</div>
            <div className="text-3xl font-bold">{metrics.successRate}%</div>
          </div>
          <div className="bg-gradient-to-br from-purple-500 to-purple-600 text-white rounded-lg p-4">
            <div className="text-sm opacity-90 mb-1">평균 작업시간</div>
            <div className="text-3xl font-bold">{(metrics.avgTaskTime / 1000).toFixed(1)}s</div>
          </div>
          <div className="bg-gradient-to-br from-orange-500 to-orange-600 text-white rounded-lg p-4">
            <div className="text-sm opacity-90 mb-1">에이전트 활용률</div>
            <div className="text-3xl font-bold">{metrics.agentUtilization}%</div>
          </div>
        </div>
      )}

      {/* Org Chart Canvas */}
      {showOrgChart && crew.agents.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-6">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
            팀 조직도
          </h4>
          <canvas
            ref={canvasRef}
            className="w-full rounded-lg bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800"
          />
        </div>
      )}

      {/* Code Export */}
      {showCodeExport && crew.agents.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-6">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
              CrewAI Python 코드
            </h4>
            <div className="flex gap-2">
              <button
                onClick={copyCode}
                className="px-3 py-1 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 rounded transition-colors text-xs flex items-center gap-1"
              >
                <Copy className="w-3 h-3" />
                복사
              </button>
              <button
                onClick={downloadCode}
                className="px-3 py-1 bg-orange-600 hover:bg-orange-700 text-white rounded transition-colors text-xs flex items-center gap-1"
              >
                <Download className="w-3 h-3" />
                다운로드
              </button>
            </div>
          </div>
          <pre className="bg-gray-900 text-green-400 p-4 rounded-lg overflow-x-auto text-xs font-mono max-h-96">
            {generateCrewAICode()}
          </pre>
        </div>
      )}

      <div className="grid grid-cols-12 gap-4">
        {/* Agents Panel */}
        <div className="col-span-12 md:col-span-5 space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
                Agents ({crew.agents.length})
              </h4>
              <button
                onClick={addAgent}
                className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
              >
                <Plus className="w-4 h-4 text-orange-600 dark:text-orange-400" />
              </button>
            </div>

            <div className="space-y-2 max-h-96 overflow-y-auto">
              {crew.agents.length === 0 ? (
                <div className="text-center py-8">
                  <Users className="w-12 h-12 text-gray-300 dark:text-gray-600 mx-auto mb-2" />
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    위의 템플릿을 선택하거나<br />+ 버튼으로 에이전트를 추가하세요
                  </p>
                </div>
              ) : (
                crew.agents.map(agent => (
                  <div
                    key={agent.id}
                    onClick={() => setSelectedAgent(agent.id === selectedAgent ? null : agent.id)}
                    className={`p-3 rounded-lg cursor-pointer transition-all ${
                      selectedAgent === agent.id
                        ? 'bg-orange-50 dark:bg-orange-900/30 border-2 border-orange-500'
                        : 'bg-gray-50 dark:bg-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600'
                    }`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <Users className="w-4 h-4 text-orange-600 dark:text-orange-400" />
                          <input
                            type="text"
                            value={agent.name}
                            onChange={(e) => updateAgent(agent.id, { name: e.target.value })}
                            className="text-sm font-medium bg-transparent border-none outline-none flex-1 text-gray-900 dark:text-white"
                            onClick={(e) => e.stopPropagation()}
                          />
                        </div>
                        <input
                          type="text"
                          value={agent.role}
                          onChange={(e) => updateAgent(agent.id, { role: e.target.value })}
                          className="text-xs text-gray-600 dark:text-gray-400 bg-transparent border-none outline-none w-full mt-1"
                          placeholder="Role"
                          onClick={(e) => e.stopPropagation()}
                        />
                        {agent.status && agent.status !== 'idle' && (
                          <div className="mt-2">
                            <div className="flex items-center gap-2 text-xs">
                              <div className={`w-2 h-2 rounded-full ${
                                agent.status === 'thinking' ? 'bg-blue-500 animate-pulse' :
                                agent.status === 'working' ? 'bg-orange-500 animate-pulse' :
                                'bg-green-500'
                              }`}></div>
                              <span className="text-gray-600 dark:text-gray-400">
                                {agent.status === 'thinking' && '생각 중...'}
                                {agent.status === 'working' && '작업 중...'}
                                {agent.status === 'done' && '완료'}
                              </span>
                            </div>
                            {agent.progress !== undefined && agent.progress > 0 && (
                              <div className="mt-1 bg-gray-200 dark:bg-gray-600 rounded-full h-1.5 overflow-hidden">
                                <div
                                  className="bg-orange-500 h-full transition-all duration-300"
                                  style={{ width: `${agent.progress}%` }}
                                ></div>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteAgent(agent.id);
                        }}
                        className="p-1 hover:bg-red-100 dark:hover:bg-red-900/30 rounded"
                      >
                        <Trash2 className="w-3 h-3 text-red-500" />
                      </button>
                    </div>

                    {selectedAgent === agent.id && (
                      <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-600 space-y-2">
                        <div>
                          <label className="text-xs text-gray-600 dark:text-gray-400">Goal</label>
                          <textarea
                            value={agent.goal}
                            onChange={(e) => updateAgent(agent.id, { goal: e.target.value })}
                            className="w-full text-xs bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded p-2 text-gray-900 dark:text-white"
                            rows={2}
                          />
                        </div>
                        <div>
                          <label className="text-xs text-gray-600 dark:text-gray-400">Backstory</label>
                          <textarea
                            value={agent.backstory}
                            onChange={(e) => updateAgent(agent.id, { backstory: e.target.value })}
                            className="w-full text-xs bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded p-2 text-gray-900 dark:text-white"
                            rows={2}
                          />
                        </div>
                        <div>
                          <label className="text-xs text-gray-600 dark:text-gray-400">LLM Model</label>
                          <select
                            value={agent.llm}
                            onChange={(e) => updateAgent(agent.id, { llm: e.target.value })}
                            className="w-full text-xs bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded p-2 text-gray-900 dark:text-white"
                          >
                            {AVAILABLE_LLMS.map(llm => (
                              <option key={llm} value={llm}>{llm}</option>
                            ))}
                          </select>
                        </div>
                        <div>
                          <label className="text-xs text-gray-600 dark:text-gray-400">Temperature ({agent.temperature})</label>
                          <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.1"
                            value={agent.temperature}
                            onChange={(e) => updateAgent(agent.id, { temperature: parseFloat(e.target.value) })}
                            className="w-full"
                          />
                        </div>
                        <div>
                          <label className="text-xs text-gray-600 dark:text-gray-400">Tools</label>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {AVAILABLE_TOOLS.map(tool => {
                              const isSelected = agent.tools.includes(tool.id);
                              return (
                                <button
                                  key={tool.id}
                                  onClick={() => {
                                    const newTools = isSelected
                                      ? agent.tools.filter(t => t !== tool.id)
                                      : [...agent.tools, tool.id];
                                    updateAgent(agent.id, { tools: newTools });
                                  }}
                                  className={`text-xs px-2 py-1 rounded transition-colors ${
                                    isSelected
                                      ? 'bg-orange-600 text-white'
                                      : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                                  }`}
                                >
                                  {tool.name}
                                </button>
                              );
                            })}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Process Settings */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
              <Settings className="w-4 h-4 inline mr-1" />
              Process Type
            </h4>
            <div className="space-y-2">
              {(['sequential', 'hierarchical', 'parallel'] as const).map(process => (
                <button
                  key={process}
                  onClick={() => setCrew({ ...crew, process })}
                  className={`w-full text-left p-3 rounded-lg transition-colors ${
                    crew.process === process
                      ? 'bg-gradient-to-r from-orange-500 to-red-500 text-white'
                      : 'bg-gray-50 dark:bg-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 text-gray-900 dark:text-white'
                  }`}
                >
                  <div className="font-medium text-sm capitalize">{process}</div>
                  <div className={`text-xs ${crew.process === process ? 'text-orange-100' : 'text-gray-600 dark:text-gray-400'}`}>
                    {process === 'sequential' && '작업을 순차적으로 실행'}
                    {process === 'hierarchical' && '관리자가 작업을 위임'}
                    {process === 'parallel' && '작업을 동시 병렬 실행'}
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Tasks Panel */}
        <div className="col-span-12 md:col-span-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 h-full">
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
                Tasks ({crew.tasks.length})
              </h4>
              <button
                onClick={addTask}
                disabled={crew.agents.length === 0}
                className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Plus className="w-4 h-4 text-orange-600 dark:text-orange-400" />
              </button>
            </div>

            <div className="space-y-2 max-h-96 overflow-y-auto mb-4">
              {crew.tasks.length === 0 ? (
                <div className="text-center py-8">
                  <Target className="w-12 h-12 text-gray-300 dark:text-gray-600 mx-auto mb-2" />
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    {crew.agents.length === 0
                      ? '먼저 에이전트를 추가하세요'
                      : '+ 버튼으로 작업을 추가하세요'}
                  </p>
                </div>
              ) : (
                crew.tasks.map((task, index) => {
                  const agent = crew.agents.find(a => a.id === task.agent);
                  return (
                    <div key={task.id} className={`p-3 rounded-lg ${
                      task.status === 'pending' ? 'bg-gray-50 dark:bg-gray-700' :
                      task.status === 'running' ? 'bg-orange-50 dark:bg-orange-900/20 border-2 border-orange-500' :
                      'bg-green-50 dark:bg-green-900/20 border-2 border-green-500'
                    }`}>
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-bold text-gray-500">#{index + 1}</span>
                          {task.status === 'completed' && <CheckCircle className="w-4 h-4 text-green-600" />}
                          {task.status === 'running' && <RefreshCw className="w-4 h-4 text-orange-600 animate-spin" />}
                          {task.status === 'pending' && <Target className="w-4 h-4 text-gray-400" />}
                        </div>
                        <button
                          onClick={() => deleteTask(task.id)}
                          className="p-1 hover:bg-red-100 dark:hover:bg-red-900/30 rounded"
                        >
                          <Trash2 className="w-3 h-3 text-red-500" />
                        </button>
                      </div>

                      <textarea
                        value={task.description}
                        onChange={(e) => updateTask(task.id, { description: e.target.value })}
                        className="w-full text-xs bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded p-2 mb-2 text-gray-900 dark:text-white"
                        placeholder="Task description"
                        rows={2}
                      />

                      <textarea
                        value={task.expectedOutput}
                        onChange={(e) => updateTask(task.id, { expectedOutput: e.target.value })}
                        className="w-full text-xs bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded p-2 mb-2 text-gray-900 dark:text-white"
                        placeholder="Expected output"
                        rows={2}
                      />

                      <select
                        value={task.agent}
                        onChange={(e) => updateTask(task.id, { agent: e.target.value })}
                        className="w-full text-xs bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded p-2 mb-2 text-gray-900 dark:text-white"
                      >
                        <option value="">Select agent</option>
                        {crew.agents.map(a => (
                          <option key={a.id} value={a.id}>{a.name} ({a.role})</option>
                        ))}
                      </select>

                      {agent && (
                        <div className="text-xs text-gray-600 dark:text-gray-400">
                          👤 {agent.name}
                        </div>
                      )}
                    </div>
                  );
                })
              )}
            </div>

            {/* Control Buttons */}
            <div className="space-y-2">
              <button
                onClick={runCrew}
                disabled={isRunning || crew.agents.length === 0 || crew.tasks.length === 0}
                className="w-full px-4 py-3 bg-gradient-to-r from-orange-600 to-red-600 text-white rounded-lg hover:from-orange-700 hover:to-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2 font-semibold"
              >
                {isRunning ? (
                  <>
                    <RefreshCw className="w-5 h-5 animate-spin" />
                    실행 중...
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5" />
                    Crew 실행
                  </>
                )}
              </button>

              {isRunning && (
                <button
                  onClick={() => setIsPaused(!isPaused)}
                  className="w-full px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors flex items-center justify-center gap-2"
                >
                  <Pause className="w-4 h-4" />
                  {isPaused ? '재개' : '일시정지'}
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Execution Log */}
        <div className="col-span-12 md:col-span-3">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 h-full">
            <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
              <FileText className="w-4 h-4 inline mr-1" />
              Execution Log
            </h4>
            <div className="h-96 overflow-y-auto space-y-1 bg-gray-900 rounded-lg p-3">
              {executionLog.length === 0 ? (
                <p className="text-xs text-gray-500">
                  Crew를 실행하면 로그가 여기에 표시됩니다
                </p>
              ) : (
                executionLog.map((log, idx) => (
                  <p key={idx} className="text-xs text-green-400 font-mono">
                    {log}
                  </p>
                ))
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Quick Guide */}
      <div className="mt-6 bg-gradient-to-r from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <Briefcase className="w-5 h-5 text-orange-600 dark:text-orange-400 mt-0.5 flex-shrink-0" />
          <div className="text-sm text-orange-800 dark:text-orange-200 space-y-1">
            <p className="font-semibold mb-2">🚀 빠른 시작 가이드:</p>
            <p>1️⃣ 위의 템플릿 중 하나를 선택하거나 직접 에이전트를 추가하세요</p>
            <p>2️⃣ 각 에이전트의 역할, 목표, 도구를 설정하세요</p>
            <p>3️⃣ 작업을 생성하고 적절한 에이전트에게 할당하세요</p>
            <p>4️⃣ Process Type을 선택하세요 (Sequential, Hierarchical, Parallel)</p>
            <p>5️⃣ "Crew 실행" 버튼을 클릭하여 시뮬레이션을 시작하세요</p>
            <p>6️⃣ "코드 생성"을 클릭하여 실제 CrewAI Python 코드를 다운로드하세요</p>
          </div>
        </div>
      </div>
    </div>
  );
}
