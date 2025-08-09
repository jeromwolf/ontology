export class ModuleRegistry {
  private modules = [
    {
      id: 'ontology',
      name: '온톨로지',
      version: '1.0.0',
      chapters: 16,
      simulators: 4
    },
    {
      id: 'llm',
      name: 'LLM 이해와 활용',
      version: '1.0.0',
      chapters: 8,
      simulators: 6
    },
    {
      id: 'rag',
      name: 'RAG 시스템',
      version: '1.0.0',
      chapters: 6,
      simulators: 5
    },
    {
      id: 'stock-analysis',
      name: '주식분석',
      version: '1.0.0',
      chapters: 10,
      simulators: 8
    },
    {
      id: 'autonomous-mobility',
      name: '자율주행',
      version: '1.0.0',
      chapters: 8,
      simulators: 5
    },
    {
      id: 'medical-ai',
      name: '의료 AI',
      version: '1.0.0',
      chapters: 8,
      simulators: 6
    },
    {
      id: 'system-design',
      name: '시스템 설계',
      version: '1.0.0',
      chapters: 8,
      simulators: 6
    },
    {
      id: 'neo4j',
      name: 'Neo4j 그래프 DB',
      version: '1.0.0',
      chapters: 8,
      simulators: 5
    },
    {
      id: 'quantum-computing',
      name: '양자 컴퓨팅',
      version: '1.0.0',
      chapters: 8,
      simulators: 4
    },
    {
      id: 'bioinformatics',
      name: '바이오인포매틱스',
      version: '1.0.0',
      chapters: 8,
      simulators: 4
    },
    {
      id: 'web3',
      name: 'Web3 & 블록체인',
      version: '1.0.0',
      chapters: 8,
      simulators: 5
    },
    {
      id: 'smart-factory',
      name: '스마트 팩토리',
      version: '1.0.0',
      chapters: 8,
      simulators: 4
    },
    {
      id: 'physical-ai',
      name: 'Physical AI',
      version: '1.0.0',
      chapters: 8,
      simulators: 4
    },
    {
      id: 'ai-automation',
      name: 'AI 자동화',
      version: '1.0.0',
      chapters: 8,
      simulators: 4
    },
    {
      id: 'agent-mcp',
      name: 'Agent & MCP',
      version: '1.0.0',
      chapters: 8,
      simulators: 4
    },
    {
      id: 'multi-agent',
      name: '멀티 에이전트',
      version: '1.0.0',
      chapters: 8,
      simulators: 4
    },
    {
      id: 'english-conversation',
      name: '영어 회화',
      version: '1.0.0',
      chapters: 8,
      simulators: 5
    },
    {
      id: 'linear-algebra',
      name: '선형대수학',
      version: '1.0.0',
      chapters: 8,
      simulators: 4
    }
  ]

  async getAllModuleStatuses() {
    return this.modules.map(module => ({
      ...module,
      lastUpdate: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000), // Random date within 30 days
      accuracyScore: 70 + Math.random() * 30, // 70-100
      outdatedChapters: Math.floor(Math.random() * 3),
      brokenLinks: Math.floor(Math.random() * 2),
      deprecatedCode: Math.floor(Math.random() * 2),
      simulatorHealth: Math.random() > 0.7 ? 'healthy' : Math.random() > 0.3 ? 'warning' : 'critical',
      updateFrequency: module.id === 'stock-analysis' ? 'daily' : 
                       ['llm', 'autonomous-mobility', 'medical-ai'].includes(module.id) ? 'weekly' : 'monthly',
      nextUpdate: new Date(Date.now() + Math.random() * 7 * 24 * 60 * 60 * 1000) // Random date within 7 days
    }))
  }

  async getModuleById(id: string) {
    const module = this.modules.find(m => m.id === id)
    if (!module) return null
    
    return {
      ...module,
      lastUpdate: new Date(),
      accuracyScore: 85,
      outdatedChapters: 1,
      brokenLinks: 0,
      deprecatedCode: 1,
      simulatorHealth: 'healthy' as const,
      updateFrequency: 'weekly' as const,
      nextUpdate: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000)
    }
  }
  
  async refreshModuleStatus(id: string) {
    // Mock refresh - in production, actually update the status
    return this.getModuleById(id)
  }
  
  async updateValidationResults(issues: any[]) {
    // Mock update - in production, update database
    console.log(`Updated validation results with ${issues.length} issues`)
    return true
  }
}