export const moduleMetadata = {
  id: 'content-manager',
  title: 'Content Manager',
  description: 'AI 기반 콘텐츠 검증 및 자동 업데이트 시스템',
  icon: '🔄',
  gradient: 'from-green-500 to-emerald-600',
  category: 'System Management',
  difficulty: 'Advanced',
  estimatedHours: 0, // Management tool, not learning content
  isTool: true, // Mark as system tool
  chapters: [], // No chapters - this is a dashboard
  simulators: [], // Content Manager dashboard itself
  features: [
    'Real-time module health monitoring',
    'Automated content validation',
    'AI-powered update suggestions',
    'Broken link detection',
    'Deprecated code scanning',
    'Source attribution tracking',
  ],
  capabilities: {
    validation: [
      'Outdated dates and statistics',
      'Broken external links',
      'Deprecated APIs and code',
      'Missing simulators',
    ],
    automation: [
      'Daily news scanning',
      'Research paper monitoring',
      'Automatic relevance detection',
      'Confidence scoring',
    ],
    management: [
      'Module-specific strategies',
      'Update approval workflow',
      'Rollback capability',
      'Version tracking',
    ],
  },
  updateStrategies: {
    'stock-analysis': { frequency: 'daily', sources: ['Bloomberg', 'Reuters', 'Yahoo Finance'] },
    'llm': { frequency: 'weekly', sources: ['ArXiv', 'Hugging Face', 'OpenAI'] },
    'medical-ai': { frequency: 'weekly', sources: ['PubMed', 'Nature Medicine', 'FDA'] },
    'system-design': { frequency: 'monthly', sources: ['High Scalability', 'AWS Blog'] },
  },
}
