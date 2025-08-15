'use client';

import { useState, useEffect } from 'react';
import { Volume2, Pause, MessageCircle, Users, Globe, Copy, CheckCircle, Play } from 'lucide-react';

export default function Chapter3() {
  const [copiedCode, setCopiedCode] = useState<string | null>(null)

  const copyCode = (code: string, id: string) => {
    navigator.clipboard.writeText(code)
    setCopiedCode(id)
    setTimeout(() => setCopiedCode(null), 2000)
  }

  const [activeTab, setActiveTab] = useState('meetings')
  const [expandedSection, setExpandedSection] = useState<string | null>(null)

  const businessTopics = [
    { id: 'meetings', name: '회의', icon: '📊' },
    { id: 'presentations', name: '프레젠테이션', icon: '📈' },
    { id: 'emails', name: '이메일', icon: '✉️' },
    { id: 'negotiations', name: '협상', icon: '🤝' },
    { id: 'networking', name: '네트워킹', icon: '🌐' },
    { id: 'phone-calls', name: '전화 통화', icon: '📞' }
  ]

  const meetingExpressions = {
    opening: [
      { expression: "Let's call this meeting to order.", korean: "회의를 시작하겠습니다." },
      { expression: "Thank you all for coming today.", korean: "오늘 참석해 주셔서 감사합니다." },
      { expression: "I'd like to welcome everyone to today's meeting.", korean: "오늘 회의에 참석하신 모든 분들을 환영합니다." },
      { expression: "Let's go around the table and introduce ourselves.", korean: "돌아가면서 자기소개를 해보겠습니다." },
      { expression: "The purpose of today's meeting is to discuss...", korean: "오늘 회의의 목적은 ...에 대해 논의하는 것입니다." }
    ],
    agenda: [
      { expression: "Let's review the agenda.", korean: "의제를 검토해보겠습니다." },
      { expression: "We have three main items on the agenda today.", korean: "오늘 의제에는 세 가지 주요 안건이 있습니다." },
      { expression: "Let's move on to the next item.", korean: "다음 안건으로 넘어가겠습니다." },
      { expression: "Are there any questions about this agenda item?", korean: "이 안건에 대해 질문이 있으신가요?" },
      { expression: "Let's table this discussion for now.", korean: "이 논의는 잠시 보류하겠습니다." }
    ],
    opinions: [
      { expression: "I think we should consider all options.", korean: "모든 선택지를 고려해야 한다고 생각합니다." },
      { expression: "From my perspective, this is the best approach.", korean: "제 관점에서는 이것이 최선의 접근법입니다." },
      { expression: "I'd like to suggest an alternative solution.", korean: "대안을 제시하고 싶습니다." },
      { expression: "I have some concerns about this proposal.", korean: "이 제안에 대해 우려되는 점이 있습니다." },
      { expression: "I couldn't agree more with that point.", korean: "그 점에 전적으로 동의합니다." }
    ],
    disagreeing: [
      { expression: "I respectfully disagree with that assessment.", korean: "그 평가에 정중히 반대합니다." },
      { expression: "I see your point, but I have a different view.", korean: "당신의 요점은 이해하지만, 저는 다른 견해를 가지고 있습니다." },
      { expression: "That's an interesting perspective, however...", korean: "흥미로운 관점이지만, 그러나..." },
      { expression: "I'm not entirely convinced by that argument.", korean: "그 논거에 완전히 설득되지는 않습니다." },
      { expression: "May I offer a counterpoint?", korean: "반박 의견을 제시해도 될까요?" }
    ],
    closing: [
      { expression: "Let's wrap up today's meeting.", korean: "오늘 회의를 마무리하겠습니다." },
      { expression: "To summarize what we've discussed...", korean: "우리가 논의한 내용을 요약하면..." },
      { expression: "What are our next steps?", korean: "다음 단계는 무엇입니까?" },
      { expression: "Who will be responsible for this action item?", korean: "이 실행 항목을 누가 담당할 것입니까?" },
      { expression: "Thank you for your time and participation.", korean: "시간을 내어 참여해 주셔서 감사합니다." }
    ]
  }

  const presentationStructure = [
    {
      section: "Opening",
      expressions: [
        { expression: "Good morning, everyone. Thank you for being here.", korean: "안녕하세요, 여러분. 참석해 주셔서 감사합니다." },
        { expression: "Today I'm going to talk about...", korean: "오늘 저는 ...에 대해 말씀드리겠습니다." },
        { expression: "My presentation will take approximately 20 minutes.", korean: "제 발표는 약 20분 정도 소요될 예정입니다." },
        { expression: "Please feel free to interrupt if you have any questions.", korean: "질문이 있으시면 언제든지 말씀해 주세요." }
      ]
    },
    {
      section: "Main Content",
      expressions: [
        { expression: "Let me start by giving you some background information.", korean: "배경 정보부터 말씀드리겠습니다." },
        { expression: "This slide shows our quarterly results.", korean: "이 슬라이드는 분기별 결과를 보여줍니다." },
        { expression: "As you can see from this chart...", korean: "이 차트에서 보시는 바와 같이..." },
        { expression: "Moving on to the next point...", korean: "다음 요점으로 넘어가서..." },
        { expression: "This brings me to my next slide.", korean: "이제 다음 슬라이드로 넘어가겠습니다." }
      ]
    },
    {
      section: "Closing",
      expressions: [
        { expression: "To sum up, our main points are...", korean: "요약하면, 주요 요점들은..." },
        { expression: "In conclusion, I'd like to emphasize...", korean: "결론적으로, 강조하고 싶은 것은..." },
        { expression: "Thank you for your attention.", korean: "경청해 주셔서 감사합니다." },
        { expression: "Are there any questions?", korean: "질문이 있으신가요?" }
      ]
    }
  ]

  const emailTemplates = [
    {
      type: "Meeting Request",
      subject: "Meeting Request - Q4 Budget Review",
      body: `Dear [Name],

I hope this email finds you well.

I would like to schedule a meeting to discuss our Q4 budget review. The meeting would cover:
• Budget allocations for each department
• Cost optimization opportunities
• Planning for Q1 next year

Would you be available next Tuesday, October 15th, at 2:00 PM? The meeting is expected to last about 60 minutes and will be held in Conference Room A.

Please let me know if this time works for you, or suggest an alternative that fits your schedule.

Best regards,
[Your Name]`,
      korean: "회의 요청 - 4분기 예산 검토"
    },
    {
      type: "Follow-up Email",
      subject: "Follow-up: Action Items from Today's Meeting",
      body: `Dear Team,

Thank you for your participation in today's project review meeting.

As discussed, here are the key action items and deadlines:

1. Market research report - Due: October 20th (John)
2. Technical specifications - Due: October 22nd (Sarah)
3. Budget proposal revision - Due: October 25th (Mike)

Please confirm receipt of this email and let me know if you have any questions about your assigned tasks.

Our next meeting is scheduled for October 30th at 10:00 AM.

Best regards,
[Your Name]`,
      korean: "후속 조치 - 오늘 회의의 실행 항목들"
    }
  ]

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-200 mb-4">
          비즈니스 영어 마스터 과정
        </h2>
        <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
          국제 비즈니스 환경에서 성공하기 위한 전문적인 영어 커뮤니케이션 스킬을 체계적으로 학습합니다. 
          회의, 프레젠테이션, 이메일, 협상 등 실무에서 바로 활용할 수 있는 실전 표현들을 마스터하세요.
        </p>
      </div>

      {/* Business Topics Navigation */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950/20 dark:to-indigo-950/20 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
          🏢 비즈니스 영어 핵심 영역
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          {businessTopics.map(topic => (
            <button
              key={topic.id}
              onClick={() => setActiveTab(topic.id)}
              className={`p-3 rounded-lg text-center transition-colors ${
                activeTab === topic.id
                  ? 'bg-blue-500 text-white'
                  : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-blue-100 dark:hover:bg-blue-900/50'
              }`}
            >
              <div className="text-xl mb-1">{topic.icon}</div>
              <div className="text-xs font-medium">{topic.name}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Meeting Content */}
      {activeTab === 'meetings' && (
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
              📊 효과적인 회의 진행을 위한 필수 표현
            </h3>
            
            {Object.entries(meetingExpressions).map(([category, expressions]) => (
              <div key={category} className="mb-6">
                <button
                  onClick={() => setExpandedSection(expandedSection === category ? null : category)}
                  className="w-full text-left p-3 bg-blue-50 dark:bg-blue-950/20 rounded-lg hover:bg-blue-100 dark:hover:bg-blue-900/30 transition-colors"
                >
                  <h4 className="font-medium text-gray-800 dark:text-gray-200 capitalize">
                    {category === 'opening' && '🚀 회의 시작'}
                    {category === 'agenda' && '📋 의제 관리'}
                    {category === 'opinions' && '💭 의견 표현'}
                    {category === 'disagreeing' && '🤔 정중한 반대'}
                    {category === 'closing' && '🏁 회의 마무리'}
                  </h4>
                </button>
                
                {expandedSection === category && (
                  <div className="mt-3 space-y-3 pl-4">
                    {expressions.map((item, idx) => (
                      <div key={idx} className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                        <p className="font-medium text-gray-800 dark:text-gray-200 mb-1">
                          "{item.expression}"
                        </p>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          {item.korean}
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Meeting Role Play Scenarios */}
          <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-950/20 dark:to-emerald-950/20 rounded-xl p-6">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
              🎭 회의 역할극 시나리오
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">팀 리더 역할</h4>
                <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                  <li>• 회의 시작과 마무리 진행</li>
                  <li>• 의제 관리와 시간 조절</li>
                  <li>• 팀원들의 참여 유도</li>
                  <li>• 결정사항 정리와 후속 조치 배정</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">팀원 역할</h4>
                <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                  <li>• 적극적인 의견 표현</li>
                  <li>• 건설적인 질문하기</li>
                  <li>• 정중한 반대 의견 제시</li>
                  <li>• 실행 가능한 제안하기</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Presentation Content */}
      {activeTab === 'presentations' && (
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
              📈 임팩트 있는 프레젠테이션 구성법
            </h3>
            
            {presentationStructure.map((section, idx) => (
              <div key={idx} className="mb-6 p-4 bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-950/20 dark:to-pink-950/20 rounded-lg">
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">
                  {idx + 1}. {section.section}
                </h4>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                  {section.expressions.map((expr, exprIdx) => (
                    <div key={exprIdx} className="bg-white dark:bg-gray-800 rounded-lg p-3">
                      <p className="font-medium text-gray-800 dark:text-gray-200 text-sm mb-1">
                        "{expr.expression}"
                      </p>
                      <p className="text-xs text-gray-600 dark:text-gray-400">
                        {expr.korean}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* Presentation Tips */}
          <div className="bg-amber-50 dark:bg-amber-950/20 rounded-xl p-6">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
              💡 프레젠테이션 성공 비법
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">💬 언어적 요소</h4>
                <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                  <li>• 명확하고 간단한 문장 사용</li>
                  <li>• 핵심 키워드 반복 강조</li>
                  <li>• 논리적 순서로 내용 전개</li>
                  <li>• 청중과의 상호작용 유도</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">👥 청중 관리</h4>
                <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                  <li>• 아이컨택으로 집중도 유지</li>
                  <li>• 적절한 제스처 활용</li>
                  <li>• 질문으로 참여 유도</li>
                  <li>• 피드백에 열린 자세</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">📊 시각 자료</h4>
                <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                  <li>• 한 슬라이드 하나의 메시지</li>
                  <li>• 글보다는 시각적 요소 활용</li>
                  <li>• 일관된 디자인 유지</li>
                  <li>• 데이터는 그래프로 표현</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Email Content */}
      {activeTab === 'emails' && (
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
              ✉️ 프로페셔널 이메일 작성법
            </h3>
            
            {emailTemplates.map((template, idx) => (
              <div key={idx} className="mb-6 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">
                  {template.type} - {template.korean}
                </h4>
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4 font-mono text-sm">
                  <div className="border-b border-gray-200 dark:border-gray-600 pb-2 mb-3">
                    <strong>Subject:</strong> {template.subject}
                  </div>
                  <pre className="whitespace-pre-wrap text-gray-600 dark:text-gray-400">
                    {template.body}
                  </pre>
                </div>
              </div>
            ))}
          </div>

          {/* Email Writing Guidelines */}
          <div className="bg-gradient-to-r from-cyan-50 to-blue-50 dark:from-cyan-950/20 dark:to-blue-950/20 rounded-xl p-6">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
              📝 이메일 작성 가이드라인
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">DO's ✅</h4>
                <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                  <li>• 명확하고 구체적인 제목 작성</li>
                  <li>• 정중하고 전문적인 톤 유지</li>
                  <li>• 핵심 내용을 먼저 제시</li>
                  <li>• 액션 아이템을 명확히 명시</li>
                  <li>• 마감일과 책임자 지정</li>
                  <li>• 감사 인사로 마무리</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">DON'Ts ❌</h4>
                <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                  <li>• 너무 길거나 복잡한 문장</li>
                  <li>• 모호하거나 불명확한 표현</li>
                  <li>• 감정적이거나 비판적인 톤</li>
                  <li>• 중요한 정보 누락</li>
                  <li>• 맞춤법이나 문법 오류</li>
                  <li>• 불필요한 전체 답장</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Negotiation Content */}
      {activeTab === 'negotiations' && (
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
              🤝 Win-Win 협상 전략과 표현법
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="p-4 bg-green-50 dark:bg-green-950/20 rounded-lg">
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">협상 시작</h4>
                <ul className="space-y-2 text-sm">
                  <li className="text-gray-800 dark:text-gray-200">
                    <strong>"Let's find a solution that works for both parties."</strong>
                    <br />
                    <span className="text-gray-600 dark:text-gray-400">양측 모두에게 효과적인 해결책을 찾아봅시다.</span>
                  </li>
                  <li className="text-gray-800 dark:text-gray-200">
                    <strong>"What are your main concerns about this proposal?"</strong>
                    <br />
                    <span className="text-gray-600 dark:text-gray-400">이 제안에 대한 주요 우려사항은 무엇입니까?</span>
                  </li>
                </ul>
              </div>
              
              <div className="p-4 bg-blue-50 dark:bg-blue-950/20 rounded-lg">
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">조건 제시</h4>
                <ul className="space-y-2 text-sm">
                  <li className="text-gray-800 dark:text-gray-200">
                    <strong>"We could consider that if you're willing to..."</strong>
                    <br />
                    <span className="text-gray-600 dark:text-gray-400">당신이 ...을 기꺼이 한다면 그것을 고려할 수 있습니다.</span>
                  </li>
                  <li className="text-gray-800 dark:text-gray-200">
                    <strong>"How about we meet in the middle?"</strong>
                    <br />
                    <span className="text-gray-600 dark:text-gray-400">중간에서 만나는 것은 어떨까요?</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Final Success Tips */}
      <div className="bg-gradient-to-r from-emerald-500 to-teal-600 rounded-xl p-6 text-white">
        <h3 className="text-lg font-semibold mb-3">🚀 비즈니스 영어 성공을 위한 최종 팁</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-emerald-100">
          <div>
            <h4 className="font-semibold mb-2">💪 자신감 키우기</h4>
            <p className="text-sm">지속적인 연습과 실전 적용을 통해 자연스러운 비즈니스 영어 구사능력을 기르세요.</p>
          </div>
          <div>
            <h4 className="font-semibold mb-2">🌐 문화적 맥락 이해</h4>
            <p className="text-sm">언어뿐만 아니라 비즈니스 문화와 에티켓을 함께 학습하여 효과적인 소통을 하세요.</p>
          </div>
          <div>
            <h4 className="font-semibold mb-2">📈 지속적 발전</h4>
            <p className="text-sm">피드백을 적극 수용하고 새로운 표현을 계속 학습하여 전문성을 향상시키세요.</p>
          </div>
        </div>
      </div>
    </div>
  )
}

