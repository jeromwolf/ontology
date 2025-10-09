'use client';

import React from 'react';

export default function Section2() {
  return (
    <section>
      <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
        CrewAI 실전 코드
      </h3>
      <div className="bg-gray-900 rounded-xl p-6 text-white">
        <pre className="overflow-x-auto">
          <code className="text-sm">{`from crewai import Agent, Task, Crew, Process

# 1. 에이전트 정의
researcher = Agent(
    role='Senior Research Analyst',
    goal='Find accurate information about {topic}',
    backstory='Expert researcher with 10 years experience',
    tools=[search_tool, web_scraper],
    verbose=True
)

writer = Agent(
    role='Content Writer',
    goal='Create engaging content based on research',
    backstory='Professional writer specializing in tech',
    tools=[writing_tool],
    verbose=True
)

editor = Agent(
    role='Editor',
    goal='Ensure high quality and accuracy',
    backstory='Meticulous editor with attention to detail',
    verbose=True
)

# 2. 작업 정의
research_task = Task(
    description='Research latest trends in {topic}',
    expected_output='Comprehensive research report',
    agent=researcher
)

writing_task = Task(
    description='Write article based on research',
    expected_output='1000-word article',
    agent=writer
)

editing_task = Task(
    description='Edit and polish the article',
    expected_output='Final polished article',
    agent=editor
)

# 3. Crew 구성 및 실행
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    process=Process.sequential,
    verbose=True
)

result = crew.kickoff(inputs={'topic': 'AI Agents'})`}</code>
        </pre>
      </div>
    </section>
  );
}
