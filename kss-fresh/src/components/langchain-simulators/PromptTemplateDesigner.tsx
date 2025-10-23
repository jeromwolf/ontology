'use client'

import React, { useState, useEffect } from 'react'
import { Copy, Download, Sparkles, BookOpen, RefreshCw } from 'lucide-react'

interface Variable {
  name: string
  value: string
  description?: string
}

interface Template {
  name: string
  description: string
  template: string
  variables: Variable[]
  category: string
}

const TEMPLATE_LIBRARY: Template[] = [
  {
    name: 'Few-Shot Classification',
    description: 'Classify text using examples',
    category: 'Few-Shot',
    template: `Classify the sentiment of the following text as positive, negative, or neutral.

Examples:
Text: "I love this product!"
Sentiment: positive

Text: "This is terrible."
Sentiment: negative

Text: "It's okay."
Sentiment: neutral

Text: "{text}"
Sentiment:`,
    variables: [
      { name: 'text', value: '', description: 'Text to classify' }
    ]
  },
  {
    name: 'ReAct Agent',
    description: 'Reasoning and Acting framework',
    category: 'Agent',
    template: `Answer the following question by reasoning step by step.

Question: {question}

Thought: I need to think about this carefully.
Action: search[{query}]
Observation: [Results will appear here]
Thought: Based on the observation, I can now answer.
Answer: [Final answer]`,
    variables: [
      { name: 'question', value: '', description: 'Question to answer' },
      { name: 'query', value: '', description: 'Search query' }
    ]
  },
  {
    name: 'Chain of Thought',
    description: 'Step-by-step reasoning',
    category: 'Reasoning',
    template: `Let's solve this problem step by step.

Problem: {problem}

Step 1: Understanding the problem
{step1}

Step 2: Breaking it down
{step2}

Step 3: Solution
{step3}

Final Answer:`,
    variables: [
      { name: 'problem', value: '', description: 'Problem statement' },
      { name: 'step1', value: '', description: 'First step' },
      { name: 'step2', value: '', description: 'Second step' },
      { name: 'step3', value: '', description: 'Third step' }
    ]
  },
  {
    name: 'Context-QA',
    description: 'Question answering with context',
    category: 'QA',
    template: `Answer the question based on the context below.

Context: {context}

Question: {question}

Answer: Let me analyze the context and provide an accurate answer.`,
    variables: [
      { name: 'context', value: '', description: 'Background information' },
      { name: 'question', value: '', description: 'Question to answer' }
    ]
  },
  {
    name: 'Summarization',
    description: 'Summarize long text',
    category: 'Summarization',
    template: `Summarize the following text in {length} sentences.

Text: {text}

Summary:`,
    variables: [
      { name: 'text', value: '', description: 'Text to summarize' },
      { name: 'length', value: '3', description: 'Number of sentences' }
    ]
  },
  {
    name: 'Code Generation',
    description: 'Generate code from description',
    category: 'Code',
    template: `Generate {language} code for the following task.

Task: {task}

Requirements:
- {requirement1}
- {requirement2}

Code:
\`\`\`{language}`,
    variables: [
      { name: 'language', value: 'python', description: 'Programming language' },
      { name: 'task', value: '', description: 'Task description' },
      { name: 'requirement1', value: '', description: 'First requirement' },
      { name: 'requirement2', value: '', description: 'Second requirement' }
    ]
  }
]

export default function PromptTemplateDesigner() {
  const [template, setTemplate] = useState('')
  const [variables, setVariables] = useState<Variable[]>([])
  const [preview, setPreview] = useState('')
  const [selectedTemplate, setSelectedTemplate] = useState<string | null>(null)
  const [showLibrary, setShowLibrary] = useState(false)

  useEffect(() => {
    generatePreview()
  }, [template, variables])

  const generatePreview = () => {
    let result = template
    variables.forEach(v => {
      result = result.replace(new RegExp(`\\{${v.name}\\}`, 'g'), v.value || `[${v.name}]`)
    })
    setPreview(result)
  }

  const extractVariables = (text: string): Variable[] => {
    const matches = text.match(/\{([^}]+)\}/g)
    if (!matches) return []

    const uniqueVars = Array.from(new Set(matches.map(m => m.slice(1, -1))))
    return uniqueVars.map(name => ({
      name,
      value: variables.find(v => v.name === name)?.value || '',
      description: variables.find(v => v.name === name)?.description || ''
    }))
  }

  const handleTemplateChange = (text: string) => {
    setTemplate(text)
    setVariables(extractVariables(text))
  }

  const loadTemplate = (tmpl: Template) => {
    setTemplate(tmpl.template)
    setVariables(tmpl.variables)
    setSelectedTemplate(tmpl.name)
    setShowLibrary(false)
  }

  const updateVariable = (name: string, value: string) => {
    setVariables(variables.map(v =>
      v.name === name ? { ...v, value } : v
    ))
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
    alert('Copied to clipboard!')
  }

  const exportAsCode = () => {
    const code = `from langchain.prompts import PromptTemplate

# Define template
template = """${template}"""

# Create prompt
prompt = PromptTemplate(
    template=template,
    input_variables=[${variables.map(v => `"${v.name}"`).join(', ')}]
)

# Format prompt
formatted = prompt.format(
    ${variables.map(v => `${v.name}="${v.value || 'your_value'}"`).join(',\n    ')}
)

print(formatted)`

    navigator.clipboard.writeText(code)
    alert('Code exported to clipboard!')
  }

  const categories = Array.from(new Set(TEMPLATE_LIBRARY.map(t => t.category)))

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-amber-400 to-orange-500 bg-clip-text text-transparent">
            📝 Prompt Template Designer
          </h1>
          <p className="text-gray-300 text-lg">
            Design, test, and optimize your LangChain prompt templates with real-time preview.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Editor */}
          <div className="space-y-4">
            <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-bold flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-amber-500" />
                  Template Editor
                </h3>
                <button
                  onClick={() => setShowLibrary(!showLibrary)}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded flex items-center gap-2"
                >
                  <BookOpen className="w-4 h-4" />
                  Library
                </button>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Prompt Template (use {'{variable}'} for variables)
                  </label>
                  <textarea
                    value={template}
                    onChange={(e) => handleTemplateChange(e.target.value)}
                    placeholder="Enter your prompt template here...

Example:
Translate the following {language} text to English:

Text: {text}

Translation:"
                    className="w-full px-4 py-3 bg-gray-900 border border-gray-600 rounded-lg font-mono text-sm"
                    rows={12}
                  />
                </div>

                <div className="flex gap-2">
                  <button
                    onClick={() => copyToClipboard(template)}
                    disabled={!template}
                    className="flex-1 px-4 py-2 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 rounded flex items-center justify-center gap-2"
                  >
                    <Copy className="w-4 h-4" />
                    Copy Template
                  </button>
                  <button
                    onClick={exportAsCode}
                    disabled={!template}
                    className="flex-1 px-4 py-2 bg-amber-600 hover:bg-amber-700 disabled:bg-gray-800 rounded flex items-center justify-center gap-2"
                  >
                    <Download className="w-4 h-4" />
                    Export Code
                  </button>
                </div>
              </div>
            </div>

            {/* Variables */}
            {variables.length > 0 && (
              <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
                <h3 className="text-xl font-bold mb-4">Variables ({variables.length})</h3>

                <div className="space-y-4">
                  {variables.map(v => (
                    <div key={v.name}>
                      <label className="block text-sm font-medium mb-2">
                        {v.name}
                        {v.description && (
                          <span className="text-gray-400 text-xs ml-2">({v.description})</span>
                        )}
                      </label>
                      <input
                        type="text"
                        value={v.value}
                        onChange={(e) => updateVariable(v.name, e.target.value)}
                        placeholder={`Enter ${v.name}...`}
                        className="w-full px-4 py-2 bg-gray-900 border border-gray-600 rounded"
                      />
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Preview & Library */}
          <div className="space-y-4">
            {/* Preview */}
            <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-bold">Real-time Preview</h3>
                <button
                  onClick={generatePreview}
                  className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded flex items-center gap-2"
                >
                  <RefreshCw className="w-4 h-4" />
                  Refresh
                </button>
              </div>

              <div className="bg-gray-900 border border-gray-600 rounded-lg p-4 min-h-[400px]">
                <pre className="whitespace-pre-wrap text-sm font-mono">
                  {preview || 'Enter a template to see preview...'}
                </pre>
              </div>

              <button
                onClick={() => copyToClipboard(preview)}
                disabled={!preview}
                className="w-full mt-4 px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-800 rounded flex items-center justify-center gap-2"
              >
                <Copy className="w-4 h-4" />
                Copy Formatted Prompt
              </button>
            </div>

            {/* Template Library */}
            {showLibrary && (
              <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-6">
                <h3 className="text-xl font-bold mb-4">Template Library</h3>

                {categories.map(category => (
                  <div key={category} className="mb-6">
                    <h4 className="text-lg font-semibold mb-3 text-amber-400">{category}</h4>
                    <div className="space-y-2">
                      {TEMPLATE_LIBRARY.filter(t => t.category === category).map(tmpl => (
                        <button
                          key={tmpl.name}
                          onClick={() => loadTemplate(tmpl)}
                          className={`w-full text-left px-4 py-3 bg-gray-700/50 hover:bg-gray-600 border border-gray-600 rounded-lg transition-all ${
                            selectedTemplate === tmpl.name ? 'ring-2 ring-amber-500' : ''
                          }`}
                        >
                          <div className="font-medium">{tmpl.name}</div>
                          <div className="text-sm text-gray-400 mt-1">{tmpl.description}</div>
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Tips */}
            <div className="bg-blue-900/20 backdrop-blur border border-blue-700 rounded-xl p-6">
              <h3 className="text-xl font-bold mb-3 text-blue-400">💡 Tips</h3>
              <ul className="space-y-2 text-sm text-gray-300">
                <li>• Use {'{variable}'} syntax to define dynamic variables</li>
                <li>• Test with real data in the preview panel</li>
                <li>• Few-shot examples improve model performance</li>
                <li>• Clear instructions lead to better results</li>
                <li>• Export to Python code for easy integration</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
