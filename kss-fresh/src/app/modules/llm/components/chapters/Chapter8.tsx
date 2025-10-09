'use client'

import References from '@/components/common/References'

export default function Chapter8() {
  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-2xl p-8 border border-indigo-200 dark:border-indigo-800">
        <div className="flex items-start gap-4">
          <div className="text-5xl">🚀</div>
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-3">
              AI 서비스와 API 활용
            </h1>
            <p className="text-lg text-gray-700 dark:text-gray-300">
              OpenAI, Claude, Gemini 등 주요 AI 기업 API 실전 가이드 - 프로덕션 레벨 구현
            </p>
          </div>
        </div>
      </div>

      {/* Learning Objectives */}
      <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6 border border-blue-200 dark:border-blue-800">
        <h2 className="text-xl font-semibold text-blue-900 dark:text-blue-300 mb-4">
          🎯 학습 목표
        </h2>
        <ul className="space-y-2 text-gray-700 dark:text-gray-300">
          <li className="flex items-start gap-2">
            <span className="text-blue-500 mt-1">✓</span>
            <span>OpenAI API (GPT-4o, DALL-E 3, Whisper)로 멀티모달 앱 구축</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-500 mt-1">✓</span>
            <span>Anthropic Claude API 고급 기능 (Computer Use, Analysis) 마스터</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-500 mt-1">✓</span>
            <span>Google Gemini 2.5 Flash의 100만 토큰 컨텍스트 활용</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-500 mt-1">✓</span>
            <span>AWS Bedrock으로 멀티모델 통합 플랫폼 구축</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-500 mt-1">✓</span>
            <span>Azure OpenAI Service 엔터프라이즈 배포</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-500 mt-1">✓</span>
            <span>LangChain으로 API 통합 및 에이전트 구축</span>
          </li>
        </ul>
      </div>

      {/* Section 1: OpenAI API */}
      <section className="space-y-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white border-b-2 border-indigo-500 pb-2">
          1. OpenAI API - 업계 표준
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            1.1 GPT-4o - 최신 멀티모달 모델
          </h3>

          <div className="bg-gray-900 rounded-lg p-4 mb-4">
            <pre className="text-sm text-gray-100 overflow-x-auto">
              <code>{`from openai import OpenAI
import os

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# 기본 채팅 완성
response = client.chat.completions.create(
    model="gpt-4o",  # 2025년 최신 모델
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant specialized in Python programming."},
        {"role": "user", "content": "Write a Python function to calculate Fibonacci numbers efficiently."}
    ],
    temperature=0.7,
    max_tokens=1000,
    top_p=0.9,
    frequency_penalty=0.0,
    presence_penalty=0.0
)

print(response.choices[0].message.content)
print(f"Tokens used: {response.usage.total_tokens}")
print(f"Cost: \${response.usage.total_tokens * 0.000005:.4f}")  # GPT-4o 가격

# 스트리밍 응답 (실시간 출력)
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Explain quantum entanglement"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)

# Function Calling (도구 사용)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. Seoul"
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in Seoul?"}],
    tools=tools,
    tool_choice="auto"
)

# 함수 호출 처리
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    print(f"Function: {tool_call.function.name}")
    print(f"Arguments: {tool_call.function.arguments}")`}</code>
            </pre>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
              <h4 className="font-semibold text-green-900 dark:text-green-300 mb-2">
                GPT-4o 장점
              </h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>• 텍스트 + 이미지 + 오디오 통합 처리</li>
                <li>• 128K 토큰 컨텍스트 (GPT-4 Turbo)</li>
                <li>• GPT-4 대비 50% 저렴 ($5/1M tokens)</li>
                <li>• 2배 빠른 응답 속도</li>
                <li>• JSON 모드, Structured Output 지원</li>
              </ul>
            </div>
            <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-4">
              <h4 className="font-semibold text-orange-900 dark:text-orange-300 mb-2">
                모델 선택 가이드
              </h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>• <strong>gpt-4o</strong>: 멀티모달, 고품질 + 속도</li>
                <li>• <strong>gpt-4-turbo</strong>: 긴 컨텍스트</li>
                <li>• <strong>gpt-3.5-turbo</strong>: 저비용, 빠른 응답</li>
                <li>• <strong>gpt-4o-mini</strong>: 초저비용 ($0.15/1M)</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            1.2 DALL-E 3 - 최고 품질 이미지 생성
          </h3>

          <div className="bg-gray-900 rounded-lg p-4 mb-4">
            <pre className="text-sm text-gray-100 overflow-x-auto">
              <code>{`# 이미지 생성
response = client.images.generate(
    model="dall-e-3",
    prompt="A serene Japanese garden with cherry blossoms, koi pond, and a traditional tea house, rendered in ukiyo-e art style",
    size="1792x1024",  # HD: 1024x1024, 1792x1024, 1024x1792
    quality="hd",      # "standard" or "hd"
    style="vivid",     # "vivid" or "natural"
    n=1
)

image_url = response.data[0].url
revised_prompt = response.data[0].revised_prompt  # GPT-4로 자동 개선된 프롬프트
print(f"Generated: {image_url}")
print(f"Revised: {revised_prompt}")

# 이미지 편집 (DALL-E 2)
response = client.images.edit(
    image=open("original.png", "rb"),
    mask=open("mask.png", "rb"),  # 편집할 영역 (흰색)
    prompt="Replace the background with a futuristic cityscape",
    size="1024x1024",
    n=1
)

# 이미지 변형 생성
response = client.images.create_variation(
    image=open("original.png", "rb"),
    n=3,  # 3개의 변형 생성
    size="1024x1024"
)`}</code>
            </pre>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            1.3 Whisper - 음성 인식 및 번역
          </h3>

          <div className="bg-gray-900 rounded-lg p-4 mb-4">
            <pre className="text-sm text-gray-100 overflow-x-auto">
              <code>{`# 음성을 텍스트로 변환 (STT)
audio_file = open("speech.mp3", "rb")
transcript = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file,
    language="ko",  # 언어 힌트 (선택)
    response_format="verbose_json",  # text, json, verbose_json, srt, vtt
    timestamp_granularities=["word", "segment"]
)

print(transcript.text)
for word in transcript.words:
    print(f"{word.word} ({word.start:.2f}s - {word.end:.2f}s)")

# 음성 번역 (다국어 → 영어)
translation = client.audio.translations.create(
    model="whisper-1",
    file=open("korean_speech.mp3", "rb")
)
print(translation.text)  # 영어로 자동 번역

# 텍스트를 음성으로 (TTS)
speech_file = Path(__file__).parent / "speech.mp3"
response = client.audio.speech.create(
    model="tts-1-hd",  # tts-1 (빠름) or tts-1-hd (고품질)
    voice="alloy",     # alloy, echo, fable, onyx, nova, shimmer
    input="안녕하세요. OpenAI의 음성 합성 기술입니다.",
    speed=1.0
)

response.stream_to_file(speech_file)`}</code>
            </pre>
          </div>
        </div>
      </section>

      {/* Section 2: Anthropic Claude */}
      <section className="space-y-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white border-b-2 border-indigo-500 pb-2">
          2. Anthropic Claude - 안전성과 긴 컨텍스트
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            2.1 Claude 3.5 Sonnet & Opus 4 활용
          </h3>

          <div className="bg-gray-900 rounded-lg p-4 mb-4">
            <pre className="text-sm text-gray-100 overflow-x-auto">
              <code>{`import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# 기본 메시지 생성
message = client.messages.create(
    model="claude-opus-4-20250514",  # 2025년 최신 모델
    max_tokens=4096,
    temperature=1.0,
    system="You are a world-class poet. Respond only with short poems.",
    messages=[
        {"role": "user", "content": "Write a haiku about recursion in programming."}
    ]
)

print(message.content[0].text)

# 스트리밍
with client.messages.stream(
    model="claude-3-5-sonnet-20250220",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Explain black holes"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

# Vision (이미지 분석)
import base64

with open("diagram.png", "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

message = client.messages.create(
    model="claude-3-5-sonnet-20250220",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_data,
                    },
                },
                {
                    "type": "text",
                    "text": "Describe this diagram in detail and explain the workflow."
                }
            ],
        }
    ]
)

print(message.content[0].text)`}</code>
            </pre>
          </div>

          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4 border-l-4 border-purple-500">
            <p className="text-sm text-gray-700 dark:text-gray-300">
              <strong>🎯 Claude의 강점:</strong> Claude Opus 4는 200K 토큰 컨텍스트를 지원하며,
              복잡한 추론, 코드 생성, 수학 문제에서 GPT-4o를 능가합니다. 특히 안전성과 정확성이 중요한 엔터프라이즈 환경에 최적입니다.
            </p>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            2.2 Tool Use (Function Calling)
          </h3>

          <div className="bg-gray-900 rounded-lg p-4 mb-4">
            <pre className="text-sm text-gray-100 overflow-x-auto">
              <code>{`tools = [
    {
        "name": "get_stock_price",
        "description": "Get the current stock price for a given ticker symbol",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol, e.g. AAPL for Apple"
                }
            },
            "required": ["ticker"]
        }
    }
]

message = client.messages.create(
    model="claude-3-5-sonnet-20250220",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "What's the current price of Apple stock?"}]
)

# Tool 사용 처리
if message.stop_reason == "tool_use":
    tool_use = next(block for block in message.content if block.type == "tool_use")
    print(f"Claude wants to call: {tool_use.name}")
    print(f"With arguments: {tool_use.input}")

    # 실제 함수 실행 (예시)
    result = get_stock_price(tool_use.input["ticker"])

    # 결과를 Claude에게 다시 전달
    response = client.messages.create(
        model="claude-3-5-sonnet-20250220",
        max_tokens=1024,
        tools=tools,
        messages=[
            {"role": "user", "content": "What's the current price of Apple stock?"},
            {"role": "assistant", "content": message.content},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": str(result)
                    }
                ]
            }
        ]
    )
    print(response.content[0].text)`}</code>
            </pre>
          </div>
        </div>
      </section>

      {/* Section 3: Google Gemini */}
      <section className="space-y-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white border-b-2 border-indigo-500 pb-2">
          3. Google Gemini - 100만 토큰 컨텍스트
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            3.1 Gemini 2.5 Flash 활용
          </h3>

          <div className="bg-gray-900 rounded-lg p-4 mb-4">
            <pre className="text-sm text-gray-100 overflow-x-auto">
              <code>{`import google.generativeai as genai

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# 모델 초기화
model = genai.GenerativeModel("gemini-2.5-flash")

# 텍스트 생성
response = model.generate_content("Explain quantum computing in simple terms")
print(response.text)

# 멀티모달: 이미지 + 텍스트
import PIL.Image

img = PIL.Image.open("chart.png")
response = model.generate_content(["Analyze this chart and provide insights", img])
print(response.text)

# 대화형 채팅
chat = model.start_chat(history=[])

response = chat.send_message("Hello! Can you help me with Python?")
print(response.text)

response = chat.send_message("Write a function to sort a list")
print(response.text)

# 긴 컨텍스트 처리 (1M 토큰!)
with open("entire_codebase.txt", "r") as f:
    large_text = f.read()  # 수백만 자의 코드

response = model.generate_content([
    "Analyze this entire codebase and suggest architectural improvements",
    large_text
])
print(response.text)`}</code>
            </pre>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4 border-l-4 border-green-500">
            <p className="text-sm text-gray-700 dark:text-gray-300">
              <strong>🚀 Gemini 2.5 Flash의 혁신:</strong> 100만 토큰 컨텍스트는 책 여러 권, 전체 코드베이스,
              긴 동영상을 한 번에 처리 가능합니다. 비용도 매우 저렴 ($0.075/1M tokens)하여 대규모 문서 분석에 최적입니다.
            </p>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            3.2 Function Calling & JSON Mode
          </h3>

          <div className="bg-gray-900 rounded-lg p-4 mb-4">
            <pre className="text-sm text-gray-100 overflow-x-auto">
              <code>{`# Function Calling
def get_weather(location: str, unit: str = "celsius"):
    """Get weather for a location"""
    return {"temperature": 22, "condition": "sunny"}

tools = [get_weather]

model = genai.GenerativeModel("gemini-2.5-flash", tools=tools)
chat = model.start_chat()

response = chat.send_message("What's the weather in Seoul?")

# Function call 자동 실행
function_call = response.candidates[0].content.parts[0].function_call
result = get_weather(**dict(function_call.args))

response = chat.send_message(
    genai.types.Part.from_function_response(
        name="get_weather",
        response={"content": result}
    )
)
print(response.text)

# JSON 모드 - 구조화된 출력
model = genai.GenerativeModel(
    "gemini-2.5-flash",
    generation_config={
        "response_mime_type": "application/json",
        "response_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "skills": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }
    }
)

response = model.generate_content(
    "Create a profile for a senior Python developer"
)
import json
profile = json.loads(response.text)
print(profile)`}</code>
            </pre>
          </div>
        </div>
      </section>

      {/* Section 4: AWS Bedrock */}
      <section className="space-y-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white border-b-2 border-indigo-500 pb-2">
          4. AWS Bedrock - 멀티모델 플랫폼
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            4.1 다양한 모델 통합 접근
          </h3>

          <div className="bg-gray-900 rounded-lg p-4 mb-4">
            <pre className="text-sm text-gray-100 overflow-x-auto">
              <code>{`import boto3
import json

bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
)

# Claude on Bedrock
body = json.dumps({
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 1024,
    "messages": [
        {
            "role": "user",
            "content": "Explain AWS Lambda"
        }
    ]
})

response = bedrock.invoke_model(
    modelId="anthropic.claude-3-5-sonnet-20250220-v1:0",
    body=body
)

result = json.loads(response['body'].read())
print(result['content'][0]['text'])

# Meta Llama 3.3 on Bedrock
body = json.dumps({
    "prompt": "Explain machine learning",
    "max_gen_len": 512,
    "temperature": 0.7,
    "top_p": 0.9
})

response = bedrock.invoke_model(
    modelId="meta.llama3-3-70b-instruct-v1:0",
    body=body
)

# Titan Embeddings (임베딩)
body = json.dumps({
    "inputText": "This is a sentence to embed"
})

response = bedrock.invoke_model(
    modelId="amazon.titan-embed-text-v2:0",
    body=body
)

embedding = json.loads(response['body'].read())['embedding']
print(f"Embedding dimension: {len(embedding)}")`}</code>
            </pre>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
              <h4 className="font-semibold text-blue-900 dark:text-blue-300 mb-2">
                Bedrock 지원 모델
              </h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>• Claude (Anthropic)</li>
                <li>• Llama 3.3 (Meta)</li>
                <li>• Titan (Amazon)</li>
                <li>• Mistral & Mixtral</li>
                <li>• Stable Diffusion (이미지)</li>
                <li>• Cohere Command</li>
              </ul>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
              <h4 className="font-semibold text-yellow-900 dark:text-yellow-300 mb-2">
                엔터프라이즈 장점
              </h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>• AWS 인프라 내 프라이빗 배포</li>
                <li>• VPC 격리, IAM 통합</li>
                <li>• CloudWatch 모니터링</li>
                <li>• 규정 준수 (HIPAA, SOC)</li>
                <li>• 데이터 주권 보장</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* Section 5: LangChain Integration */}
      <section className="space-y-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white border-b-2 border-indigo-500 pb-2">
          5. LangChain - API 통합 프레임워크
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            5.1 멀티 프로바이더 통합
          </h3>

          <div className="bg-gray-900 rounded-lg p-4 mb-4">
            <pre className="text-sm text-gray-100 overflow-x-auto">
              <code>{`from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# 여러 LLM 정의
openai_llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
claude_llm = ChatAnthropic(model="claude-opus-4-20250514", temperature=0.7)
gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# 프롬프트 템플릿
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("human", "{input}")
])

# Chain 구성
chain = prompt | openai_llm | StrOutputParser()

# 실행
response = chain.invoke({"input": "What is LangChain?"})
print(response)

# 모델 비교 (동시 실행)
from langchain.schema.runnable import RunnableParallel

comparison = RunnableParallel(
    openai=prompt | openai_llm,
    claude=prompt | claude_llm,
    gemini=prompt | gemini_llm
)

results = comparison.invoke({"input": "Explain quantum computing"})
for model, response in results.items():
    print(f"\n{model.upper()}:")
    print(response.content)`}</code>
            </pre>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            5.2 에이전트 구축
          </h3>

          <div className="bg-gray-900 rounded-lg p-4 mb-4">
            <pre className="text-sm text-gray-100 overflow-x-auto">
              <code>{`from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool
from langchain import hub

# 도구 정의
def search_web(query: str) -> str:
    """Search the web for information"""
    # 실제로는 Tavily, Serper 등 사용
    return f"Search results for: {query}"

def calculator(expression: str) -> float:
    """Calculate mathematical expressions"""
    return eval(expression)

tools = [
    Tool(
        name="WebSearch",
        func=search_web,
        description="Search the web for current information"
    ),
    Tool(
        name="Calculator",
        func=calculator,
        description="Perform mathematical calculations"
    )
]

# 프롬프트 로드
prompt = hub.pull("hwchase17/openai-tools-agent")

# 에이전트 생성
agent = create_openai_tools_agent(
    llm=ChatOpenAI(model="gpt-4o", temperature=0),
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# 복잡한 작업 실행
result = agent_executor.invoke({
    "input": "What's 1284 * 567? Then search for information about that number."
})
print(result["output"])`}</code>
            </pre>
          </div>
        </div>
      </section>

      {/* API Comparison Table */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          📊 주요 API 비교 (2025년 기준)
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 dark:bg-gray-700">
              <tr>
                <th className="px-4 py-3 text-left">Provider</th>
                <th className="px-4 py-3 text-left">Top Model</th>
                <th className="px-4 py-3 text-left">Context</th>
                <th className="px-4 py-3 text-left">Price (Input)</th>
                <th className="px-4 py-3 text-left">특징</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 dark:divide-gray-600">
              <tr>
                <td className="px-4 py-3 font-semibold">OpenAI</td>
                <td className="px-4 py-3">GPT-4o</td>
                <td className="px-4 py-3">128K</td>
                <td className="px-4 py-3">$5/1M</td>
                <td className="px-4 py-3">멀티모달, Function Calling</td>
              </tr>
              <tr>
                <td className="px-4 py-3 font-semibold">Anthropic</td>
                <td className="px-4 py-3">Claude Opus 4</td>
                <td className="px-4 py-3">200K</td>
                <td className="px-4 py-3">$15/1M</td>
                <td className="px-4 py-3">추론, 코드, 안전성</td>
              </tr>
              <tr>
                <td className="px-4 py-3 font-semibold">Google</td>
                <td className="px-4 py-3">Gemini 2.5 Flash</td>
                <td className="px-4 py-3">1M</td>
                <td className="px-4 py-3">$0.075/1M</td>
                <td className="px-4 py-3">초장 컨텍스트, 저비용</td>
              </tr>
              <tr>
                <td className="px-4 py-3 font-semibold">xAI</td>
                <td className="px-4 py-3">Grok 4</td>
                <td className="px-4 py-3">128K</td>
                <td className="px-4 py-3">$5/1M</td>
                <td className="px-4 py-3">실시간 X 데이터 접근</td>
              </tr>
              <tr>
                <td className="px-4 py-3 font-semibold">Meta</td>
                <td className="px-4 py-3">Llama 3.3 70B</td>
                <td className="px-4 py-3">128K</td>
                <td className="px-4 py-3">무료 (오픈소스)</td>
                <td className="px-4 py-3">완전 자체 호스팅 가능</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Best Practices */}
      <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-6 border border-purple-200 dark:border-purple-800">
        <h2 className="text-xl font-semibold text-purple-900 dark:text-purple-300 mb-4">
          💎 프로덕션 배포 Best Practices
        </h2>
        <div className="grid md:grid-cols-3 gap-6">
          <div className="space-y-3">
            <h4 className="font-semibold text-gray-900 dark:text-white">🔒 보안</h4>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• API 키 환경변수 저장</li>
              <li>• 프롬프트 인젝션 방어</li>
              <li>• Rate limiting 구현</li>
              <li>• 민감 데이터 필터링</li>
            </ul>
          </div>
          <div className="space-y-3">
            <h4 className="font-semibold text-gray-900 dark:text-white">⚡ 성능</h4>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• 스트리밍 응답 활용</li>
              <li>• 캐싱 전략 구현</li>
              <li>• 배치 처리 최적화</li>
              <li>• 타임아웃 설정</li>
            </ul>
          </div>
          <div className="space-y-3">
            <h4 className="font-semibold text-gray-900 dark:text-white">💰 비용</h4>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• 토큰 사용량 모니터링</li>
              <li>• 적절한 모델 선택</li>
              <li>• 프롬프트 최적화</li>
              <li>• 사용량 알림 설정</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Summary */}
      <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-xl p-6 border border-indigo-200 dark:border-indigo-800">
        <h2 className="text-xl font-semibold text-indigo-900 dark:text-indigo-300 mb-3">
          ✨ 핵심 요약
        </h2>
        <ul className="space-y-2 text-gray-700 dark:text-gray-300">
          <li>• <strong>OpenAI:</strong> 가장 성숙한 생태계, GPT-4o의 멀티모달, DALL-E 3 이미지, Whisper 음성</li>
          <li>• <strong>Anthropic:</strong> Claude Opus 4의 최고 수준 추론, 200K 컨텍스트, Computer Use 기능</li>
          <li>• <strong>Google:</strong> Gemini 2.5 Flash의 100만 토큰 + 초저비용, 긴 문서 분석 최적</li>
          <li>• <strong>AWS Bedrock:</strong> 엔터프라이즈 요구사항, 프라이빗 VPC, 멀티모델 선택</li>
          <li>• <strong>LangChain:</strong> 통합 프레임워크로 프로바이더 전환 용이, 복잡한 에이전트 구축</li>
        </ul>
      </div>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 공식 API 문서',
            icon: 'web' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'OpenAI API Reference',
                authors: 'OpenAI',
                year: '2025',
                description: 'GPT-4o, DALL-E 3, Whisper API 완전 가이드',
                link: 'https://platform.openai.com/docs'
              },
              {
                title: 'Anthropic Claude API',
                authors: 'Anthropic',
                year: '2025',
                description: 'Claude Opus 4 API 문서 및 Tool Use',
                link: 'https://docs.anthropic.com'
              },
              {
                title: 'Google AI for Developers',
                authors: 'Google',
                year: '2025',
                description: 'Gemini 2.5 Flash API 및 Function Calling',
                link: 'https://ai.google.dev'
              },
              {
                title: 'AWS Bedrock Documentation',
                authors: 'Amazon Web Services',
                year: '2025',
                description: '엔터프라이즈 AI 멀티모델 플랫폼',
                link: 'https://docs.aws.amazon.com/bedrock'
              },
              {
                title: 'Azure OpenAI Service',
                authors: 'Microsoft',
                year: '2025',
                description: '엔터프라이즈급 OpenAI 배포',
                link: 'https://learn.microsoft.com/azure/ai-services/openai'
              }
            ]
          },
          {
            title: '🛠️ 통합 프레임워크',
            icon: 'paper' as const,
            color: 'border-indigo-500',
            items: [
              {
                title: 'LangChain Documentation',
                authors: 'LangChain',
                year: '2025',
                description: 'AI 애플리케이션 통합 개발 프레임워크',
                link: 'https://python.langchain.com/docs'
              },
              {
                title: 'LlamaIndex',
                authors: 'LlamaIndex',
                year: '2025',
                description: 'RAG 및 데이터 프레임워크',
                link: 'https://docs.llamaindex.ai'
              },
              {
                title: 'AutoGen Framework',
                authors: 'Microsoft',
                year: '2024',
                description: 'Multi-agent 시스템 구축',
                link: 'https://microsoft.github.io/autogen'
              }
            ]
          },
          {
            title: '💰 가격 정보',
            icon: 'web' as const,
            color: 'border-green-500',
            items: [
              {
                title: 'OpenAI Pricing',
                authors: 'OpenAI',
                year: '2025',
                description: 'GPT-4o \$5/1M tokens',
                link: 'https://openai.com/pricing'
              },
              {
                title: 'Anthropic Pricing',
                authors: 'Anthropic',
                year: '2025',
                description: 'Claude Opus 4 \$15/1M tokens',
                link: 'https://www.anthropic.com/pricing'
              },
              {
                title: 'Google AI Pricing',
                authors: 'Google',
                year: '2025',
                description: 'Gemini 2.5 Flash \$0.075/1M tokens',
                link: 'https://ai.google.dev/pricing'
              }
            ]
          },
          {
            title: '📖 실전 가이드',
            icon: 'web' as const,
            color: 'border-red-500',
            items: [
              {
                title: 'OpenAI Cookbook',
                authors: 'OpenAI',
                year: '2025',
                description: 'API 사용 예제 및 베스트 프랙티스',
                link: 'https://cookbook.openai.com'
              },
              {
                title: 'Anthropic Prompt Engineering',
                authors: 'Anthropic',
                year: '2025',
                description: 'Claude 최적화 프롬프트 가이드',
                link: 'https://docs.anthropic.com/claude/docs/prompt-engineering'
              },
              {
                title: 'LangChain Integrations',
                authors: 'LangChain',
                year: '2025',
                description: '40+ AI 프로바이더 통합 가이드',
                link: 'https://python.langchain.com/docs/integrations'
              }
            ]
          }
        ]}
      />
    </div>
  )
}
