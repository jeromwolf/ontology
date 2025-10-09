'use client'

import References from '@/components/common/References'

export default function Chapter7() {
  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-2xl p-8 border border-yellow-200 dark:border-yellow-800">
        <div className="flex items-start gap-4">
          <div className="text-5xl">🤗</div>
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-3">
              Hugging Face 실전 활용
            </h1>
            <p className="text-lg text-gray-700 dark:text-gray-300">
              허깅페이스 플랫폼으로 모델 개발부터 배포까지 - 실무 중심 완전 가이드
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
            <span>Transformers 라이브러리로 모델 로드, 파인튜닝, 추론 마스터</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-500 mt-1">✓</span>
            <span>Datasets로 대규모 데이터셋 효율적 처리</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-500 mt-1">✓</span>
            <span>Tokenizers로 커스텀 토크나이저 구축</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-500 mt-1">✓</span>
            <span>Spaces로 웹 데모 앱 배포 실습</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-500 mt-1">✓</span>
            <span>AutoTrain으로 No-code 모델 학습</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-500 mt-1">✓</span>
            <span>Inference API를 활용한 프로덕션 배포</span>
          </li>
        </ul>
      </div>

      {/* Section 1: Transformers Library */}
      <section className="space-y-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white border-b-2 border-yellow-500 pb-2">
          1. Transformers 라이브러리 - 핵심 도구
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            1.1 기본 사용법: 모델 로드와 추론
          </h3>

          <p className="text-gray-700 dark:text-gray-300 mb-4">
            Hugging Face의 Transformers는 10만+ 사전학습 모델에 3줄 코드로 접근할 수 있는 혁명적 라이브러리입니다.
          </p>

          <div className="bg-gray-900 rounded-lg p-4 mb-4">
            <pre className="text-sm text-gray-100 overflow-x-auto">
              <code>{`# 텍스트 생성 - GPT-2 예제
from transformers import pipeline

# Pipeline API - 가장 간단한 방법
generator = pipeline('text-generation', model='gpt2')
result = generator(
    "Artificial Intelligence is",
    max_length=50,
    num_return_sequences=3
)
print(result)

# 감정 분석 - 한국어 모델
sentiment = pipeline(
    'sentiment-analysis',
    model='beomi/KcELECTRA-base-v2022'
)
result = sentiment("이 영화 정말 재미있어요!")
# Output: [{'label': 'POSITIVE', 'score': 0.9987}]`}</code>
            </pre>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4 border-l-4 border-yellow-500">
            <p className="text-sm text-gray-700 dark:text-gray-300">
              <strong>💡 Pro Tip:</strong> pipeline() API는 토크나이저, 모델, 후처리를 자동으로 처리합니다.
              프로토타이핑에 최적이지만, 세밀한 제어가 필요하면 AutoModel과 AutoTokenizer를 직접 사용하세요.
            </p>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            1.2 고급 사용: 직접 모델 제어
          </h3>

          <div className="bg-gray-900 rounded-lg p-4 mb-4">
            <pre className="text-sm text-gray-100 overflow-x-auto">
              <code>{`from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 모델과 토크나이저 로드
model_name = "meta-llama/Llama-3.3-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # 메모리 효율
    device_map="auto",            # 멀티 GPU 자동 분산
    load_in_8bit=True             # 양자화로 메모리 절약
)

# 토큰화
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Explain quantum computing simply."}
]
text = tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

# 생성
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)`}</code>
            </pre>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
              <h4 className="font-semibold text-purple-900 dark:text-purple-300 mb-2">
                주요 파라미터 설명
              </h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li><code className="bg-purple-100 dark:bg-purple-900 px-1 rounded">temperature</code>: 창의성 조절 (0.1-2.0)</li>
                <li><code className="bg-purple-100 dark:bg-purple-900 px-1 rounded">top_p</code>: 누적 확률 샘플링</li>
                <li><code className="bg-purple-100 dark:bg-purple-900 px-1 rounded">top_k</code>: 상위 K개 토큰 제한</li>
                <li><code className="bg-purple-100 dark:bg-purple-900 px-1 rounded">repetition_penalty</code>: 반복 억제</li>
              </ul>
            </div>
            <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
              <h4 className="font-semibold text-green-900 dark:text-green-300 mb-2">
                메모리 최적화 기법
              </h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li><code className="bg-green-100 dark:bg-green-900 px-1 rounded">load_in_8bit</code>: 8비트 양자화</li>
                <li><code className="bg-green-100 dark:bg-green-900 px-1 rounded">load_in_4bit</code>: 4비트 양자화 (QLoRA)</li>
                <li><code className="bg-green-100 dark:bg-green-900 px-1 rounded">device_map="auto"</code>: 자동 GPU 분산</li>
                <li><code className="bg-green-100 dark:bg-green-900 px-1 rounded">torch.compile()</code>: PyTorch 2.0 가속</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            1.3 파인튜닝: Trainer API 활용
          </h3>

          <div className="bg-gray-900 rounded-lg p-4 mb-4">
            <pre className="text-sm text-gray-100 overflow-x-auto">
              <code>{`from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset

# 데이터셋 로드 (감정 분석 예제)
dataset = load_dataset("yelp_review_full")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 토큰화 함수
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 모델 초기화
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=5  # 5점 평점
)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_steps=500,
    logging_dir='./logs',
    logging_steps=100,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,  # Mixed precision training
    gradient_checkpointing=True,  # 메모리 절약
)

# Trainer 초기화 및 학습
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics  # 커스텀 메트릭 함수
)

# 학습 실행
trainer.train()

# 모델 저장 및 Hub 업로드
trainer.save_model("./my-finetuned-model")
model.push_to_hub("your-username/yelp-sentiment-model")`}</code>
            </pre>
          </div>
        </div>
      </section>

      {/* Section 2: Datasets */}
      <section className="space-y-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white border-b-2 border-yellow-500 pb-2">
          2. Datasets - 대규모 데이터 처리
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            2.1 Hub에서 데이터셋 로드
          </h3>

          <div className="bg-gray-900 rounded-lg p-4 mb-4">
            <pre className="text-sm text-gray-100 overflow-x-auto">
              <code>{`from datasets import load_dataset

# 공개 데이터셋 로드
dataset = load_dataset("squad")  # Stanford Question Answering

# 특정 설정(configuration) 지정
dataset = load_dataset("glue", "mrpc")  # GLUE 벤치마크의 MRPC 태스크

# 스트리밍 모드 (메모리 절약)
dataset = load_dataset("c4", "en", streaming=True)

# 로컬 파일 로드
dataset = load_dataset("csv", data_files="./my_data.csv")
dataset = load_dataset("json", data_files="./data/*.jsonl")

# 커스텀 split
dataset = load_dataset("imdb", split="train[:80%]")  # 훈련 데이터의 80%만`}</code>
            </pre>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            2.2 데이터 전처리 및 변환
          </h3>

          <div className="bg-gray-900 rounded-lg p-4 mb-4">
            <pre className="text-sm text-gray-100 overflow-x-auto">
              <code>{`# map() - 가장 많이 사용하는 변환 함수
def preprocess_function(examples):
    # 배치 처리로 속도 최적화
    return tokenizer(
        examples["question"],
        examples["context"],
        truncation=True,
        max_length=384,
        padding="max_length"
    )

dataset = dataset.map(
    preprocess_function,
    batched=True,  # 배치 처리 활성화
    num_proc=4,    # 멀티프로세싱
    remove_columns=dataset.column_names,  # 원본 컬럼 제거
    load_from_cache_file=True  # 캐싱 활용
)

# filter() - 조건에 맞는 데이터만 선택
long_texts = dataset.filter(lambda x: len(x["text"]) > 1000)

# select() - 인덱스로 선택
small_dataset = dataset.select(range(1000))

# train_test_split() - 데이터 분할
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# shuffle() - 셔플
dataset = dataset.shuffle(seed=42)`}</code>
            </pre>
          </div>

          <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-4 border-l-4 border-orange-500">
            <p className="text-sm text-gray-700 dark:text-gray-300">
              <strong>⚡ 성능 최적화 팁:</strong> <code className="bg-orange-100 dark:bg-orange-900 px-1 rounded">batched=True</code>와
              <code className="bg-orange-100 dark:bg-orange-900 px-1 rounded ml-1">num_proc</code>을 함께 사용하면
              대규모 데이터셋 처리 속도가 10-100배 빨라집니다. Arrow 포맷 덕분에 디스크 I/O도 최소화됩니다.
            </p>
          </div>
        </div>
      </section>

      {/* Section 3: Tokenizers */}
      <section className="space-y-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white border-b-2 border-yellow-500 pb-2">
          3. Tokenizers - 커스텀 토크나이저 구축
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            3.1 빠른 토크나이저 학습
          </h3>

          <div className="bg-gray-900 rounded-lg p-4 mb-4">
            <pre className="text-sm text-gray-100 overflow-x-auto">
              <code>{`from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.processors import TemplateProcessing

# BPE 토크나이저 생성 (GPT 스타일)
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# 정규화: 소문자 변환, 악센트 제거
tokenizer.normalizer = normalizers.Sequence([
    NFD(),
    Lowercase(),
    StripAccents()
])

# 사전 토큰화: 공백 기준
tokenizer.pre_tokenizer = Whitespace()

# 학습 준비
trainer = BpeTrainer(
    vocab_size=30000,
    min_frequency=2,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)

# 파일로부터 학습
files = ["path/to/corpus1.txt", "path/to/corpus2.txt"]
tokenizer.train(files, trainer)

# 후처리: BERT 스타일 [CLS], [SEP] 추가
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],
)

# 저장
tokenizer.save("my_tokenizer.json")

# Transformers와 통합
from transformers import PreTrainedTokenizerFast
fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="my_tokenizer.json")`}</code>
            </pre>
          </div>
        </div>
      </section>

      {/* Section 4: Spaces */}
      <section className="space-y-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white border-b-2 border-yellow-500 pb-2">
          4. Spaces - 웹 데모 앱 배포
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            4.1 Gradio로 즉시 배포
          </h3>

          <div className="bg-gray-900 rounded-lg p-4 mb-4">
            <pre className="text-sm text-gray-100 overflow-x-auto">
              <code>{`import gradio as gr
from transformers import pipeline

# 모델 로드
pipe = pipeline("text-generation", model="gpt2")

# 인터페이스 함수
def generate_text(prompt, max_length=100, temperature=0.7):
    result = pipe(
        prompt,
        max_length=max_length,
        temperature=temperature,
        num_return_sequences=1
    )
    return result[0]['generated_text']

# Gradio 인터페이스 생성
demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="프롬프트", placeholder="텍스트를 입력하세요..."),
        gr.Slider(minimum=50, maximum=500, value=100, label="최대 길이"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature")
    ],
    outputs=gr.Textbox(label="생성된 텍스트"),
    title="GPT-2 텍스트 생성기",
    description="GPT-2 모델을 사용한 자동 텍스트 생성 데모",
    examples=[
        ["Once upon a time", 150, 0.9],
        ["In a galaxy far far away", 200, 0.8]
    ]
)

# 로컬 실행
if __name__ == "__main__":
    demo.launch()

# Hugging Face Spaces에 배포
# 1. Spaces 저장소 생성: https://huggingface.co/new-space
# 2. Git clone
# 3. app.py에 위 코드 저장
# 4. requirements.txt 생성:
#    transformers
#    torch
#    gradio
# 5. git push → 자동 배포!`}</code>
            </pre>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            4.2 Streamlit으로 고급 앱 구축
          </h3>

          <div className="bg-gray-900 rounded-lg p-4 mb-4">
            <pre className="text-sm text-gray-100 overflow-x-auto">
              <code>{`import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

st.title("감정 분석 앱")
st.write("텍스트의 긍정/부정을 분석합니다")

# 사이드바 설정
with st.sidebar:
    st.header("설정")
    threshold = st.slider("신뢰도 임계값", 0.0, 1.0, 0.5)

# 메인 영역
text = st.text_area("분석할 텍스트를 입력하세요", height=150)

if st.button("분석"):
    if text:
        model = load_model()
        result = model(text)[0]

        # 결과 표시
        st.subheader("결과")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("감정", result['label'])
        with col2:
            st.metric("신뢰도", f"{result['score']:.2%}")

        # 시각화
        if result['score'] >= threshold:
            st.success("높은 신뢰도로 분류되었습니다!")
        else:
            st.warning("신뢰도가 낮습니다. 재검토가 필요합니다.")
    else:
        st.error("텍스트를 입력해주세요")`}</code>
            </pre>
          </div>
        </div>
      </section>

      {/* Section 5: AutoTrain */}
      <section className="space-y-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white border-b-2 border-yellow-500 pb-2">
          5. AutoTrain - No-code 모델 학습
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            AutoTrain은 코드 없이 브라우저에서 클릭만으로 모델을 파인튜닝할 수 있는 플랫폼입니다.
          </p>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <h4 className="font-semibold text-gray-900 dark:text-white">
                📊 지원 태스크
              </h4>
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li>• Text Classification (감정 분석, 주제 분류)</li>
                <li>• Token Classification (NER, POS tagging)</li>
                <li>• Question Answering (SQuAD 스타일)</li>
                <li>• Summarization (요약)</li>
                <li>• Translation (번역)</li>
                <li>• Image Classification (이미지 분류)</li>
                <li>• Object Detection (객체 탐지)</li>
                <li>• Tabular Data (테이블 데이터)</li>
              </ul>
            </div>

            <div className="space-y-4">
              <h4 className="font-semibold text-gray-900 dark:text-white">
                🚀 사용 방법
              </h4>
              <ol className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li>1. huggingface.co/autotrain 접속</li>
                <li>2. 프로젝트 생성 및 태스크 선택</li>
                <li>3. 데이터셋 업로드 (CSV, JSON)</li>
                <li>4. 베이스 모델 선택 (BERT, GPT, etc.)</li>
                <li>5. 하이퍼파라미터 자동 튜닝</li>
                <li>6. 학습 시작 (GPU 자동 할당)</li>
                <li>7. 모델 평가 및 배포</li>
              </ol>
            </div>
          </div>

          <div className="mt-6 bg-green-50 dark:bg-green-900/20 rounded-lg p-4 border-l-4 border-green-500">
            <p className="text-sm text-gray-700 dark:text-gray-300">
              <strong>💰 비용:</strong> 무료 플랜은 월 5시간 GPU 제공.
              Pro 플랜($9/월)은 무제한 학습 + 우선 GPU 접근 + 프라이빗 모델 지원.
            </p>
          </div>
        </div>
      </section>

      {/* Section 6: Inference API */}
      <section className="space-y-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white border-b-2 border-yellow-500 pb-2">
          6. Inference API - 프로덕션 배포
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            6.1 REST API로 모델 호출
          </h3>

          <div className="bg-gray-900 rounded-lg p-4 mb-4">
            <pre className="text-sm text-gray-100 overflow-x-auto">
              <code>{`import requests

API_URL = "https://api-inference.huggingface.co/models/gpt2"
headers = {"Authorization": "Bearer YOUR_HF_TOKEN"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# 텍스트 생성
output = query({
    "inputs": "The future of AI is",
    "parameters": {
        "max_new_tokens": 50,
        "temperature": 0.7,
        "top_p": 0.9
    }
})
print(output)

# 배치 처리
batch_output = query({
    "inputs": [
        "First prompt",
        "Second prompt",
        "Third prompt"
    ]
})

# 이미지 분류
import base64
with open("image.jpg", "rb") as f:
    img_data = base64.b64encode(f.read()).decode()

API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"
output = query({"inputs": img_data})`}</code>
            </pre>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            6.2 Python 클라이언트 사용
          </h3>

          <div className="bg-gray-900 rounded-lg p-4 mb-4">
            <pre className="text-sm text-gray-100 overflow-x-auto">
              <code>{`from huggingface_hub import InferenceClient

# 클라이언트 초기화
client = InferenceClient(token="YOUR_HF_TOKEN")

# 텍스트 생성
response = client.text_generation(
    "Explain quantum computing",
    model="meta-llama/Llama-3.3-70B-Instruct",
    max_new_tokens=200,
    temperature=0.7,
    stream=True  # 스트리밍 응답
)

for token in response:
    print(token, end="")

# 채팅 완성
messages = [
    {"role": "user", "content": "What is machine learning?"}
]
response = client.chat_completion(
    messages,
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    max_tokens=500
)
print(response.choices[0].message.content)

# 임베딩 생성
embeddings = client.feature_extraction(
    "This is a sentence to embed",
    model="sentence-transformers/all-MiniLM-L6-v2"
)
print(f"Embedding dimension: {len(embeddings[0])}")

# 제로샷 분류
result = client.zero_shot_classification(
    "This movie was amazing!",
    labels=["positive", "negative", "neutral"],
    model="facebook/bart-large-mnli"
)
print(result)`}</code>
            </pre>
          </div>

          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 border-l-4 border-blue-500">
            <p className="text-sm text-gray-700 dark:text-gray-300">
              <strong>🔑 API 토큰 생성:</strong> huggingface.co/settings/tokens에서
              Read 권한 토큰 생성. 환경변수 <code className="bg-blue-100 dark:bg-blue-900 px-1 rounded">HF_TOKEN</code>으로 저장 권장.
            </p>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            6.3 전용 엔드포인트 (Dedicated Endpoints)
          </h3>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg p-6">
              <h4 className="font-semibold text-purple-900 dark:text-purple-300 mb-3">
                무료 Inference API
              </h4>
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li>✓ 10만+ 공개 모델 접근</li>
                <li>✓ 무제한 요청 (rate limit 有)</li>
                <li>✓ 콜드 스타트 지연 有</li>
                <li>✓ 공유 인프라</li>
                <li>✓ 개발/프로토타이핑 최적</li>
              </ul>
            </div>

            <div className="bg-gradient-to-br from-green-50 to-teal-50 dark:from-green-900/20 dark:to-teal-900/20 rounded-lg p-6">
              <h4 className="font-semibold text-green-900 dark:text-green-300 mb-3">
                Dedicated Endpoints (유료)
              </h4>
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li>✓ 전용 GPU/CPU 할당</li>
                <li>✓ 콜드 스타트 없음</li>
                <li>✓ 예측 가능한 레이턴시</li>
                <li>✓ 오토스케일링 지원</li>
                <li>✓ 프로덕션 환경 최적</li>
                <li>💰 $0.60/hr (GPU) ~</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* Best Practices */}
      <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-6 border border-purple-200 dark:border-purple-800">
        <h2 className="text-xl font-semibold text-purple-900 dark:text-purple-300 mb-4">
          💎 실무 Best Practices
        </h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="space-y-3">
            <h4 className="font-semibold text-gray-900 dark:text-white">개발 단계</h4>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• <strong>모델 선택:</strong> Hub에서 태스크별 리더보드 확인</li>
              <li>• <strong>데이터:</strong> 최소 1,000개 샘플로 시작</li>
              <li>• <strong>검증:</strong> 10-20% 홀드아웃 세트 필수</li>
              <li>• <strong>버전 관리:</strong> Git LFS로 모델 체크포인트 관리</li>
            </ul>
          </div>
          <div className="space-y-3">
            <h4 className="font-semibold text-gray-900 dark:text-white">프로덕션 단계</h4>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• <strong>모니터링:</strong> 입출력 로깅, 레이턴시 추적</li>
              <li>• <strong>A/B 테스팅:</strong> Spaces로 여러 모델 비교</li>
              <li>• <strong>비용 최적화:</strong> 양자화, 지식 증류 적용</li>
              <li>• <strong>보안:</strong> 프라이빗 모델 + 액세스 토큰 관리</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Additional Resources */}
      <div className="bg-gray-50 dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          📚 추가 학습 리소스
        </h2>
        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">공식 문서</h4>
            <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
              <li>• <a href="https://huggingface.co/docs/transformers" className="text-blue-600 hover:underline">Transformers 문서</a></li>
              <li>• <a href="https://huggingface.co/docs/datasets" className="text-blue-600 hover:underline">Datasets 문서</a></li>
              <li>• <a href="https://huggingface.co/docs/tokenizers" className="text-blue-600 hover:underline">Tokenizers 문서</a></li>
              <li>• <a href="https://huggingface.co/docs/hub" className="text-blue-600 hover:underline">Hub 가이드</a></li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">실습 코스</h4>
            <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
              <li>• <a href="https://huggingface.co/learn/nlp-course" className="text-blue-600 hover:underline">NLP Course (무료)</a></li>
              <li>• <a href="https://huggingface.co/learn/deep-rl-course" className="text-blue-600 hover:underline">Deep RL Course</a></li>
              <li>• <a href="https://github.com/huggingface/transformers/tree/main/examples" className="text-blue-600 hover:underline">공식 예제 모음</a></li>
            </ul>
          </div>
        </div>
      </div>

      {/* Summary */}
      <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-xl p-6 border border-yellow-200 dark:border-yellow-800">
        <h2 className="text-xl font-semibold text-yellow-900 dark:text-yellow-300 mb-3">
          ✨ 핵심 요약
        </h2>
        <ul className="space-y-2 text-gray-700 dark:text-gray-300">
          <li>• <strong>Transformers:</strong> Pipeline API로 빠른 프로토타이핑, Trainer로 전문 파인튜닝</li>
          <li>• <strong>Datasets:</strong> Arrow 포맷 덕분에 테라바이트 데이터도 효율적 처리</li>
          <li>• <strong>Tokenizers:</strong> Rust 기반 초고속 토크나이저, 커스텀 빌드 가능</li>
          <li>• <strong>Spaces:</strong> Gradio/Streamlit으로 git push 한 번에 배포</li>
          <li>• <strong>AutoTrain:</strong> 코드 없이 브라우저에서 모델 학습</li>
          <li>• <strong>Inference API:</strong> 서버리스 추론, 무료 → 전용 엔드포인트 확장</li>
        </ul>
      </div>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 공식 문서',
            icon: 'web' as const,
            color: 'border-orange-500',
            items: [
              {
                title: 'Transformers Documentation',
                authors: 'Hugging Face',
                year: '2025',
                description: 'Transformers 라이브러리 공식 문서',
                link: 'https://huggingface.co/docs/transformers'
              },
              {
                title: 'Datasets Documentation',
                authors: 'Hugging Face',
                year: '2025',
                description: 'Datasets 라이브러리 완벽 가이드',
                link: 'https://huggingface.co/docs/datasets'
              },
              {
                title: 'Tokenizers Documentation',
                authors: 'Hugging Face',
                year: '2025',
                description: 'Rust 기반 고속 토크나이저',
                link: 'https://huggingface.co/docs/tokenizers'
              },
              {
                title: 'Hugging Face Hub',
                authors: 'Hugging Face',
                year: '2025',
                description: '200,000+ 모델과 데이터셋 허브',
                link: 'https://huggingface.co/models'
              }
            ]
          },
          {
            title: '🎓 학습 코스',
            icon: 'paper' as const,
            color: 'border-yellow-500',
            items: [
              {
                title: 'NLP Course',
                authors: 'Hugging Face',
                year: '2025',
                description: '무료 NLP 전문 코스 (한국어 지원)',
                link: 'https://huggingface.co/learn/nlp-course'
              },
              {
                title: 'Deep Reinforcement Learning Course',
                authors: 'Hugging Face',
                year: '2024',
                description: '강화학습 실전 가이드',
                link: 'https://huggingface.co/learn/deep-rl-course'
              },
              {
                title: 'Fine-tuning Guide',
                authors: 'Hugging Face',
                year: '2025',
                description: '모델 파인튜닝 완벽 가이드',
                link: 'https://huggingface.co/docs/transformers/training'
              }
            ]
          },
          {
            title: '🛠️ 실전 리소스',
            icon: 'web' as const,
            color: 'border-red-500',
            items: [
              {
                title: 'Spaces Documentation',
                authors: 'Hugging Face',
                year: '2025',
                description: 'ML 앱 무료 배포 플랫폼',
                link: 'https://huggingface.co/docs/hub/spaces'
              },
              {
                title: 'AutoTrain',
                authors: 'Hugging Face',
                year: '2025',
                description: 'No-code 모델 학습 플랫폼',
                link: 'https://huggingface.co/autotrain'
              },
              {
                title: 'Inference Endpoints',
                authors: 'Hugging Face',
                year: '2025',
                description: '프로덕션 AI API 서비스',
                link: 'https://huggingface.co/inference-endpoints'
              },
              {
                title: 'Transformers Examples',
                authors: 'Hugging Face',
                year: '2025',
                description: '공식 예제 코드 모음',
                link: 'https://github.com/huggingface/transformers/tree/main/examples'
              }
            ]
          }
        ]}
      />
    </div>
  )
}
