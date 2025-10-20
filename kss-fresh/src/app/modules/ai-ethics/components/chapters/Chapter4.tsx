import React from 'react';
import { Lock, Shield, Key, Database, UserCheck } from 'lucide-react';
import References from '../References';

export default function Chapter4() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold mb-6 text-gray-900 dark:text-white">프라이버시와 보안</h1>

      <div className="bg-gradient-to-r from-rose-100 to-pink-100 dark:from-rose-900/30 dark:to-pink-900/30 p-6 rounded-lg mb-8">
        <p className="text-lg text-gray-800 dark:text-gray-200 leading-relaxed">
          AI 시대의 프라이버시는 단순히 데이터를 "숨기는" 것이 아닙니다.
          Differential Privacy, Federated Learning, Homomorphic Encryption 등 혁신 기술로
          개인정보를 보호하면서도 유용한 AI를 학습할 수 있습니다.
        </p>
      </div>

      {/* 3대 프라이버시 보호 기술 */}
      <section className="mb-12">
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Shield className="w-8 h-8 text-rose-600" />
          프라이버시 보호 3대 기술
        </h2>

        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg border-l-4 border-blue-500">
            <div className="flex items-center gap-3 mb-3">
              <Lock className="w-6 h-6 text-blue-600" />
              <h3 className="text-2xl font-bold text-gray-900 dark:text-white">1. Differential Privacy (차등 프라이버시)</h3>
            </div>

            <p className="text-gray-700 dark:text-gray-300 mb-4">
              데이터셋에 노이즈를 추가하여 개별 데이터를 식별할 수 없게 만들면서,
              전체 통계적 패턴은 유지. Apple, Google이 사용자 데이터 수집에 활용.
            </p>

            <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded mb-4">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">수학적 정의 (ε-Differential Privacy)</h4>
              <p className="font-mono text-sm text-gray-800 dark:text-gray-200 mb-2">
                Pr[M(D) ∈ S] ≤ e<sup>ε</sup> * Pr[M(D') ∈ S]
              </p>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>M: 메커니즘 (쿼리 함수)</li>
                <li>D, D': 한 명의 데이터만 다른 데이터셋</li>
                <li>ε (epsilon): 프라이버시 예산 (작을수록 강한 보호, 0.01~10)</li>
                <li>e<sup>ε</sup>: 한 사람의 존재 여부가 결과에 미치는 최대 영향</li>
              </ul>
            </div>

            <div className="bg-gray-50 dark:bg-gray-900/50 p-4 rounded">
              <p className="font-semibold text-gray-900 dark:text-white mb-2">대표 메커니즘</p>
              <ul className="list-disc list-inside text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li><strong>Laplace Mechanism</strong>: 연속값 쿼리에 Laplace 노이즈 추가</li>
                <li><strong>Exponential Mechanism</strong>: 이산 선택 문제 (최빈값 등)</li>
                <li><strong>Gaussian Mechanism</strong>: 여러 쿼리 조합 시 사용</li>
              </ul>
              <p className="text-sm text-gray-700 dark:text-gray-300 mt-2">
                실전 예: Google의 RAPPOR (2014) - Chrome 사용자 통계 수집 시 ε=2 적용
              </p>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg border-l-4 border-green-500">
            <div className="flex items-center gap-3 mb-3">
              <Database className="w-6 h-6 text-green-600" />
              <h3 className="text-2xl font-bold text-gray-900 dark:text-white">2. Federated Learning (연합 학습)</h3>
            </div>

            <p className="text-gray-700 dark:text-gray-300 mb-4">
              데이터를 중앙 서버로 모으지 않고, 각 디바이스에서 모델을 학습한 뒤
              모델 파라미터만 공유. Google Keyboard, Apple Siri가 활용.
            </p>

            <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded mb-4">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">작동 원리 (FedAvg 알고리즘)</h4>
              <ol className="list-decimal list-inside text-sm text-gray-700 dark:text-gray-300 space-y-2">
                <li>서버가 전역 모델 w<sub>t</sub>를 클라이언트들에게 배포</li>
                <li>각 클라이언트 k가 로컬 데이터로 모델 업데이트: w<sub>t+1</sub><sup>k</sup></li>
                <li>클라이언트들이 업데이트된 모델을 서버로 전송</li>
                <li>서버가 가중 평균으로 전역 모델 업데이트:</li>
              </ol>
              <p className="font-mono text-sm text-gray-800 dark:text-gray-200 mt-2">
                w<sub>t+1</sub> = Σ (n<sub>k</sub>/n) * w<sub>t+1</sub><sup>k</sup>
              </p>
              <p className="text-sm text-gray-700 dark:text-gray-300 mt-2">
                (n<sub>k</sub>: 클라이언트 k의 데이터 수, n: 전체 데이터 수)
              </p>
            </div>

            <div className="bg-gray-50 dark:bg-gray-900/50 p-4 rounded">
              <p className="font-semibold text-gray-900 dark:text-white mb-2">실제 활용 사례</p>
              <ul className="list-disc list-inside text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>Google Keyboard: 다음 단어 예측 (2017~)</li>
                <li>Apple iOS: QuickType, Siri 개선 (2019~)</li>
                <li>의료 AI: 병원 간 데이터 공유 없이 질병 예측 모델 학습</li>
                <li>금융: 은행들이 협력하여 사기 탐지 모델 개발 (데이터 유출 방지)</li>
              </ul>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg border-l-4 border-purple-500">
            <div className="flex items-center gap-3 mb-3">
              <Key className="w-6 h-6 text-purple-600" />
              <h3 className="text-2xl font-bold text-gray-900 dark:text-white">3. Homomorphic Encryption (동형 암호화)</h3>
            </div>

            <p className="text-gray-700 dark:text-gray-300 mb-4">
              암호화된 데이터를 복호화하지 않고 직접 연산 가능.
              "암호화된 상태로 AI 추론"을 실현하여 클라우드 AI에서도 프라이버시 보장.
            </p>

            <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded mb-4">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">개념 증명</h4>
              <div className="font-mono text-sm text-gray-800 dark:text-gray-200 space-y-1">
                <p>평문: a = 3, b = 5</p>
                <p>암호화: Enc(a) = X, Enc(b) = Y</p>
                <p>암호화된 연산: X + Y = Enc(a + b) = Enc(8)</p>
                <p>복호화: Dec(Enc(8)) = 8 ✓</p>
              </div>
              <p className="text-sm text-gray-700 dark:text-gray-300 mt-2">
                핵심: 서버는 X, Y가 무엇인지 모르지만 덧셈 가능
              </p>
            </div>

            <div className="bg-gray-50 dark:bg-gray-900/50 p-4 rounded">
              <p className="font-semibold text-gray-900 dark:text-white mb-2">3가지 유형</p>
              <ul className="list-disc list-inside text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li><strong>Partially HE</strong>: 덧셈 또는 곱셈 중 하나만 (빠름)</li>
                <li><strong>Somewhat HE</strong>: 제한된 횟수의 덧셈+곱셈 (중간)</li>
                <li><strong>Fully HE (FHE)</strong>: 무제한 연산 (느림, 실용화 진행 중)</li>
              </ul>
              <p className="text-sm text-gray-700 dark:text-gray-300 mt-2">
                현실: FHE는 평문 대비 1000~10000배 느림 (2024 기준). 간단한 모델만 실용적.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* GDPR, CCPA 규제 */}
      <section className="mb-12">
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <UserCheck className="w-8 h-8 text-red-600" />
          글로벌 프라이버시 규제 (GDPR, CCPA, 한국 개인정보보호법)
        </h2>

        <div className="grid md:grid-cols-3 gap-6">
          <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg">
            <h3 className="text-lg font-bold mb-3 text-gray-900 dark:text-white">GDPR (EU, 2018)</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              일반 데이터 보호 규정 - 전 세계 프라이버시 법의 표준
            </p>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
              <li><strong>동의 (Consent)</strong>: 명시적·자발적 동의 필수</li>
              <li><strong>접근권</strong>: 내 데이터 열람 요청 가능</li>
              <li><strong>삭제권 (Right to be Forgotten)</strong>: 데이터 삭제 요청</li>
              <li><strong>이동권</strong>: 데이터 다운로드 및 이전</li>
              <li><strong>설명권</strong>: 자동화 결정에 대한 설명 요구</li>
            </ul>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-3">
              벌금: 최대 €2,000만 또는 전 세계 매출 4%
            </p>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-lg">
            <h3 className="text-lg font-bold mb-3 text-gray-900 dark:text-white">CCPA (미국 캘리포니아, 2020)</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              캘리포니아 소비자 프라이버시법 - 미국 최초 포괄적 프라이버시법
            </p>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
              <li><strong>알 권리</strong>: 수집되는 데이터 공개</li>
              <li><strong>판매 거부권</strong>: "Do Not Sell My Info" 옵션</li>
              <li><strong>삭제권</strong>: 데이터 삭제 요청</li>
              <li><strong>차별 금지</strong>: 권리 행사 시 서비스 차별 불가</li>
            </ul>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-3">
              적용 대상: 연매출 $2,500만 이상 또는 5만 명 이상 데이터 보유
            </p>
          </div>

          <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-lg">
            <h3 className="text-lg font-bold mb-3 text-gray-900 dark:text-white">한국 개인정보보호법 (2011, 2024 개정)</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              개인정보 보호법 + 정보통신망법 통합
            </p>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
              <li><strong>사전 동의</strong>: 수집·이용 목적 명시</li>
              <li><strong>최소 수집</strong>: 필요 최소한의 정보만</li>
              <li><strong>목적 외 사용 금지</strong>: 동의 범위 엄격 적용</li>
              <li><strong>가명·익명 처리</strong>: 과학적 연구 목적 허용</li>
              <li><strong>AI 특별 조항 (2024)</strong>: AI 학습 데이터 사용 규정</li>
            </ul>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-3">
              벌금: 매출액 3% 이하 또는 8억 원 이하
            </p>
          </div>
        </div>
      </section>

      {/* 실전 코드 */}
      <section className="mb-12">
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">실전 코드: TensorFlow Privacy & Federated Learning</h2>

        <div className="mb-6">
          <h3 className="text-xl font-bold mb-3 text-gray-900 dark:text-white">1. Differential Privacy with TensorFlow Privacy</h3>
          <div className="bg-gray-900 dark:bg-gray-950 p-6 rounded-lg overflow-x-auto">
            <pre className="text-sm text-gray-100">
              <code>{`# pip install tensorflow-privacy
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
import tensorflow as tf

# 1단계: 프라이버시 파라미터 설정
l2_norm_clip = 1.0        # Gradient clipping (민감도 제어)
noise_multiplier = 1.1    # 노이즈 강도 (높을수록 강한 프라이버시)
num_microbatches = 256    # Batch를 작게 쪼개서 노이즈 추가
learning_rate = 0.15

# 2단계: DP Optimizer 생성
optimizer = DPKerasSGDOptimizer(
    l2_norm_clip=l2_norm_clip,
    noise_multiplier=noise_multiplier,
    num_microbatches=num_microbatches,
    learning_rate=learning_rate
)

# 3단계: 모델 컴파일 (일반 모델과 동일)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, reduction=tf.losses.Reduction.NONE  # DP 필수!
    ),
    metrics=['accuracy']
)

# 4단계: 학습 (일반 학습과 동일)
model.fit(
    train_data, train_labels,
    epochs=15,
    batch_size=256,
    validation_data=(test_data, test_labels)
)

# 5단계: 프라이버시 예산 계산
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy

eps, delta = compute_dp_sgd_privacy.compute_dp_sgd_privacy(
    n=60000,                      # 전체 데이터 수
    batch_size=256,
    noise_multiplier=1.1,
    epochs=15,
    delta=1e-5                    # δ-DP에서 δ 값
)

print(f'Privacy Budget: ε = {eps:.2f} at δ = {delta}')
# 출력 예: ε = 2.92 (3 이하면 강한 프라이버시 보장)

# 6단계: 프라이버시-정확도 Trade-off 탐색
for noise in [0.5, 1.0, 1.5, 2.0]:
    optimizer = DPKerasSGDOptimizer(
        l2_norm_clip=1.0,
        noise_multiplier=noise,
        num_microbatches=256,
        learning_rate=0.15
    )
    # ... 모델 학습 및 평가
    # → 노이즈 높을수록: ε 낮음 (프라이버시↑), 정확도↓`}</code>
            </pre>
          </div>
        </div>

        <div>
          <h3 className="text-xl font-bold mb-3 text-gray-900 dark:text-white">2. Federated Learning with TensorFlow Federated (TFF)</h3>
          <div className="bg-gray-900 dark:bg-gray-950 p-6 rounded-lg overflow-x-auto">
            <pre className="text-sm text-gray-100">
              <code>{`# pip install tensorflow-federated
import tensorflow_federated as tff
import tensorflow as tf

# 1단계: 모델 생성 함수
def create_keras_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='softmax', input_shape=(784,))
    ])

def model_fn():
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=example_dataset.element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

# 2단계: Federated Averaging 프로세스 생성
iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(1.0)
)

# 3단계: 서버 초기화
state = iterative_process.initialize()

# 4단계: 클라이언트 데이터 준비 (각 클라이언트 = 개별 디바이스)
federated_train_data = [
    client_1_dataset,  # 사용자 1의 로컬 데이터
    client_2_dataset,  # 사용자 2의 로컬 데이터
    # ... 수천~수백만 클라이언트
]

# 5단계: Federated Learning 라운드 실행
NUM_ROUNDS = 10
for round_num in range(NUM_ROUNDS):
    # 무작위로 일부 클라이언트만 선택 (통신 비용 절감)
    sampled_clients = sample_clients(federated_train_data, num_clients=100)

    # 하나의 라운드 실행 (클라이언트 → 서버 → 업데이트)
    state, metrics = iterative_process.next(state, sampled_clients)

    print(f'Round {round_num}: loss={metrics["train"]["loss"]:.4f}, '
          f'accuracy={metrics["train"]["sparse_categorical_accuracy"]:.2%}')

# 6단계: 최종 모델 추출 및 배포
final_model = create_keras_model()
final_model.set_weights(state.model.trainable)

# 7단계: Secure Aggregation 추가 (암호화된 집계)
secure_sum = tff.federated_secure_sum(
    client_values,
    max_value=2**20  # 각 클라이언트 기여도 상한
)
# → 서버도 개별 클라이언트 값을 볼 수 없음 (Cryptographic Security)`}</code>
            </pre>
          </div>
        </div>
      </section>

      {/* 2024 데이터 유출 사례 */}
      <section className="mb-12">
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">2024-2025 주요 데이터 유출 사례</h2>

        <div className="space-y-4">
          <div className="bg-red-50 dark:bg-red-900/20 p-6 rounded-lg border-l-4 border-red-500">
            <h3 className="text-lg font-bold mb-2 text-gray-900 dark:text-white">OpenAI ChatGPT 대화 내역 노출 (2024.03)</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              Redis 버그로 일부 사용자가 타인의 채팅 제목 열람 가능. 결제 정보 4자리도 노출.
            </p>
            <p className="text-sm font-semibold text-gray-900 dark:text-white">교훈: 캐싱 레이어 보안 철저, 민감 정보 최소화</p>
          </div>

          <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-lg border-l-4 border-orange-500">
            <h3 className="text-lg font-bold mb-2 text-gray-900 dark:text-white">Samsung 직원 ChatGPT 사내 정보 유출 (2023)</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              반도체 설계 코드 등 기밀을 ChatGPT에 입력 → OpenAI 학습 데이터 포함 우려
            </p>
            <p className="text-sm font-semibold text-gray-900 dark:text-white">대응: 기업용 AI는 데이터 학습 옵트아웃 필수</p>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 p-6 rounded-lg border-l-4 border-yellow-500">
            <h3 className="text-lg font-bold mb-2 text-gray-900 dark:text-white">Google Bard 프롬프트 인젝션 (2024.02)</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              악의적 프롬프트로 시스템 프롬프트 추출 및 제약 우회 성공
            </p>
            <p className="text-sm font-semibold text-gray-900 dark:text-white">대책: Prompt firewall, Output filtering 강화</p>
          </div>
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 프라이버시 기술 프레임워크',
            icon: 'docs' as const,
            color: 'border-rose-500',
            items: [
              {
                title: 'The Algorithmic Foundations of Differential Privacy (Book)',
                url: 'https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf',
                description: 'Dwork & Roth의 DP 교과서 - 이론적 기초 (무료 공개)'
              },
              {
                title: 'Federated Learning: Challenges, Methods, and Future Directions',
                url: 'https://arxiv.org/abs/1908.07873',
                description: '연합학습 종합 서베이 논문 (CMU, Google, 2019)'
              },
              {
                title: 'Homomorphic Encryption Standard',
                url: 'http://homomorphicencryption.org/',
                description: 'HE 표준화 컨소시엄 - Microsoft, IBM, Intel 참여'
              },
              {
                title: 'NIST Privacy Framework',
                url: 'https://www.nist.gov/privacy-framework',
                description: '미국 표준기술연구소의 프라이버시 위험 관리 프레임워크'
              }
            ]
          },
          {
            title: '🛠️ 실전 도구',
            icon: 'tools' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'TensorFlow Privacy',
                url: 'https://github.com/tensorflow/privacy',
                description: 'DP-SGD, Privacy Accounting 등 Google 오픈소스'
              },
              {
                title: 'TensorFlow Federated',
                url: 'https://www.tensorflow.org/federated',
                description: 'Federated Learning 공식 라이브러리'
              },
              {
                title: 'OpenDP',
                url: 'https://opendp.org/',
                description: 'Harvard - 모듈식 DP 라이브러리 (Python/Rust)'
              },
              {
                title: 'PySyft',
                url: 'https://github.com/OpenMined/PySyft',
                description: 'OpenMined - Federated Learning + DP + HE 통합'
              },
              {
                title: 'Microsoft SEAL',
                url: 'https://www.microsoft.com/en-us/research/project/microsoft-seal/',
                description: 'Fully Homomorphic Encryption 라이브러리 (C++)'
              }
            ]
          },
          {
            title: '⚖️ 법규 및 규제',
            icon: 'research' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'GDPR Official Text',
                url: 'https://gdpr-info.eu/',
                description: 'EU 일반 데이터 보호 규정 전문 (2018)'
              },
              {
                title: 'CCPA Official Portal',
                url: 'https://oag.ca.gov/privacy/ccpa',
                description: '캘리포니아 소비자 프라이버시법 가이드'
              },
              {
                title: '한국 개인정보보호위원회',
                url: 'https://www.pipc.go.kr/',
                description: '개인정보보호법 해석 및 가이드라인'
              },
              {
                title: 'AI & Privacy: 2024 Legal Landscape',
                url: 'https://iapp.org/resources/article/ai-privacy-2024/',
                description: 'IAPP (국제 프라이버시 전문가 협회) - 최신 동향'
              },
              {
                title: 'EU AI Act - Data Governance Requirements',
                url: 'https://artificialintelligenceact.eu/data-governance/',
                description: 'EU AI Act의 데이터 거버넌스 조항 (2024.08)'
              }
            ]
          }
        ]}
      />
    </div>
  );
}
