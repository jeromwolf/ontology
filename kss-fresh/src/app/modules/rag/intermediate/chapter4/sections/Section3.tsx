'use client'

import { Cpu } from 'lucide-react'

export default function Section3() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-indigo-100 dark:bg-indigo-900/20 flex items-center justify-center">
          <Cpu className="text-indigo-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">4.3 모델 양자화 및 최적화</h2>
          <p className="text-gray-600 dark:text-gray-400">메모리와 연산 효율성 극대화</p>
        </div>
      </div>

      <div className="space-y-6">
        <div className="bg-indigo-50 dark:bg-indigo-900/20 p-6 rounded-xl border border-indigo-200 dark:border-indigo-700">
          <h3 className="font-bold text-indigo-800 dark:text-indigo-200 mb-4">모델 압축 기법</h3>

          <div className="prose prose-sm dark:prose-invert mb-4">
            <p className="text-gray-700 dark:text-gray-300">
              <strong>모델 양자화는 메모리 사용량을 줄이고 추론 속도를 향상시키는 강력한 기술입니다.</strong>
              32비트 부동소수점을 8비트 정수로 변환하여 모델 크기를 4분의 1로 줄이면서도 성능 저하는 최소화합니다.
            </p>
            <p className="text-gray-700 dark:text-gray-300">
              <strong>주요 압축 기법:</strong>
            </p>
            <ul className="list-disc list-inside text-gray-700 dark:text-gray-300 space-y-1">
              <li><strong>동적 양자화</strong>: 실행 시 가중치를 int8로 변환 (가장 쉬운 방법)</li>
              <li><strong>정적 양자화</strong>: 사전에 캘리브레이션하여 더 높은 압축률 달성</li>
              <li><strong>ONNX 변환</strong>: 프레임워크 독립적이고 최적화된 포맷</li>
              <li><strong>프루닝</strong>: 중요하지 않은 연결을 제거하여 희소 모델 생성</li>
            </ul>
            <div className="bg-yellow-100 dark:bg-yellow-900/20 p-3 rounded-lg mt-3">
              <p className="text-sm text-yellow-800 dark:text-yellow-200">
                <strong>💡 실무 팁:</strong> 양자화 시 정확도를 반드시 검증하세요. 일반적으로 2-5% 정도의 성능 하락은
                4배의 속도 향상을 위한 합리적인 트레이드오프입니다.
              </p>
            </div>
          </div>

          <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg overflow-hidden border border-slate-200 dark:border-slate-700">
            <pre className="text-sm text-slate-800 dark:text-slate-200 overflow-x-auto max-h-96 overflow-y-auto font-mono">
{`import torch
import torch.quantization as quant
from transformers import AutoModel, AutoTokenizer
import onnx
import onnxruntime as ort

class ModelOptimizer:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def quantize_dynamic(self, output_path: str):
        """동적 양자화 (int8)"""
        print("동적 양자화 시작...")

        # PyTorch 동적 양자화
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear},  # 선형 레이어만 양자화
            dtype=torch.qint8   # int8로 압축
        )

        # 모델 저장
        torch.save(quantized_model.state_dict(), f"{output_path}/quantized_model.pt")

        # 메모리 사용량 비교
        original_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())

        compression_ratio = original_size / quantized_size

        return {
            "original_size_mb": original_size / (1024**2),
            "quantized_size_mb": quantized_size / (1024**2),
            "compression_ratio": compression_ratio,
            "model": quantized_model
        }

    def convert_to_onnx(self, output_path: str, optimize: bool = True):
        """ONNX 변환 및 최적화"""
        print("ONNX 변환 시작...")

        # 더미 입력 생성
        dummy_input = torch.randint(0, 1000, (1, 512))  # [batch_size, seq_len]

        # ONNX 변환
        torch.onnx.export(
            self.model,
            dummy_input,
            f"{output_path}/model.onnx",
            input_names=['input_ids'],
            output_names=['last_hidden_state'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'last_hidden_state': {0: 'batch_size', 1: 'sequence_length'}
            },
            opset_version=14
        )

        if optimize:
            # ONNX 그래프 최적화
            from onnxruntime.tools import optimizer
            optimized_model = optimizer.optimize_model(
                f"{output_path}/model.onnx",
                model_type='bert',
                use_gpu=False,
                opt_level=99  # 최대 최적화
            )
            optimized_model.save_model_to_file(f"{output_path}/optimized_model.onnx")

        return f"{output_path}/optimized_model.onnx" if optimize else f"{output_path}/model.onnx"

    def benchmark_models(self, test_queries: List[str]):
        """모델 성능 벤치마크"""
        results = {}

        # 원본 모델 벤치마크
        print("원본 모델 벤치마크...")
        original_times = []
        for query in test_queries:
            start_time = time.time()
            inputs = self.tokenizer(query, return_tensors='pt', truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            original_times.append(time.time() - start_time)

        results['original'] = {
            'avg_time': np.mean(original_times),
            'std_time': np.std(original_times)
        }

        # 양자화된 모델 벤치마크
        quantized_info = self.quantize_dynamic("./temp")
        quantized_model = quantized_info['model']

        print("양자화된 모델 벤치마크...")
        quantized_times = []
        for query in test_queries:
            start_time = time.time()
            inputs = self.tokenizer(query, return_tensors='pt', truncation=True)
            with torch.no_grad():
                outputs = quantized_model(**inputs)
            quantized_times.append(time.time() - start_time)

        results['quantized'] = {
            'avg_time': np.mean(quantized_times),
            'std_time': np.std(quantized_times),
            'compression_ratio': quantized_info['compression_ratio']
        }

        # ONNX 모델 벤치마크
        onnx_path = self.convert_to_onnx("./temp")
        session = ort.InferenceSession(onnx_path)

        print("ONNX 모델 벤치마크...")
        onnx_times = []
        for query in test_queries:
            start_time = time.time()
            inputs = self.tokenizer(query, return_tensors='np', truncation=True)
            outputs = session.run(None, {'input_ids': inputs['input_ids']})
            onnx_times.append(time.time() - start_time)

        results['onnx'] = {
            'avg_time': np.mean(onnx_times),
            'std_time': np.std(onnx_times)
        }

        return results

# 사용 예시
optimizer = ModelOptimizer("sentence-transformers/all-MiniLM-L6-v2")

# 테스트 쿼리
test_queries = [
    "인공지능의 기본 원리는 무엇인가요?",
    "머신러닝과 딥러닝의 차이점을 설명해주세요.",
    "자연어 처리의 주요 기술들은?",
    "컴퓨터 비전의 응용 분야는?",
    "강화학습의 핵심 개념은?"
] * 20  # 100개 쿼리

# 벤치마크 실행
benchmark_results = optimizer.benchmark_models(test_queries)

print("\\n=== 모델 성능 비교 ===")
for model_type, metrics in benchmark_results.items():
    print(f"{model_type.upper()}:")
    print(f"  평균 처리 시간: {metrics['avg_time']:.4f}초")
    print(f"  표준 편차: {metrics['std_time']:.4f}초")
    if 'compression_ratio' in metrics:
        print(f"  압축 비율: {metrics['compression_ratio']:.2f}x")
    print()`}
            </pre>
          </div>
        </div>

        <div className="bg-cyan-50 dark:bg-cyan-900/20 p-6 rounded-xl border border-cyan-200 dark:border-cyan-700">
          <h3 className="font-bold text-cyan-800 dark:text-cyan-200 mb-4">모델 양자화 벤치마크 결과</h3>

          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-gray-900 dark:text-white mb-3">🎯 실제 모델별 압축 효과</h4>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">BERT-base (110M 파라미터)</p>
                  <div className="space-y-1 text-xs">
                    <div className="flex justify-between">
                      <span>원본 크기:</span>
                      <span className="font-mono">438MB</span>
                    </div>
                    <div className="flex justify-between">
                      <span>INT8 양자화:</span>
                      <span className="font-mono text-green-600">112MB (4x 압축)</span>
                    </div>
                    <div className="flex justify-between">
                      <span>속도 향상:</span>
                      <span className="font-mono text-blue-600">2.3x 빠름</span>
                    </div>
                    <div className="flex justify-between">
                      <span>정확도 손실:</span>
                      <span className="font-mono text-orange-600">-1.2%</span>
                    </div>
                  </div>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">T5-base (220M 파라미터)</p>
                  <div className="space-y-1 text-xs">
                    <div className="flex justify-between">
                      <span>원본 크기:</span>
                      <span className="font-mono">892MB</span>
                    </div>
                    <div className="flex justify-between">
                      <span>INT8 양자화:</span>
                      <span className="font-mono text-green-600">230MB (3.9x 압축)</span>
                    </div>
                    <div className="flex justify-between">
                      <span>속도 향상:</span>
                      <span className="font-mono text-blue-600">2.8x 빠름</span>
                    </div>
                    <div className="flex justify-between">
                      <span>정확도 손실:</span>
                      <span className="font-mono text-orange-600">-2.5%</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-gray-900 dark:text-white mb-2">💾 메모리 효율성 개선</h4>
              <div className="relative h-4 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                <div className="absolute left-0 top-0 h-full bg-red-500 flex items-center justify-center text-xs text-white font-bold" style={{width: '100%'}}>
                  원본: 4GB RAM 필요
                </div>
              </div>
              <div className="relative h-4 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden mt-2">
                <div className="absolute left-0 top-0 h-full bg-green-500 flex items-center justify-center text-xs text-white font-bold" style={{width: '25%'}}>
                  양자화: 1GB RAM
                </div>
              </div>
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
                모바일 디바이스에서도 실행 가능한 수준으로 메모리 사용량 감소
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
