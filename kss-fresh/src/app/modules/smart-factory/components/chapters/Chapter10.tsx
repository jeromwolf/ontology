'use client';

import {
  Eye, Brain, BarChart3, Target, Code
} from 'lucide-react';
import CodeEditor from '../CodeEditor';
import Link from 'next/link';
import References from '@/components/common/References';

export default function Chapter10() {
  return (
    <div className="space-y-8">
      <div className="grid lg:grid-cols-2 gap-8">
        <div className="bg-white dark:bg-gray-800 p-6 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <Eye className="w-6 h-6 text-slate-600" />
            머신 비전 시스템 구성
          </h3>
          <div className="space-y-4">
            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-400 rounded">
              <h4 className="font-semibold text-blue-800 dark:text-blue-300 mb-2">조명 시스템 (Lighting)</h4>
              <ul className="text-sm text-blue-700 dark:text-blue-400 space-y-1">
                <li>• <strong>백라이트:</strong> 외곽선, 구멍, 투명도 검사</li>
                <li>• <strong>링라이트:</strong> 표면 결함, 스크래치 검출</li>
                <li>• <strong>돔라이트:</strong> 반사 제거, 균일한 조명</li>
                <li>• <strong>동축조명:</strong> 평면 반사체 검사</li>
              </ul>
            </div>
            
            <div className="p-4 bg-green-50 dark:bg-green-900/20 border-l-4 border-green-400 rounded">
              <h4 className="font-semibold text-green-800 dark:text-green-300 mb-2">렌즈 및 카메라</h4>
              <ul className="text-sm text-green-700 dark:text-green-400 space-y-1">
                <li>• <strong>해상도:</strong> 최소 결함 크기의 3-5배</li>
                <li>• <strong>시야각(FOV):</strong> 검사 영역 최적화</li>
                <li>• <strong>작업거리:</strong> 렌즈-객체 간 거리 설정</li>
                <li>• <strong>센서 타입:</strong> CCD vs CMOS 선택</li>
              </ul>
            </div>
            
            <div className="p-4 bg-purple-50 dark:bg-purple-900/20 border-l-4 border-purple-400 rounded">
              <h4 className="font-semibold text-purple-800 dark:text-purple-300 mb-2">이미지 처리 알고리즘</h4>
              <ul className="text-sm text-purple-700 dark:text-purple-400 space-y-1">
                <li>• <strong>전처리:</strong> 노이즈 제거, 히스토그램 평활화</li>
                <li>• <strong>특징 추출:</strong> 에지, 텍스처, 컬러 분석</li>
                <li>• <strong>패턴 매칭:</strong> 템플릿 매칭, 정규화</li>
                <li>• <strong>분류:</strong> SVM, CNN, Random Forest</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 p-6 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <Brain className="w-6 h-6 text-slate-600" />
            AI 기반 결함 검출 모델
          </h3>
          <div className="space-y-4">
            <div className="p-4 bg-red-50 dark:bg-red-900/20 border rounded">
              <h4 className="font-semibold text-red-800 dark:text-red-300 mb-2">YOLO (You Only Look Once)</h4>
              <div className="text-sm text-red-700 dark:text-red-400 space-y-2">
                <p><strong>장점:</strong> 실시간 객체 검출, 높은 속도</p>
                <p><strong>용도:</strong> 표면 결함, 이물질, 치수 이상</p>
                <p><strong>정확도:</strong> mAP 85-92% (YOLOv8 기준)</p>
              </div>
            </div>
            
            <div className="p-4 bg-orange-50 dark:bg-orange-900/20 border rounded">
              <h4 className="font-semibold text-orange-800 dark:text-orange-300 mb-2">Faster R-CNN</h4>
              <div className="text-sm text-orange-700 dark:text-orange-400 space-y-2">
                <p><strong>장점:</strong> 높은 정확도, 정밀한 위치 검출</p>
                <p><strong>용도:</strong> 복잡한 결함, 다중 클래스 분류</p>
                <p><strong>정확도:</strong> mAP 90-95%, 속도 50-100ms</p>
              </div>
            </div>
            
            <div className="p-4 bg-green-50 dark:bg-green-900/20 border rounded">
              <h4 className="font-semibold text-green-800 dark:text-green-300 mb-2">Semantic Segmentation</h4>
              <div className="text-sm text-green-700 dark:text-green-400 space-y-2">
                <p><strong>장점:</strong> 픽셀 단위 정밀 분석</p>
                <p><strong>용도:</strong> 복잡한 형상, 불규칙 결함</p>
                <p><strong>모델:</strong> U-Net, DeepLab, FCN</p>
              </div>
            </div>
            
            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border rounded">
              <h4 className="font-semibold text-blue-800 dark:text-blue-300 mb-2">Anomaly Detection</h4>
              <div className="text-sm text-blue-700 dark:text-blue-400 space-y-2">
                <p><strong>장점:</strong> 사전 학습 없는 이상 탐지</p>
                <p><strong>용도:</strong> 새로운 결함 유형 발견</p>
                <p><strong>방법:</strong> VAE, GAN, One-Class SVM</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 시뮬레이터 체험 섹션 */}
      <div className="mt-8 p-6 bg-gradient-to-r from-rose-50 to-pink-50 dark:from-rose-900/20 dark:to-pink-900/20 rounded-xl border border-rose-200 dark:border-rose-800">
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-rose-900 dark:text-rose-200 mb-2">
            🎮 AI 품질 관리 시뮬레이터 체험
          </h3>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-rose-800/20 p-4 rounded-lg border border-rose-200 dark:border-rose-700">
              <h4 className="font-medium text-rose-800 dark:text-rose-300 mb-2">품질 관리 비전 시스템</h4>
              <p className="text-sm text-rose-700 dark:text-rose-400 mb-3">
                AI 기반 실시간 불량품 검출 시스템을 체험해보세요.
              </p>
              <Link
                href="/modules/smart-factory/simulators/quality-control-vision?from=/modules/smart-factory/quality-management-ai"
                className="inline-flex items-center gap-2 px-3 py-2 bg-rose-600 hover:bg-rose-700 text-white rounded-lg transition-colors text-sm"
              >
                <span>시뮬레이터 열기</span>
                <span>→</span>
              </Link>
            </div>
            <div className="bg-white dark:bg-pink-800/20 p-4 rounded-lg border border-pink-200 dark:border-pink-700">
              <h4 className="font-medium text-pink-800 dark:text-pink-300 mb-2">SPC 통계 공정 관리</h4>
              <p className="text-sm text-pink-700 dark:text-pink-400 mb-3">
                실시간 품질 데이터 분석과 관리도를 체험해보세요.
              </p>
              <Link
                href="/modules/smart-factory/simulators/spc-control-system?from=/modules/smart-factory/quality-management-ai"
                className="inline-flex items-center gap-2 px-3 py-2 bg-pink-600 hover:bg-pink-700 text-white rounded-lg transition-colors text-sm"
              >
                <span>시뮬레이터 열기</span>
                <span>→</span>
              </Link>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 p-8 rounded-xl border border-purple-200 dark:border-purple-800">
        <h3 className="text-2xl font-bold text-purple-900 dark:text-purple-200 mb-6 flex items-center gap-3">
          <BarChart3 className="w-8 h-8" />
          SPC (Statistical Process Control) 시스템
        </h3>
        <div className="grid md:grid-cols-3 gap-6">
          <div className="bg-white dark:bg-purple-800/30 p-6 rounded-lg border border-purple-300 dark:border-purple-600">
            <h4 className="font-bold text-purple-800 dark:text-purple-200 mb-4">관리도 종류</h4>
            <div className="space-y-3 text-sm">
              <div className="p-3 bg-purple-100 dark:bg-purple-700/30 rounded">
                <h5 className="font-semibold text-purple-700 dark:text-purple-300">Xbar-R Chart</h5>
                <p className="text-purple-600 dark:text-purple-400">연속 변수, 공정 평균과 범위 관리</p>
              </div>
              <div className="p-3 bg-purple-100 dark:bg-purple-700/30 rounded">
                <h5 className="font-semibold text-purple-700 dark:text-purple-300">p-Chart</h5>
                <p className="text-purple-600 dark:text-purple-400">불량률 관리, 샘플 크기 가변</p>
              </div>
              <div className="p-3 bg-purple-100 dark:bg-purple-700/30 rounded">
                <h5 className="font-semibold text-purple-700 dark:text-purple-300">c-Chart</h5>
                <p className="text-purple-600 dark:text-purple-400">단위당 결점 수 관리</p>
              </div>
              <div className="p-3 bg-purple-100 dark:bg-purple-700/30 rounded">
                <h5 className="font-semibold text-purple-700 dark:text-purple-300">CUSUM</h5>
                <p className="text-purple-600 dark:text-purple-400">누적합 관리도, 미세 변화 검출</p>
              </div>
            </div>
          </div>
          
          <div className="bg-white dark:bg-purple-800/30 p-6 rounded-lg border border-purple-300 dark:border-purple-600">
            <h4 className="font-bold text-purple-800 dark:text-purple-200 mb-4">관리 한계선</h4>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between items-center p-2 bg-red-100 dark:bg-red-800/30 rounded">
                <span className="text-red-700 dark:text-red-300">UCL (상한선)</span>
                <span className="font-mono text-red-600 dark:text-red-400">μ + 3σ</span>
              </div>
              <div className="flex justify-between items-center p-2 bg-green-100 dark:bg-green-800/30 rounded">
                <span className="text-green-700 dark:text-green-300">CL (중심선)</span>
                <span className="font-mono text-green-600 dark:text-green-400">μ</span>
              </div>
              <div className="flex justify-between items-center p-2 bg-red-100 dark:bg-red-800/30 rounded">
                <span className="text-red-700 dark:text-red-300">LCL (하한선)</span>
                <span className="font-mono text-red-600 dark:text-red-400">μ - 3σ</span>
              </div>
              <div className="p-3 bg-yellow-100 dark:bg-yellow-800/30 rounded">
                <h5 className="font-semibold text-yellow-700 dark:text-yellow-300 mb-1">이상 징후 패턴</h5>
                <ul className="text-xs text-yellow-600 dark:text-yellow-400">
                  <li>• 연속 7점이 중심선 한쪽에 위치</li>
                  <li>• 연속 2점이 관리한계선 밖에 위치</li>
                  <li>• 증가/감소 트렌드 지속</li>
                </ul>
              </div>
            </div>
          </div>
          
          <div className="bg-white dark:bg-purple-800/30 p-6 rounded-lg border border-purple-300 dark:border-purple-600">
            <h4 className="font-bold text-purple-800 dark:text-purple-200 mb-4">공정 능력 지수</h4>
            <div className="space-y-3 text-sm">
              <div className="p-3 bg-purple-100 dark:bg-purple-700/30 rounded">
                <h5 className="font-semibold text-purple-700 dark:text-purple-300">Cp (공정 능력)</h5>
                <p className="font-mono text-purple-600 dark:text-purple-400">Cp = (USL - LSL) / 6σ</p>
                <p className="text-xs text-purple-500 dark:text-purple-400">≥ 1.33 우수, ≥ 1.67 세계적</p>
              </div>
              <div className="p-3 bg-purple-100 dark:bg-purple-700/30 rounded">
                <h5 className="font-semibold text-purple-700 dark:text-purple-300">Cpk (공정 성능)</h5>
                <p className="font-mono text-purple-600 dark:text-purple-400">Cpk = min(Cpu, Cpl)</p>
                <p className="text-xs text-purple-500 dark:text-purple-400">중심 이동 고려한 실제 능력</p>
              </div>
              <div className="p-3 bg-purple-100 dark:bg-purple-700/30 rounded">
                <h5 className="font-semibold text-purple-700 dark:text-purple-300">Pp/Ppk (성능 지수)</h5>
                <p className="font-mono text-purple-600 dark:text-purple-400">전체 변동 기준</p>
                <p className="text-xs text-purple-500 dark:text-purple-400">장기 공정 성능 평가</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-800 p-8 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">
        <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Target className="w-8 h-8 text-amber-600" />
          6시그마 DMAIC 방법론
        </h3>
        <div className="grid md:grid-cols-5 gap-6">
          <div className="text-center">
            <div className="w-16 h-16 bg-blue-500 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-white font-bold text-lg">D</span>
            </div>
            <h4 className="font-bold text-gray-900 dark:text-white mb-2">Define</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">문제 정의</p>
            <ul className="text-xs text-gray-500 dark:text-gray-500 space-y-1">
              <li>• 고객 요구사항 분석</li>
              <li>• 프로젝트 범위 설정</li>
              <li>• CTQ 정의</li>
            </ul>
          </div>
          
          <div className="text-center">
            <div className="w-16 h-16 bg-green-500 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-white font-bold text-lg">M</span>
            </div>
            <h4 className="font-bold text-gray-900 dark:text-white mb-2">Measure</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">측정</p>
            <ul className="text-xs text-gray-500 dark:text-gray-500 space-y-1">
              <li>• 현재 성능 측정</li>
              <li>• 데이터 수집 계획</li>
              <li>• MSA 실시</li>
            </ul>
          </div>
          
          <div className="text-center">
            <div className="w-16 h-16 bg-yellow-500 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-white font-bold text-lg">A</span>
            </div>
            <h4 className="font-bold text-gray-900 dark:text-white mb-2">Analyze</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">분석</p>
            <ul className="text-xs text-gray-500 dark:text-gray-500 space-y-1">
              <li>• 근본 원인 분석</li>
              <li>• 통계적 분석</li>
              <li>• 가설 검정</li>
            </ul>
          </div>
          
          <div className="text-center">
            <div className="w-16 h-16 bg-purple-500 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-white font-bold text-lg">I</span>
            </div>
            <h4 className="font-bold text-gray-900 dark:text-white mb-2">Improve</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">개선</p>
            <ul className="text-xs text-gray-500 dark:text-gray-500 space-y-1">
              <li>• 개선안 도출</li>
              <li>• 파일럿 테스트</li>
              <li>• DOE 실시</li>
            </ul>
          </div>
          
          <div className="text-center">
            <div className="w-16 h-16 bg-red-500 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-white font-bold text-lg">C</span>
            </div>
            <h4 className="font-bold text-gray-900 dark:text-white mb-2">Control</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">통제</p>
            <ul className="text-xs text-gray-500 dark:text-gray-500 space-y-1">
              <li>• 관리 시스템 구축</li>
              <li>• 표준화</li>
              <li>• 지속적 모니터링</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="bg-green-50 dark:bg-green-900/20 p-8 rounded-xl border border-green-200 dark:border-green-800">
        <h3 className="text-2xl font-bold text-green-900 dark:text-green-200 mb-6 flex items-center gap-3">
          <Code className="w-8 h-8" />
          실습: OpenCV & YOLO 제품 결함 분류 시스템
        </h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-green-800/30 p-6 rounded-lg">
            <h4 className="font-bold text-green-800 dark:text-green-200 mb-4">1단계: 이미지 전처리</h4>
            <CodeEditor 
              code={`import cv2
import numpy as np

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    
    # 가우시안 블러로 노이즈 제거
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # 히스토그램 평활화
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0).apply(l)
    img = cv2.merge([l, a, b])
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    
    return img`}
              language="python"
              title="1단계: 이미지 전처리"
              filename="image_preprocessing.py"
              maxHeight="350px"
            />
          </div>
          
          <div className="bg-white dark:bg-green-800/30 p-6 rounded-lg">
            <h4 className="font-bold text-green-800 dark:text-green-200 mb-4">2단계: YOLO 모델 설정</h4>
            <CodeEditor 
              code={`from ultralytics import YOLO

# 사전 훈련된 모델 로드
model = YOLO('yolov8n.pt')

# 커스텀 데이터셋으로 파인튜닝
model.train(
    data='defect_dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='defect_detector'
)

# 추론 실행
results = model.predict('test_image.jpg')

# 결과 처리
for result in results:
    boxes = result.boxes
    for box in boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        print(f"결함 클래스: {class_id}, 신뢰도: {confidence:.2f}")`}
              language="python"
              title="2단계: YOLO 모델 설정"
              filename="yolo_defect_detection.py"
              maxHeight="350px"
            />
          </div>
          
          <div className="bg-white dark:bg-green-800/30 p-6 rounded-lg">
            <h4 className="font-bold text-green-800 dark:text-green-200 mb-4">3단계: SPC 차트 생성</h4>
            <CodeEditor 
              code={`import matplotlib.pyplot as plt
import numpy as np

def create_spc_chart(data, title="SPC Chart"):
    # 평균과 표준편차 계산
    mean = np.mean(data)
    std = np.std(data)
    
    # 관리 한계선 계산
    ucl = mean + 3 * std
    lcl = mean - 3 * std
    
    # 차트 그리기
    plt.figure(figsize=(12, 6))
    plt.plot(data, 'b-o', markersize=4)
    plt.axhline(y=mean, color='g', linestyle='-', label='CL')
    plt.axhline(y=ucl, color='r', linestyle='--', label='UCL')
    plt.axhline(y=lcl, color='r', linestyle='--', label='LCL')
    
    # 관리 한계 벗어난 점 표시
    out_of_control = (data > ucl) | (data < lcl)
    plt.scatter(np.where(out_of_control)[0], 
               data[out_of_control], 
               color='red', s=50, zorder=5)
    
    plt.title(title)
    plt.xlabel('Sample Number')
    plt.ylabel('Measurement')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()`}
              language="python"
              title="3단계: SPC 차트 생성"
              filename="spc_chart.py"
              maxHeight="350px"
            />
          </div>
          
          <div className="bg-white dark:bg-green-800/30 p-6 rounded-lg">
            <h4 className="font-bold text-green-800 dark:text-green-200 mb-4">4단계: 통합 품질 시스템</h4>
            <CodeEditor 
              code={`class QualityControlSystem:
    def __init__(self):
        self.yolo_model = YOLO('defect_detector.pt')
        self.spc_data = []
        
    def inspect_product(self, image_path):
        # YOLO로 결함 검출
        results = self.yolo_model.predict(image_path)
        
        defects = []
        for result in results:
            for box in result.boxes:
                defect = {
                    'class': int(box.cls[0]),
                    'confidence': float(box.conf[0]),
                    'bbox': box.xyxy[0].tolist()
                }
                defects.append(defect)
        
        # SPC 데이터 업데이트
        defect_count = len(defects)
        self.spc_data.append(defect_count)
        
        # 품질 판정
        if defect_count == 0:
            status = "PASS"
        elif any(d['confidence'] > 0.8 for d in defects):
            status = "REJECT"
        else:
            status = "REVIEW"
            
        return {
            'status': status,
            'defects': defects,
            'spc_alert': self.check_spc_control()
        }
    
    def check_spc_control(self):
        if len(self.spc_data) < 7:
            return False
            
        recent_data = self.spc_data[-7:]
        mean = np.mean(self.spc_data)
        
        # 7점 연속 중심선 한쪽 체크
        return all(x > mean for x in recent_data) or \\
               all(x < mean for x in recent_data)`}
              language="python"
              title="4단계: 통합 품질 시스템"
              filename="quality_control_system.py"
              maxHeight="350px"
            />
          </div>
        </div>
      </div>

      <References
        sections={[
          {
            title: '📚 공식 표준 & 문서',
            icon: 'web' as const,
            color: 'border-emerald-500',
            items: [
              {
                title: 'ISO 9001:2015 - Quality Management Systems',
                url: 'https://www.iso.org/iso-9001-quality-management.html',
                description: '품질 경영 시스템 국제 표준 - 프로세스 관리 및 지속적 개선 요구사항'
              },
              {
                title: 'ASQ Six Sigma Handbook',
                url: 'https://asq.org/quality-resources/six-sigma',
                description: '미국 품질협회(ASQ)의 Six Sigma 공식 가이드북 - DMAIC, 통계적 도구, 사례'
              },
              {
                title: 'ISO 3951 - Statistical Sampling Procedures',
                url: 'https://www.iso.org/standard/53467.html',
                description: '계량형 샘플링 검사 절차 - 평균, 표준편차 기준 품질 관리'
              },
              {
                title: 'NIST Engineering Statistics Handbook',
                url: 'https://www.itl.nist.gov/div898/handbook/',
                description: '미국 국립표준기술연구소의 공정 관리 통계 기법 완벽 가이드 - SPC, DOE 포함'
              },
              {
                title: 'AIAG MSA Manual (Measurement System Analysis)',
                url: 'https://www.aiag.org/quality/automotive-core-tools/msa',
                description: '자동차산업협회의 측정 시스템 분석 매뉴얼 - 게이지 R&R 방법론'
              }
            ]
          },
          {
            title: '🔬 핵심 논문 & 연구',
            icon: 'research' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'Deep Learning for Defect Detection in Manufacturing (IEEE Transactions, 2020)',
                url: 'https://ieeexplore.ieee.org/document/9088181',
                description: 'CNN 기반 제조 결함 검출 시스템 - 98.7% 정확도, 실시간 처리 (15ms/frame)'
              },
              {
                title: 'YOLO-based Surface Defect Detection (Journal of Manufacturing Systems, 2021)',
                url: 'https://www.sciencedirect.com/science/article/abs/pii/S0278612521000868',
                description: 'YOLOv5를 활용한 표면 결함 실시간 검출 시스템 - 금속, 직물, PCB 적용'
              },
              {
                title: 'Anomaly Detection in Manufacturing using VAE (Nature Scientific Reports, 2022)',
                url: 'https://www.nature.com/articles/s41598-022-08892-w',
                description: 'Variational Autoencoder 기반 비지도 학습 이상 탐지 - 새로운 결함 유형 자동 발견'
              },
              {
                title: 'Transfer Learning for Industrial Quality Inspection (Computers in Industry, 2021)',
                url: 'https://www.sciencedirect.com/science/article/abs/pii/S0166361521000403',
                description: '소량 데이터 환경에서 전이 학습을 활용한 품질 검사 시스템 구축 연구'
              }
            ]
          },
          {
            title: '🛠️ 실전 도구 & 플랫폼',
            icon: 'tools' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'Cognex Vision Systems - In-Sight & VisionPro',
                url: 'https://www.cognex.com/products/machine-vision',
                description: '산업용 머신 비전 시스템 세계 1위 - AOI, OCR, 바코드 읽기, 결함 검출 통합'
              },
              {
                title: 'Keyence CV-X Series - AI Vision System',
                url: 'https://www.keyence.com/products/vision/vision-sys/cv-x/',
                description: 'AI 기반 산업용 비전 검사 시스템 - 딥러닝 통합, 조명-카메라-제어 일체형'
              },
              {
                title: 'Ultralytics YOLOv8 - Object Detection',
                url: 'https://github.com/ultralytics/ultralytics',
                description: '최신 YOLO 모델 오픈소스 - 실시간 결함 검출에 최적화, Python API 제공'
              },
              {
                title: 'OpenCV - Computer Vision Library',
                url: 'https://opencv.org/',
                description: '오픈소스 컴퓨터 비전 라이브러리 - 이미지 전처리, 특징 추출, 패턴 매칭 기능'
              },
              {
                title: 'Minitab - Statistical Process Control Software',
                url: 'https://www.minitab.com/en-us/products/minitab/',
                description: '통계 공정 관리(SPC) 전문 소프트웨어 - 관리도, 공정 능력 분석, Six Sigma 도구'
              }
            ]
          }
        ]}
      />
    </div>
  );
}