'use client'

export default function Chapter3() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          AI & 딥러닝 응용
        </h2>
        
        <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 rounded-xl p-6 mb-6">
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
            자율주행의 핵심은 AI입니다. 수백만 개의 파라미터를 가진 신경망이 실시간으로
            복잡한 도로 상황을 이해하고 판단합니다. Tesla의 FSD, Waymo의 PaLM 2 등
            최첨단 AI 모델들이 어떻게 운전을 학습하는지 알아봅시다.
          </p>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🎯 객체 탐지 모델
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-4">Two-Stage Detectors</h4>
            <div className="space-y-4">
              <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                <h5 className="font-bold text-purple-600 dark:text-purple-400 mb-2">Faster R-CNN</h5>
                <pre className="text-gray-700 dark:text-gray-300 font-mono text-xs overflow-x-auto">
{`# Faster R-CNN 구조
1. Backbone (ResNet/VGG)
2. RPN (Region Proposal Network)
3. ROI Pooling
4. Classification + Bbox Regression

# 장점: 높은 정확도
# 단점: 느린 속도 (5-10 FPS)`}</pre>
              </div>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-4">One-Stage Detectors</h4>
            <div className="space-y-4">
              <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                <h5 className="font-bold text-green-600 dark:text-green-400 mb-2">YOLOv8</h5>
                <pre className="text-gray-700 dark:text-gray-300 font-mono text-xs overflow-x-auto">
{`# YOLO 실시간 처리
class YOLO:
    def detect(self, image):
        # 그리드별 객체 예측
        predictions = self.backbone(image)
        
        # NMS로 중복 제거
        boxes = non_max_suppression(predictions)
        
        return boxes

# 장점: 빠른 속도 (30-60 FPS)
# 단점: 상대적으로 낮은 정확도`}</pre>
              </div>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🖼️ Semantic Segmentation
        </h3>
        
        <div className="bg-gray-50 dark:bg-gray-900 rounded-xl p-6 mb-6">
          <div className="space-y-6">
            <div>
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">FCN (Fully Convolutional Network)</h4>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <pre className="text-gray-700 dark:text-gray-300 font-mono text-sm overflow-x-auto">
{`# FCN으로 픽셀별 분류
def semantic_segmentation(image):
    # Encoder: 특징 추출
    features = resnet_encoder(image)
    
    # Decoder: 업샘플링
    segmap = upsample_decoder(features)
    
    # 클래스별 확률 맵
    return softmax(segmap, dim=1)

# 도로, 차선, 보행자, 차량 등을 픽셀 단위로 분류`}</pre>
              </div>
            </div>
            
            <div>
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">DeepLab v3+</h4>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <pre className="text-gray-700 dark:text-gray-300 font-mono text-sm overflow-x-auto">
{`# Atrous Convolution으로 다양한 스케일 처리
class ASPP(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1)  # 1x1
        self.conv2 = nn.Conv2d(in_ch, out_ch, 3, dilation=6)
        self.conv3 = nn.Conv2d(in_ch, out_ch, 3, dilation=12)
        self.conv4 = nn.Conv2d(in_ch, out_ch, 3, dilation=18)
    
    def forward(self, x):
        return torch.cat([self.conv1(x), self.conv2(x), 
                         self.conv3(x), self.conv4(x)], dim=1)`}</pre>
              </div>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🔮 행동 예측 AI
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6 mb-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">
              Trajectory Prediction
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
              <pre className="text-gray-700 dark:text-gray-300 font-mono text-xs overflow-x-auto">
{`# LSTM 기반 궤적 예측
class TrajectoryLSTM(nn.Module):
    def __init__(self):
        self.lstm = nn.LSTM(input_size=4, 
                           hidden_size=128,
                           num_layers=2)
        self.output = nn.Linear(128, 2)  # x, y
    
    def forward(self, trajectory_history):
        # 과거 5초 궤적으로 미래 3초 예측
        out, _ = self.lstm(trajectory_history)
        future_traj = self.output(out)
        return future_traj`}</pre>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">
              Attention Mechanism
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
              <pre className="text-gray-700 dark:text-gray-300 font-mono text-xs overflow-x-auto">
{`# Transformer로 다중 에이전트 상호작용
class MultiAgentAttention(nn.Module):
    def forward(self, agent_features):
        # 자차와 주변 차량들 간의 관계 모델링
        Q = self.query_proj(agent_features)
        K = self.key_proj(agent_features) 
        V = self.value_proj(agent_features)
        
        attention = softmax(Q @ K.T / sqrt(d_k))
        context = attention @ V
        
        return context`}</pre>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🚀 End-to-End 학습
        </h3>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700 mb-6">
          <h4 className="font-bold text-gray-900 dark:text-white mb-4">Tesla FSD 접근법</h4>
          <div className="space-y-4">
            <div className="bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 rounded-lg p-4">
              <h5 className="font-bold text-red-600 dark:text-red-400 mb-2">Neural Network Architecture</h5>
              <pre className="text-gray-700 dark:text-gray-300 font-mono text-xs overflow-x-auto">
{`# Tesla HydraNets - Multi-Task Learning
class HydraNet(nn.Module):
    def __init__(self):
        self.backbone = EfficientNet()  # 공유 특징 추출기
        
        # 각 태스크별 헤드
        self.detection_head = DetectionHead()
        self.segmentation_head = SegmentationHead() 
        self.depth_head = DepthHead()
        self.planning_head = PlanningHead()
    
    def forward(self, multi_camera_input):
        # 8개 카메라 입력 융합
        features = self.backbone(multi_camera_input)
        
        # 동시 처리
        detections = self.detection_head(features)
        segmentation = self.segmentation_head(features)
        depth = self.depth_head(features)
        trajectory = self.planning_head(features)
        
        return detections, segmentation, depth, trajectory`}</pre>
            </div>
          </div>
        </div>
      </section>

      <section className="border-t border-gray-200 dark:border-gray-700 pt-8">
        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          ⚡ Edge Computing 최적화
        </h3>
        
        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
            <h4 className="font-bold text-blue-700 dark:text-blue-400 mb-2">
              Model Quantization
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              FP32 → INT8 변환으로 4배 속도 향상
            </p>
          </div>
          
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
            <h4 className="font-bold text-green-700 dark:text-green-400 mb-2">
              Neural Architecture Search
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              하드웨어 제약에 맞는 최적 구조 자동 설계
            </p>
          </div>
          
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
            <h4 className="font-bold text-purple-700 dark:text-purple-400 mb-2">
              Knowledge Distillation
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              큰 모델의 지식을 작은 모델로 전이
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}