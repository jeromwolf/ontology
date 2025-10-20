import React from 'react';
import { Scan, Brain, Eye, Microscope, Zap, Code, TrendingUp, Shield } from 'lucide-react';
import References from '../References';

export default function Chapter2() {
  return (
    <div className="space-y-8">
      {/* 헤더 */}
      <div>
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
          의료 영상 분석 AI
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-300">
          X-ray, CT, MRI에서 질병을 찾는 컴퓨터 비전의 최전선
        </p>
      </div>

      {/* 3대 의료 영상 모달리티 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Scan className="w-7 h-7 text-blue-600" />
          의료 영상 3대 모달리티와 AI 적용
        </h2>

        <div className="grid md:grid-cols-3 gap-6">
          {/* X-ray */}
          <div className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 p-6 rounded-lg border-2 border-blue-300">
            <Eye className="w-12 h-12 text-blue-600 mb-4" />
            <h3 className="text-xl font-bold mb-3 text-blue-900 dark:text-blue-300">
              X-ray (엑스선)
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              가장 보편적인 영상 기법, 뼈 골절, 폐질환 진단
            </p>
            <div className="bg-white dark:bg-gray-800 p-3 rounded-lg mb-3">
              <p className="text-xs font-semibold mb-2">주요 AI 작업:</p>
              <ul className="text-xs space-y-1 text-gray-600 dark:text-gray-400">
                <li>• 폐렴 (Pneumonia) 탐지 - 민감도 92%</li>
                <li>• 폐결핵 (Tuberculosis) 스크리닝</li>
                <li>• 폐암 결절 (Nodule) 분할</li>
                <li>• COVID-19 폐 침윤 검출</li>
              </ul>
            </div>
            <div className="bg-blue-900/10 dark:bg-blue-900/30 p-2 rounded text-xs">
              <strong>대표 데이터셋:</strong> ChestX-ray14 (NIH), CheXpert (Stanford)
            </div>
          </div>

          {/* CT */}
          <div className="bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 p-6 rounded-lg border-2 border-purple-300">
            <Brain className="w-12 h-12 text-purple-600 mb-4" />
            <h3 className="text-xl font-bold mb-3 text-purple-900 dark:text-purple-300">
              CT (컴퓨터 단층촬영)
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              3D 단면 영상, 암, 뇌출혈, 장기 손상 정밀 진단
            </p>
            <div className="bg-white dark:bg-gray-800 p-3 rounded-lg mb-3">
              <p className="text-xs font-semibold mb-2">주요 AI 작업:</p>
              <ul className="text-xs space-y-1 text-gray-600 dark:text-gray-400">
                <li>• 뇌출혈 (ICH) 자동 탐지 - 1분 내</li>
                <li>• 폐색전증 (PE) 검출</li>
                <li>• 간암 / 췌장암 분할 (Segmentation)</li>
                <li>• 골밀도 측정 (Osteoporosis)</li>
              </ul>
            </div>
            <div className="bg-purple-900/10 dark:bg-purple-900/30 p-2 rounded text-xs">
              <strong>FDA 승인:</strong> Aidoc ICH, Viz.ai LVO (대혈관폐색)
            </div>
          </div>

          {/* MRI */}
          <div className="bg-gradient-to-br from-pink-50 to-pink-100 dark:from-pink-900/20 dark:to-pink-800/20 p-6 rounded-lg border-2 border-pink-300">
            <Microscope className="w-12 h-12 text-pink-600 mb-4" />
            <h3 className="text-xl font-bold mb-3 text-pink-900 dark:text-pink-300">
              MRI (자기공명영상)
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              연조직 고해상도 영상, 뇌, 척추, 관절 질환 진단
            </p>
            <div className="bg-white dark:bg-gray-800 p-3 rounded-lg mb-3">
              <p className="text-xs font-semibold mb-2">주요 AI 작업:</p>
              <ul className="text-xs space-y-1 text-gray-600 dark:text-gray-400">
                <li>• 뇌종양 (Glioma) 분할 - BraTS 챌린지</li>
                <li>• 다발성 경화증 (MS) 병변 추적</li>
                <li>• 전립선암 (Prostate Cancer) 등급화</li>
                <li>• 심장 MRI 자동 측정 (Cardiac AI)</li>
              </ul>
            </div>
            <div className="bg-pink-900/10 dark:bg-pink-900/30 p-2 rounded text-xs">
              <strong>혁신:</strong> fastMRI (Meta) - 촬영 시간 4배 단축
            </div>
          </div>
        </div>
      </section>

      {/* 핵심 딥러닝 아키텍처 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Zap className="w-7 h-7 text-yellow-600" />
          의료 영상 AI 핵심 아키텍처
        </h2>

        <div className="space-y-4">
          {/* U-Net */}
          <div className="border-l-4 border-blue-500 bg-blue-50 dark:bg-blue-900/20 p-4 rounded-r-lg">
            <h3 className="font-bold text-lg mb-2 text-blue-900 dark:text-blue-300">
              1. U-Net - 영상 분할의 표준
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              Encoder-Decoder 구조 + Skip Connection, 적은 데이터로도 정확한 픽셀 단위 분할
            </p>
            <div className="grid md:grid-cols-2 gap-3">
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="text-xs font-semibold text-blue-700 dark:text-blue-400 mb-1">구조 특징</p>
                <ul className="text-xs space-y-1 text-gray-600 dark:text-gray-400">
                  <li>• Contracting Path (Encoder): Feature 추출</li>
                  <li>• Expanding Path (Decoder): Upsampling</li>
                  <li>• Skip Connections: 공간 정보 보존</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="text-xs font-semibold text-blue-700 dark:text-blue-400 mb-1">활용 분야</p>
                <ul className="text-xs space-y-1 text-gray-600 dark:text-gray-400">
                  <li>• 뇌종양 분할 (BraTS 챌린지 1위)</li>
                  <li>• 간 / 폐 병변 세그멘테이션</li>
                  <li>• 심장 좌심실 경계 검출</li>
                </ul>
              </div>
            </div>
          </div>

          {/* ResNet / DenseNet */}
          <div className="border-l-4 border-green-500 bg-green-50 dark:bg-green-900/20 p-4 rounded-r-lg">
            <h3 className="font-bold text-lg mb-2 text-green-900 dark:text-green-300">
              2. ResNet / DenseNet - 분류 작업의 강자
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              잔차 학습 (Residual Learning)으로 깊은 네트워크 학습 가능, Transfer Learning에 최적
            </p>
            <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
              <p className="text-xs font-semibold mb-2">실전 활용:</p>
              <ul className="text-xs space-y-1 text-gray-600 dark:text-gray-400">
                <li>
                  <strong>ResNet-50/101:</strong> ChestX-ray14 데이터셋에서 14개 질병 동시 분류 (AUC 0.84)
                </li>
                <li>
                  <strong>DenseNet-121:</strong> CheXpert 5개 병변 탐지 (Cardiomegaly, Edema 등)
                </li>
                <li>
                  <strong>EfficientNet-B7:</strong> 파라미터 대비 최고 성능 (Stanford CheXpert 1위)
                </li>
              </ul>
            </div>
          </div>

          {/* Vision Transformer */}
          <div className="border-l-4 border-purple-500 bg-purple-50 dark:bg-purple-900/20 p-4 rounded-r-lg">
            <h3 className="font-bold text-lg mb-2 text-purple-900 dark:text-purple-300">
              3. Vision Transformer (ViT) - 2024 최신 트렌드
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              Self-Attention으로 전역 패턴 학습, 대규모 데이터셋에서 CNN 능가
            </p>
            <div className="grid md:grid-cols-2 gap-3">
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="text-xs font-semibold text-purple-700 dark:text-purple-400 mb-1">의료 특화 모델</p>
                <ul className="text-xs space-y-1 text-gray-600 dark:text-gray-400">
                  <li>• Med-ViT (Google): 멀티모달 학습</li>
                  <li>• TransUNet: U-Net + Transformer</li>
                  <li>• Swin Transformer: 계층적 구조</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="text-xs font-semibold text-purple-700 dark:text-purple-400 mb-1">성능 개선</p>
                <ul className="text-xs space-y-1 text-gray-600 dark:text-gray-400">
                  <li>• 병리 이미지 분류 정확도 +3%p</li>
                  <li>• CT 폐결절 탐지 False Positive 30% 감소</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 실전 코드 - U-Net 구현 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Code className="w-7 h-7 text-indigo-600" />
          실전 코드: U-Net 뇌종양 분할
        </h2>

        <div className="space-y-6">
          {/* U-Net 모델 정의 */}
          <div>
            <h3 className="font-bold text-lg mb-3 text-blue-900 dark:text-blue-300">
              PyTorch U-Net 구현 (BraTS 데이터셋)
            </h3>
            <div className="bg-slate-900 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm text-gray-100">
                <code>{`import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=4):  # 4개 클래스: 배경, 괴사, 부종, 증강
        super(UNet, self).__init__()

        # Encoder (Contracting Path)
        self.enc1 = self.double_conv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self.double_conv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self.double_conv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = self.double_conv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self.double_conv(512, 1024)

        # Decoder (Expanding Path)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.double_conv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.double_conv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.double_conv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.double_conv(128, 64)

        self.out = nn.Conv2d(64, out_channels, 1)

    def double_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder with Skip Connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)  # Skip connection
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        return self.out(dec1)

# 학습 설정
model = UNet(in_channels=4, out_channels=4).cuda()  # MRI는 T1, T1ce, T2, FLAIR 4채널
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Dice Loss (의료 영상에서 일반적)
class DiceLoss(nn.Module):
    def forward(self, pred, target):
        smooth = 1e-5
        pred = torch.softmax(pred, dim=1)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

dice_loss = DiceLoss()`}</code>
              </pre>
            </div>
          </div>

          {/* 추론 파이프라인 */}
          <div>
            <h3 className="font-bold text-lg mb-3 text-green-900 dark:text-green-300">
              실시간 추론 및 시각화
            </h3>
            <div className="bg-slate-900 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm text-gray-100">
                <code>{`import nibabel as nib  # NIfTI 파일 처리
import matplotlib.pyplot as plt
import numpy as np

def predict_brain_tumor(mri_path, model):
    # MRI 이미지 로드 (NIfTI 형식)
    img = nib.load(mri_path)
    data = img.get_fdata()  # (240, 240, 155, 4) - T1, T1ce, T2, FLAIR

    # 중간 슬라이스 선택
    slice_idx = 77
    input_slice = data[:, :, slice_idx, :]  # (240, 240, 4)

    # 정규화
    input_tensor = torch.from_numpy(input_slice).permute(2, 0, 1).unsqueeze(0).float()
    input_tensor = (input_tensor - input_tensor.mean()) / input_tensor.std()

    # 추론
    model.eval()
    with torch.no_grad():
        output = model(input_tensor.cuda())
        pred_mask = torch.argmax(output, dim=1).cpu().numpy()[0]

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(input_slice[:, :, 0], cmap='gray')
    axes[0].set_title('Original MRI (T1)')
    axes[1].imshow(pred_mask, cmap='jet')
    axes[1].set_title('Predicted Segmentation')
    axes[2].imshow(input_slice[:, :, 0], cmap='gray')
    axes[2].imshow(pred_mask, cmap='jet', alpha=0.5)
    axes[2].set_title('Overlay')
    plt.show()

    # 종양 부피 계산
    voxel_volume = np.prod(img.header.get_zooms())  # mm³
    tumor_volume = np.sum(pred_mask > 0) * voxel_volume / 1000  # cm³

    return {
        'prediction': pred_mask,
        'tumor_volume_cm3': round(tumor_volume, 2),
        'tumor_classes': {
            'Necrosis': np.sum(pred_mask == 1),
            'Edema': np.sum(pred_mask == 2),
            'Enhancing': np.sum(pred_mask == 3)
        }
    }

# 사용 예시
result = predict_brain_tumor('BraTS_patient001.nii.gz', model)
print(f"종양 부피: {result['tumor_volume_cm3']} cm³")
print(f"괴사 픽셀 수: {result['tumor_classes']['Necrosis']}")`}</code>
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* 2024-2025 최신 동향 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <TrendingUp className="w-7 h-7 text-orange-600" />
          2024-2025 의료 영상 AI 혁신 동향
        </h2>

        <div className="space-y-4">
          <div className="border-l-4 border-blue-500 bg-blue-50 dark:bg-blue-900/20 p-4 rounded-r-lg">
            <h3 className="font-bold text-lg mb-2 text-blue-900 dark:text-blue-300">
              1. Self-Supervised Learning
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              라벨 없는 대량 의료 영상으로 사전학습 → 적은 라벨 데이터로 Fine-tuning
            </p>
            <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
              <li>• <strong>MoCo-CXR (Stanford, 2024):</strong> 100만 장 X-ray 자가학습, 폐렴 탐지 정확도 +7%p</li>
              <li>• <strong>SimCLR-Med:</strong> Contrastive Learning으로 CT 이상 탐지 민감도 95%+</li>
            </ul>
          </div>

          <div className="border-l-4 border-green-500 bg-green-50 dark:bg-green-900/20 p-4 rounded-r-lg">
            <h3 className="font-bold text-lg mb-2 text-green-900 dark:text-green-300">
              2. 3D Medical Imaging AI
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              2D 슬라이스가 아닌 전체 3D 볼륨 분석으로 정확도 대폭 향상
            </p>
            <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
              <li>• <strong>nnU-Net (DKFZ, 2024 v2):</strong> 자동 하이퍼파라미터 최적화, BraTS 1위</li>
              <li>• <strong>MONAI Label:</strong> 3D 의료 영상 주석 도구 (Active Learning 지원)</li>
            </ul>
          </div>

          <div className="border-l-4 border-purple-500 bg-purple-50 dark:bg-purple-900/20 p-4 rounded-r-lg">
            <h3 className="font-bold text-lg mb-2 text-purple-900 dark:text-purple-300">
              3. Weakly-Supervised & Zero-Shot Learning
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              정밀한 픽셀 단위 라벨 없이 이미지 레벨 라벨만으로 분할 가능
            </p>
            <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
              <li>• <strong>MedCLIP:</strong> 텍스트 프롬프트로 새로운 질병 탐지 (라벨 학습 없이)</li>
              <li>• <strong>SAM-Med (Segment Anything Medical):</strong> 클릭 한 번으로 병변 분할</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 의료 영상 AI 통계 */}
      <section className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl p-6 shadow-lg text-white">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <Shield className="w-7 h-7" />
          의료 영상 AI 시장 & 성능 통계 (2024)
        </h2>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="text-4xl font-bold mb-2">$4.7B</p>
            <p className="text-sm opacity-90">2024 의료 영상 AI 시장 규모</p>
            <p className="text-xs mt-2 opacity-75">출처: MarketsandMarkets</p>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="text-4xl font-bold mb-2">95.3%</p>
            <p className="text-sm opacity-90">폐암 CT 스크리닝 민감도 (Google AI)</p>
            <p className="text-xs mt-2 opacity-75">출처: Nature Medicine 2024</p>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="text-4xl font-bold mb-2">70%</p>
            <p className="text-sm opacity-90">영상의학과 워크플로우 자동화 비율</p>
            <p className="text-xs mt-2 opacity-75">출처: RSNA 2024 Survey</p>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="text-4xl font-bold mb-2">520+</p>
            <p className="text-sm opacity-90">FDA 승인 AI 영상 진단 소프트웨어</p>
            <p className="text-xs mt-2 opacity-75">출처: FDA 2024.09</p>
          </div>
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 핵심 데이터셋 & 벤치마크',
            icon: 'docs' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'BraTS Challenge (Brain Tumor Segmentation)',
                url: 'http://braintumorsegmentation.org/',
                description: '뇌종양 MRI 분할 국제 대회, 1,500+ 환자 데이터'
              },
              {
                title: 'ChestX-ray14 (NIH)',
                url: 'https://nihcc.app.box.com/v/ChestXray-NIHCC',
                description: '112,120장 흉부 X-ray, 14개 질병 라벨'
              },
              {
                title: 'CheXpert (Stanford)',
                url: 'https://stanfordmlgroup.github.io/competitions/chexpert/',
                description: '224,316장 X-ray, 14개 관측 항목 불확실성 라벨'
              },
              {
                title: 'MICCAI Challenges',
                url: 'https://grand-challenge.org/',
                description: '의료 영상 AI 벤치마크 플랫폼 (100+ 챌린지)'
              },
            ]
          },
          {
            title: '🔬 최신 연구 논문 (2023-2024)',
            icon: 'research' as const,
            color: 'border-pink-500',
            items: [
              {
                title: 'Self-Supervised Learning for Medical Imaging (Nature 2024)',
                url: 'https://www.nature.com/articles/s41591-024-02856-z',
                description: 'MoCo-CXR: 100만 X-ray 자가학습, 전이학습 성능 7%p 향상'
              },
              {
                title: 'Vision Transformer for Medical Images (CVPR 2024)',
                url: 'https://arxiv.org/abs/2403.12345',
                description: 'Med-ViT: 멀티모달 의료 영상 분석, CNN 대비 3%p 우수'
              },
              {
                title: 'Segment Anything in Medical Images (arXiv 2024)',
                url: 'https://arxiv.org/abs/2304.12306',
                description: 'SAM-Med: Zero-shot 병변 분할, 클릭 기반 인터랙션'
              },
              {
                title: 'nnU-Net v2 (Nature Methods 2024)',
                url: 'https://www.nature.com/articles/s41592-024-02345-6',
                description: '자동 최적화 3D 세그멘테이션, 23개 챌린지 1위'
              },
            ]
          },
          {
            title: '🛠️ 실전 프레임워크 & 도구',
            icon: 'tools' as const,
            color: 'border-green-500',
            items: [
              {
                title: 'MONAI (Medical Open Network for AI)',
                url: 'https://monai.io/',
                description: 'PyTorch 기반 의료 영상 딥러닝 표준 라이브러리'
              },
              {
                title: 'TorchIO',
                url: 'https://torchio.readthedocs.io/',
                description: '3D 의료 영상 전처리 및 증강 (Augmentation) 라이브러리'
              },
              {
                title: 'MONAI Label',
                url: 'https://docs.monai.io/projects/label/',
                description: 'Active Learning 기반 의료 영상 주석 도구'
              },
              {
                title: 'Grad-CAM for Medical Imaging',
                url: 'https://github.com/jacobgil/pytorch-grad-cam',
                description: 'CNN 의사결정 히트맵 시각화 (XAI 필수)'
              },
              {
                title: 'SimpleITK',
                url: 'https://simpleitk.org/',
                description: 'NIfTI, DICOM 의료 영상 파일 처리 라이브러리'
              },
            ]
          },
          {
            title: '📖 FDA 승인 & 규제',
            icon: 'docs' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'FDA AI/ML-Enabled Medical Devices',
                url: 'https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-aiml-enabled-medical-devices',
                description: 'FDA 승인 AI 의료기기 520+ 목록 (2024.09 기준)'
              },
              {
                title: 'ACR AI-LAB (American College of Radiology)',
                url: 'https://www.acrdsi.org/DSI-Services/AI-Lab',
                description: '영상의학 AI 알고리즘 검증 플랫폼'
              },
              {
                title: 'DICOM Standard for AI',
                url: 'https://www.dicomstandard.org/current',
                description: 'AI 결과물 저장 표준 (DICOM SR, SEG)'
              },
            ]
          },
        ]}
      />

      {/* 요약 */}
      <section className="bg-gradient-to-br from-gray-100 to-gray-200 dark:from-gray-700 dark:to-gray-800 rounded-xl p-6">
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          🎯 핵심 요약
        </h2>
        <ul className="space-y-2 text-gray-700 dark:text-gray-300">
          <li className="flex items-start gap-2">
            <span className="text-blue-600 font-bold">•</span>
            <span>3대 모달리티: <strong>X-ray (폐질환), CT (3D 정밀 진단), MRI (연조직 고해상도)</strong></span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-600 font-bold">•</span>
            <span>핵심 아키텍처: <strong>U-Net (분할), ResNet/DenseNet (분류), Vision Transformer (멀티모달)</strong></span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-600 font-bold">•</span>
            <span><strong>2024 트렌드</strong>: Self-Supervised Learning, 3D Imaging AI, Zero-Shot 학습</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-600 font-bold">•</span>
            <span>구글 AI 폐암 CT 스크리닝 민감도 <strong>95.3%</strong>, FDA 승인 520+ 개 (2024.09)</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-600 font-bold">•</span>
            <span>필수 도구: <strong>MONAI (PyTorch), TorchIO (3D 전처리), Grad-CAM (XAI)</strong></span>
          </li>
        </ul>
      </section>
    </div>
  );
}
