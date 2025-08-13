'use client';

import { useState } from 'react';
import { 
  Terminal,
  Copy,
  Check
} from 'lucide-react';

export default function Chapter4() {
  const [copiedCode, setCopiedCode] = useState<string | null>(null);

  const copyToClipboard = (code: string, id: string) => {
    navigator.clipboard.writeText(code);
    setCopiedCode(id);
    setTimeout(() => setCopiedCode(null), 2000);
  };

  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4">CNN 아키텍처의 진화</h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          Convolutional Neural Networks(CNN)는 컴퓨터 비전 분야에 혁명을 일으켰습니다.
          LeNet부터 최신 EfficientNet까지, CNN 아키텍처는 계속 발전하고 있습니다.
        </p>

        <div className="bg-gradient-to-r from-teal-50 to-cyan-50 dark:from-teal-900/20 dark:to-cyan-900/20 rounded-lg p-6 mb-6">
          <h3 className="text-lg font-semibold mb-4 text-teal-900 dark:text-teal-100">주요 CNN 아키텍처</h3>
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <div className="w-2 h-2 bg-teal-500 rounded-full"></div>
              <span className="font-medium">AlexNet (2012)</span> - ImageNet 대회 우승, 딥러닝 시대 개막
            </div>
            <div className="flex items-center gap-3">
              <div className="w-2 h-2 bg-teal-500 rounded-full"></div>
              <span className="font-medium">VGGNet (2014)</span> - 단순하고 깊은 구조의 효과 증명
            </div>
            <div className="flex items-center gap-3">
              <div className="w-2 h-2 bg-teal-500 rounded-full"></div>
              <span className="font-medium">ResNet (2015)</span> - Skip connection으로 매우 깊은 네트워크 학습
            </div>
            <div className="flex items-center gap-3">
              <div className="w-2 h-2 bg-teal-500 rounded-full"></div>
              <span className="font-medium">EfficientNet (2019)</span> - 효율적인 스케일링으로 높은 성능
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">Vision Transformer</h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          Transformer 아키텍처를 컴퓨터 비전에 적용한 Vision Transformer(ViT)는 
          CNN의 대안으로 주목받고 있습니다. 이미지를 패치로 나누어 시퀀스로 처리합니다.
        </p>

        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Terminal className="w-4 h-4 text-gray-600 dark:text-gray-400" />
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Python - Vision Transformer 사용</span>
            </div>
            <button
              onClick={() => copyToClipboard(`from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

# 사전 학습된 ViT 모델 로드
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# 이미지 전처리
image = Image.open('sample.jpg')
inputs = processor(images=image, return_tensors="pt")

# 예측
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()

print(f"예측된 클래스: {model.config.id2label[predicted_class]}")`, 'dl-vision-1')}
              className="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded"
            >
              {copiedCode === 'dl-vision-1' ? (
                <Check className="w-4 h-4 text-green-600" />
              ) : (
                <Copy className="w-4 h-4 text-gray-600 dark:text-gray-400" />
              )}
            </button>
          </div>
          <pre className="text-sm text-gray-800 dark:text-gray-200 overflow-x-auto">
            <code>{`from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

# 사전 학습된 ViT 모델 로드
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# 이미지 전처리
image = Image.open('sample.jpg')
inputs = processor(images=image, return_tensors="pt")

# 예측
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()

print(f"예측된 클래스: {model.config.id2label[predicted_class]}")`}</code>
          </pre>
        </div>
      </section>
    </div>
  );
}