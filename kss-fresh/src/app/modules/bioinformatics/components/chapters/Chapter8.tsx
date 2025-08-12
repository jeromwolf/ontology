'use client'

import { useState } from 'react'
import { Copy, CheckCircle, Activity } from 'lucide-react'

export default function Chapter8() {
  const [copiedCode, setCopiedCode] = useState<string | null>(null)

  const copyCode = (code: string, id: string) => {
    navigator.clipboard.writeText(code)
    setCopiedCode(id)
    setTimeout(() => setCopiedCode(null), 2000)
  }

  const mlCode = `# 딥러닝을 사용한 유전체 변이 예측
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class GenomicCNN(nn.Module):
    """DNA 서열에서 변이 효과 예측 CNN 모델"""
    
    def __init__(self, seq_length=1000, num_filters=128):
        super(GenomicCNN, self).__init__()
        
        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(4, num_filters, kernel_size=12, padding=5)
        self.conv2 = nn.Conv1d(num_filters, num_filters*2, kernel_size=8, padding=3)
        self.conv3 = nn.Conv1d(num_filters*2, num_filters*4, kernel_size=4, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.bn2 = nn.BatchNorm1d(num_filters*2)
        self.bn3 = nn.BatchNorm1d(num_filters*4)
        
        # Pooling
        self.pool = nn.MaxPool1d(2)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=num_filters*4, 
            num_heads=8
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(num_filters*4 * (seq_length//8), 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        
        # Dropout
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # Convolutional blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # Flatten for FC layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Output (pathogenicity score)
        x = torch.sigmoid(self.fc3(x))
        
        return x

class VariantEffectPredictor:
    """변이 효과 예측 파이프라인"""
    
    def __init__(self, model_path=None):
        self.model = GenomicCNN()
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def encode_sequence(self, sequence):
        """DNA 서열을 one-hot encoding"""
        encoding = {'A': [1,0,0,0], 
                   'C': [0,1,0,0], 
                   'G': [0,0,1,0], 
                   'T': [0,0,0,1],
                   'N': [0,0,0,0]}
        
        encoded = []
        for base in sequence.upper():
            encoded.append(encoding.get(base, [0,0,0,0]))
        
        return torch.tensor(encoded).transpose(0, 1).unsqueeze(0)
    
    def predict_variant_effect(self, ref_seq, alt_seq):
        """Reference와 Alternative 서열의 효과 비교"""
        ref_encoded = self.encode_sequence(ref_seq)
        alt_encoded = self.encode_sequence(alt_seq)
        
        with torch.no_grad():
            ref_score = self.model(ref_encoded)
            alt_score = self.model(alt_encoded)
        
        effect = {
            'reference_score': ref_score.item(),
            'alternative_score': alt_score.item(),
            'delta_score': alt_score.item() - ref_score.item(),
            'predicted_effect': 'Pathogenic' if alt_score.item() > 0.5 else 'Benign'
        }
        
        return effect`

  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          1. 딥러닝과 유전체학의 만남
        </h2>
        <p className="mb-4">
          딥러닝은 유전체 데이터의 복잡한 패턴을 학습하여 변이 효과 예측, 
          유전자 발현 예측, 질병 위험도 평가 등에 혁신을 가져왔습니다.
        </p>
        
        <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6 mb-6">
          <h3 className="font-bold mb-3">주요 응용 분야</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h4 className="font-semibold mb-2">서열 기반 예측</h4>
              <ul className="space-y-1 text-sm">
                <li>• Variant effect prediction</li>
                <li>• Splice site prediction</li>
                <li>• Promoter/enhancer identification</li>
                <li>• TFBS prediction</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-2">임상 응용</h4>
              <ul className="space-y-1 text-sm">
                <li>• Disease risk prediction</li>
                <li>• Drug response prediction</li>
                <li>• Survival analysis</li>
                <li>• Treatment recommendation</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          2. CNN for Genomics
        </h2>
        
        <div className="bg-gray-100 dark:bg-gray-900 rounded-lg p-4 mb-4">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-mono text-gray-600 dark:text-gray-400">genomic_cnn.py</span>
            <button
              onClick={() => copyCode(mlCode, 'ml')}
              className="p-2 hover:bg-gray-200 dark:hover:bg-gray-800 rounded"
            >
              {copiedCode === 'ml' ? <CheckCircle className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
            </button>
          </div>
          <pre className="text-sm overflow-x-auto">
            <code>{mlCode}</code>
          </pre>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          3. Polygenic Risk Score (PRS)
        </h2>
        <p className="mb-4">
          수천~수백만 개의 유전 변이를 종합하여 질병 위험도를 계산합니다.
        </p>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-emerald-200 dark:border-emerald-800">
          <h3 className="font-bold mb-3">PRS 계산 과정</h3>
          <ol className="space-y-2">
            <li className="flex items-start gap-2">
              <span className="text-emerald-600 font-bold">1.</span>
              <span>GWAS summary statistics 수집</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-600 font-bold">2.</span>
              <span>SNP effect size 추정</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-600 font-bold">3.</span>
              <span>LD clumping & pruning</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-600 font-bold">4.</span>
              <span>가중 합계 계산: PRS = Σ(βi × dosagei)</span>
            </li>
          </ol>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          4. 정밀의학 응용
        </h2>
        
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
          <h3 className="font-bold mb-3 flex items-center gap-2">
            <Activity className="w-5 h-5 text-blue-600" />
            개인 맞춤 의학
          </h3>
          <ul className="space-y-2">
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-1">•</span>
              <div>
                <strong>약물유전체학:</strong> 개인별 약물 반응 예측
              </div>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-1">•</span>
              <div>
                <strong>암 유전체학:</strong> 종양 특이적 치료법 선택
              </div>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-1">•</span>
              <div>
                <strong>희귀질환:</strong> 유전자 진단 및 치료
              </div>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-1">•</span>
              <div>
                <strong>예방의학:</strong> 질병 위험도 기반 건강관리
              </div>
            </li>
          </ul>
        </div>
      </section>
    </div>
  )
}