'use client'

import { Layers, Network } from 'lucide-react'

export default function CNNandRNN() {
  return (
    <>
      {/* CNN Section */}
      <section>
        <h2 className="text-3xl font-bold mb-6">4. CNN (Convolutional Neural Network)</h2>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700 mb-6">
          <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Layers className="text-blue-500" />
            CNN의 핵심 구성 요소
          </h3>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold mb-2">Convolution Layer</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                필터(커널)를 사용해 특징을 추출하는 층
              </p>
              <ul className="text-sm space-y-1">
                <li>• 파라미터 공유</li>
                <li>• 지역적 연결성</li>
                <li>• 이동 불변성</li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-semibold mb-2">Pooling Layer</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                공간 크기를 줄이고 중요한 정보 보존
              </p>
              <ul className="text-sm space-y-1">
                <li>• Max Pooling</li>
                <li>• Average Pooling</li>
                <li>• 계산량 감소</li>
              </ul>
            </div>
          </div>
        </div>

        <CNNCode />
      </section>

      {/* RNN Section */}
      <section className="mt-12">
        <h2 className="text-3xl font-bold mb-6">5. RNN (Recurrent Neural Network)</h2>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700 mb-6">
          <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Network className="text-green-500" />
            RNN과 그 변형들
          </h3>
          
          <div className="grid md:grid-cols-3 gap-4">
            <RNNCard
              title="Vanilla RNN"
              description="기본 순환 신경망"
              pros={["간단한 구조", "시퀀스 처리"]}
              cons={["Vanishing Gradient", "장기 의존성 문제"]}
              color="blue"
            />
            
            <RNNCard
              title="LSTM"
              description="Long Short-Term Memory"
              pros={["장기 기억 가능", "Gate 메커니즘"]}
              cons={["복잡한 구조", "계산 비용"]}
              color="green"
            />
            
            <RNNCard
              title="GRU"
              description="Gated Recurrent Unit"
              pros={["LSTM보다 간단", "빠른 학습"]}
              cons={["LSTM보다 표현력 약함"]}
              color="purple"
            />
          </div>
        </div>

        <RNNCode />
      </section>
    </>
  )
}

function CNNCode() {
  return (
    <div className="bg-gray-900 rounded-xl p-6">
      <h3 className="text-white font-semibold mb-4">CNN 구현 예제</h3>
      <pre className="bg-gray-800 p-4 rounded-lg overflow-x-auto">
        <code className="text-sm text-gray-300">{`import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Conv Block 1
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        
        # Conv Block 2
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        
        # Conv Block 3
        x = self.pool(F.relu(self.conv3(x)))  # 8x8 -> 4x4
        
        # Flatten
        x = x.view(-1, 128 * 4 * 4)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# 유명한 CNN 아키텍처들
class ResBlock(nn.Module):
    """ResNet의 Residual Block"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out`}</code>
      </pre>
    </div>
  )
}

function RNNCard({ title, description, pros, cons, color }: {
  title: string
  description: string
  pros: string[]
  cons: string[]
  color: string
}) {
  const colorClasses = {
    blue: 'from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20',
    green: 'from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20',
    purple: 'from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20'
  }

  return (
    <div className={`bg-gradient-to-br ${colorClasses[color as keyof typeof colorClasses]} p-4 rounded-lg`}>
      <h4 className="font-semibold mb-1">{title}</h4>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">{description}</p>
      <div className="text-xs">
        <div className="mb-1">
          <span className="text-green-600 dark:text-green-400">장점:</span>
          <ul className="mt-1">
            {pros.map((pro, i) => <li key={i}>• {pro}</li>)}
          </ul>
        </div>
        <div>
          <span className="text-red-600 dark:text-red-400">단점:</span>
          <ul className="mt-1">
            {cons.map((con, i) => <li key={i}>• {con}</li>)}
          </ul>
        </div>
      </div>
    </div>
  )
}

function RNNCode() {
  return (
    <div className="bg-gray-900 rounded-xl p-6">
      <h3 className="text-white font-semibold mb-4">RNN 구현 예제</h3>
      <pre className="bg-gray-800 p-4 rounded-lg overflow-x-auto">
        <code className="text-sm text-gray-300">{`import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, 
                         batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# 양방향 LSTM
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, 
                           bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        
        # Concatenate the final forward and backward hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        return self.fc(self.dropout(hidden))`}</code>
      </pre>
    </div>
  )
}