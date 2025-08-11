'use client'

import { useState } from 'react'

export default function NeuralNetworkBasics() {
  const [activeTab, setActiveTab] = useState('perceptron')

  return (
    <section>
      <h2 className="text-3xl font-bold mb-6">2. 신경망의 기초</h2>
      
      <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700 mb-6">
        <div className="flex gap-2 mb-4 flex-wrap">
          {['perceptron', 'activation', 'backprop', 'optimization'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                activeTab === tab
                  ? 'bg-indigo-500 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
              }`}
            >
              {tab === 'perceptron' && '퍼셉트론'}
              {tab === 'activation' && '활성화 함수'}
              {tab === 'backprop' && '역전파'}
              {tab === 'optimization' && '최적화'}
            </button>
          ))}
        </div>

        {activeTab === 'perceptron' && <PerceptronSection />}
        {activeTab === 'activation' && <ActivationSection />}
        {activeTab === 'backprop' && <BackpropSection />}
        {activeTab === 'optimization' && <OptimizationSection />}
      </div>
    </section>
  )
}

function PerceptronSection() {
  return (
    <div>
      <h3 className="text-lg font-semibold mb-3 text-indigo-600 dark:text-indigo-400">퍼셉트론 (Perceptron)</h3>
      
      <div className="grid md:grid-cols-2 gap-4 mb-4">
        <div>
          <h4 className="font-semibold mb-2">단일 퍼셉트론</h4>
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
            가장 간단한 형태의 인공 뉴런. 입력값에 가중치를 곱하고 편향을 더한 후 활성화 함수를 적용
          </p>
          <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded">
            <p className="font-mono text-sm">
              y = f(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)
            </p>
            <p className="text-xs mt-2">
              • w: 가중치 (weight)<br/>
              • x: 입력값 (input)<br/>
              • b: 편향 (bias)<br/>
              • f: 활성화 함수
            </p>
          </div>
        </div>
        
        <div>
          <h4 className="font-semibold mb-2">다층 퍼셉트론 (MLP)</h4>
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
            여러 층의 퍼셉트론을 연결한 신경망. 은닉층을 통해 비선형 문제 해결 가능
          </p>
          <ul className="text-sm space-y-1">
            <li>• 입력층 (Input Layer)</li>
            <li>• 은닉층 (Hidden Layers)</li>
            <li>• 출력층 (Output Layer)</li>
            <li>• 완전 연결 (Fully Connected)</li>
          </ul>
        </div>
      </div>
      
      <PerceptronCode />
    </div>
  )
}

function PerceptronCode() {
  return (
    <div className="bg-gray-900 rounded-lg p-4">
      <pre className="text-sm text-gray-300 overflow-x-auto">
{`import numpy as np

# 단순 퍼셉트론 구현
class Perceptron:
    def __init__(self, input_size, learning_rate=0.01):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.lr = learning_rate
    
    def activation(self, x):
        # 계단 함수 (step function)
        return 1 if x >= 0 else 0
    
    def predict(self, x):
        z = np.dot(self.weights, x) + self.bias
        return self.activation(z)
    
    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            for xi, yi in zip(X, y):
                y_pred = self.predict(xi)
                error = yi - y_pred
                
                # 가중치 업데이트
                self.weights += self.lr * error * xi
                self.bias += self.lr * error

# XOR 문제를 위한 다층 퍼셉트론
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(2, 4)  # 입력 2 -> 은닉 4
        self.output = nn.Linear(4, 1)  # 은닉 4 -> 출력 1
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

# 학습
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

model = MLP()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(1000):
    output = model(X)
    loss = criterion(output, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')`}</pre>
    </div>
  )
}

function ActivationSection() {
  return (
    <div>
      <h3 className="text-lg font-semibold mb-3 text-purple-600 dark:text-purple-400">활성화 함수 (Activation Functions)</h3>
      
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
        활성화 함수는 신경망에 비선형성을 추가하여 복잡한 패턴을 학습할 수 있게 합니다.
      </p>
      
      <div className="grid md:grid-cols-2 gap-4 mb-4">
        <ActivationCard
          name="Sigmoid"
          formula="σ(x) = 1 / (1 + e^(-x))"
          color="blue"
          pros={["출력: 0~1 사이", "확률 해석 가능"]}
          cons={["Vanishing Gradient", "출력이 중심이 아님"]}
        />
        
        <ActivationCard
          name="ReLU"
          formula="f(x) = max(0, x)"
          color="green"
          pros={["계산 효율적", "Vanishing Gradient 완화"]}
          cons={["Dying ReLU 문제", "음수 입력 무시"]}
        />
        
        <ActivationCard
          name="Tanh"
          formula="tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))"
          color="purple"
          pros={["출력: -1~1", "중심이 0"]}
          cons={["여전히 Vanishing Gradient", "계산 비용"]}
        />
        
        <ActivationCard
          name="Leaky ReLU"
          formula="f(x) = max(0.01x, x)"
          color="orange"
          pros={["Dying ReLU 해결", "음수 입력도 학습"]}
          cons={["α 값 선택 필요", "성능 향상 미미할 수 있음"]}
        />
      </div>
      
      <ActivationCode />
    </div>
  )
}

function ActivationCard({ name, formula, color, pros, cons }: {
  name: string
  formula: string
  color: string
  pros: string[]
  cons: string[]
}) {
  const colorClasses = {
    blue: 'from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20',
    green: 'from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20',
    purple: 'from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20',
    orange: 'from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20'
  }

  const textColorClasses = {
    blue: 'text-blue-700 dark:text-blue-400',
    green: 'text-green-700 dark:text-green-400',
    purple: 'text-purple-700 dark:text-purple-400',
    orange: 'text-orange-700 dark:text-orange-400'
  }

  return (
    <div className={`bg-gradient-to-br ${colorClasses[color as keyof typeof colorClasses]} p-4 rounded-lg`}>
      <h4 className={`font-semibold ${textColorClasses[color as keyof typeof textColorClasses]} mb-2`}>{name}</h4>
      <p className="text-sm mb-2">{formula}</p>
      <ul className="text-xs space-y-1">
        {pros.map((pro, i) => <li key={i}>✓ {pro}</li>)}
        {cons.map((con, i) => <li key={i}>✗ {con}</li>)}
      </ul>
    </div>
  )
}

function ActivationCode() {
  return (
    <div className="bg-gray-900 rounded-lg p-4">
      <pre className="text-sm text-gray-300 overflow-x-auto">
{`import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 활성화 함수 비교
x = torch.linspace(-5, 5, 100)

# 다양한 활성화 함수
activations = {
    'Sigmoid': torch.sigmoid(x),
    'Tanh': torch.tanh(x),
    'ReLU': F.relu(x),
    'Leaky ReLU': F.leaky_relu(x, 0.01),
    'ELU': F.elu(x),
    'GELU': F.gelu(x)
}

# 시각화
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.ravel()

for idx, (name, y) in enumerate(activations.items()):
    axes[idx].plot(x.numpy(), y.numpy())
    axes[idx].set_title(name)
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_xlabel('x')
    axes[idx].set_ylabel('f(x)')

plt.tight_layout()
plt.show()`}</pre>
    </div>
  )
}

function BackpropSection() {
  return (
    <div>
      <h3 className="text-lg font-semibold mb-3 text-green-600 dark:text-green-400">역전파 (Backpropagation)</h3>
      
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
        역전파는 신경망의 가중치를 업데이트하는 핵심 알고리즘입니다. 
        출력층에서 시작해 입력층 방향으로 오차를 전파하며 각 가중치의 기울기를 계산합니다.
      </p>
      
      <div className="bg-gray-50 dark:bg-gray-900/50 p-4 rounded-lg mb-4">
        <h4 className="font-semibold mb-2">역전파 과정</h4>
        <ol className="list-decimal list-inside space-y-2 text-sm">
          <li><strong>순전파 (Forward Pass)</strong>: 입력에서 출력까지 계산</li>
          <li><strong>손실 계산</strong>: 예측값과 실제값의 차이 측정</li>
          <li><strong>역전파 (Backward Pass)</strong>: 체인룰을 사용해 기울기 계산</li>
          <li><strong>가중치 업데이트</strong>: 경사하강법으로 파라미터 갱신</li>
        </ol>
      </div>
      
      <BackpropCode />
    </div>
  )
}

function BackpropCode() {
  return (
    <div className="bg-gray-900 rounded-lg p-4">
      <pre className="text-sm text-gray-300 overflow-x-auto">
{`# 간단한 역전파 구현
import numpy as np

class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        # 가중치 초기화 (Xavier)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2/input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2/hidden_size)
        self.b2 = np.zeros((1, output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        # 순전파
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        
        # 출력층 오차
        dz2 = self.a2 - y
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # 은닉층 오차
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.sigmoid_derivative(self.a1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # 가중치 업데이트
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # 순전파
            output = self.forward(X)
            
            # 손실 계산
            loss = np.mean((output - y)**2)
            
            # 역전파
            self.backward(X, y, learning_rate)
            
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss:.4f}')`}</pre>
    </div>
  )
}

function OptimizationSection() {
  return (
    <div>
      <h3 className="text-lg font-semibold mb-3 text-orange-600 dark:text-orange-400">최적화 알고리즘</h3>
      
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
        최적화 알고리즘은 손실 함수를 최소화하기 위해 가중치를 업데이트하는 방법을 결정합니다.
      </p>
      
      <div className="grid md:grid-cols-2 gap-4 mb-4">
        <OptimizerCard
          name="SGD"
          description="확률적 경사하강법"
          formula="θ = θ - α∇J(θ)"
          pros={["간단한 구현", "메모리 효율적"]}
          cons={["느린 수렴", "국소 최소값 문제"]}
        />
        
        <OptimizerCard
          name="Adam"
          description="Adaptive Moment Estimation"
          formula="적응적 학습률 + 모멘텀"
          pros={["빠른 수렴", "적응적 학습률"]}
          cons={["하이퍼파라미터 민감", "메모리 사용량"]}
        />
      </div>
      
      <OptimizationCode />
    </div>
  )
}

function OptimizerCard({ name, description, formula, pros, cons }: {
  name: string
  description: string
  formula: string
  pros: string[]
  cons: string[]
}) {
  return (
    <div className="bg-white dark:bg-gray-700 p-4 rounded-lg border border-gray-200 dark:border-gray-600">
      <h4 className="font-semibold mb-1">{name}</h4>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">{description}</p>
      <p className="text-xs font-mono mb-2">{formula}</p>
      <div className="text-xs">
        <div className="mb-1">
          <span className="text-green-600 dark:text-green-400">장점:</span>
          {pros.map((pro, i) => <span key={i}> {pro}{i < pros.length-1 ? ',' : ''}</span>)}
        </div>
        <div>
          <span className="text-red-600 dark:text-red-400">단점:</span>
          {cons.map((con, i) => <span key={i}> {con}{i < cons.length-1 ? ',' : ''}</span>)}
        </div>
      </div>
    </div>
  )
}

function OptimizationCode() {
  return (
    <div className="bg-gray-900 rounded-lg p-4">
      <pre className="text-sm text-gray-300 overflow-x-auto">
{`# PyTorch 최적화 알고리즘 비교
import torch
import torch.nn as nn
import torch.optim as optim

# 모델 정의
model = nn.Sequential(
    nn.Linear(10, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

# 다양한 최적화 알고리즘
optimizers = {
    'SGD': optim.SGD(model.parameters(), lr=0.01),
    'SGD with Momentum': optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
    'Adam': optim.Adam(model.parameters(), lr=0.001),
    'AdamW': optim.AdamW(model.parameters(), lr=0.001),
    'RMSprop': optim.RMSprop(model.parameters(), lr=0.001),
    'Adagrad': optim.Adagrad(model.parameters(), lr=0.01)
}

# 학습률 스케줄러
scheduler = optim.lr_scheduler.StepLR(optimizers['Adam'], step_size=30, gamma=0.1)

# 학습 루프
for epoch in range(100):
    for batch_x, batch_y in dataloader:
        # 순전파
        output = model(batch_x)
        loss = criterion(output, batch_y)
        
        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 학습률 업데이트
    scheduler.step()`}</pre>
    </div>
  )
}