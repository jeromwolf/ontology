'use client'

export default function DeepLearningFrameworks() {
  return (
    <section>
      <h2 className="text-3xl font-bold mb-6">3. TensorFlow vs PyTorch</h2>
      
      <div className="grid md:grid-cols-2 gap-6 mb-6">
        <div className="bg-gradient-to-br from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 p-6 rounded-xl">
          <h3 className="text-xl font-semibold mb-4 text-orange-700 dark:text-orange-400">TensorFlow 2.x</h3>
          <ul className="space-y-2 text-sm">
            <li>✓ Google 개발</li>
            <li>✓ 프로덕션 배포 강점</li>
            <li>✓ TensorFlow Lite (모바일)</li>
            <li>✓ TensorFlow.js (웹)</li>
            <li>✓ Keras 통합</li>
            <li>• 산업계 선호</li>
          </ul>
        </div>
        
        <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 p-6 rounded-xl">
          <h3 className="text-xl font-semibold mb-4 text-blue-700 dark:text-blue-400">PyTorch</h3>
          <ul className="space-y-2 text-sm">
            <li>✓ Facebook 개발</li>
            <li>✓ Dynamic Graph</li>
            <li>✓ 연구 친화적</li>
            <li>✓ 디버깅 용이</li>
            <li>✓ Pythonic API</li>
            <li>• 학계 선호</li>
          </ul>
        </div>
      </div>

      {/* 프레임워크 비교 코드 */}
      <div className="bg-gray-900 rounded-xl p-6">
        <h3 className="text-white font-semibold mb-4">동일한 모델을 두 프레임워크로 구현</h3>
        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <h4 className="text-orange-400 font-semibold mb-2">TensorFlow</h4>
            <pre className="bg-gray-800 p-4 rounded-lg overflow-x-auto">
              <code className="text-sm text-gray-300">{`import tensorflow as tf

# 모델 정의
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 컴파일
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 학습
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32
)`}</code>
            </pre>
          </div>
          
          <div>
            <h4 className="text-blue-400 font-semibold mb-2">PyTorch</h4>
            <pre className="bg-gray-800 p-4 rounded-lg overflow-x-auto">
              <code className="text-sm text-gray-300">{`import torch
import torch.nn as nn

# 모델 정의
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# 학습 루프
model = Net()
optimizer = optim.Adam(model.parameters())
for epoch in range(10):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()`}</code>
            </pre>
          </div>
        </div>
      </div>

      {/* 프레임워크 선택 가이드 */}
      <div className="mt-6 bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 p-6 rounded-xl">
        <h3 className="text-lg font-semibold mb-4">프레임워크 선택 가이드</h3>
        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <h4 className="font-semibold text-purple-700 dark:text-purple-400 mb-2">TensorFlow를 선택하세요:</h4>
            <ul className="text-sm space-y-1">
              <li>• 프로덕션 배포가 중요한 경우</li>
              <li>• 모바일/웹 배포가 필요한 경우</li>
              <li>• TensorBoard를 활용하고 싶은 경우</li>
              <li>• 대규모 분산 학습이 필요한 경우</li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold text-pink-700 dark:text-pink-400 mb-2">PyTorch를 선택하세요:</h4>
            <ul className="text-sm space-y-1">
              <li>• 연구 및 프로토타이핑이 목적인 경우</li>
              <li>• 동적 그래프가 필요한 경우</li>
              <li>• 디버깅이 중요한 경우</li>
              <li>• 최신 논문 구현이 필요한 경우</li>
            </ul>
          </div>
        </div>
      </div>
    </section>
  )
}