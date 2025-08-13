'use client'

import React from 'react'

export default function Chapter4() {
  return (
    <div className="prose prose-lg dark:prose-invert max-w-none">
      <h2>강화학습과 제어</h2>
      
      <h3>1. 강화학습 기초</h3>
      <p>
        Physical AI는 시행착오를 통해 최적의 행동을 학습합니다. 
        강화학습은 로봇이 환경과 상호작용하며 스스로 학습하는 핵심 기술입니다.
      </p>

      <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-lg my-6">
        <h4 className="font-semibold mb-3">Q-Learning 예제</h4>
        <pre className="bg-white dark:bg-gray-900 p-3 rounded text-sm overflow-x-auto">
{`import numpy as np

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, gamma=0.99):
        self.q_table = np.zeros((state_size, action_size))
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_size)  # 탐색
        return np.argmax(self.q_table[state])  # 활용
        
    def update(self, state, action, reward, next_state):
        # Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state, action] = new_q
        
        self.epsilon *= self.epsilon_decay`}
        </pre>
      </div>
    </div>
  )
}