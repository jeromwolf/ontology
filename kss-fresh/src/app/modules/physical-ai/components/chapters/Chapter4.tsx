'use client';

import React from 'react';
import { Brain, Gamepad2, Target, Zap, TrendingUp, RefreshCw, Award } from 'lucide-react';

export default function Chapter4() {
  return (
    <div className="prose prose-lg dark:prose-invert max-w-none">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-2xl p-8 mb-8 border border-orange-200 dark:border-orange-800">
        <div className="flex items-center gap-4 mb-4">
          <div className="w-12 h-12 bg-orange-500 rounded-xl flex items-center justify-center">
            <Brain className="w-6 h-6 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white m-0">강화학습과 로봇 제어</h1>
        </div>
        <p className="text-xl text-gray-700 dark:text-gray-300 m-0">
          시행착오를 통한 학습 - 로봇이 스스로 최적의 행동을 찾는다
        </p>
      </div>

      {/* Introduction */}
      <section className="my-8">
        <h2 className="flex items-center gap-2">
          <Gamepad2 className="text-orange-600" />
          강화학습이란?
        </h2>

        <div className="bg-gradient-to-r from-orange-50 to-yellow-50 dark:from-orange-900/20 dark:to-yellow-900/20 p-6 rounded-lg border-l-4 border-orange-500 mb-6">
          <h3 className="text-xl font-bold mb-4">🎮 게임처럼 학습하는 AI</h3>
          <p className="mb-4">
            <strong>강화학습 (Reinforcement Learning)</strong>은 AI가 <strong>보상과 벌점</strong>을 통해
            스스로 최적의 전략을 찾아가는 학습 방법입니다. 마치 게임을 반복하며 고수가 되는 것과 같습니다.
          </p>

          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg text-center">
              <div className="text-3xl mb-2">🎯</div>
              <h4 className="font-bold text-sm mb-2">목표 (Goal)</h4>
              <p className="text-xs">최대한 높은 보상 획득</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg text-center">
              <div className="text-3xl mb-2">🔄</div>
              <h4 className="font-bold text-sm mb-2">시행착오 (Trial & Error)</h4>
              <p className="text-xs">실패하며 배운다</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg text-center">
              <div className="text-3xl mb-2">🏆</div>
              <h4 className="font-bold text-sm mb-2">최적화 (Optimization)</h4>
              <p className="text-xs">점점 나아지는 전략</p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg mb-6">
          <h4 className="font-bold mb-3">📚 지도학습 vs 강화학습</h4>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
              <h5 className="font-bold mb-2 text-blue-600">지도학습 (Supervised Learning)</h5>
              <ul className="text-sm space-y-2">
                <li>✅ 정답이 주어진 데이터로 학습</li>
                <li>✅ "이 이미지는 고양이입니다"</li>
                <li>❌ 새로운 상황에 약함</li>
                <li>🎯 용도: 이미지 분류, 번역</li>
              </ul>
            </div>
            <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg border-2 border-orange-500">
              <h5 className="font-bold mb-2 text-orange-600">강화학습 (Reinforcement Learning)</h5>
              <ul className="text-sm space-y-2">
                <li>✅ 보상으로만 학습 (정답 없음)</li>
                <li>✅ "이 행동이 좋았나? 나빴나?"</li>
                <li>✅ 새로운 상황에 적응 가능</li>
                <li>🎯 용도: 로봇 제어, 게임 AI</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* 1. Q-Learning */}
      <section className="my-8">
        <h2 className="flex items-center gap-2">
          <Target className="text-blue-600" />
          1. Q-Learning - 가장 기본적인 강화학습
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg mb-6">
          <h3 className="text-xl font-bold mb-4">🧮 Q-Table: 모든 상황별 최선의 행동 저장</h3>

          <div className="bg-blue-50 dark:bg-blue-900/20 p-5 rounded-lg mb-6">
            <h4 className="font-bold mb-3">Q-Learning 핵심 개념</h4>
            <div className="space-y-3 text-sm">
              <div>
                <strong className="text-blue-600">• State (상태)</strong>: 로봇이 처한 현재 상황
                <p className="mt-1 ml-4">예: "로봇 팔이 물체로부터 10cm 떨어져 있음"</p>
              </div>
              <div>
                <strong className="text-green-600">• Action (행동)</strong>: 로봇이 취할 수 있는 동작
                <p className="mt-1 ml-4">예: "왼쪽으로 5cm 이동", "그리퍼 닫기"</p>
              </div>
              <div>
                <strong className="text-purple-600">• Reward (보상)</strong>: 행동의 결과에 대한 점수
                <p className="mt-1 ml-4">예: 물체를 성공적으로 잡으면 +10, 떨어뜨리면 -5</p>
              </div>
              <div>
                <strong className="text-orange-600">• Q-Value</strong>: 특정 상태에서 특정 행동의 기대 보상
                <p className="mt-1 ml-4">Q(상태, 행동) = "이 상황에서 이 행동이 얼마나 좋은가?"</p>
              </div>
            </div>
          </div>

          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto mb-4">
            <pre className="text-sm">
{`# Q-Learning: 로봇 팔이 물체 잡기 학습
import numpy as np

class RobotArmQLearning:
    def __init__(self, num_positions=10, num_actions=4):
        # Q-Table 초기화 (상태 x 행동)
        self.q_table = np.zeros((num_positions, num_actions))

        # 하이퍼파라미터
        self.learning_rate = 0.1  # α: 얼마나 빨리 학습?
        self.discount_factor = 0.99  # γ: 미래 보상을 얼마나 중시?
        self.epsilon = 1.0  # 탐색 확률 (초기엔 랜덤)
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        # 행동: 0=왼쪽, 1=오른쪽, 2=잡기, 3=놓기
        self.actions = ['LEFT', 'RIGHT', 'GRASP', 'RELEASE']

    def choose_action(self, state):
        # ε-greedy 전략: 탐색 vs 활용
        if np.random.random() < self.epsilon:
            # 탐색: 랜덤 행동 (새로운 전략 시도)
            return np.random.randint(0, len(self.actions))
        else:
            # 활용: Q-Table에서 최선의 행동 선택
            return np.argmax(self.q_table[state])

    def update_q_value(self, state, action, reward, next_state):
        # Bellman 방정식: Q-Learning의 핵심
        # Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]

        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])

        # TD Error (Temporal Difference)
        td_target = reward + self.discount_factor * max_next_q
        td_error = td_target - current_q

        # Q-Value 업데이트
        new_q = current_q + self.learning_rate * td_error
        self.q_table[state, action] = new_q

        # ε 감소 (점점 탐색 줄이고 활용 증가)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, episodes=1000):
        for episode in range(episodes):
            state = 0  # 시작 위치
            total_reward = 0

            for step in range(50):  # 최대 50스텝
                # 행동 선택
                action = self.choose_action(state)

                # 환경에서 행동 실행
                next_state, reward, done = self.env_step(state, action)

                # Q-Value 업데이트
                self.update_q_value(state, action, reward, next_state)

                total_reward += reward
                state = next_state

                if done:
                    break

            if episode % 100 == 0:
                print(f"Episode {episode}: Total Reward = {total_reward:.2f}, ε = {self.epsilon:.3f}")

    def env_step(self, state, action):
        # 환경 시뮬레이션 (실제로는 물리 엔진 또는 실제 로봇)
        target_position = 7  # 물체 위치

        if action == 0:  # LEFT
            next_state = max(0, state - 1)
        elif action == 1:  # RIGHT
            next_state = min(9, state + 1)
        elif action == 2:  # GRASP
            if state == target_position:
                return state, 10.0, True  # 성공!
            else:
                return state, -1.0, False  # 실패
        else:  # RELEASE
            return state, -0.1, False

        # 목표에 가까워질수록 작은 보상
        distance_reward = -abs(next_state - target_position) * 0.1

        return next_state, distance_reward, False

# 학습 실행
agent = RobotArmQLearning()
agent.train(episodes=1000)

# 학습된 정책으로 실행
state = 0
for step in range(10):
    action = agent.choose_action(state)
    print(f"State {state}: Action = {agent.actions[action]}")
    state, _, done = agent.env_step(state, action)
    if done:
        print("✅ 물체를 성공적으로 잡았습니다!")
        break`}
            </pre>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-l-4 border-green-500">
            <h4 className="font-bold mb-2">📈 학습 과정</h4>
            <div className="space-y-2 text-sm">
              <div className="flex items-start gap-2">
                <span className="font-bold text-green-600">Episode 1-100:</span>
                <span>랜덤하게 움직이며 환경 탐색 (ε ≈ 1.0)</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="font-bold text-blue-600">Episode 100-500:</span>
                <span>좋은 행동 패턴 발견 시작 (ε ≈ 0.5)</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="font-bold text-purple-600">Episode 500-1000:</span>
                <span>최적 전략 활용, 거의 항상 성공 (ε ≈ 0.01)</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 2. Deep Q-Network (DQN) */}
      <section className="my-8">
        <h2 className="flex items-center gap-2">
          <Zap className="text-purple-600" />
          2. Deep Q-Network (DQN) - 딥러닝 + 강화학습
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg mb-6">
          <h3 className="text-xl font-bold mb-4">🧠 신경망으로 Q-Value 예측</h3>

          <div className="bg-purple-50 dark:bg-purple-900/20 p-5 rounded-lg mb-6">
            <h4 className="font-bold mb-3">Q-Learning의 한계와 DQN의 해결책</h4>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h5 className="font-bold text-sm mb-2 text-red-600">Q-Learning 한계</h5>
                <ul className="text-sm space-y-1">
                  <li>❌ 상태가 많으면 Q-Table 폭발</li>
                  <li>❌ 연속적 상태 처리 불가</li>
                  <li>❌ 예: 로봇 관절 각도 (무한대)</li>
                </ul>
              </div>
              <div className="border-l-2 border-purple-300 pl-4">
                <h5 className="font-bold text-sm mb-2 text-green-600">DQN 해결책</h5>
                <ul className="text-sm space-y-1">
                  <li>✅ 신경망이 Q-Value 근사</li>
                  <li>✅ 연속 상태도 처리 가능</li>
                  <li>✅ 이미지 입력도 가능 (CNN)</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto mb-4">
            <pre className="text-sm">
{`# DQN: 이미지로 로봇 제어 학습
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DQNetwork(nn.Module):
    """신경망으로 Q-Value 예측"""
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()

        # 3층 신경망
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)  # 각 행동의 Q-Value
        return q_values

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # 주 네트워크와 타겟 네트워크 (Double DQN)
        self.q_network = DQNetwork(state_size, action_size)
        self.target_network = DQNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # 경험 재현 메모리 (Experience Replay)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64

        # 학습 파라미터
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def remember(self, state, action, reward, next_state, done):
        """경험을 메모리에 저장"""
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()

    def replay(self):
        """경험 재현으로 학습"""
        if len(self.memory) < self.batch_size:
            return

        # 랜덤 샘플링
        batch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in batch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

            # 현재 Q-Value
            q_values = self.q_network(state_tensor)
            q_value = q_values[0][action]

            # 타겟 Q-Value (Double DQN)
            with torch.no_grad():
                next_q_values = self.target_network(next_state_tensor)
                max_next_q = torch.max(next_q_values)
                target_q = reward if done else reward + self.gamma * max_next_q

            # 손실 계산 및 역전파
            loss = nn.MSELoss()(q_value, torch.FloatTensor([target_q]))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # ε 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        """타겟 네트워크 업데이트 (매 N 에피소드마다)"""
        self.target_network.load_state_dict(self.q_network.state_dict())

# 사용 예시: 로봇 팔 제어
state_size = 6  # 6 DOF 로봇 팔 관절 각도
action_size = 4  # 상하좌우 이동

agent = DQNAgent(state_size, action_size)

for episode in range(1000):
    state = env.reset()
    total_reward = 0

    for step in range(200):
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)

        # 경험 저장
        agent.remember(state, action, reward, next_state, done)

        # 학습
        agent.replay()

        state = next_state
        total_reward += reward

        if done:
            break

    # 타겟 네트워크 업데이트 (매 10 에피소드)
    if episode % 10 == 0:
        agent.update_target_network()

    print(f"Episode {episode}: Reward = {total_reward:.2f}")`}
            </pre>
          </div>

          <div className="bg-cyan-50 dark:bg-cyan-900/20 p-4 rounded-lg border-l-4 border-cyan-500">
            <h4 className="font-bold mb-2">🎮 DQN의 혁신적 기술</h4>
            <ul className="text-sm space-y-2">
              <li>
                <strong className="text-cyan-700 dark:text-cyan-300">• Experience Replay</strong>
                <p className="mt-1">과거 경험을 메모리에 저장해 반복 학습 → 데이터 효율성 향상</p>
              </li>
              <li>
                <strong className="text-cyan-700 dark:text-cyan-300">• Target Network</strong>
                <p className="mt-1">별도의 타겟 네트워크로 학습 안정화 → 진동 방지</p>
              </li>
              <li>
                <strong className="text-cyan-700 dark:text-cyan-300">• CNN 통합</strong>
                <p className="mt-1">이미지를 직접 입력받아 시각 정보로 학습 가능</p>
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* 3. PPO (Proximal Policy Optimization) */}
      <section className="my-8">
        <h2 className="flex items-center gap-2">
          <Award className="text-green-600" />
          3. PPO - 현대 로봇의 표준 알고리즘
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg mb-6">
          <h3 className="text-xl font-bold mb-4">🏆 OpenAI가 선택한 알고리즘</h3>

          <div className="bg-green-50 dark:bg-green-900/20 p-5 rounded-lg mb-6">
            <h4 className="font-bold mb-3">왜 PPO가 로봇 제어의 표준인가?</h4>
            <p className="text-sm mb-4">
              <strong>PPO (Proximal Policy Optimization)</strong>는 OpenAI가 개발한 알고리즘으로,
              <strong>안정성과 성능</strong>을 동시에 갖춘 최고의 강화학습 방법입니다.
            </p>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-800 p-4 rounded">
                <h5 className="font-bold text-sm mb-2">✅ 장점</h5>
                <ul className="text-sm space-y-1">
                  <li>• 학습 안정적 (폭발 없음)</li>
                  <li>• 샘플 효율적</li>
                  <li>• 구현 간단</li>
                  <li>• 다양한 작업에 적용 가능</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded">
                <h5 className="font-bold text-sm mb-2">🎯 사용 사례</h5>
                <ul className="text-sm space-y-1">
                  <li>• Tesla Bot 보행 학습</li>
                  <li>• Boston Dynamics 동작</li>
                  <li>• OpenAI Dota 2 챔피언</li>
                  <li>• ChatGPT RLHF 학습</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
            <pre className="text-sm">
{`# PPO를 이용한 휴머노이드 보행 학습
import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    """Actor (정책)와 Critic (가치) 네트워크"""
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()

        # 공유 레이어
        self.shared = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # Actor: 행동 확률 분포 출력
        self.actor_mean = nn.Linear(256, action_size)
        self.actor_std = nn.Parameter(torch.ones(action_size) * 0.1)

        # Critic: 상태 가치 출력
        self.critic = nn.Linear(256, 1)

    def forward(self, state):
        shared_features = self.shared(state)

        # Actor 출력
        action_mean = self.actor_mean(shared_features)
        action_std = self.actor_std.expand_as(action_mean)
        action_dist = Normal(action_mean, action_std)

        # Critic 출력
        state_value = self.critic(shared_features)

        return action_dist, state_value

class PPOAgent:
    def __init__(self, state_size, action_size):
        self.model = ActorCritic(state_size, action_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

        self.clip_epsilon = 0.2  # PPO 클리핑 파라미터
        self.gamma = 0.99
        self.gae_lambda = 0.95  # GAE (Generalized Advantage Estimation)

    def get_action(self, state):
        """정책에 따라 행동 샘플링"""
        state_tensor = torch.FloatTensor(state)
        action_dist, _ = self.model(state_tensor)

        action = action_dist.sample()
        log_prob = action_dist.log_prob(action).sum()

        return action.numpy(), log_prob

    def update(self, states, actions, old_log_probs, rewards, dones):
        """PPO 핵심: Clipped Surrogate Objective"""
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)

        # Advantage 계산 (GAE)
        advantages = self.compute_gae(states, rewards, dones)
        returns = advantages + self.compute_value(states)

        # PPO 업데이트 (여러 에포크 반복)
        for _ in range(10):
            # 현재 정책으로 log_prob 재계산
            action_dist, values = self.model(states)
            new_log_probs = action_dist.log_prob(actions).sum(dim=1)

            # Importance Sampling Ratio
            ratio = torch.exp(new_log_probs - old_log_probs)

            # Clipped Surrogate Loss (PPO 핵심!)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic Loss
            critic_loss = nn.MSELoss()(values.squeeze(), returns)

            # 총 손실
            loss = actor_loss + 0.5 * critic_loss

            # 역전파
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

    def compute_gae(self, states, rewards, dones):
        """Generalized Advantage Estimation"""
        with torch.no_grad():
            _, values = self.model(states)
            values = values.squeeze()

        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        return torch.FloatTensor(advantages)

# 휴머노이드 보행 학습
state_size = 44  # 관절 각도, 속도, 센서 데이터
action_size = 17  # 17개 관절 토크 제어

agent = PPOAgent(state_size, action_size)

for iteration in range(1000):
    states, actions, log_probs, rewards, dones = collect_trajectories(agent)
    agent.update(states, actions, log_probs, rewards, dones)

    print(f"Iteration {iteration}: Avg Reward = {np.mean(rewards):.2f}")`}
            </pre>
          </div>
        </div>
      </section>

      {/* 4. Model Predictive Control (MPC) */}
      <section className="my-8">
        <h2 className="flex items-center gap-2">
          <RefreshCw className="text-indigo-600" />
          4. MPC - 정밀한 로봇 제어
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg mb-6">
          <h3 className="text-xl font-bold mb-4">🎯 미래를 예측하며 최적 경로 계산</h3>

          <div className="bg-indigo-50 dark:bg-indigo-900/20 p-5 rounded-lg mb-6">
            <h4 className="font-bold mb-3">MPC란?</h4>
            <p className="text-sm mb-4">
              <strong>Model Predictive Control (모델 예측 제어)</strong>는 로봇의 물리 모델을 사용해
              <strong>미래 궤적을 시뮬레이션</strong>하고, 가장 좋은 경로를 선택하는 제어 방법입니다.
            </p>
            <div className="bg-white dark:bg-gray-800 p-4 rounded text-sm">
              <strong>작동 원리</strong>:
              <ol className="mt-2 space-y-1">
                <li>1. 현재 상태에서 미래 N스텝 시뮬레이션</li>
                <li>2. 최적화로 최선의 행동 시퀀스 찾기</li>
                <li>3. 첫 번째 행동만 실행</li>
                <li>4. 반복 (Receding Horizon)</li>
              </ol>
            </div>
          </div>

          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto mb-4">
            <pre className="text-sm">
{`# MPC for 로봇 팔 궤적 최적화
from scipy.optimize import minimize
import numpy as np

class MPCController:
    def __init__(self, horizon=10, dt=0.1):
        self.horizon = horizon  # 예측 범위
        self.dt = dt  # 시간 간격

    def predict_trajectory(self, current_state, actions):
        """물리 모델로 미래 궤적 예측"""
        trajectory = [current_state]
        state = current_state.copy()

        for action in actions:
            # 로봇 동역학 모델 (간단한 예시)
            velocity = state[3:6]  # 속도
            acceleration = action  # 제어 입력

            # 오일러 적분
            new_velocity = velocity + acceleration * self.dt
            new_position = state[0:3] + new_velocity * self.dt

            state = np.concatenate([new_position, new_velocity])
            trajectory.append(state)

        return np.array(trajectory)

    def cost_function(self, actions, current_state, target_state):
        """비용 함수: 목표와의 차이 + 제어 에너지"""
        actions = actions.reshape(self.horizon, 3)  # (N, 3) 형태로 변환

        # 미래 궤적 예측
        trajectory = self.predict_trajectory(current_state, actions)

        # 목표 도달 비용
        final_state = trajectory[-1]
        position_error = np.linalg.norm(final_state[0:3] - target_state[0:3])

        # 제어 에너지 비용 (작은 힘 선호)
        control_cost = np.sum(actions**2) * 0.01

        # 경로 스무스니스 (급격한 변화 방지)
        smoothness_cost = np.sum(np.diff(actions, axis=0)**2) * 0.1

        total_cost = position_error + control_cost + smoothness_cost
        return total_cost

    def compute_optimal_action(self, current_state, target_state):
        """최적화로 최선의 행동 시퀀스 계산"""
        # 초기 추정 (영 입력)
        initial_guess = np.zeros(self.horizon * 3)

        # 최적화 실행
        result = minimize(
            self.cost_function,
            initial_guess,
            args=(current_state, target_state),
            method='SLSQP',
            bounds=[(-1, 1)] * (self.horizon * 3)  # 액션 범위
        )

        # 첫 번째 행동만 반환 (Receding Horizon)
        optimal_actions = result.x.reshape(self.horizon, 3)
        return optimal_actions[0]

# 사용 예시: 로봇 팔을 목표 위치로 이동
mpc = MPCController(horizon=10, dt=0.1)

current_state = np.array([0, 0, 0, 0, 0, 0])  # [x, y, z, vx, vy, vz]
target_state = np.array([1, 1, 1, 0, 0, 0])

for step in range(100):
    # 최적 제어 계산
    action = mpc.compute_optimal_action(current_state, target_state)

    # 실제 로봇에 명령 전달
    robot.apply_force(action)

    # 상태 업데이트 (실제 센서 측정)
    current_state = robot.get_state()

    # 목표 도달 확인
    if np.linalg.norm(current_state[0:3] - target_state[0:3]) < 0.01:
        print("✅ 목표 도달!")
        break`}
            </pre>
          </div>

          <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border-l-4 border-blue-500">
            <h4 className="font-bold mb-2">🏭 MPC 응용 사례</h4>
            <ul className="text-sm space-y-2">
              <li>
                <strong>• Boston Dynamics Atlas</strong>
                <p className="mt-1">전신 제어: 100Hz로 미래 궤적 최적화해 파쿠르 동작 수행</p>
              </li>
              <li>
                <strong>• Tesla Autopilot</strong>
                <p className="mt-1">차선 유지: 앞 차량과의 간격을 예측하며 최적 속도 계산</p>
              </li>
              <li>
                <strong>• 산업 로봇 팔</strong>
                <p className="mt-1">정밀 조립: 진동 최소화하며 빠른 이동 실현</p>
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* 5. Sim2Real */}
      <section className="my-8">
        <h2 className="flex items-center gap-2">
          <TrendingUp className="text-pink-600" />
          5. Sim2Real - 가상에서 학습, 현실에서 실행
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
          <h3 className="text-xl font-bold mb-4">🌐 NVIDIA Isaac Gym의 혁명</h3>

          <div className="bg-pink-50 dark:bg-pink-900/20 p-5 rounded-lg mb-6">
            <h4 className="font-bold mb-3">왜 시뮬레이션이 필수인가?</h4>
            <p className="text-sm mb-4">
              실제 로봇으로 학습하려면 <strong>시간과 비용이 엄청납니다</strong>.
              시뮬레이션에서는 <strong>수천 배 빠르게, 무한 반복</strong> 가능합니다.
            </p>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-800 p-4 rounded">
                <h5 className="font-bold text-sm mb-2 text-red-600">실제 로봇 학습</h5>
                <ul className="text-sm space-y-1">
                  <li>❌ 1시간 = 1시간 (실시간)</li>
                  <li>❌ 로봇 고장 위험</li>
                  <li>❌ 위험한 상황 테스트 불가</li>
                  <li>❌ 비용: $1M+ (로봇 + 인력)</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded border-2 border-green-500">
                <h5 className="font-bold text-sm mb-2 text-green-600">시뮬레이션 학습</h5>
                <ul className="text-sm space-y-1">
                  <li>✅ 1시간 = 1000시간 (병렬)</li>
                  <li>✅ 무한 시행착오</li>
                  <li>✅ 극한 상황 테스트 가능</li>
                  <li>✅ 비용: GPU 렌탈 비용만</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-r from-green-50 to-cyan-50 dark:from-green-900/20 dark:to-cyan-900/20 p-5 rounded-lg">
            <h4 className="font-bold mb-3">🎯 Sim2Real 성공 사례</h4>
            <div className="space-y-3 text-sm">
              <div>
                <strong className="text-green-700 dark:text-green-300">• NVIDIA Isaac Gym</strong>
                <p className="mt-1">GPU로 4096개 로봇을 동시에 시뮬레이션. 하루에 수백 년치 경험 학습.</p>
              </div>
              <div>
                <strong className="text-blue-700 dark:text-blue-300">• OpenAI Dactyl</strong>
                <p className="mt-1">로봇 손이 루빅스 큐브를 푸는 법을 시뮬레이션에서 학습 후 실제로 성공.</p>
              </div>
              <div>
                <strong className="text-purple-700 dark:text-purple-300">• Tesla Bot</strong>
                <p className="mt-1">가상 공장에서 수천 번 넘어지며 보행 학습. 실제 로봇은 즉시 걸음.</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Summary */}
      <div className="bg-gradient-to-r from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 border-l-4 border-orange-500 p-6 rounded-lg my-8">
        <h3 className="text-xl font-bold mb-3">📌 핵심 요약</h3>
        <ul className="space-y-2 text-sm">
          <li>✅ <strong>Q-Learning</strong>: 기본 강화학습, Q-Table로 최적 행동 저장</li>
          <li>✅ <strong>DQN</strong>: 신경망으로 Q-Value 근사, 이미지 입력 가능</li>
          <li>✅ <strong>PPO</strong>: 현대 로봇의 표준, 안정적이고 효율적</li>
          <li>✅ <strong>MPC</strong>: 미래 예측으로 정밀 제어, Boston Dynamics 핵심</li>
          <li>✅ <strong>Sim2Real</strong>: 시뮬레이션 학습 → 실제 로봇 배포</li>
        </ul>
      </div>

      {/* Next Chapter */}
      <div className="bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-500 p-6 rounded-lg">
        <h3 className="text-xl font-bold mb-2">다음 단계: IoT & Edge Computing</h3>
        <p className="text-gray-700 dark:text-gray-300">
          다음 챕터에서는 로봇이 <strong>실시간으로 판단</strong>하기 위한
          IoT 센서 네트워크와 엣지 AI 칩 기술을 학습합니다.
        </p>
      </div>
    </div>
  )
}