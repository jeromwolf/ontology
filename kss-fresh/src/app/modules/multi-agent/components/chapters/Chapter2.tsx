'use client';

import React from 'react';
import { MessageSquare } from 'lucide-react';
import References from '@/components/common/References';

export default function Chapter2() {
  return (
    <div className="space-y-8">
      {/* A2A 통신 */}
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          Agent-to-Agent Communication Protocol
        </h2>
        <div className="prose prose-lg dark:prose-invert max-w-none">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            A2A 통신은 에이전트 간 <strong>정보 교환, 작업 조정, 협력적 문제 해결</strong>을 가능하게 하는 
            핵심 메커니즘입니다. 효율적인 통신 프로토콜은 시스템의 성능과 확장성을 결정합니다.
          </p>
        </div>
      </section>

      <section className="bg-cyan-50 dark:bg-cyan-900/20 rounded-xl p-6">
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          통신 패턴과 프로토콜
        </h3>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-3">동기식 통신</h4>
            <div className="space-y-2 text-sm">
              <div className="p-2 bg-gray-50 dark:bg-gray-700 rounded">
                <p className="font-medium">Request-Response</p>
                <p className="text-gray-600 dark:text-gray-400">직접적인 질의응답 패턴</p>
              </div>
              <div className="p-2 bg-gray-50 dark:bg-gray-700 rounded">
                <p className="font-medium">RPC (Remote Procedure Call)</p>
                <p className="text-gray-600 dark:text-gray-400">원격 함수 호출 방식</p>
              </div>
            </div>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-3">비동기식 통신</h4>
            <div className="space-y-2 text-sm">
              <div className="p-2 bg-gray-50 dark:bg-gray-700 rounded">
                <p className="font-medium">Publish-Subscribe</p>
                <p className="text-gray-600 dark:text-gray-400">이벤트 기반 메시징</p>
              </div>
              <div className="p-2 bg-gray-50 dark:bg-gray-700 rounded">
                <p className="font-medium">Message Queue</p>
                <p className="text-gray-600 dark:text-gray-400">큐를 통한 비동기 처리</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          메시지 포맷과 프로토콜
        </h3>
        <div className="bg-gray-900 rounded-xl p-6 text-white">
          <pre className="overflow-x-auto">
            <code className="text-sm">{`// FIPA ACL (Agent Communication Language) 예시
{
  "performative": "REQUEST",
  "sender": "agent-1@system",
  "receiver": "agent-2@system",
  "content": {
    "action": "analyze_data",
    "params": {
      "dataset": "sales_2024",
      "metrics": ["revenue", "growth"]
    }
  },
  "language": "JSON",
  "protocol": "fipa-request",
  "conversation-id": "conv-123",
  "reply-with": "req-456",
  "timestamp": "2024-01-15T10:30:00Z"
}`}</code>
          </pre>
        </div>
      </section>

      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          통신 신뢰성과 보안
        </h3>
        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-4">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">메시지 보장</h4>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <li>• At-most-once</li>
              <li>• At-least-once</li>
              <li>• Exactly-once</li>
            </ul>
          </div>
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">인증/인가</h4>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <li>• Agent 신원 확인</li>
              <li>• 권한 검증</li>
              <li>• 암호화 통신</li>
            </ul>
          </div>
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">장애 처리</h4>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <li>• Timeout 관리</li>
              <li>• Retry 전략</li>
              <li>• Fallback 메커니즘</li>
            </ul>
          </div>
        </div>
      </section>

      <References
        sections={[
          {
            title: 'Communication Protocols & Standards',
            icon: 'book',
            color: 'border-orange-500',
            items: [
              {
                title: 'FIPA ACL: Agent Communication Language',
                description: 'FIPA의 표준 agent 통신 언어 규격',
                link: 'http://www.fipa.org/repository/aclspecs.html'
              },
              {
                title: 'KQML: Knowledge Query and Manipulation Language',
                description: 'Agent 간 지식 교환을 위한 프로토콜',
                link: 'http://www.cs.umbc.edu/kqml/'
              },
              {
                title: 'gRPC: High Performance RPC Framework',
                description: 'Google의 고성능 RPC 프레임워크',
                link: 'https://grpc.io/'
              },
              {
                title: 'MQTT: Message Queuing Telemetry Transport',
                description: 'IoT를 위한 경량 메시지 프로토콜',
                link: 'https://mqtt.org/'
              }
            ]
          },
          {
            title: 'A2A Communication Research',
            icon: 'paper',
            color: 'border-purple-500',
            items: [
              {
                title: 'Emergent Communication in Multi-Agent Reinforcement Learning',
                authors: 'Jakob Foerster, Ioannis Alexandros Assael, et al.',
                year: '2016',
                description: 'Multi-agent 학습에서 자발적 통신 언어 출현',
                link: 'https://arxiv.org/abs/1605.06676'
              },
              {
                title: 'Learning to Communicate with Deep Multi-Agent RL',
                authors: 'Jakob Foerster, Yannis M. Assael, et al.',
                year: '2016',
                description: 'Deep RL을 통한 agent 간 통신 학습',
                link: 'https://arxiv.org/abs/1605.06676'
              },
              {
                title: 'CommNet: Learning Multiagent Communication with Backpropagation',
                authors: 'Sainbayar Sukhbaatar, Arthur Szlam, Rob Fergus',
                year: '2016',
                description: 'End-to-end 통신 학습 아키텍처',
                link: 'https://arxiv.org/abs/1605.07736'
              },
              {
                title: 'TarMAC: Targeted Multi-Agent Communication',
                authors: 'Abhishek Das, Théophile Gervet, et al.',
                year: '2019',
                description: 'Attention 기반 선택적 통신 메커니즘',
                link: 'https://arxiv.org/abs/1810.11187'
              }
            ]
          },
          {
            title: 'Message Queue Systems',
            icon: 'web',
            color: 'border-blue-500',
            items: [
              {
                title: 'Apache Kafka',
                description: '분산 이벤트 스트리밍 플랫폼',
                link: 'https://kafka.apache.org/'
              },
              {
                title: 'RabbitMQ',
                description: 'AMQP 기반 메시지 브로커',
                link: 'https://www.rabbitmq.com/'
              },
              {
                title: 'Redis Pub/Sub',
                description: 'Redis 기반 Pub/Sub 메시징',
                link: 'https://redis.io/docs/manual/pubsub/'
              },
              {
                title: 'NATS.io',
                description: '고성능 클라우드 네이티브 메시징',
                link: 'https://nats.io/'
              }
            ]
          },
          {
            title: 'Security & Reliability',
            icon: 'web',
            color: 'border-green-500',
            items: [
              {
                title: 'Securing Agent Communication with TLS',
                description: 'Agent 통신 보안 베스트 프랙티스',
                link: 'https://www.cloudflare.com/learning/ssl/what-is-ssl/'
              },
              {
                title: 'Message Authentication Codes (MAC)',
                description: '메시지 인증 및 무결성 보장',
                link: 'https://csrc.nist.gov/glossary/term/message_authentication_code'
              },
              {
                title: 'Exactly-Once Semantics in Distributed Systems',
                description: '메시지 전달 보장 메커니즘',
                link: 'https://kafka.apache.org/documentation/#semantics'
              }
            ]
          }
        ]}
      />
    </div>
  );
}