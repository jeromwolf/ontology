import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'AI Ethics & Governance - KSS',
  description: '책임감 있는 AI 개발과 윤리적 거버넌스 체계를 학습합니다.',
};

export default function AIEthicsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}