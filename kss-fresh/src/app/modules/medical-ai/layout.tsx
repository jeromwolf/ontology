import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Medical AI - 의료 AI 기술',
  description: '의료 영상 분석, 진단 보조, 신약 개발 AI 기술',
};

export default function MedicalAILayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
