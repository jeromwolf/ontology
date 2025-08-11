import { Metadata } from 'next';
import { moduleMetadata } from './metadata';

export const metadata: Metadata = {
  title: `${moduleMetadata.title} - KSS`,
  description: moduleMetadata.description,
};

export default function Layout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}