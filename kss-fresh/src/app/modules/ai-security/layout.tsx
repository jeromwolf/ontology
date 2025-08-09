import { ReactNode } from 'react';

export default function AISecurityLayout({
  children,
}: {
  children: ReactNode;
}) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-red-50 via-gray-50 to-red-50 dark:from-gray-900 dark:via-red-950 dark:to-gray-900">
      <div className="container mx-auto px-4 py-8">
        {children}
      </div>
    </div>
  );
}