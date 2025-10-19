'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { BookOpen, FileText } from 'lucide-react';
import KSSLogo from './icons/KSSLogo';
import UserMenu from './auth/UserMenu';

export default function Navigation() {
  const pathname = usePathname();

  const navItems = [
    { href: '/#modules', label: '모듈', icon: BookOpen },
    { href: '/papers', label: '논문', icon: FileText },
  ];

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-white/80 dark:bg-gray-900/80 backdrop-blur-md border-b border-gray-200 dark:border-gray-700">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <Link href="/" className="flex items-center gap-2">
              <KSSLogo className="w-8 h-8" />
              <span className="font-bold text-xl">KSS</span>
            </Link>
            
            <div className="ml-10 flex items-baseline space-x-4">
              {navItems.map((item) => {
                const Icon = item.icon;
                const isActive = pathname === item.href || 
                                (item.href !== '/' && pathname.startsWith(item.href));
                
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={`
                      flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors
                      ${isActive 
                        ? 'bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white' 
                        : 'text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800'
                      }
                    `}
                  >
                    <Icon className="w-4 h-4" />
                    {item.label}
                  </Link>
                );
              })}
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            {/* User Menu */}
            <UserMenu />
          </div>
        </div>
      </div>
    </nav>
  );
}