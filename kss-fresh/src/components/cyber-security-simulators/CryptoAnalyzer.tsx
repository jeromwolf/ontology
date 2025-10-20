import React, { useState } from 'react';
import { Lock, Unlock, Key, Hash } from 'lucide-react';
import SimulatorNav from './SimulatorNav';

export default function CryptoAnalyzer() {
  const [inputText, setInputText] = useState<string>('Hello, World!');
  const [encryptionKey, setEncryptionKey] = useState<string>('MySecretKey123');
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<string>('aes');
  const [encryptedText, setEncryptedText] = useState<string>('');
  const [decryptedText, setDecryptedText] = useState<string>('');
  const [hashOutput, setHashOutput] = useState<string>('');

  const algorithms = [
    {
      id: 'aes',
      name: 'AES-256',
      type: 'symmetric',
      desc: '대칭키 암호화 (가장 안전)',
    },
    {
      id: 'des',
      name: 'DES',
      type: 'symmetric',
      desc: '레거시 대칭키 (취약)',
    },
    {
      id: 'rsa',
      name: 'RSA-2048',
      type: 'asymmetric',
      desc: '공개키 암호화',
    },
    {
      id: 'caesar',
      name: 'Caesar Cipher',
      type: 'classical',
      desc: '고전 암호 (교육용)',
    },
  ];

  const hashAlgorithms = [
    { id: 'md5', name: 'MD5', desc: '128-bit (취약, 권장안함)' },
    { id: 'sha1', name: 'SHA-1', desc: '160-bit (취약)' },
    { id: 'sha256', name: 'SHA-256', desc: '256-bit (안전)' },
    { id: 'sha512', name: 'SHA-512', desc: '512-bit (매우 안전)' },
  ];

  // Caesar Cipher implementation (simple example)
  const caesarEncrypt = (text: string, shift: number = 3): string => {
    return text
      .split('')
      .map((char) => {
        if (char.match(/[a-z]/i)) {
          const code = char.charCodeAt(0);
          const base = code >= 65 && code <= 90 ? 65 : 97;
          return String.fromCharCode(((code - base + shift) % 26) + base);
        }
        return char;
      })
      .join('');
  };

  const caesarDecrypt = (text: string, shift: number = 3): string => {
    return caesarEncrypt(text, 26 - shift);
  };

  // Mock encryption (Base64 for demonstration)
  const mockEncrypt = (text: string, key: string, algorithm: string): string => {
    if (algorithm === 'caesar') {
      return caesarEncrypt(text, 3);
    }
    // For other algorithms, use Base64 as mock
    const combined = `${algorithm}:${key}:${text}`;
    return btoa(combined);
  };

  const mockDecrypt = (ciphertext: string, key: string, algorithm: string): string => {
    if (algorithm === 'caesar') {
      return caesarDecrypt(ciphertext, 3);
    }
    try {
      const decoded = atob(ciphertext);
      const parts = decoded.split(':');
      if (parts.length === 3 && parts[0] === algorithm && parts[1] === key) {
        return parts[2];
      }
      return '⚠️ 복호화 실패: 잘못된 키 또는 알고리즘';
    } catch {
      return '⚠️ 복호화 실패: 유효하지 않은 암호문';
    }
  };

  // Mock hash function
  const mockHash = (text: string, algorithm: string): string => {
    const hash = btoa(text + algorithm);
    switch (algorithm) {
      case 'md5':
        return hash.substring(0, 32).padEnd(32, '0');
      case 'sha1':
        return hash.substring(0, 40).padEnd(40, '0');
      case 'sha256':
        return hash.substring(0, 64).padEnd(64, '0');
      case 'sha512':
        return hash.substring(0, 128).padEnd(128, '0');
      default:
        return hash;
    }
  };

  const handleEncrypt = () => {
    const result = mockEncrypt(inputText, encryptionKey, selectedAlgorithm);
    setEncryptedText(result);
    setDecryptedText('');
  };

  const handleDecrypt = () => {
    if (!encryptedText) {
      alert('먼저 암호화를 실행하세요');
      return;
    }
    const result = mockDecrypt(encryptedText, encryptionKey, selectedAlgorithm);
    setDecryptedText(result);
  };

  const handleHash = (algorithm: string) => {
    const result = mockHash(inputText, algorithm);
    setHashOutput(result);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-slate-900 to-gray-900 text-white relative">
      <SimulatorNav />
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-4 flex items-center gap-3">
            <Key className="w-10 h-10 text-green-500" />
            Cryptography Analyzer
          </h1>
          <p className="text-xl text-gray-300">암호화/복호화 및 해시 함수 시뮬레이터</p>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {/* Encryption/Decryption */}
          <div className="bg-gray-800 rounded-xl p-6">
            <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
              <Lock className="w-7 h-7 text-blue-500" />
              암호화/복호화
            </h2>

            {/* Input */}
            <div className="mb-4">
              <label className="block text-sm font-semibold mb-2">평문 (Plain Text)</label>
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                className="w-full bg-gray-900 border border-gray-700 rounded-lg p-3 text-white h-24 resize-none"
                placeholder="암호화할 텍스트를 입력하세요"
              />
            </div>

            {/* Algorithm Selection */}
            <div className="mb-4">
              <label className="block text-sm font-semibold mb-2">암호화 알고리즘</label>
              <div className="grid grid-cols-2 gap-2">
                {algorithms.map((algo) => (
                  <button
                    key={algo.id}
                    onClick={() => setSelectedAlgorithm(algo.id)}
                    className={`p-3 rounded-lg text-left transition-all ${
                      selectedAlgorithm === algo.id
                        ? 'bg-blue-600 border-2 border-blue-400'
                        : 'bg-gray-900 border-2 border-gray-700 hover:border-gray-600'
                    }`}
                  >
                    <div className="font-bold text-sm">{algo.name}</div>
                    <div className="text-xs text-gray-400">{algo.desc}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Key */}
            <div className="mb-4">
              <label className="block text-sm font-semibold mb-2">암호화 키</label>
              <input
                type="text"
                value={encryptionKey}
                onChange={(e) => setEncryptionKey(e.target.value)}
                className="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-3 text-white"
                placeholder="암호화 키를 입력하세요"
              />
            </div>

            {/* Actions */}
            <div className="flex gap-2 mb-4">
              <button
                onClick={handleEncrypt}
                className="flex-1 bg-blue-600 hover:bg-blue-700 py-3 rounded-lg font-semibold flex items-center justify-center gap-2"
              >
                <Lock className="w-5 h-5" />
                암호화
              </button>
              <button
                onClick={handleDecrypt}
                className="flex-1 bg-green-600 hover:bg-green-700 py-3 rounded-lg font-semibold flex items-center justify-center gap-2"
              >
                <Unlock className="w-5 h-5" />
                복호화
              </button>
            </div>

            {/* Encrypted Text */}
            {encryptedText && (
              <div className="mb-4">
                <label className="block text-sm font-semibold mb-2">암호문 (Cipher Text)</label>
                <div className="bg-black rounded-lg p-3 font-mono text-xs break-all text-green-400">
                  {encryptedText}
                </div>
              </div>
            )}

            {/* Decrypted Text */}
            {decryptedText && (
              <div>
                <label className="block text-sm font-semibold mb-2">복호화 결과</label>
                <div
                  className={`rounded-lg p-3 ${
                    decryptedText.includes('⚠️')
                      ? 'bg-red-900/30 text-red-300'
                      : 'bg-green-900/30 text-green-300'
                  }`}
                >
                  {decryptedText}
                </div>
              </div>
            )}
          </div>

          {/* Hash Functions */}
          <div className="bg-gray-800 rounded-xl p-6">
            <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
              <Hash className="w-7 h-7 text-purple-500" />
              해시 함수
            </h2>

            <div className="mb-4">
              <label className="block text-sm font-semibold mb-2">해시할 텍스트</label>
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                className="w-full bg-gray-900 border border-gray-700 rounded-lg p-3 text-white h-24 resize-none"
                placeholder="해시할 텍스트를 입력하세요"
              />
            </div>

            <div className="mb-4">
              <label className="block text-sm font-semibold mb-2">해시 알고리즘 선택</label>
              <div className="grid grid-cols-2 gap-2">
                {hashAlgorithms.map((algo) => (
                  <button
                    key={algo.id}
                    onClick={() => handleHash(algo.id)}
                    className="p-3 rounded-lg bg-gray-900 border-2 border-gray-700 hover:border-purple-500 transition-all text-left"
                  >
                    <div className="font-bold text-sm">{algo.name}</div>
                    <div className="text-xs text-gray-400">{algo.desc}</div>
                  </button>
                ))}
              </div>
            </div>

            {hashOutput && (
              <div>
                <label className="block text-sm font-semibold mb-2">해시 출력</label>
                <div className="bg-black rounded-lg p-3 font-mono text-xs break-all text-purple-400 mb-2">
                  {hashOutput}
                </div>
                <div className="text-xs text-gray-400">길이: {hashOutput.length} characters</div>
              </div>
            )}

            <div className="mt-6 bg-purple-900/20 border-2 border-purple-500 rounded-lg p-4">
              <h3 className="font-bold text-purple-400 mb-2">💡 해시 함수 특징</h3>
              <ul className="text-sm text-gray-300 space-y-1">
                <li>• 단방향 함수 (복호화 불가능)</li>
                <li>• 입력이 같으면 출력도 항상 같음</li>
                <li>• 고정된 길이의 출력</li>
                <li>• 충돌 저항성 (서로 다른 입력 → 다른 출력)</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Comparison Table */}
        <div className="mt-6 bg-gray-800 rounded-xl p-6">
          <h2 className="text-2xl font-bold mb-6">암호화 알고리즘 비교</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-3 px-4">알고리즘</th>
                  <th className="text-left py-3 px-4">타입</th>
                  <th className="text-left py-3 px-4">키 길이</th>
                  <th className="text-left py-3 px-4">보안 수준</th>
                  <th className="text-left py-3 px-4">사용 사례</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-gray-700">
                  <td className="py-3 px-4 font-semibold">AES-256</td>
                  <td className="py-3 px-4">대칭키</td>
                  <td className="py-3 px-4">256-bit</td>
                  <td className="py-3 px-4">
                    <span className="text-green-400">매우 안전</span>
                  </td>
                  <td className="py-3 px-4">파일 암호화, VPN</td>
                </tr>
                <tr className="border-b border-gray-700">
                  <td className="py-3 px-4 font-semibold">DES</td>
                  <td className="py-3 px-4">대칭키</td>
                  <td className="py-3 px-4">56-bit</td>
                  <td className="py-3 px-4">
                    <span className="text-red-400">취약</span>
                  </td>
                  <td className="py-3 px-4">레거시 시스템</td>
                </tr>
                <tr className="border-b border-gray-700">
                  <td className="py-3 px-4 font-semibold">RSA-2048</td>
                  <td className="py-3 px-4">공개키</td>
                  <td className="py-3 px-4">2048-bit</td>
                  <td className="py-3 px-4">
                    <span className="text-green-400">안전</span>
                  </td>
                  <td className="py-3 px-4">디지털 서명, SSL/TLS</td>
                </tr>
                <tr className="border-b border-gray-700">
                  <td className="py-3 px-4 font-semibold">Caesar</td>
                  <td className="py-3 px-4">고전</td>
                  <td className="py-3 px-4">N/A</td>
                  <td className="py-3 px-4">
                    <span className="text-yellow-400">교육용</span>
                  </td>
                  <td className="py-3 px-4">암호학 학습</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
