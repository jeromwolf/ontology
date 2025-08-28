// PDF 처리 유틸리티 함수들

export const MAX_PDF_SIZE = 10 * 1024 * 1024; // 10MB

export function validatePDFFile(file: File): { valid: boolean; error?: string } {
  // 파일 타입 확인
  if (!file.type.includes('pdf') && !file.name.toLowerCase().endsWith('.pdf')) {
    return { valid: false, error: 'PDF 파일이 아닙니다' };
  }
  
  // 파일 크기 확인
  if (file.size > MAX_PDF_SIZE) {
    return { 
      valid: false, 
      error: `파일이 너무 큽니다. 최대 ${MAX_PDF_SIZE / 1024 / 1024}MB까지 지원합니다.` 
    };
  }
  
  // 파일명 확인
  if (file.name.length > 255) {
    return { valid: false, error: '파일명이 너무 깁니다' };
  }
  
  return { valid: true };
}

// 텍스트 청킹 함수
export function chunkText(text: string, chunkSize: number = 1000, overlap: number = 200): string[] {
  const chunks: string[] = [];
  let start = 0;
  
  while (start < text.length) {
    const end = start + chunkSize;
    const chunk = text.slice(start, end);
    chunks.push(chunk);
    
    // 다음 청크는 overlap만큼 겹치게
    start = end - overlap;
  }
  
  return chunks;
}

// 텍스트 정제 함수
export function cleanPDFText(text: string): string {
  return text
    // 연속된 공백을 하나로
    .replace(/\s+/g, ' ')
    // 특수문자 정리
    .replace(/[^\x20-\x7E\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F\uA960-\uA97F\uD7B0-\uD7FF]/g, ' ')
    // 연속된 줄바꿈 정리
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

// PDF 메타데이터 포맷팅
export function formatPDFMetadata(metadata: any): string {
  const lines: string[] = ['📄 PDF 문서 정보:'];
  
  if (metadata.title) lines.push(`제목: ${metadata.title}`);
  if (metadata.author) lines.push(`작성자: ${metadata.author}`);
  if (metadata.subject) lines.push(`주제: ${metadata.subject}`);
  if (metadata.creator) lines.push(`생성 프로그램: ${metadata.creator}`);
  if (metadata.producer) lines.push(`PDF 변환기: ${metadata.producer}`);
  if (metadata.creationDate) {
    try {
      const date = new Date(metadata.creationDate);
      lines.push(`작성일: ${date.toLocaleDateString('ko-KR')}`);
    } catch (e) {
      // 날짜 파싱 실패시 무시
    }
  }
  
  return lines.join('\n');
}