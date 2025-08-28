'use client'

// 간단한 PDF 텍스트 추출기 - PDF.js 없이 브라우저 API 사용
export async function extractTextFromPDFSimple(file: File): Promise<string> {
  try {
    // PDF 파일 기본 정보
    const fileInfo = `파일명: ${file.name}
크기: ${(file.size / 1024).toFixed(1)}KB
타입: ${file.type}
수정일: ${new Date(file.lastModified).toLocaleString('ko-KR')}

⚠️ 간단한 PDF 리더 모드
`;

    // PDF 파일을 텍스트로 읽기 시도 (일부 PDF에서만 작동)
    try {
      const text = await file.text();
      
      // PDF에서 읽을 수 있는 텍스트 추출
      const readableText = text
        .replace(/[^\x20-\x7E\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F\uA960-\uA97F\uD7B0-\uD7FF]/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
      
      if (readableText.length > 100) {
        return fileInfo + '\n추출된 텍스트:\n' + readableText;
      }
    } catch (e) {
      // 텍스트 추출 실패 무시
    }

    // 기본 메시지
    return fileInfo + `
이 PDF는 암호화되었거나 이미지 기반일 수 있습니다.

PDF 텍스트 추출 대안:
1. Adobe Acrobat Reader에서 텍스트 복사
2. Google Drive에서 Google Docs로 변환
3. 온라인 PDF to Text 변환기 사용
4. OCR 도구 사용 (이미지 PDF의 경우)

또는 다른 형식으로 업로드해주세요:
- TXT, MD: 일반 텍스트 파일
- JSON: 구조화된 데이터
- CSV: 표 형식 데이터
- XML: 마크업 데이터`;
  } catch (error) {
    console.error('간단한 PDF 추출 오류:', error);
    throw new Error('PDF 파일을 읽을 수 없습니다. 다른 형식으로 변환해주세요.');
  }
}