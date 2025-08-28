import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;
    
    if (!file) {
      return NextResponse.json({ error: '파일이 없습니다' }, { status: 400 });
    }
    
    // 파일 크기 체크
    const MAX_SIZE = 5 * 1024 * 1024; // 5MB
    if (file.size > MAX_SIZE) {
      return NextResponse.json({ 
        error: '파일이 너무 큽니다',
        details: `최대 5MB까지 지원합니다. 현재: ${(file.size / 1024 / 1024).toFixed(1)}MB`
      }, { status: 400 });
    }
    
    console.log('간단한 PDF 처리:', file.name, file.size);
    
    // 현재는 시뮬레이션 텍스트 반환
    const simulatedText = `
[PDF 파일: ${file.name}]

현재 서버에서 PDF 처리 중 문제가 발생하고 있습니다.
임시로 다음과 같은 대안을 사용해주세요:

1. PDF를 텍스트 파일로 변환
   - Google Docs에서 열기 → 파일 → 다운로드 → 텍스트(.txt)
   - Adobe Acrobat → 파일 → 내보내기 → 텍스트
   
2. 온라인 변환 도구 사용
   - smallpdf.com
   - ilovepdf.com
   - pdf2txt.com

3. 샘플 텍스트 파일 사용
   - /sample-rag-data.txt 다운로드 후 업로드

파일 정보:
- 이름: ${file.name}
- 크기: ${(file.size / 1024).toFixed(1)}KB
- 타입: ${file.type}
`;
    
    return NextResponse.json({ 
      text: simulatedText,
      pageCount: 1,
      metadata: {
        fileName: file.name,
        fileSize: file.size,
        fileType: file.type
      }
    });
    
  } catch (error: any) {
    console.error('간단한 PDF 처리 오류:', error);
    return NextResponse.json({ 
      error: 'PDF 처리 실패',
      details: error.message 
    }, { status: 500 });
  }
}