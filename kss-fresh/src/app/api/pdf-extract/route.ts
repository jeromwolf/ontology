import { NextRequest, NextResponse } from 'next/server';
import { cleanPDFText, MAX_PDF_SIZE } from '@/lib/pdf-utils';

// Dynamic import to avoid issues with Next.js
async function extractPDFText(buffer: Buffer): Promise<{ text: string; pages: number; info: any }> {
  try {
    console.log('pdf-parse 라이브러리 로딩...');
    
    // pdf-parse 동적 import
    let pdfParse;
    try {
      pdfParse = (await import('pdf-parse')).default;
    } catch (importError) {
      console.error('pdf-parse import 실패:', importError);
      // 대체 방법 시도
      pdfParse = require('pdf-parse');
    }
    
    console.log('PDF 파싱 중... 버퍼 크기:', buffer.length);
    
    // 파싱 옵션 설정
    const options = {
      max: 0, // 모든 페이지 파싱
      version: 'default'
    };
    
    const data = await pdfParse(buffer, options);
    
    console.log('PDF 파싱 완료 - 페이지 수:', data.numpages, ', 텍스트 길이:', data.text.length);
    return {
      text: data.text || '',
      pages: data.numpages || 0,
      info: data.info || {}
    };
  } catch (error) {
    console.error('extractPDFText 내부 오류:', error);
    throw error;
  }
}

export async function POST(request: NextRequest) {
  console.log('PDF API 호출됨 - 타임스탬프:', new Date().toISOString());
  
  try {
    // FormData 파싱 시작
    console.log('FormData 파싱 시작...');
    const formData = await request.formData();
    const file = formData.get('file') as File;
    
    if (!file) {
      console.error('파일이 전송되지 않음');
      return NextResponse.json({ error: 'No file provided' }, { status: 400 });
    }
    
    // 파일 타입 확인
    if (!file.type.includes('pdf') && !file.name.toLowerCase().endsWith('.pdf')) {
      console.error('PDF가 아닌 파일:', file.type, file.name);
      return NextResponse.json({ error: 'PDF 파일이 아닙니다' }, { status: 400 });
    }
    
    // 파일 크기 확인
    if (file.size > MAX_PDF_SIZE) {
      console.error('파일 크기 초과:', file.size);
      return NextResponse.json({ 
        error: '파일이 너무 큽니다', 
        details: `최대 ${MAX_PDF_SIZE / 1024 / 1024}MB까지 지원합니다. 현재 파일: ${(file.size / 1024 / 1024).toFixed(1)}MB` 
      }, { status: 400 });
    }
    
    // 파일 정보 로깅
    console.log('PDF 파일 정보:', {
      name: file.name,
      size: `${(file.size / 1024).toFixed(1)}KB`,
      type: file.type
    });
    
    // File을 Buffer로 변환
    console.log('ArrayBuffer 변환 중...');
    const arrayBuffer = await file.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);
    console.log('Buffer 생성 완료:', buffer.length, 'bytes');
    
    try {
      console.log('PDF 파싱 함수 호출...');
      // PDF 텍스트 추출
      const { text, pages, info } = await extractPDFText(buffer);
      
      console.log('PDF 추출 결과:', {
        pages,
        textLength: text.length,
        hasText: text.trim().length > 0
      });
      
      // 빈 텍스트 체크
      if (!text || text.trim().length === 0) {
        console.warn('추출된 텍스트가 비어있음');
        return NextResponse.json({ 
          text: `PDF 파일 "${file.name}"에서 텍스트를 찾을 수 없습니다.

이미지 기반 PDF이거나 스캔된 문서일 수 있습니다.
OCR(광학 문자 인식)이 필요할 수 있습니다.

파일 정보:
- 페이지 수: ${pages}
- 크기: ${(file.size / 1024).toFixed(1)}KB`,
          pageCount: pages,
          metadata: {
            fileName: file.name,
            fileSize: file.size,
            fileType: file.type,
            isEmpty: true
          }
        });
      }
      
      // 텍스트 정제
      console.log('텍스트 정제 중...');
      const cleanedText = cleanPDFText(text);
      console.log('정제 완료, 최종 텍스트 길이:', cleanedText.length);
      
      // 성공적으로 텍스트 추출
      return NextResponse.json({ 
        text: cleanedText,
        pageCount: pages,
        metadata: {
          fileName: file.name,
          fileSize: file.size,
          fileType: file.type,
          title: info?.Title || '',
          author: info?.Author || '',
          subject: info?.Subject || '',
          creator: info?.Creator || '',
          producer: info?.Producer || '',
          creationDate: info?.CreationDate || '',
          modificationDate: info?.ModificationDate || '',
          textLength: cleanedText.length
        }
      });
      
    } catch (parseError: any) {
      console.error('PDF 파싱 중 오류 발생:', parseError.message, parseError.stack);
      
      // 암호화된 PDF나 손상된 파일 처리
      if (parseError.message?.includes('encrypted')) {
        return NextResponse.json({ 
          error: 'PDF가 암호로 보호되어 있습니다', 
          details: '비밀번호가 필요한 PDF 파일입니다.' 
        }, { status: 400 });
      }
      
      if (parseError.message?.includes('Invalid')) {
        return NextResponse.json({ 
          error: '유효하지 않은 PDF 파일', 
          details: '파일이 손상되었거나 올바른 PDF 형식이 아닙니다.' 
        }, { status: 400 });
      }
      
      // 기타 파싱 오류
      return NextResponse.json({ 
        error: 'PDF 파싱 실패', 
        details: parseError.message || 'PDF 파일을 읽을 수 없습니다.' 
      }, { status: 500 });
    }
    
  } catch (error: any) {
    console.error('PDF API 전체 오류:', error.message, error.stack);
    return NextResponse.json({ 
      error: 'PDF 처리 실패', 
      details: error.message || 'Unknown error' 
    }, { status: 500 });
  }
}