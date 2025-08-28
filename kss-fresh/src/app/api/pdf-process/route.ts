import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  console.log('[PDF Process API] 호출됨:', new Date().toISOString());
  
  try {
    // FormData 파싱
    const formData = await request.formData();
    const file = formData.get('file') as File;
    
    if (!file) {
      return NextResponse.json({ error: '파일이 없습니다' }, { status: 400 });
    }
    
    console.log('[PDF Process API] 파일 정보:', {
      name: file.name,
      size: file.size,
      type: file.type
    });
    
    // PDF 파일 확인
    const isPDF = file.type === 'application/pdf' || 
                  file.name.toLowerCase().endsWith('.pdf');
    
    if (!isPDF) {
      return NextResponse.json({ 
        error: 'PDF 파일이 아닙니다',
        details: `파일 타입: ${file.type}` 
      }, { status: 400 });
    }
    
    // 파일 크기 체크 (10MB)
    const MAX_SIZE = 10 * 1024 * 1024;
    if (file.size > MAX_SIZE) {
      return NextResponse.json({ 
        error: '파일이 너무 큽니다',
        details: `최대 10MB까지 지원합니다. 현재: ${(file.size / 1024 / 1024).toFixed(1)}MB`
      }, { status: 400 });
    }
    
    try {
      // Buffer로 변환
      const buffer = Buffer.from(await file.arrayBuffer());
      console.log('[PDF Process API] Buffer 생성:', buffer.length, 'bytes');
      
      // pdf-parse 시도
      let pdfParse;
      try {
        pdfParse = require('pdf-parse');
      } catch (err) {
        console.error('[PDF Process API] pdf-parse 로드 실패:', err);
        // 시뮬레이션 모드로 전환
        return NextResponse.json({
          text: generateSimulationText(file.name),
          pageCount: 1,
          metadata: {
            fileName: file.name,
            fileSize: file.size,
            fileType: file.type,
            isSimulation: true
          }
        });
      }
      
      // PDF 파싱 시작
      console.log('[PDF Process API] PDF 파싱 시작...');
      const startTime = Date.now();
      
      // 타임아웃 설정 (10초)
      const parsePromise = pdfParse(buffer);
      const timeoutPromise = new Promise((_, reject) => 
        setTimeout(() => reject(new Error('PDF 파싱 타임아웃')), 10000)
      );
      
      const data = await Promise.race([parsePromise, timeoutPromise]) as any;
      const parseTime = Date.now() - startTime;
      
      console.log(`[PDF Process API] 파싱 완료 (${parseTime}ms):`, {
        pages: data.numpages,
        textLength: data.text?.length || 0
      });
      
      // 텍스트가 없는 경우
      if (!data.text || data.text.trim().length === 0) {
        return NextResponse.json({
          text: `PDF 파일 "${file.name}"에서 텍스트를 찾을 수 없습니다.

이미지 기반 PDF이거나 스캔된 문서일 수 있습니다.
텍스트가 포함된 PDF 파일을 사용해주세요.

파일 정보:
- 페이지 수: ${data.numpages || 0}
- 크기: ${(file.size / 1024).toFixed(1)}KB`,
          pageCount: data.numpages || 0,
          metadata: {
            fileName: file.name,
            fileSize: file.size,
            fileType: file.type,
            isEmpty: true
          }
        });
      }
      
      // 텍스트 정제
      const cleanedText = data.text
        .replace(/\s+/g, ' ')
        .replace(/\n{3,}/g, '\n\n')
        .trim();
      
      return NextResponse.json({
        text: cleanedText,
        pageCount: data.numpages || 1,
        metadata: {
          fileName: file.name,
          fileSize: file.size,
          fileType: file.type,
          title: data.info?.Title || '',
          author: data.info?.Author || '',
          textLength: cleanedText.length,
          parseTime: parseTime
        }
      });
      
    } catch (parseError: any) {
      console.error('[PDF Process API] 파싱 오류:', parseError);
      
      // 타임아웃 오류
      if (parseError.message?.includes('타임아웃')) {
        return NextResponse.json({
          error: 'PDF 처리 시간 초과',
          details: 'PDF 파일이 너무 복잡하거나 크기가 큽니다. 더 작은 파일로 시도해주세요.'
        }, { status: 408 });
      }
      
      // 시뮬레이션 텍스트 반환
      return NextResponse.json({
        text: generateSimulationText(file.name),
        pageCount: 1,
        metadata: {
          fileName: file.name,
          fileSize: file.size,
          fileType: file.type,
          isSimulation: true,
          error: parseError.message
        }
      });
    }
    
  } catch (error: any) {
    console.error('[PDF Process API] 전체 오류:', error);
    return NextResponse.json({ 
      error: 'PDF 처리 실패', 
      details: error.message || 'Unknown error' 
    }, { status: 500 });
  }
}

// 시뮬레이션 텍스트 생성
function generateSimulationText(fileName: string): string {
  return `[PDF 시뮬레이션 모드]

파일: ${fileName}

현재 PDF 처리 서버에 일시적인 문제가 있어 시뮬레이션 텍스트를 표시합니다.

RAG 시스템 테스트를 위한 샘플 텍스트:

1. RAG (Retrieval-Augmented Generation) 개요
RAG는 검색 기반 생성 모델로, 대규모 언어 모델(LLM)의 한계를 극복하기 위해 개발되었습니다.
외부 지식 베이스에서 관련 정보를 검색한 후, 이를 바탕으로 더 정확하고 신뢰할 수 있는 답변을 생성합니다.

2. RAG의 주요 구성 요소
- 문서 전처리: 텍스트를 청크로 분할하고 임베딩으로 변환
- 벡터 데이터베이스: 임베딩을 저장하고 유사도 검색 수행
- 검색 모듈: 쿼리에 가장 관련된 문서 청크 검색
- 생성 모듈: 검색된 컨텍스트를 활용하여 답변 생성

3. RAG의 장점
- 최신 정보 활용: 지식 베이스를 업데이트하여 최신 정보 제공
- 환각 현상 감소: 실제 문서를 기반으로 답변하여 정확도 향상
- 출처 추적 가능: 답변의 근거가 되는 문서 확인 가능
- 도메인 특화: 특정 분야의 전문 지식 활용 가능

4. RAG 구현 시 고려사항
- 청킹 전략: 문서를 어떻게 분할할 것인가?
- 임베딩 모델: 어떤 임베딩 모델을 사용할 것인가?
- 검색 알고리즘: 단순 유사도 검색 vs 하이브리드 검색
- 컨텍스트 길이: LLM의 컨텍스트 윈도우 크기 고려

이 시뮬레이션 텍스트로 RAG 시스템의 기본 기능을 테스트할 수 있습니다.`;
}