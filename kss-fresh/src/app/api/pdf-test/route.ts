import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  console.log('PDF 테스트 API 호출');
  
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;
    
    if (!file) {
      return NextResponse.json({ error: 'No file' }, { status: 400 });
    }
    
    console.log('파일 받음:', {
      name: file.name,
      size: file.size,
      type: file.type
    });
    
    // 간단히 파일 정보만 반환
    return NextResponse.json({ 
      success: true,
      fileName: file.name,
      fileSize: file.size,
      fileType: file.type,
      message: 'PDF 파일 수신 확인'
    });
    
  } catch (error: any) {
    console.error('테스트 API 오류:', error);
    return NextResponse.json({ 
      error: error.message 
    }, { status: 500 });
  }
}