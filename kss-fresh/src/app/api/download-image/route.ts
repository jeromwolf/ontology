import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const { imageUrl, filename } = await request.json();

    if (!imageUrl || !filename) {
      return NextResponse.json(
        { error: 'imageUrl and filename are required' },
        { status: 400 }
      );
    }

    // 이미지 다운로드
    const imageResponse = await fetch(imageUrl);
    if (!imageResponse.ok) {
      throw new Error(`Failed to fetch image: ${imageResponse.status}`);
    }

    const buffer = await imageResponse.arrayBuffer();
    
    // 파일 확장자 확인 및 MIME 타입 설정
    const extension = filename.split('.').pop()?.toLowerCase() || 'png';
    const mimeType = extension === 'jpg' || extension === 'jpeg' 
      ? 'image/jpeg' 
      : 'image/png';

    // 브라우저에 파일로 다운로드되도록 헤더 설정
    return new NextResponse(buffer, {
      status: 200,
      headers: {
        'Content-Type': mimeType,
        'Content-Disposition': `attachment; filename="${filename}"`,
        'Content-Length': buffer.byteLength.toString(),
      },
    });

  } catch (error) {
    console.error('Image download error:', error);
    return NextResponse.json(
      { error: 'Failed to download image' },
      { status: 500 }
    );
  }
}