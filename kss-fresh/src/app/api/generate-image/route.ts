import { NextRequest, NextResponse } from 'next/server';
import OpenAI from 'openai';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

export async function POST(request: NextRequest) {
  let prompt = '';
  
  try {
    const body = await request.json();
    const { prompt: userPrompt, size = '1024x1024', quality = 'standard' } = body;
    prompt = userPrompt || '';

    if (!prompt) {
      return NextResponse.json(
        { error: 'Prompt is required' },
        { status: 400 }
      );
    }

    const response = await openai.images.generate({
      model: "dall-e-3",
      prompt: prompt,
      n: 1,
      size: size as any,
      quality: quality as any,
    });

    const imageUrl = response.data?.[0]?.url;

    // 이미지를 다운로드하고 public 폴더에 저장
    if (imageUrl) {
      try {
        const imageResponse = await fetch(imageUrl);
        const buffer = Buffer.from(await imageResponse.arrayBuffer());
        
        // 폴더 생성 (없으면)
        const fs = require('fs');
        const path = require('path');
        const uploadsDir = path.join(process.cwd(), 'public/images/generated');
        if (!fs.existsSync(uploadsDir)) {
          fs.mkdirSync(uploadsDir, { recursive: true });
        }
        
        // 파일명 생성 (timestamp 기반)
        const fileName = `generated-${Date.now()}.png`;
        const filePath = path.join(uploadsDir, fileName);
        const publicPath = `/images/generated/${fileName}`;
        
        // 파일 저장
        fs.writeFileSync(filePath, buffer);
        
        return NextResponse.json({
          success: true,
          imageUrl: imageUrl,
          localPath: publicPath,
          prompt: prompt
        });
      } catch (saveError) {
        console.error('Failed to save image:', saveError);
        // 저장 실패해도 원본 URL은 반환
        return NextResponse.json({
          success: true,
          imageUrl: imageUrl,
          localPath: null,
          prompt: prompt,
          warning: 'Image generated but failed to save locally'
        });
      }
    }

    return NextResponse.json(
      { error: 'Failed to generate image' },
      { status: 500 }
    );

  } catch (error: any) {
    console.error('Image generation error:', error);
    
    // OpenAI 특정 에러 처리
    if (error?.code === 'billing_hard_limit_reached') {
      return NextResponse.json({
        success: false,
        error: 'OpenAI 계정의 결제 한도에 도달했습니다.\n\n대안으로 Unsplash 검색 탭을 사용하여 고품질 무료 이미지를 찾아보세요.',
        errorType: 'billing_limit',
        prompt: prompt
      }, { status: 402 });
    }
    
    if (error?.code === 'insufficient_quota') {
      return NextResponse.json({
        success: false,
        error: 'OpenAI API 사용량이 초과되었습니다. 잠시 후 다시 시도해주세요.',
        errorType: 'quota_exceeded',
        prompt: prompt
      }, { status: 429 });
    }
    
    if (error?.status === 429) {
      return NextResponse.json({
        success: false,
        error: 'API 요청 한도를 초과했습니다. 잠시 후 다시 시도해주세요.',
        errorType: 'rate_limit',
        prompt: prompt
      }, { status: 429 });
    }
    
    return NextResponse.json({
      success: false,
      error: '이미지 생성 중 오류가 발생했습니다. 다시 시도해주세요.',
      errorType: 'general_error',
      prompt: prompt
    }, { status: 500 });
  }
}