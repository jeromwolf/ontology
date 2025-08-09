import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const query = searchParams.get('query');
    const page = searchParams.get('page') || '1';
    const perPage = searchParams.get('per_page') || '20';

    if (!query) {
      return NextResponse.json(
        { error: 'Query parameter is required' },
        { status: 400 }
      );
    }

    if (!process.env.UNSPLASH_ACCESS_KEY) {
      return NextResponse.json(
        { error: 'Unsplash API key not configured' },
        { status: 500 }
      );
    }

    const unsplashUrl = `https://api.unsplash.com/search/photos?query=${encodeURIComponent(query)}&page=${page}&per_page=${perPage}&orientation=landscape`;
    
    const response = await fetch(unsplashUrl, {
      headers: {
        'Authorization': `Client-ID ${process.env.UNSPLASH_ACCESS_KEY}`
      }
    });

    if (!response.ok) {
      throw new Error(`Unsplash API error: ${response.status}`);
    }

    const data = await response.json();
    
    // 응답 데이터를 우리 형식에 맞게 변환
    const formattedResults = {
      total: data.total,
      total_pages: data.total_pages,
      results: data.results.map((photo: any) => ({
        id: photo.id,
        description: photo.description || photo.alt_description || '',
        urls: {
          raw: photo.urls.raw,
          full: photo.urls.full,
          regular: photo.urls.regular,
          small: photo.urls.small,
          thumb: photo.urls.thumb
        },
        user: {
          name: photo.user.name,
          username: photo.user.username,
          profile_url: `https://unsplash.com/@${photo.user.username}`
        },
        download_url: photo.links.download_location,
        html_url: photo.links.html,
        width: photo.width,
        height: photo.height,
        color: photo.color,
        tags: photo.tags?.map((tag: any) => tag.title) || []
      }))
    };

    return NextResponse.json(formattedResults);

  } catch (error) {
    console.error('Image search error:', error);
    return NextResponse.json(
      { error: 'Failed to search images' },
      { status: 500 }
    );
  }
}

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
      throw new Error('Failed to fetch image');
    }

    const buffer = Buffer.from(await imageResponse.arrayBuffer());
    
    // 폴더 생성 및 파일 저장
    const fs = require('fs');
    const path = require('path');
    const uploadsDir = path.join(process.cwd(), 'public/images/unsplash');
    
    if (!fs.existsSync(uploadsDir)) {
      fs.mkdirSync(uploadsDir, { recursive: true });
    }
    
    const safeFilename = filename.replace(/[^a-zA-Z0-9.-]/g, '_');
    const filePath = path.join(uploadsDir, safeFilename);
    const publicPath = `/images/unsplash/${safeFilename}`;
    
    fs.writeFileSync(filePath, buffer);
    
    return NextResponse.json({
      success: true,
      localPath: publicPath,
      filename: safeFilename
    });

  } catch (error) {
    console.error('Image download error:', error);
    return NextResponse.json(
      { error: 'Failed to download image' },
      { status: 500 }
    );
  }
}