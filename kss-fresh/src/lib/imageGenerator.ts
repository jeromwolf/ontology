// Image generation utility for system tools
export interface ImageGenerationOptions {
  prompt: string;
  size?: '1024x1024' | '1024x1792' | '1792x1024';
  quality?: 'standard' | 'hd';
  style?: 'vivid' | 'natural';
}

export interface ImageGenerationResult {
  success: boolean;
  imageUrl?: string;
  localPath?: string;
  prompt: string;
  error?: string;
}

export async function generateImage(options: ImageGenerationOptions): Promise<ImageGenerationResult> {
  try {
    const response = await fetch('/api/generate-image', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(options),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    return result;

  } catch (error) {
    console.error('Image generation failed:', error);
    return {
      success: false,
      prompt: options.prompt,
      error: error instanceof Error ? error.message : 'Unknown error'
    };
  }
}

// 교육 컨텐츠용 프롬프트 템플릿
export const educationalPrompts = {
  transformer: (style: string = 'technical diagram') => 
    `Create a ${style} showing the Transformer architecture with encoder and decoder stacks, attention mechanisms, and data flow arrows. Clean, educational, suitable for learning materials.`,
  
  attention: (style: string = 'visualization') => 
    `Create a ${style} explaining attention mechanism in neural networks, showing how tokens attend to each other with connecting lines and weights.`,
  
  neuralNetwork: (layers: number = 3, style: string = 'clean diagram') => 
    `Create a ${style} of a ${layers}-layer neural network with nodes, connections, and clear labels. Educational style, suitable for learning materials.`,
  
  dataFlow: (concept: string, style: string = 'flowchart') => 
    `Create a ${style} showing the data flow of ${concept} with clear arrows, labels, and processing steps. Technical but easy to understand.`
};

// System Tools에서 사용할 수 있는 래퍼 함수
export async function generateEducationalImage(
  concept: keyof typeof educationalPrompts | string,
  customPrompt?: string,
  options?: Partial<ImageGenerationOptions>
): Promise<ImageGenerationResult> {
  const prompt = customPrompt || 
    (typeof concept === 'string' && concept in educationalPrompts 
      ? educationalPrompts[concept as keyof typeof educationalPrompts]()
      : concept
    );

  return await generateImage({
    prompt,
    size: '1024x1024',
    quality: 'standard',
    style: 'natural',
    ...options
  });
}