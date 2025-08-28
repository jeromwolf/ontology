// PDF.js 설정을 위한 중앙화된 설정 파일

export const configurePdfWorker = async () => {
  if (typeof window === 'undefined') return;

  const pdfjsLib = await import('pdfjs-dist');
  
  // 여러 방법으로 worker 설정 시도
  try {
    // 방법 1: CDN (가장 안정적)
    pdfjsLib.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.js`;
    console.log('PDF.js Worker 설정 완료 (CDN):', pdfjsLib.version);
  } catch (error) {
    console.error('PDF.js Worker 설정 실패:', error);
    
    // 방법 2: 로컬 파일 (fallback)
    try {
      pdfjsLib.GlobalWorkerOptions.workerSrc = '/pdf.worker.min.js';
      console.log('PDF.js Worker 설정 완료 (로컬)');
    } catch (localError) {
      console.error('로컬 Worker 설정도 실패:', localError);
    }
  }
  
  return pdfjsLib;
};