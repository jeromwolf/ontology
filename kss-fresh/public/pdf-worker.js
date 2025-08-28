// PDF.js worker loader for Next.js
// This file helps load the PDF.js worker properly in Next.js environment

if (typeof window !== 'undefined' && 'Worker' in window) {
  // Import the worker from CDN
  importScripts('https://cdnjs.cloudflare.com/ajax/libs/pdf.js/5.3.93/pdf.worker.min.js');
}