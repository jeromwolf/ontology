'use client'

import { AudioLines } from 'lucide-react'

export default function Section3() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-green-100 dark:bg-green-900/20 flex items-center justify-center">
          <AudioLines className="text-green-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">5.3 오디오/음성 데이터 처리</h2>
          <p className="text-gray-600 dark:text-gray-400">음성 인식 및 오디오 기반 검색</p>
        </div>
      </div>

      <div className="space-y-6">
        <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl border border-green-200 dark:border-green-700">
          <h3 className="font-bold text-green-800 dark:text-green-200 mb-4">오디오 RAG 시스템</h3>

          <div className="prose prose-sm dark:prose-invert mb-4">
            <p className="text-gray-700 dark:text-gray-300">
              <strong>오디오 RAG는 음성 인식과 자연어 처리를 결합한 음향 기반 정보 검색 시스템입니다.</strong>
              OpenAI의 Whisper 모델을 활용하여 99% 정확도의 다국어 음성 인식을 구현하며,
              실시간 스트리밍과 배치 처리 모두를 지원합니다.
            </p>
            <p className="text-gray-700 dark:text-gray-300">
              <strong>오디오 RAG 핵심 기술:</strong>
            </p>
            <ul className="list-disc list-inside text-gray-700 dark:text-gray-300 space-y-1">
              <li><strong>Whisper ASR</strong>: 680,000시간 다국어 데이터로 훈련된 최고 성능 모델</li>
              <li><strong>단어별 타임스탬프</strong>: 정확한 시간 동기화로 구간별 검색 가능</li>
              <li><strong>MFCC 특성 추출</strong>: 13차원 멜-켑스트럼 계수로 음향 특성 분석</li>
              <li><strong>화자 분리</strong>: 다중 화자 환경에서 개별 발언자 구분</li>
            </ul>

            <div className="bg-emerald-50 dark:bg-emerald-900/20 p-4 rounded-lg border border-emerald-200 dark:border-emerald-700 mt-4">
              <h4 className="font-bold text-emerald-800 dark:text-emerald-200 mb-2">🎯 성능 벤치마크</h4>
              <div className="overflow-x-auto">
                <table className="min-w-full text-sm">
                  <thead>
                    <tr className="border-b border-emerald-300 dark:border-emerald-600">
                      <th className="text-left py-2 text-emerald-800 dark:text-emerald-200">모델</th>
                      <th className="text-left py-2 text-emerald-800 dark:text-emerald-200">WER</th>
                      <th className="text-left py-2 text-emerald-800 dark:text-emerald-200">처리속도</th>
                      <th className="text-left py-2 text-emerald-800 dark:text-emerald-200">언어지원</th>
                    </tr>
                  </thead>
                  <tbody className="text-emerald-700 dark:text-emerald-300">
                    <tr>
                      <td className="py-1">Whisper-base</td>
                      <td className="py-1">5.1%</td>
                      <td className="py-1">~6x실시간</td>
                      <td className="py-1">99개 언어</td>
                    </tr>
                    <tr>
                      <td className="py-1">Whisper-large</td>
                      <td className="py-1">2.8%</td>
                      <td className="py-1">~2x실시간</td>
                      <td className="py-1">99개 언어</td>
                    </tr>
                    <tr>
                      <td className="py-1">Google STT</td>
                      <td className="py-1">4.2%</td>
                      <td className="py-1">실시간</td>
                      <td className="py-1">125개 언어</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border border-blue-200 dark:border-blue-700 mt-4">
              <h4 className="font-bold text-blue-800 dark:text-blue-200 mb-2">🔊 실제 적용 분야</h4>
              <div className="grid md:grid-cols-3 gap-3 text-sm">
                <div>
                  <strong className="text-blue-800 dark:text-blue-200">콜센터 분석</strong>
                  <ul className="list-disc list-inside ml-2 text-blue-700 dark:text-blue-300 mt-1">
                    <li>고객 불만 키워드 검출</li>
                    <li>상담원 스크립트 준수 체크</li>
                    <li>감정 분석 및 품질 관리</li>
                  </ul>
                </div>
                <div>
                  <strong className="text-blue-800 dark:text-blue-200">회의 스마트 요약</strong>
                  <ul className="list-disc list-inside ml-2 text-blue-700 dark:text-blue-300 mt-1">
                    <li>핵심 결정사항 자동 추출</li>
                    <li>액션 아이템 할당 추적</li>
                    <li>다음 회의 어젠다 생성</li>
                  </ul>
                </div>
                <div>
                  <strong className="text-blue-800 dark:text-blue-200">방송 아카이빙</strong>
                  <ul className="list-disc list-inside ml-2 text-blue-700 dark:text-blue-300 mt-1">
                    <li>뉴스 자동 색인 생성</li>
                    <li>인물 발언 검색 시스템</li>
                    <li>실시간 자막 및 번역</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg overflow-hidden border border-slate-200 dark:border-slate-700">
            <pre className="text-sm text-slate-800 dark:text-slate-200 overflow-x-auto max-h-96 overflow-y-auto font-mono">
{`import whisper
import torch
import librosa
import numpy as np
from typing import List, Dict
from datetime import timedelta
import re

class AudioRAGSystem:
    def __init__(self, whisper_model_size: str = "base"):
        # Whisper 모델 로드 (음성 인식용)
        self.whisper_model = whisper.load_model(whisper_model_size)

        # 오디오 세그먼트 저장
        self.audio_segments = []
        self.transcripts = []

    def transcribe_audio(self, audio_path: str, language: str = "ko") -> Dict:
        """오디오 파일 전사"""
        print(f"오디오 전사 시작: {audio_path}")

        try:
            # Whisper로 전사
            result = self.whisper_model.transcribe(
                audio_path,
                language=language,
                word_timestamps=True  # 단어별 타임스탬프
            )

            return {
                'text': result['text'],
                'language': result['language'],
                'segments': result['segments'],
                'audio_path': audio_path,
                'success': True
            }

        except Exception as e:
            print(f"전사 실패: {e}")
            return {
                'text': '',
                'error': str(e),
                'success': False
            }

    def extract_audio_features(self, audio_path: str) -> Dict:
        """오디오 특성 추출"""
        try:
            # 오디오 로드
            y, sr = librosa.load(audio_path)

            # 기본 특성들
            features = {
                'duration': librosa.get_duration(y=y, sr=sr),
                'sample_rate': sr,
                'rms_energy': float(np.mean(librosa.feature.rms(y=y))),
                'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(y))),
                'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
                'tempo': float(librosa.beat.tempo(y=y, sr=sr)[0])
            }

            # MFCC 특성 (음성 인식에 중요)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = np.mean(mfccs, axis=1).tolist()
            features['mfcc_std'] = np.std(mfccs, axis=1).tolist()

            return features

        except Exception as e:
            print(f"특성 추출 실패: {e}")
            return {}

    def segment_transcript(self, transcript_data: Dict, segment_duration: int = 30) -> List[Dict]:
        """전사 텍스트를 의미있는 세그먼트로 분할"""
        segments = []

        if not transcript_data.get('segments'):
            return segments

        current_segment = {
            'start_time': 0,
            'end_time': segment_duration,
            'text': '',
            'words': [],
            'speaker_changes': 0
        }

        for segment in transcript_data['segments']:
            segment_start = segment['start']
            segment_end = segment['end']
            segment_text = segment['text'].strip()

            # 세그먼트 시간 범위 확인
            if segment_start < current_segment['end_time']:
                current_segment['text'] += ' ' + segment_text
                current_segment['words'].extend(segment.get('words', []))
            else:
                # 현재 세그먼트 저장
                if current_segment['text'].strip():
                    current_segment['word_count'] = len(current_segment['text'].split())
                    current_segment['summary'] = self.summarize_audio_segment(current_segment['text'])
                    segments.append(current_segment.copy())

                # 새 세그먼트 시작
                segment_num = int(segment_start // segment_duration)
                current_segment = {
                    'start_time': segment_num * segment_duration,
                    'end_time': (segment_num + 1) * segment_duration,
                    'text': segment_text,
                    'words': segment.get('words', []),
                    'speaker_changes': 0
                }

        # 마지막 세그먼트
        if current_segment['text'].strip():
            current_segment['word_count'] = len(current_segment['text'].split())
            current_segment['summary'] = self.summarize_audio_segment(current_segment['text'])
            segments.append(current_segment)

        return segments

    def summarize_audio_segment(self, text: str) -> str:
        """오디오 세그먼트 요약"""
        if len(text) < 50:
            return text

        # 문장 분리
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if len(sentences) <= 2:
            return text

        # 첫 문장과 마지막 문장으로 요약 구성
        summary = f"{sentences[0]}... {sentences[-1]}"
        return summary[:150]  # 150자로 제한

    def search_audio_content(self, query: str, audio_segments: List[Dict]) -> List[Dict]:
        """오디오 내용에서 검색"""
        results = []
        query_words = query.lower().split()

        for segment in audio_segments:
            segment_text = segment.get('text', '').lower()

            # 키워드 매칭 점수
            matches = sum(1 for word in query_words if word in segment_text)
            keyword_score = matches / len(query_words) if query_words else 0

            # 텍스트 포함 점수 (부분 문자열)
            substring_score = 0.5 if query.lower() in segment_text else 0

            # 최종 점수
            final_score = max(keyword_score, substring_score)

            if final_score > 0.2:  # 20% 이상 매칭시 포함
                # 관련 구문 추출
                relevant_phrases = self.extract_relevant_phrases(
                    segment_text, query_words
                )

                results.append({
                    'segment': segment,
                    'score': final_score,
                    'start_time': segment['start_time'],
                    'end_time': segment['end_time'],
                    'relevant_text': relevant_phrases,
                    'context': segment.get('summary', '')
                })

        # 점수순 정렬
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

    def extract_relevant_phrases(self, text: str, query_words: List[str],
                                context_window: int = 50) -> List[str]:
        """쿼리 관련 구문 추출"""
        phrases = []
        text_lower = text.lower()

        for word in query_words:
            if word in text_lower:
                # 단어 위치 찾기
                start_pos = text_lower.find(word)

                # 컨텍스트 윈도우 적용
                start = max(0, start_pos - context_window)
                end = min(len(text), start_pos + len(word) + context_window)

                phrase = text[start:end].strip()
                if phrase and phrase not in phrases:
                    phrases.append(phrase)

        return phrases[:3]  # 최대 3개 구문

    def process_audio_file(self, audio_path: str) -> Dict:
        """오디오 파일 전체 처리"""
        print("1. 오디오 전사 중...")
        transcript = self.transcribe_audio(audio_path)

        if not transcript['success']:
            return {'error': '전사 실패', 'transcript': transcript}

        print("2. 오디오 특성 추출 중...")
        features = self.extract_audio_features(audio_path)

        print("3. 세그먼트 생성 중...")
        segments = self.segment_transcript(transcript, segment_duration=30)

        return {
            'audio_path': audio_path,
            'transcript': transcript,
            'features': features,
            'segments': segments,
            'total_duration': features.get('duration', 0),
            'segment_count': len(segments)
        }

# 사용 예시
audio_rag = AudioRAGSystem(whisper_model_size="small")

# 오디오 파일 처리
# audio_data = audio_rag.process_audio_file('/path/to/lecture.mp3')

# if 'error' not in audio_data:
#     print(f"처리 완료:")
#     print(f"  총 길이: {audio_data['total_duration']:.1f}초")
#     print(f"  세그먼트 수: {audio_data['segment_count']}개")
#     print(f"  전체 텍스트: {len(audio_data['transcript']['text'])}자")
#
#     # 검색 테스트
#     results = audio_rag.search_audio_content("머신러닝", audio_data['segments'])
#
#     print(f"\\n검색 결과: {len(results)}건")
#     for i, result in enumerate(results[:3]):
#         print(f"{i+1}. 시간: {result['start_time']}s-{result['end_time']}s")
#         print(f"   점수: {result['score']:.3f}")
#         print(f"   내용: {result['context'][:100]}...")
# else:
#     print(f"처리 실패: {audio_data['error']}")`}
            </pre>
          </div>
        </div>
      </div>
    </section>
  )
}
