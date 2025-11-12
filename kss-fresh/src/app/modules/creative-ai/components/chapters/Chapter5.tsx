'use client'

import React from 'react'
import { Music, Headphones, Radio, Mic, Sparkles, Wand2 } from 'lucide-react'

export default function Chapter5() {
  return (
    <div className="max-w-4xl mx-auto px-6 py-8">
      {/* Hero Section */}
      <div className="mb-12">
        <div className="inline-block px-4 py-2 bg-purple-500/10 border border-purple-500/20 rounded-full mb-4">
          <span className="text-purple-400 text-sm font-medium">Chapter 5</span>
        </div>
        <h1 className="text-5xl font-bold mb-6 bg-gradient-to-r from-purple-400 via-pink-400 to-rose-400 bg-clip-text text-transparent">
          AI 음악 생성
        </h1>
        <p className="text-xl text-gray-300 leading-relaxed">
          Suno, Udio, MusicGen 등 AI 음악 생성 도구로 누구나 쉽게 작곡가가 될 수 있습니다.
          가사, 장르, 분위기를 지정하면 완성도 높은 음악이 몇 분 안에 생성됩니다.
        </p>
      </div>

      {/* 1. AI 음악 생성 도구 비교 */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
            <Music className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-3xl font-bold text-white">AI 음악 생성 도구 비교</h2>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full bg-gray-800/50 border border-gray-700 rounded-xl overflow-hidden">
            <thead className="bg-gradient-to-r from-purple-500/20 to-pink-500/20">
              <tr>
                <th className="px-6 py-4 text-left text-white font-bold">도구</th>
                <th className="px-6 py-4 text-left text-white font-bold">특징</th>
                <th className="px-6 py-4 text-left text-white font-bold">가격</th>
                <th className="px-6 py-4 text-left text-white font-bold">장점</th>
                <th className="px-6 py-4 text-left text-white font-bold">용도</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-700">
              <tr className="hover:bg-gray-700/30 transition-colors">
                <td className="px-6 py-4">
                  <div className="flex items-center gap-2">
                    <span className="text-2xl">🎵</span>
                    <span className="font-bold text-purple-400">Suno</span>
                  </div>
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  가사 + 보컬 생성<br/>
                  2분 완곡 가능<br/>
                  v3.5 최신 모델
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  무료: 50곡/월<br/>
                  Pro: $10/월 (500곡)
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  가장 자연스러운 보컬<br/>
                  다양한 장르 지원
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  팝송, K-pop, 힙합<br/>
                  광고 음악
                </td>
              </tr>
              <tr className="hover:bg-gray-700/30 transition-colors">
                <td className="px-6 py-4">
                  <div className="flex items-center gap-2">
                    <span className="text-2xl">🎹</span>
                    <span className="font-bold text-pink-400">Udio</span>
                  </div>
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  고품질 음악 생성<br/>
                  섹션별 편집 가능<br/>
                  리믹스 지원
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  무료: 10곡/월<br/>
                  Standard: $10/월<br/>
                  Pro: $30/월
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  높은 음질<br/>
                  정밀한 제어
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  BGM, OST<br/>
                  전자음악
                </td>
              </tr>
              <tr className="hover:bg-gray-700/30 transition-colors">
                <td className="px-6 py-4">
                  <div className="flex items-center gap-2">
                    <span className="text-2xl">🤖</span>
                    <span className="font-bold text-rose-400">MusicGen<br/>(Meta)</span>
                  </div>
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  오픈소스<br/>
                  로컬 실행 가능<br/>
                  멜로디 조건부 생성
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  무료 (오픈소스)<br/>
                  Hugging Face 데모
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  완전 무료<br/>
                  커스터마이징 가능
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  배경음악<br/>
                  루프 음악
                </td>
              </tr>
              <tr className="hover:bg-gray-700/30 transition-colors">
                <td className="px-6 py-4">
                  <div className="flex items-center gap-2">
                    <span className="text-2xl">🎸</span>
                    <span className="font-bold text-orange-400">Stable Audio</span>
                  </div>
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  Stability AI 제작<br/>
                  3분 길이 지원<br/>
                  44.1kHz 스테레오
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  무료: 45초<br/>
                  Pro: $11.99/월 (3분)
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  높은 음질<br/>
                  긴 길이 지원
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  영상 BGM<br/>
                  게임 음악
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* 2. Suno 마스터 */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-pink-500 to-rose-500 rounded-lg flex items-center justify-center">
            <Headphones className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-3xl font-bold text-white">Suno - 보컬 음악 생성</h2>
        </div>

        <div className="bg-gradient-to-br from-gray-800 to-gray-900 border border-pink-500/30 rounded-xl p-8">
          <p className="text-gray-300 mb-6">
            Suno는 가장 인기 있는 AI 음악 생성 도구입니다. 가사와 보컬이 포함된 완전한 곡을 2분 안에 생성합니다.
            v3.5 모델은 놀라울 정도로 자연스러운 보컬과 다양한 장르를 지원합니다.
          </p>

          <div className="space-y-6">
            {/* 기본 사용법 */}
            <div className="bg-gray-800/50 border border-pink-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-pink-400 mb-4">1. Custom Mode (커스텀 모드)</h3>
              <p className="text-gray-300 text-sm mb-4">
                가사와 스타일을 직접 지정하여 원하는 곡 생성
              </p>

              <div className="space-y-4">
                <div>
                  <h4 className="font-bold text-white mb-2">Style of Music (음악 스타일)</h4>
                  <div className="bg-gray-900/50 rounded-lg p-4">
                    <p className="text-sm text-gray-300 mb-2">장르, 분위기, 악기를 조합하여 입력</p>
                    <div className="bg-black/30 rounded p-3">
                      <p className="text-xs text-purple-400 mb-2"># 예시:</p>
                      <ul className="text-xs text-gray-300 space-y-1">
                        <li>• <span className="text-pink-400">pop ballad, emotional, piano</span></li>
                        <li>• <span className="text-rose-400">upbeat electronic dance music, synth, energetic</span></li>
                        <li>• <span className="text-orange-400">acoustic folk, guitar, warm vocals</span></li>
                        <li>• <span className="text-purple-400">k-pop, catchy, female vocals, upbeat</span></li>
                      </ul>
                    </div>
                  </div>
                </div>

                <div>
                  <h4 className="font-bold text-white mb-2">Lyrics (가사)</h4>
                  <div className="bg-gray-900/50 rounded-lg p-4">
                    <p className="text-sm text-gray-300 mb-3">
                      [Verse], [Chorus], [Bridge] 태그로 구조 지정
                    </p>
                    <div className="bg-black/30 rounded p-3">
                      <p className="text-xs font-mono text-gray-300 leading-relaxed">
                        [Verse 1]<br/>
                        Walking down the empty street<br/>
                        Memories beneath my feet<br/>
                        Every step a story told<br/>
                        In the city of the bold<br/>
                        <br/>
                        [Chorus]<br/>
                        We'll rise again, stronger than before<br/>
                        Through the pain, we'll find what we're looking for<br/>
                        <br/>
                        [Bridge]<br/>
                        When the night falls down<br/>
                        We'll light up this town
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* 고급 기능 */}
            <div className="bg-gray-800/50 border border-purple-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-purple-400 mb-4">2. 고급 스타일 태그</h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <h4 className="font-bold text-pink-400 mb-2">장르 (Genre)</h4>
                  <ul className="text-gray-300 text-sm space-y-1">
                    <li>• pop, rock, jazz, blues</li>
                    <li>• hip hop, rap, R&B</li>
                    <li>• electronic, EDM, house, techno</li>
                    <li>• country, folk, bluegrass</li>
                    <li>• k-pop, j-pop</li>
                    <li>• classical, opera</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-bold text-rose-400 mb-2">분위기 (Mood)</h4>
                  <ul className="text-gray-300 text-sm space-y-1">
                    <li>• happy, upbeat, cheerful</li>
                    <li>• sad, melancholic, emotional</li>
                    <li>• dark, mysterious, ominous</li>
                    <li>• energetic, powerful, aggressive</li>
                    <li>• calm, peaceful, relaxing</li>
                    <li>• romantic, dreamy</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-bold text-orange-400 mb-2">보컬 (Vocals)</h4>
                  <ul className="text-gray-300 text-sm space-y-1">
                    <li>• male vocals, female vocals</li>
                    <li>• deep voice, high pitched</li>
                    <li>• raspy, smooth, soulful</li>
                    <li>• harmonized, choir</li>
                    <li>• rap verses, sung chorus</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-bold text-purple-400 mb-2">악기 (Instruments)</h4>
                  <ul className="text-gray-300 text-sm space-y-1">
                    <li>• piano, guitar, drums</li>
                    <li>• synthesizer, bass</li>
                    <li>• violin, cello, orchestra</li>
                    <li>• trumpet, saxophone</li>
                    <li>• acoustic, electric</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* 실전 예시 */}
            <div className="bg-gray-800/50 border border-rose-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-rose-400 mb-4">3. 실전 프롬프트 예시</h3>
              <div className="space-y-4">
                <div className="bg-gray-900/50 rounded-lg p-4">
                  <h4 className="font-bold text-purple-400 mb-2">예시 1: K-pop 스타일</h4>
                  <div className="bg-black/30 rounded p-3">
                    <p className="text-xs text-pink-400 mb-1">Style:</p>
                    <p className="text-xs font-mono text-gray-300 mb-3">
                      k-pop, energetic, catchy, female vocals, electronic beats, powerful
                    </p>
                    <p className="text-xs text-pink-400 mb-1">Lyrics:</p>
                    <p className="text-xs font-mono text-gray-300">
                      [Verse 1] Dancing through the neon lights...<br/>
                      [Chorus] We shine brighter than the stars tonight...
                    </p>
                  </div>
                </div>

                <div className="bg-gray-900/50 rounded-lg p-4">
                  <h4 className="font-bold text-pink-400 mb-2">예시 2: 감성 발라드</h4>
                  <div className="bg-black/30 rounded p-3">
                    <p className="text-xs text-rose-400 mb-1">Style:</p>
                    <p className="text-xs font-mono text-gray-300 mb-3">
                      emotional ballad, piano, orchestral, powerful male vocals, slow tempo
                    </p>
                    <p className="text-xs text-rose-400 mb-1">Lyrics:</p>
                    <p className="text-xs font-mono text-gray-300">
                      [Verse 1] In the silence of the night...<br/>
                      [Chorus] I'll remember you forever...
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 3. Udio - 고품질 음악 */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-rose-500 to-orange-500 rounded-lg flex items-center justify-center">
            <Radio className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-3xl font-bold text-white">Udio - 프로페셔널 음악 제작</h2>
        </div>

        <div className="bg-gradient-to-br from-rose-900/20 to-orange-900/20 border border-rose-500/30 rounded-xl p-8">
          <p className="text-gray-300 mb-6">
            Udio는 Suno의 강력한 경쟁자로, 더 높은 음질과 섹션별 편집 기능을 제공합니다.
            특히 전자음악, EDM, 힙합 장르에 강점이 있습니다.
          </p>

          <div className="space-y-6">
            {/* Udio 특징 */}
            <div className="bg-gray-800/50 border border-rose-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-rose-400 mb-4">Udio만의 특징</h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-gray-900/50 rounded-lg p-4">
                  <h4 className="font-bold text-purple-400 mb-2">1. 섹션별 편집</h4>
                  <p className="text-gray-300 text-sm">
                    인트로, 아웃트로, 특정 구간을 선택하여 재생성 또는 연장 가능
                  </p>
                </div>
                <div className="bg-gray-900/50 rounded-lg p-4">
                  <h4 className="font-bold text-pink-400 mb-2">2. Remix 기능</h4>
                  <p className="text-gray-300 text-sm">
                    기존 곡의 스타일을 바꾸거나 특정 부분만 수정 가능
                  </p>
                </div>
                <div className="bg-gray-900/50 rounded-lg p-4">
                  <h4 className="font-bold text-rose-400 mb-2">3. 고음질 출력</h4>
                  <p className="text-gray-300 text-sm">
                    320kbps MP3, 48kHz 샘플레이트로 스튜디오급 품질
                  </p>
                </div>
                <div className="bg-gray-900/50 rounded-lg p-4">
                  <h4 className="font-bold text-orange-400 mb-2">4. 상업적 이용</h4>
                  <p className="text-gray-300 text-sm">
                    Pro 플랜 가입 시 생성한 음악을 상업적으로 사용 가능
                  </p>
                </div>
              </div>
            </div>

            {/* 사용법 */}
            <div className="bg-gray-800/50 border border-orange-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-orange-400 mb-4">Udio 프롬프트 작성법</h3>
              <p className="text-gray-300 text-sm mb-4">
                Udio는 더 자세하고 기술적인 설명을 선호합니다.
              </p>

              <div className="bg-gray-900/50 rounded-lg p-4">
                <p className="text-xs text-purple-400 mb-2"># 좋은 프롬프트 예시:</p>
                <div className="bg-black/30 rounded p-3">
                  <p className="text-xs font-mono text-gray-300 leading-relaxed">
                    A futuristic synthwave track with pulsing basslines, arpeggiated synthesizers,
                    and nostalgic 80s drums. Tempo: 128 BPM. Key: C minor. Features a soaring lead
                    synth melody in the chorus with heavy reverb and sidechain compression.
                  </p>
                </div>
              </div>

              <div className="mt-4 bg-gray-900/50 rounded-lg p-4">
                <h4 className="font-bold text-white mb-3">포함하면 좋은 요소:</h4>
                <ul className="text-gray-300 text-sm space-y-1">
                  <li>• <strong className="text-purple-400">BPM (템포):</strong> 60-180 사이 값 (120 BPM = 표준 댄스)</li>
                  <li>• <strong className="text-pink-400">Key (조성):</strong> C major, A minor 등</li>
                  <li>• <strong className="text-rose-400">구조:</strong> Intro-Verse-Chorus-Bridge-Outro</li>
                  <li>• <strong className="text-orange-400">이펙트:</strong> reverb, delay, distortion, compression</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 4. MusicGen (Meta) */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-orange-500 to-yellow-500 rounded-lg flex items-center justify-center">
            <Mic className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-3xl font-bold text-white">MusicGen - 오픈소스 AI 음악</h2>
        </div>

        <div className="bg-gradient-to-br from-orange-900/20 to-yellow-900/20 border border-orange-500/30 rounded-xl p-8">
          <p className="text-gray-300 mb-6">
            Meta(구 Facebook)가 공개한 오픈소스 음악 생성 모델. 완전 무료이며, 로컬에서 실행 가능합니다.
            보컬은 없지만 배경음악, 루프 음악 생성에 적합합니다.
          </p>

          <div className="space-y-6">
            {/* Hugging Face 데모 */}
            <div className="bg-gray-800/50 border border-orange-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-orange-400 mb-4">1. Hugging Face 데모 (무료)</h3>
              <p className="text-gray-300 text-sm mb-4">
                설치 없이 브라우저에서 바로 사용 가능
              </p>

              <div className="bg-gray-900/50 rounded-lg p-4">
                <p className="text-xs text-purple-400 mb-2">접속 URL:</p>
                <a
                  href="https://huggingface.co/spaces/facebook/MusicGen"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-green-400 hover:text-green-300 text-sm underline"
                >
                  https://huggingface.co/spaces/facebook/MusicGen
                </a>

                <div className="mt-4 bg-black/30 rounded p-3">
                  <p className="text-xs text-pink-400 mb-2"># 프롬프트 예시:</p>
                  <ul className="text-xs text-gray-300 space-y-1">
                    <li>• <span className="text-purple-400">upbeat electronic dance music with synthesizers</span></li>
                    <li>• <span className="text-pink-400">calm piano melody for meditation</span></li>
                    <li>• <span className="text-rose-400">energetic rock guitar riffs with drums</span></li>
                    <li>• <span className="text-orange-400">ambient soundscape with nature sounds</span></li>
                  </ul>
                </div>
              </div>
            </div>

            {/* 로컬 설치 */}
            <div className="bg-gray-800/50 border border-yellow-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-yellow-400 mb-4">2. 로컬 설치 (고급)</h3>
              <p className="text-gray-300 text-sm mb-4">
                Python 환경에서 MusicGen 설치 및 실행
              </p>

              <div className="bg-gray-900/50 rounded-lg p-4">
                <p className="text-xs text-purple-400 mb-2"># 설치 (Python 3.9+)</p>
                <div className="bg-black/30 rounded p-3 mb-4">
                  <p className="text-xs font-mono text-gray-300">
                    pip install audiocraft
                  </p>
                </div>

                <p className="text-xs text-pink-400 mb-2"># Python 스크립트:</p>
                <div className="bg-black/30 rounded p-3">
                  <p className="text-xs font-mono text-gray-300 leading-relaxed">
                    from audiocraft.models import MusicGen<br/>
                    from audiocraft.data.audio import audio_write<br/>
                    <br/>
                    model = MusicGen.get_pretrained('facebook/musicgen-medium')<br/>
                    model.set_generation_params(duration=30)  # 30초 생성<br/>
                    <br/>
                    descriptions = ['upbeat electronic dance music']<br/>
                    wav = model.generate(descriptions)<br/>
                    <br/>
                    # WAV 파일 저장<br/>
                    audio_write('output', wav[0].cpu(), model.sample_rate)
                  </p>
                </div>
              </div>

              <div className="mt-4 bg-blue-900/20 border border-blue-500/30 rounded-lg p-4">
                <p className="text-blue-300 text-sm">
                  💡 <strong>모델 크기:</strong> small (300M), medium (1.5GB), large (3.3GB)<br/>
                  medium 권장 (품질 vs 속도 균형)
                </p>
              </div>
            </div>

            {/* Melody Conditioning */}
            <div className="bg-gray-800/50 border border-purple-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-purple-400 mb-4">3. Melody Conditioning (고급 기능)</h3>
              <p className="text-gray-300 text-sm mb-4">
                기존 멜로디(MP3, WAV)를 입력하면 그 멜로디를 따르는 음악 생성
              </p>

              <div className="bg-gray-900/50 rounded-lg p-4">
                <p className="text-xs font-mono text-gray-300 leading-relaxed">
                  from audiocraft.models import MusicGen<br/>
                  import torchaudio<br/>
                  <br/>
                  model = MusicGen.get_pretrained('facebook/musicgen-melody')<br/>
                  <br/>
                  # 멜로디 파일 로드<br/>
                  melody, sr = torchaudio.load('melody.mp3')<br/>
                  <br/>
                  # 멜로디 기반 음악 생성<br/>
                  descriptions = ['rock song with electric guitar']<br/>
                  wav = model.generate_with_chroma(descriptions, melody[None], sr)
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 5. 상업적 사용 & 저작권 */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-yellow-500 to-green-500 rounded-lg flex items-center justify-center">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-3xl font-bold text-white">상업적 사용 & 저작권</h2>
        </div>

        <div className="bg-gradient-to-br from-yellow-900/20 to-green-900/20 border border-yellow-500/30 rounded-xl p-8">
          <div className="space-y-6">
            {/* 도구별 라이선스 */}
            <div className="bg-gray-800/50 border border-yellow-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-yellow-400 mb-4">도구별 상업적 사용 가능 여부</h3>
              <div className="space-y-4">
                <div className="bg-gray-900/50 rounded-lg p-4">
                  <h4 className="font-bold text-purple-400 mb-2">Suno Pro ($10/월)</h4>
                  <p className="text-gray-300 text-sm mb-2">
                    ✅ <strong className="text-green-400">상업적 사용 가능</strong> (Pro 플랜)
                  </p>
                  <ul className="text-gray-400 text-xs space-y-1">
                    <li>• 무료 플랜: 비상업적 용도만</li>
                    <li>• Pro 플랜: YouTube, Spotify 업로드 가능</li>
                    <li>• 저작권: 사용자에게 귀속</li>
                  </ul>
                </div>

                <div className="bg-gray-900/50 rounded-lg p-4">
                  <h4 className="font-bold text-pink-400 mb-2">Udio Standard/Pro</h4>
                  <p className="text-gray-300 text-sm mb-2">
                    ✅ <strong className="text-green-400">상업적 사용 가능</strong> (Standard 이상)
                  </p>
                  <ul className="text-gray-400 text-xs space-y-1">
                    <li>• 광고 음악, 영상 BGM 사용 가능</li>
                    <li>• 음원 스트리밍 플랫폼 업로드 가능</li>
                  </ul>
                </div>

                <div className="bg-gray-900/50 rounded-lg p-4">
                  <h4 className="font-bold text-rose-400 mb-2">MusicGen (Meta)</h4>
                  <p className="text-gray-300 text-sm mb-2">
                    ✅ <strong className="text-green-400">완전 무료 & 오픈소스</strong>
                  </p>
                  <ul className="text-gray-400 text-xs space-y-1">
                    <li>• CC BY-NC 4.0 라이선스 (비상업적)</li>
                    <li>• 상업적 사용 시 Meta에 별도 문의 필요</li>
                    <li>• 연구 및 개인 프로젝트: 자유롭게 사용</li>
                  </ul>
                </div>

                <div className="bg-gray-900/50 rounded-lg p-4">
                  <h4 className="font-bold text-orange-400 mb-2">Stable Audio Pro</h4>
                  <p className="text-gray-300 text-sm mb-2">
                    ✅ <strong className="text-green-400">상업적 사용 가능</strong> (Pro 플랜)
                  </p>
                  <ul className="text-gray-400 text-xs space-y-1">
                    <li>• 무제한 생성 횟수</li>
                    <li>• 3분 길이 지원</li>
                    <li>• 저작권: 사용자 소유</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* 주의사항 */}
            <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-red-400 mb-4 flex items-center gap-2">
                ⚠️ 주의사항
              </h3>
              <ul className="space-y-2 text-gray-300 text-sm">
                <li>
                  • <strong className="text-white">유사성 문제:</strong> AI가 학습한 곡과 유사한 멜로디가 나올 수 있음
                </li>
                <li>
                  • <strong className="text-white">저작권 표기:</strong> AI 생성 음악임을 명시하는 것이 권장됨
                </li>
                <li>
                  • <strong className="text-white">라이선스 확인:</strong> 플랫폼별 이용 약관을 반드시 확인
                </li>
                <li>
                  • <strong className="text-white">프롬프트 저작권:</strong> 가사에 타인의 저작물을 사용하지 말 것
                </li>
              </ul>
            </div>

            {/* 합법적 사용 사례 */}
            <div className="bg-green-900/20 border border-green-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-green-400 mb-4">✅ 합법적 사용 사례</h3>
              <div className="grid md:grid-cols-2 gap-4">
                <ul className="space-y-2 text-gray-300 text-sm">
                  <li>• YouTube 영상 배경음악</li>
                  <li>• 팟캐스트 인트로/아웃트로</li>
                  <li>• 게임 BGM</li>
                  <li>• 광고 음악</li>
                </ul>
                <ul className="space-y-2 text-gray-300 text-sm">
                  <li>• 영화/드라마 OST</li>
                  <li>• 프레젠테이션 배경음악</li>
                  <li>• 앱/웹사이트 사운드</li>
                  <li>• 전시회/이벤트 음악</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* References */}
      <section className="mb-16">
        <h2 className="text-3xl font-bold text-white mb-6 flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
            📚
          </div>
          References
        </h2>

        <div className="space-y-4">
          <div className="bg-gray-800/50 border border-purple-500/30 rounded-xl p-6">
            <h3 className="text-lg font-bold text-purple-400 mb-4">🎵 AI 음악 생성 플랫폼</h3>
            <ul className="space-y-3">
              <li>
                <a
                  href="https://suno.ai/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-purple-400 hover:text-purple-300 transition-colors"
                >
                  Suno AI - AI Music Generator
                </a>
                <p className="text-sm text-gray-400 mt-1">가장 인기 있는 AI 음악 생성 서비스</p>
              </li>
              <li>
                <a
                  href="https://udio.com/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-purple-400 hover:text-purple-300 transition-colors"
                >
                  Udio - Professional Music Creation
                </a>
                <p className="text-sm text-gray-400 mt-1">고품질 음악 생성 및 섹션별 편집</p>
              </li>
              <li>
                <a
                  href="https://stableaudio.com/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-purple-400 hover:text-purple-300 transition-colors"
                >
                  Stable Audio - Stability AI
                </a>
                <p className="text-sm text-gray-400 mt-1">3분 길이 고음질 음악 생성</p>
              </li>
            </ul>
          </div>

          <div className="bg-gray-800/50 border border-pink-500/30 rounded-xl p-6">
            <h3 className="text-lg font-bold text-pink-400 mb-4">🤖 오픈소스 도구</h3>
            <ul className="space-y-3">
              <li>
                <a
                  href="https://github.com/facebookresearch/audiocraft"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-pink-400 hover:text-pink-300 transition-colors"
                >
                  AudioCraft (MusicGen) - Meta AI
                </a>
                <p className="text-sm text-gray-400 mt-1">오픈소스 음악 생성 라이브러리</p>
              </li>
              <li>
                <a
                  href="https://huggingface.co/spaces/facebook/MusicGen"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-pink-400 hover:text-pink-300 transition-colors"
                >
                  MusicGen Demo - Hugging Face
                </a>
                <p className="text-sm text-gray-400 mt-1">무료 온라인 MusicGen 데모</p>
              </li>
              <li>
                <a
                  href="https://github.com/suno-ai/bark"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-pink-400 hover:text-pink-300 transition-colors"
                >
                  Bark - Suno AI (Text-to-Audio)
                </a>
                <p className="text-sm text-gray-400 mt-1">오픈소스 음성 및 사운드 생성</p>
              </li>
            </ul>
          </div>

          <div className="bg-gray-800/50 border border-rose-500/30 rounded-xl p-6">
            <h3 className="text-lg font-bold text-rose-400 mb-4">📖 학습 리소스</h3>
            <ul className="space-y-3">
              <li>
                <a
                  href="https://www.youtube.com/results?search_query=suno+ai+tutorial"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-rose-400 hover:text-rose-300 transition-colors"
                >
                  Suno AI Tutorial - YouTube
                </a>
                <p className="text-sm text-gray-400 mt-1">Suno 사용법 영상 튜토리얼</p>
              </li>
              <li>
                <a
                  href="https://www.reddit.com/r/SunoAI/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-rose-400 hover:text-rose-300 transition-colors"
                >
                  r/SunoAI - Reddit Community
                </a>
                <p className="text-sm text-gray-400 mt-1">Suno 사용자 커뮤니티 및 팁 공유</p>
              </li>
              <li>
                <a
                  href="https://aituts.com/ai-music-generators/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-rose-400 hover:text-rose-300 transition-colors"
                >
                  AI Music Generators Comparison
                </a>
                <p className="text-sm text-gray-400 mt-1">AI 음악 도구 비교 및 리뷰</p>
              </li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  )
}
