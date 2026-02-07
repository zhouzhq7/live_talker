# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] - 2025-02-07

### Phase 3: VAD Upgrade - TEN-VAD ✅

#### Added
- **TEN-VAD** - Lightweight voice activity detection (default)
  - Library size: only 306KB (86% smaller than Silero)
  - 32% faster than Silero VAD
  - Lower end-of-sentence detection latency
  - Cross-platform support (Linux/macOS/Windows)
  - Fully offline operation

#### Changed
- **Default VAD method** changed from `silero` to `ten`
- Updated `config.py` with TEN-VAD configuration:
  - `ten_hop_size`: Frame hop size (256 = 16ms at 16kHz)
  - `ten_threshold`: Detection threshold
- Updated `audio/vad.py` to support TEN-VAD as an option
- TEN-VAD falls back to Silero if not available

#### Tests
- Added 13 unit tests for TEN-VAD
- All tests passing ✅

#### Documentation
- Updated README.md with TEN-VAD features
- Added comparison with Silero VAD

### Phase 2: TTS Upgrade - MeloTTS ✅

#### Added
- **MeloTTS** - Fully open source TTS engine (default)
  - No FFmpeg dependency required
  - Supports ZH, EN, ES, FR, JP, KR languages
  - Chinese-English mixed synthesis
  - Speed adjustment (0.5x - 2.0x)
  - Completely offline operation
  - ~300MB model size

#### Changed
- **Default TTS engine** changed from `edge` to `melotts`
- Updated `config.py` with MeloTTS configuration:
  - `melotts_language`: Language code (ZH/EN/ES/FR/JP/KR)
  - `melotts_speaker`: Speaker ID
  - `melotts_speed`: Speech speed (0.5 - 2.0)
- Edge-TTS is now optional (no longer requires FFmpeg)

#### Tests
- Added 22 unit tests for MeloTTS
- All tests passing ✅

#### Documentation
- Updated README.md with MeloTTS features
- Added migration guide from Edge-TTS to MeloTTS
- Updated installation instructions (FFmpeg no longer required)

### Phase 1: ASR Upgrade - SenseVoice ✅

#### Added
- **SenseVoice ASR** - Alibaba's latest speech recognition model
  - Support for 50+ languages
  - Emotion recognition (HAPPY, SAD, ANGRY, NEUTRAL)
  - Audio event detection (Speech, Applause, Laughter, Music)
  - Built-in VAD support
  - Fast inference: 70ms for 10s audio (15x faster than Whisper)
  - New `transcribe_with_emotion()` method for emotion-aware transcription

#### Changed
- **Default ASR engine** changed from `funasr` to `sensevoice`
- Updated `config.py` with SenseVoice configuration options:
  - `sensevoice_model`: Model name (default: "iic/SenseVoiceSmall")
  - `sensevoice_device`: Device (cpu/cuda)
  - `sensevoice_language`: Language code (auto/zh/en/yue/ja/ko/nospeech)
  - `sensevoice_enable_vad`: Enable built-in VAD

#### Tests
- Added 25 unit tests for SenseVoice
- Added integration tests
- Added performance tests (RTF < 0.1 target)
- All tests passing ✅

#### Documentation
- Updated README.md with SenseVoice features and usage
- Updated improvement_plan documentation
- Added migration guide from FunASR to SenseVoice

---

## [1.0.0] - 2025-01-XX

### Initial Release

#### Features
- Real-time voice conversation system
- ASR: FunASR, Whisper, FireRedASR support
- TTS: Edge-TTS, Pyttsx3 support
- VAD: Silero, WebRTC, Energy-based
- LLM: Deepseek API integration
- Qt GUI client
- Voice interruption detection
- Conversation history management
