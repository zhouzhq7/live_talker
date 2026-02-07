# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] - 2025-02-07

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
