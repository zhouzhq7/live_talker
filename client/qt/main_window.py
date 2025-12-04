"""
Main Window - Main Window Class
Integrates all UI components
"""

import sys
from pathlib import Path
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QGraphicsDropShadowEffect)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QFont, QColor

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from widgets.voice_sphere import VoiceSphere
from widgets.status_bar import StatusBar
from widgets.control_buttons import ControlButtons
from talker_thread import TalkerThread


class MainWindow(QMainWindow):
    """Main window class"""
    
    def __init__(self):
        super().__init__()
        self.talker_thread = None
        self.setupUI()
        self.setupConnections()
    
    def setupUI(self):
        """Setup UI"""
        self.setWindowTitle("Live Talker")
        self.setMinimumSize(400, 700)
        
        # Set gradient background
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #FFB6C1, stop:1 #87CEEB);
            }
        """)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Top status bar
        self.status_bar = StatusBar()
        main_layout.addWidget(self.status_bar)
        
        # Center area (with stretch)
        center_layout = QVBoxLayout()
        center_layout.addStretch()
        
        # Voice sphere
        self.voice_sphere = VoiceSphere()
        self.voice_sphere.setMinimumSize(300, 300)
        self.voice_sphere.setMaximumSize(400, 400)
        
        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setColor(QColor(0, 0, 0, 50))
        shadow.setOffset(0, 5)
        self.voice_sphere.setGraphicsEffect(shadow)
        
        center_layout.addWidget(self.voice_sphere, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Status text
        self.status_label = QLabel("Click to start conversation")
        status_font = QFont()
        status_font.setPointSize(16)
        status_font.setBold(True)
        self.status_label.setFont(status_font)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                color: rgba(0, 0, 0, 0.7);
                background-color: rgba(255, 255, 255, 0.3);
                border-radius: 20px;
                padding: 10px 20px;
            }
        """)
        center_layout.addWidget(self.status_label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Loading dots animation
        self.dots_label = QLabel("")
        self.dots_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.dots_label.setStyleSheet("color: rgba(0, 0, 0, 0.5); font-size: 20px;")
        center_layout.addWidget(self.dots_label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        center_layout.addStretch()
        main_layout.addLayout(center_layout, stretch=1)
        
        # Bottom control buttons
        self.control_buttons = ControlButtons()
        main_layout.addWidget(self.control_buttons)
        
        # Set initial state
        self.updateStatus("idle", "Click to start conversation")
    
    def setupConnections(self):
        """Setup signal-slot connections"""
        # Control button signals
        self.control_buttons.microphone_clicked.connect(self.onMicrophoneClicked)
        self.control_buttons.cancel_clicked.connect(self.onCancelClicked)
        
        # Status bar buttons (optional features)
        self.status_bar.scene_button.clicked.connect(self.onSceneButtonClicked)
        self.status_bar.text_button.clicked.connect(self.onTextButtonClicked)
    
    @pyqtSlot()
    def onMicrophoneClicked(self):
        """Microphone button clicked"""
        if self.talker_thread is None or not self.talker_thread.isRunning():
            self.startTalker()
        else:
            self.stopTalker()
    
    @pyqtSlot()
    def onCancelClicked(self):
        """Cancel button clicked"""
        self.stopTalker()
    
    @pyqtSlot()
    def onSceneButtonClicked(self):
        """Scene selection button clicked"""
        # TODO: Implement scene selection feature
        pass
    
    @pyqtSlot()
    def onTextButtonClicked(self):
        """Text mode button clicked"""
        # TODO: Implement text mode toggle
        pass
    
    def startTalker(self):
        """Start Talker"""
        if self.talker_thread and self.talker_thread.isRunning():
            return
        
        self.talker_thread = TalkerThread()
        
        # Connect signals
        self.talker_thread.state_changed.connect(self.onStateChanged)
        self.talker_thread.user_speech_detected.connect(self.onUserSpeechDetected)
        self.talker_thread.asr_result.connect(self.onASRResult)
        self.talker_thread.llm_thinking.connect(self.onLLMThinking)
        self.talker_thread.llm_response.connect(self.onLLMResponse)
        self.talker_thread.tts_synthesizing.connect(self.onTTSSynthesizing)
        self.talker_thread.audio_playing.connect(self.onAudioPlaying)
        self.talker_thread.audio_finished.connect(self.onAudioFinished)
        self.talker_thread.error_occurred.connect(self.onError)
        
        # Start thread
        self.talker_thread.start()
        self.control_buttons.setMicrophoneActive(True)
    
    def stopTalker(self):
        """Stop Talker"""
        if self.talker_thread:
            self.talker_thread.stop_talker()
            self.talker_thread.wait(3000)  # Wait up to 3 seconds
            self.talker_thread = None
        
        self.updateStatus("idle", "Click to start conversation")
        self.control_buttons.setMicrophoneActive(False)
    
    @pyqtSlot(str)
    def onStateChanged(self, state: str):
        """State changed"""
        state_texts = {
            "idle": "Click to start conversation",
            "listening": "Listening...",
            "thinking": "Thinking...",
            "speaking": "Speaking..."
        }
        self.updateStatus(state, state_texts.get(state, "Unknown state"))
    
    @pyqtSlot()
    def onUserSpeechDetected(self):
        """User speech detected"""
        self.updateStatus("listening", "Listening...")
    
    @pyqtSlot(str)
    def onASRResult(self, text: str):
        """ASR recognition result"""
        self.status_label.setText(f"You said: {text}")
    
    @pyqtSlot()
    def onLLMThinking(self):
        """LLM thinking"""
        self.updateStatus("thinking", "Thinking...")
    
    @pyqtSlot(str)
    def onLLMResponse(self, text: str):
        """LLM response"""
        self.status_label.setText(f"AI: {text[:50]}...")
    
    @pyqtSlot()
    def onTTSSynthesizing(self):
        """TTS synthesizing"""
        self.updateStatus("thinking", "Synthesizing speech...")
    
    @pyqtSlot()
    def onAudioPlaying(self):
        """Audio playing"""
        self.updateStatus("speaking", "Speaking...")
    
    @pyqtSlot()
    def onAudioFinished(self):
        """Audio finished"""
        self.updateStatus("listening", "Listening...")
    
    @pyqtSlot(str)
    def onError(self, error_msg: str):
        """Error handler"""
        self.status_label.setText(f"Error: {error_msg}")
        self.updateStatus("idle", "Error occurred")
    
    def updateStatus(self, state: str, text: str):
        """Update status"""
        self.voice_sphere.setState(state)
        self.status_label.setText(text)
        
        # Update dots animation
        if state == "idle":
            self.dots_label.setText("")
        else:
            # Start dots animation
            self.startDotsAnimation()
    
    def startDotsAnimation(self):
        """Start dots animation"""
        if not hasattr(self, '_dots_timer'):
            self._dots_timer = QTimer()
            self._dots_timer.timeout.connect(self.updateDots)
            self._dots_index = 0
        
        if not self._dots_timer.isActive():
            self._dots_timer.start(500)  # Update every 500ms
    
    def updateDots(self):
        """Update dots display"""
        dots = ["", ".", "..", "..."]
        self._dots_index = (self._dots_index + 1) % len(dots)
        self.dots_label.setText(dots[self._dots_index])
    
    def closeEvent(self, event):
        """Window close event"""
        self.stopTalker()
        event.accept()

