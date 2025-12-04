"""
Control Buttons Widget - Bottom control buttons
Microphone and cancel buttons only
"""

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont


class ControlButtons(QWidget):
    """Bottom control buttons component"""
    
    # Signal definitions
    microphone_clicked = pyqtSignal()
    cancel_clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUI()
    
    def setupUI(self):
        """Setup UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Button container
        button_layout = QHBoxLayout()
        button_layout.setSpacing(30)
        button_layout.addStretch()
        
        # Microphone button
        self.mic_button = self.createButton("🎤", "Microphone")
        self.mic_button.clicked.connect(self.microphone_clicked.emit)
        button_layout.addWidget(self.mic_button)
        
        # Cancel button
        self.cancel_button = self.createButton("❌", "Cancel")
        self.cancel_button.clicked.connect(self.cancel_clicked.emit)
        button_layout.addWidget(self.cancel_button)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # AI generated content notice
        ai_label = QLabel("内容由 AI 生成")
        ai_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(10)
        font.setItalic(True)
        ai_label.setFont(font)
        ai_label.setStyleSheet("color: rgba(0, 0, 0, 0.5);")
        layout.addWidget(ai_label)
    
    def createButton(self, icon_text: str, tooltip: str) -> QPushButton:
        """Create circular button"""
        button = QPushButton(icon_text)
        button.setToolTip(tooltip)
        button.setFixedSize(60, 60)
        button.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 0.3);
                border: 2px solid rgba(255, 255, 255, 0.5);
                border-radius: 30px;
                font-size: 24px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.5);
                border-color: rgba(255, 255, 255, 0.7);
            }
            QPushButton:pressed {
                background-color: rgba(255, 255, 255, 0.7);
            }
        """)
        return button
    
    def setMicrophoneActive(self, active: bool):
        """Set microphone button active state"""
        if active:
            self.mic_button.setStyleSheet("""
                QPushButton {
                    background-color: rgba(255, 100, 100, 0.6);
                    border: 2px solid rgba(255, 100, 100, 0.8);
                    border-radius: 30px;
                    font-size: 24px;
                }
                QPushButton:hover {
                    background-color: rgba(255, 100, 100, 0.8);
                }
            """)
        else:
            self.mic_button.setStyleSheet("""
                QPushButton {
                    background-color: rgba(255, 255, 255, 0.3);
                    border: 2px solid rgba(255, 255, 255, 0.5);
                    border-radius: 30px;
                    font-size: 24px;
                }
                QPushButton:hover {
                    background-color: rgba(255, 255, 255, 0.5);
                }
            """)

