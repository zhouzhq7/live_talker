"""
Status Bar Widget - Top status bar
Displays time, notifications, network status, etc.
"""

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt, QTimer, QDateTime
from PyQt6.QtGui import QFont


class StatusBar(QWidget):
    """Top status bar component"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUI()
        
        # Timer to update time
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.updateTime)
        self._timer.start(1000)  # Update every second
        self.updateTime()
    
    def setupUI(self):
        """Setup UI"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 10, 20, 10)
        layout.setSpacing(15)
        
        # Time label
        self.time_label = QLabel()
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.time_label.setFont(font)
        self.time_label.setStyleSheet("color: #333333;")
        layout.addWidget(self.time_label)
        
        # Notification icon (placeholder)
        self.notification_label = QLabel("🔔")
        self.notification_label.setStyleSheet("font-size: 18px;")
        layout.addWidget(self.notification_label)
        
        # Network status (placeholder)
        self.network_label = QLabel("📶")
        self.network_label.setStyleSheet("font-size: 18px;")
        layout.addWidget(self.network_label)
        
        # Battery status (placeholder)
        self.battery_label = QLabel("🔋")
        self.battery_label.setStyleSheet("font-size: 18px;")
        layout.addWidget(self.battery_label)
        
        # Stretch space
        layout.addStretch()
        
        # Scene selection button
        self.scene_button = QPushButton("选择情景")
        self.scene_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 0.2);
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 15px;
                padding: 8px 16px;
                color: #333333;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.3);
            }
            QPushButton:pressed {
                background-color: rgba(255, 255, 255, 0.4);
            }
        """)
        layout.addWidget(self.scene_button)
        
        # Text mode button
        self.text_button = QPushButton("字")
        self.text_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 0.2);
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 15px;
                padding: 8px 16px;
                color: #333333;
                font-size: 14px;
                min-width: 40px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.3);
            }
        """)
        layout.addWidget(self.text_button)
    
    def updateTime(self):
        """Update time display"""
        current_time = QDateTime.currentDateTime()
        time_str = current_time.toString("hh:mm")
        self.time_label.setText(time_str)

