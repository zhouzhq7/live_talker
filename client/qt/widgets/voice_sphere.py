"""
Voice Sphere Widget - Central gradient sphere component
Displays voice interaction state with animation support
"""

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QPointF, QRectF, pyqtProperty
from PyQt6.QtGui import QPainter, QColor, QPen


class VoiceSphere(QWidget):
    """Gradient sphere component with animation support"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pulse_radius = 0.0
        self._base_radius = 150
        self._animation_duration = 2000  # 2 second animation cycle
        
        # Create pulse animation
        self._pulse_animation = QPropertyAnimation(self, b"pulseRadius")
        self._pulse_animation.setDuration(self._animation_duration)
        self._pulse_animation.setEasingCurve(QEasingCurve.Type.InOutSine)
        self._pulse_animation.setLoopCount(-1)  # Infinite loop
        
        # State
        self._state = "idle"  # idle, listening, thinking, speaking
        
    def pulseRadius(self):
        """Pulse radius property"""
        return self._pulse_radius
    
    def setPulseRadius(self, radius):
        """Set pulse radius"""
        self._pulse_radius = radius
        self.update()
    
    pulseRadius = pyqtProperty(float, pulseRadius, setPulseRadius)
    
    def setState(self, state: str):
        """
        Set state
        
        Args:
            state: idle, listening, thinking, speaking
        """
        self._state = state
        
        if state == "idle":
            self._pulse_animation.stop()
            self._pulse_radius = 0.0
        elif state in ["listening", "thinking", "speaking"]:
            if self._pulse_animation.state() != QPropertyAnimation.State.Running:
                self._pulse_animation.setStartValue(0.0)
                self._pulse_animation.setEndValue(20.0)
                self._pulse_animation.start()
        
        self.update()
    
    def paintEvent(self, event):
        """Paint sphere"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Get center point and radius
        center = QPointF(self.rect().center())
        radius = min(self.width(), self.height()) // 2 - 20
        
        # Adjust radius based on state
        if self._state != "idle":
            radius += self._pulse_radius
        
        # Set solid colors based on state (muted, pastel colors)
        if self._state == "idle":
            # Idle state: muted gray-blue
            color = QColor(180, 190, 200, 150)  # Soft gray-blue
        elif self._state == "listening":
            # Listening state: muted blue
            color = QColor(150, 180, 210, 180)  # Soft blue
        elif self._state == "thinking":
            # Thinking state: muted purple-gray
            color = QColor(170, 160, 190, 180)  # Soft purple-gray
        elif self._state == "speaking":
            # Speaking state: muted pink-gray
            color = QColor(200, 170, 180, 180)  # Soft pink-gray
        else:
            color = QColor(180, 190, 200, 150)  # Default soft gray-blue
        
        # Draw sphere with solid color
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        sphere_rect = QRectF(center.x() - radius, center.y() - radius, 
                             radius * 2, radius * 2)
        painter.drawEllipse(sphere_rect)

