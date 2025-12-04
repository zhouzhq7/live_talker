"""
Voice Sphere Widget - Central gradient sphere component
Displays voice interaction state with animation support
"""

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QPointF, QRectF, pyqtProperty
from PyQt6.QtGui import QPainter, QRadialGradient, QColor, QPen


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
        
        # Create radial gradient (PyQt6 requires QPointF)
        gradient = QRadialGradient(center, radius)
        
        # Set colors based on state
        if self._state == "idle":
            # Static gradient: purple to pink
            gradient.setColorAt(0.0, QColor(138, 43, 226, 180))  # Purple
            gradient.setColorAt(0.5, QColor(255, 20, 147, 150))  # Pink
            gradient.setColorAt(1.0, QColor(135, 206, 250, 120))  # Light blue
        elif self._state == "listening":
            # Listening state: brighter blue
            gradient.setColorAt(0.0, QColor(100, 149, 237, 200))  # Blue
            gradient.setColorAt(0.5, QColor(255, 20, 147, 180))  # Pink
            gradient.setColorAt(1.0, QColor(135, 206, 250, 150))  # Light blue
        elif self._state == "thinking":
            # Thinking state: purple
            gradient.setColorAt(0.0, QColor(138, 43, 226, 200))  # Purple
            gradient.setColorAt(0.5, QColor(186, 85, 211, 180))  # Medium purple
            gradient.setColorAt(1.0, QColor(221, 160, 221, 150))  # Light purple
        elif self._state == "speaking":
            # Speaking state: pink
            gradient.setColorAt(0.0, QColor(255, 20, 147, 200))  # Pink
            gradient.setColorAt(0.5, QColor(255, 105, 180, 180))  # Light pink
            gradient.setColorAt(1.0, QColor(255, 182, 193, 150))  # Pale pink
        
        # Draw sphere
        painter.setBrush(gradient)
        painter.setPen(Qt.PenStyle.NoPen)
        sphere_rect = QRectF(center.x() - radius, center.y() - radius, 
                             radius * 2, radius * 2)
        painter.drawEllipse(sphere_rect)
        
        # Draw outer glow (optional)
        if self._state != "idle":
            glow_gradient = QRadialGradient(center, radius + 10)
            glow_gradient.setColorAt(0.0, QColor(255, 255, 255, 0))
            glow_gradient.setColorAt(0.5, QColor(255, 255, 255, 30))
            glow_gradient.setColorAt(1.0, QColor(255, 255, 255, 0))
            painter.setBrush(glow_gradient)
            glow_rect = QRectF(center.x() - radius - 10, center.y() - radius - 10,
                               (radius + 10) * 2, (radius + 10) * 2)
            painter.drawEllipse(glow_rect)

