from PySide6.QtWidgets import QCheckBox
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, Property, QRectF
from PySide6.QtGui import QPainter, QColor, QBrush, QPen

class ToggleSwitch(QCheckBox):
    """Custom sliding toggle switch (iOS-style)."""

    def __init__(self, width=50, bg_color="#777", circle_color="#DDD",
                 active_color="#4CAF50", animation_curve=QEasingCurve.OutBounce):
        super().__init__()
        self.setFixedSize(width, int(width * 0.55))
        self.setCursor(Qt.PointingHandCursor)

        self._bg_color = bg_color
        self._circle_color = circle_color
        self._active_color = active_color

        self._circle_position = 3
        self._animation = QPropertyAnimation(self, b"circle_position", self)
        self._animation.setEasingCurve(animation_curve)
        self._animation.setDuration(300)

        self.stateChanged.connect(self.start_transition)

    @Property(float)
    def circle_position(self):
        return self._circle_position

    @circle_position.setter
    def circle_position(self, pos):
        self._circle_position = pos
        self.update()

    def start_transition(self, state):
        self._animation.stop()
        if state:
            self._animation.setEndValue(self.width() - self.height() + 3)
        else:
            self._animation.setEndValue(3)
        self._animation.start()

    def hitButton(self, pos):
        return self.contentsRect().contains(pos)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        rect = QRectF(0, 0, self.width(), self.height())
        if self.isChecked():
            p.setBrush(QColor(self._active_color))
            p.setPen(Qt.NoPen)
        else:
            p.setBrush(QColor(self._bg_color))
            p.setPen(Qt.NoPen)

        p.drawRoundedRect(0, 0, self.width(), self.height(),
                          self.height() / 2, self.height() / 2)

        p.setBrush(QColor(self._circle_color))
        p.drawEllipse(self._circle_position, 3,
                      self.height() - 6, self.height() - 6)
        p.end()
