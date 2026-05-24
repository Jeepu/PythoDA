from PySide6.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QLineEdit, 
                               QDialogButtonBox, QLabel)

class SetScaleDialog(QDialog):
    def __init__(self, measured_pixels, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Scale")
        self.measured_pixels = measured_pixels
        
        layout = QVBoxLayout(self)
        form = QFormLayout()
        
        self.edit_distance_px = QLineEdit(f"{measured_pixels:.2f}")
        self.edit_distance_px.setReadOnly(True)
        form.addRow("Distance in pixels:", self.edit_distance_px)

        self.edit_known_dist = QLineEdit("1.0")
        form.addRow("Known distance:", self.edit_known_dist)

        self.edit_unit = QLineEdit("cm")
        form.addRow("Unit of length:", self.edit_unit)

        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_values(self):
        try:
            known = float(self.edit_known_dist.text())
        except ValueError:
            known = 1.0
        return known, self.edit_unit.text()