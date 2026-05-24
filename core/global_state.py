from PySide6.QtGui import QColor 

class GlobalState:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalState, cls).__new__(cls)
            cls._instance.pixels_per_unit = 1.0 
            cls._instance.unit_name = "px"      
            cls._instance.draw_color = QColor(255, 0, 0) 
        return cls._instance

    def set_scale(self, known_distance, pixel_distance, unit):
        if known_distance == 0: return
        self.pixels_per_unit = pixel_distance / known_distance
        self.unit_name = unit
        print(f">> [GlobalState] Update of scale factor: 1 {unit} = {self.pixels_per_unit:.2f} px")

    def reset(self):

        self.pixels_per_unit = 1.0
        self.unit_name = "px"
        print(">> [GlobalState] The scale has been reset to the default (1:1)")

    def px_to_real(self, pixels):
        return pixels / self.pixels_per_unit