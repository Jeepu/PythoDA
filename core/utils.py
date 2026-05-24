import cv2
import numpy as np
import torchvision
import torch
from scipy.spatial import KDTree

def apply_nms_patch():
    try:
        _original_batched_nms = torchvision.ops.batched_nms
        def _cpu_batched_nms(boxes, scores, idxs, iou_threshold):
            if boxes.numel() == 0: return torch.empty((0,), dtype=torch.int64, device=boxes.device)
            return _original_batched_nms(boxes.cpu(), scores.cpu(), idxs.cpu(), iou_threshold).to(boxes.device)
        torchvision.ops.batched_nms = _cpu_batched_nms
        torchvision.ops.boxes.batched_nms = _cpu_batched_nms
    except:
        pass

apply_nms_patch()

class ColorNamer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ColorNamer, cls).__new__(cls)
            cls._instance._init_colors()
        return cls._instance

    def _init_colors(self):
        self.colors = {
            "LightYellow": (255, 255, 224),  
            "LemonChiffon": (255, 250, 205), 
            "Cream": (255, 253, 208),       
            "Papaya": (255, 239, 213),       
            "PaleGoldenrod": (238, 232, 170),
            "Khaki": (240, 230, 140),        
            "Yellow": (255, 255, 0),         
            "Gold": (255, 215, 0), 
            "CornYellow": (255, 215, 0),  
            "Goldenrod": (218, 165, 32),
            "DarkGoldenrod": (184, 134, 11),
            "Maize": (251, 236, 93),    
            "Straw": (228, 217, 111), 
            "Saffron": (244, 196, 48), 
            "Mustard": (255, 219, 88),   
            "Amber": (255, 191, 0), 

            "Orange": (255, 165, 0),
            "DarkOrange": (255, 140, 0),
            "Coral": (255, 127, 80),    
            "Tomato": (255, 99, 71),   
            "OrangeRed": (255, 69, 0),
            "Pumpkin": (255, 117, 24), 
            "Tangerine": (242, 133, 0),  
            "Persimmon": (236, 88, 0),  

            "Cornsilk": (255, 248, 220), 
            "BlanchedAlmond": (255, 235, 205), 
            "Bisque": (255, 228, 196), 
            "Wheat": (245, 222, 179),  
            "Burlywood": (222, 184, 135), 
            "Tan": (210, 180, 140),  
            "SandyBrown": (244, 164, 96),  
            "Peru": (205, 133, 63),    
            "Chocolate": (210, 105, 30),  
            "SaddleBrown": (139, 69, 19), 
            "Sienna": (160, 82, 45),        
            "Brown": (165, 42, 42),  
            "Maroon": (128, 0, 0),   
            "Sepia": (112, 66, 20), 
            "Russet": (128, 70, 27),  
            "Buff": (240, 220, 130),         

            "GreenYellow": (173, 255, 47),  
            "Chartreuse": (127, 255, 0),    
            "LawnGreen": (124, 252, 0),     
            "Lime": (0, 255, 0),         
            "LimeGreen": (50, 205, 50),   
            "PaleGreen": (152, 251, 152),    
            "LightGreen": (144, 238, 144),  
            "MediumSpringGreen": (0, 250, 154),
            "SpringGreen": (0, 255, 127),
            "MediumSeaGreen": (60, 179, 113),
            "SeaGreen": (46, 139, 87),
            "ForestGreen": (34, 139, 34),  
            "Green": (0, 128, 0),          
            "DarkGreen": (0, 100, 0),       
            "YellowGreen": (154, 205, 50),   
            "OliveDrab": (107, 142, 35),     
            "Olive": (128, 128, 0),          
            "DarkOliveGreen": (85, 107, 47), 
            "MungGreen": (85, 107, 47),      
            "Teal": (0, 128, 128),           

            "LightSalmon": (255, 160, 122),  
            "Salmon": (250, 128, 114),       
            "IndianRed": (205, 92, 92),      
            "Crimson": (220, 20, 60),        
            "Red": (255, 0, 0),              
            "FireBrick": (178, 34, 34),      
            "DarkRed": (139, 0, 0),          
            "Pink": (255, 192, 203),         
            "HotPink": (255, 105, 180),     
            "DeepPink": (255, 20, 147),      

            "Lavender": (230, 230, 250),     
            "Thistle": (216, 191, 216),      
            "Plum": (221, 160, 221),         
            "Violet": (238, 130, 238),       
            "Orchid": (218, 112, 214),       
            "Fuchsia": (255, 0, 255),        
            "MediumOrchid": (186, 85, 211),
            "BlueViolet": (138, 43, 226),
            "DarkViolet": (148, 0, 211),
            "Purple": (128, 0, 128),         
            "Indigo": (75, 0, 130),         

            "Gainsboro": (220, 220, 220),
            "LightGray": (211, 211, 211),
            "Silver": (192, 192, 192),
            "DarkGray": (169, 169, 169),
            "Gray": (128, 128, 128),
            "DimGray": (105, 105, 105),
            "LightSlateGray": (119, 136, 153),
            "SlateGray": (112, 128, 144),
            "DarkSlateGray": (47, 79, 79),
            "Black": (20, 20, 20),           
            "White": (255, 255, 255),        
            "Snow": (255, 250, 250),
            "Ivory": (255, 255, 240),        
            "FloralWhite": (255, 250, 240)
        }
        self.names = list(self.colors.keys())
        self.rgb_values = list(self.colors.values())
        self.tree = KDTree(self.rgb_values)

    def get_name(self, rgb_tuple):

        dist, index = self.tree.query(rgb_tuple)
        return self.names[index]


def imread_unicode(path):
    """Load an image from a file path that may contain Unicode characters.

    On Windows, cv2.imread() fails when the file path contains non-ASCII
    characters because it uses the ANSI C API internally.  This function
    reads the raw bytes and decodes them via cv2.imdecode, which bypasses
    the issue.
    """
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def get_red_density(image_hsv, mask_bool):
    pixels_h = image_hsv[:, :, 0][mask_bool]
    pixels_s = image_hsv[:, :, 1][mask_bool]
    pixels_v = image_hsv[:, :, 2][mask_bool] 
    if pixels_h.size == 0: return 0.0
    mean_v = np.mean(pixels_v)
    if mean_v < 40: return 0.0
    is_red = ((pixels_h < 25) | (pixels_h > 155)) & (pixels_s > 30)
    return np.sum(is_red) / pixels_h.size

def get_green_density(image_hsv, mask_bool):
    pixels_h = image_hsv[:, :, 0][mask_bool]
    pixels_s = image_hsv[:, :, 1][mask_bool]
    if pixels_h.size == 0: return 0.0
    is_green = (pixels_h > 35) & (pixels_h < 85) & (pixels_s > 30)
    return np.sum(is_green) / pixels_h.size