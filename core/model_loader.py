# core/model_loader.py
import sys
import os
import torch
import time
import gc
import threading  


current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)

try:
    from depth_anything_v2.dpt import DepthAnythingV2
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
except ImportError as e:
    print(f"Core Error: Missing dependency library - {e}")

class ModelLoader:
    _instance = None
    _lock = threading.RLock()  

    def __new__(cls):
        with cls._lock:  
            if cls._instance is None:
                cls._instance = super(ModelLoader, cls).__new__(cls)
                if torch.cuda.is_available():
                    cls._instance.device = 'cuda'
                    props = torch.cuda.get_device_properties(0)
                    print(f"--> [Hardware] Detected GPU: {props.name} (VRAM: {props.total_memory / 1024**3:.1f} GB)")
                else:
                    cls._instance.device = 'cpu'
                    print("--> [Hardware] WARNING: GPU not found! Using CPU (Slow).")
                
                cls._instance.depth_model = None
                cls._instance.sam_model = None
                
                cls._instance.current_sam_type = None 
                
                cls._instance.sam_checkpoints = {
                    'vit_h': os.path.join(root_dir, 'checkpoints', 'sam_vit_h_4b8939.pth'),
                    'vit_l': os.path.join(root_dir, 'checkpoints', 'sam_vit_l_0b3195.pth'),
                    'vit_b': os.path.join(root_dir, 'checkpoints', 'sam_vit_b_01ec64.pth')
                }
        return cls._instance

    def load_models(self, callback_signal=None, target_sam_type='vit_b'):

        with self._lock:  
            t0 = time.time()

            if self.depth_model is None:
                if callback_signal: callback_signal.emit(10, f"Loading Depth Model on {self.device}...")
                
                dav2_cfg = {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
                self.depth_model = DepthAnythingV2(**dav2_cfg)
                
                ckpt_path = os.path.join(root_dir, 'checkpoints', 'depth_anything_v2_vitl.pth')
                self.depth_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
                self.depth_model = self.depth_model.to(self.device).eval()
                


            if self.sam_model is None or self.current_sam_type != target_sam_type:
                

                if self.sam_model is not None:
                    if callback_signal: callback_signal.emit(40, f"Unloading old SAM ({self.current_sam_type})...")
                    print(f"--> [Memory] Releasing SAM {self.current_sam_type} from VRAM...")
                    del self.sam_model
                    self.sam_model = None
                    gc.collect()
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()


                sam_ckpt = self.sam_checkpoints.get(target_sam_type)
                if not sam_ckpt or not os.path.exists(sam_ckpt):
                    error_msg = f"Missing SAM checkpoint file for {target_sam_type}!"
                    print(f"--> [Core Error] {error_msg}")
                    if callback_signal: callback_signal.emit(100, error_msg)
                    raise FileNotFoundError(error_msg)

                if callback_signal: callback_signal.emit(50, f"Loading SAM ({target_sam_type}) on {self.device}...")
                print(f"--> [Core] Loading SAM model ({target_sam_type})...")
                
                self.sam_model = sam_model_registry[target_sam_type](checkpoint=sam_ckpt)
                self.sam_model.to(device=self.device)
                
                self.current_sam_type = target_sam_type

            t1 = time.time()
            if callback_signal: callback_signal.emit(100, f"Ready. (Load Time: {t1-t0:.1f}s)")
            print(f"--> [Core] AI Engines ({self.current_sam_type}) Started on {self.device}.")

    def get_models(self, sam_type='vit_b'):

        with self._lock:  
            if self.sam_model is None or self.current_sam_type != sam_type:
                self.load_models(callback_signal=None, target_sam_type=sam_type)
                
            return self.depth_model, self.sam_model, self.device