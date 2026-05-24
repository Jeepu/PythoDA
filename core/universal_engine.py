# 文件路径: core/universal_engine.py
import cv2
import numpy as np
import gc
import torch
import traceback
from segment_anything import SamAutomaticMaskGenerator, SamPredictor

from core.base_plugin import BasePhenoPlugin
from core.model_loader import ModelLoader
from core.utils import ColorNamer, imread_unicode

class UniversalPhenoEngine(BasePhenoPlugin):
    def __init__(self, strategy):
        super().__init__()
        self.strategy = strategy
        self.color_namer = ColorNamer()

    def _clean_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self, image_path, params=None):
        self.is_running = True
        try:
            self._clean_memory()
            self.update_progress.emit(5, f"Loading image: {image_path}...")

            raw_image_orig = imread_unicode(image_path)
            if raw_image_orig is None: raise ValueError("Failed to load image.")
            orig_h, orig_w = raw_image_orig.shape[:2]

            config = self.strategy.get_config()
            max_dim = config.get('max_infer_dim', 1280)
            scale_factor = 1.0
            
            if max(orig_h, orig_w) > max_dim:
                scale_factor = max_dim / max(orig_h, orig_w)
                new_w, new_h = int(orig_w * scale_factor), int(orig_h * scale_factor)
                infer_image = cv2.resize(raw_image_orig, (new_w, new_h), interpolation=cv2.INTER_AREA)
                self.update_progress.emit(8, f"Engine: Auto-scaling to {new_w}x{new_h}...")
            else:
                infer_image = raw_image_orig.copy()
            
            h, w = infer_image.shape[:2]

            viz_params = {
                'viz_thickness': max(2, int(3 * (max(orig_h, orig_w) / 1000.0))),
                'viz_font_scale': max(0.6, 0.9 * (max(orig_h, orig_w) / 1000.0))
            }

            sam_type = "vit_b" 
            if params and "sam_type" in params:
                sam_type = params["sam_type"]
                
            self.update_progress.emit(10, f"Engine: Loading AI Models ({sam_type})...")
            
            loader = ModelLoader()
            depth_model, sam_model, device = loader.get_models(sam_type=sam_type)
            
            if depth_model is None or sam_model is None:
                raise RuntimeError("AI Models not loaded.")

            if not self.is_running: return

            self.update_progress.emit(20, "Engine: Estimating Depth...")
            
            with torch.inference_mode():
                depth = depth_model.infer_image(infer_image, 518)
                
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
            depth_uint8 = ((depth - depth.min()) / (depth.max() - depth.min()) * 255.0).astype(np.uint8)
            depth_colormap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)

            if not self.is_running: return

            self.update_progress.emit(30, "Strategy: Generating Prompts/Masks...")
            prompt_data = self.strategy.generate_prompts(infer_image, depth_uint8)

            self.update_progress.emit(40, "Engine: Running SAM Segmentation...")
            sam_mode = config.get('sam_mode', 'amg')
            image_rgb = cv2.cvtColor(infer_image, cv2.COLOR_BGR2RGB)
            
            sam_masks = []
            
            with torch.inference_mode():
                if sam_mode == 'amg':
                    sam_params = config.get('sam_params', {}).copy()
                    base_area = sam_params.get('min_mask_region_area', 0)
                    if base_area > 0:
                        sam_params['min_mask_region_area'] = int(base_area * scale_factor * scale_factor)
                    
                    mask_generator = SamAutomaticMaskGenerator(model=sam_model, **sam_params)
                    sam_masks = mask_generator.generate(image_rgb)
                    
                elif sam_mode == 'predictor':
                    predictor = SamPredictor(sam_model)
                    predictor.set_image(image_rgb)
                    sam_masks = prompt_data 

            if not self.is_running: return

            self.update_progress.emit(70, "Strategy: Analyzing Features & Biology...")
            analysis_result = self.strategy.extract_features(
                raw_image_orig, infer_image, depth_uint8, sam_masks, prompt_data, 
                scale_factor, self.color_namer, viz_params, self.update_progress
            )
            
            analyzed_data = analysis_result['analyzed_data']
            stats = analysis_result['stats']
            depth_mask_small = analysis_result['depth_mask_small']
            sam_viz_small = analysis_result['sam_viz_small']
            final_mask_orig = analysis_result['final_mask_orig']
            annotated_img_rgb = analysis_result['annotated_img_rgb']

            self.update_progress.emit(95, "Engine: Finalizing Six Views...")
            img1 = cv2.cvtColor(raw_image_orig, cv2.COLOR_BGR2RGB)
            
            depth_colormap_orig = cv2.resize(depth_colormap, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            img2 = cv2.cvtColor(depth_colormap_orig, cv2.COLOR_BGR2RGB)
            
            depth_mask_orig = cv2.resize(depth_mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            img3 = cv2.cvtColor(depth_mask_orig, cv2.COLOR_GRAY2RGB)
            
            raw_depth_cutout = cv2.bitwise_and(raw_image_orig, raw_image_orig, mask=depth_mask_orig)
            img4 = cv2.cvtColor(raw_depth_cutout, cv2.COLOR_BGR2RGB)
            
            sam_viz_orig = cv2.resize(sam_viz_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            img5 = cv2.cvtColor(sam_viz_orig, cv2.COLOR_BGR2RGB)
            
            final_cutout = cv2.bitwise_and(raw_image_orig, raw_image_orig, mask=final_mask_orig)
            img6 = cv2.cvtColor(final_cutout, cv2.COLOR_BGR2RGB)

            image_pack = {
                "main": annotated_img_rgb,
                "steps": [img1, img2, img3, img4, img5, img6]
            }

            self.result_ready.emit(image_pack, stats, analyzed_data)
            self.update_progress.emit(100, "Analysis Complete.")

        except Exception as e:
            traceback.print_exc()
            self.error_occurred.emit(str(e))
        finally:
            self._clean_memory()
            self.is_running = False