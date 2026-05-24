import cv2
import numpy as np
import math
from segment_anything import SamPredictor

from core.universal_engine import UniversalPhenoEngine
from core.model_loader import ModelLoader

class TomatoStrategy:
    """
    Tomato phenotyping strategy using Depth-Constrained Point Grid SAM 
    and Post-Geometric Prior Validation.
    """
    def __init__(self):
        # Default depth thresholds. These should be linked to the UI's dual-handle slider.
        self.depth_min = 120 
        self.depth_max = 255

    def get_config(self):
        return {
            'max_infer_dim': 1280,
            'sam_mode': 'predictor'
        }

    def generate_prompts(self, infer_image, depth_uint8):
        """
        Phase 1 & 2: Dynamic Depth Masking & Point Grid Sampling
        """
        # 1. Dynamic Depth Extraction (Replaces rigid Otsu)
        depth_mask = cv2.inRange(depth_uint8, self.depth_min, self.depth_max)

        # Morphological cleanup to remove tiny noise from depth map
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 2. Generate Uniform Point Grid over the valid depth area
        h, w = depth_mask.shape
        grid_spacing = 32  # Step size for point sampling. Decrease for smaller targets.
        prompt_points = []
        
        for y in range(grid_spacing // 2, h, grid_spacing):
            for x in range(grid_spacing // 2, w, grid_spacing):
                if depth_mask[y, x] > 0:
                    prompt_points.append([x, y])

        return {
            'depth_mask': depth_mask,
            'points': prompt_points
        }

    def extract_features(self, raw_image_orig, infer_image, depth_uint8, sam_masks, prompt_data,
                         scale_factor, color_namer, viz_params, progress_callback):
        prompt_points = prompt_data.get('points', [])
        depth_mask_small = prompt_data.get('depth_mask')
        orig_h, orig_w = raw_image_orig.shape[:2]
        h, w = infer_image.shape[:2]

        analyzed_data = []
        final_mask_accumulator_orig = np.zeros((orig_h, orig_w), dtype=np.uint8)
        sam_viz_small = np.zeros_like(infer_image)
        annotated_img_rgb = cv2.cvtColor(raw_image_orig, cv2.COLOR_BGR2RGB).copy()

        # Tracking arrays to prevent duplicate detections
        segmented_tracker = np.zeros((h, w), dtype=np.uint8)
        valid_history_masks = [] # List to store valid masks for overlap checking

        if not prompt_points:
            return {
                'analyzed_data': [], 'stats': {"Total Tomatoes": 0, "Avg Volume": 0},
                'depth_mask_small': depth_mask_small, 'sam_viz_small': sam_viz_small,
                'final_mask_orig': final_mask_accumulator_orig, 'annotated_img_rgb': annotated_img_rgb
            }

        _, sam_model, _ = ModelLoader().get_models()
        predictor = SamPredictor(sam_model)
        image_rgb = cv2.cvtColor(infer_image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)

        uid_counter = 1
        total_pts = len(prompt_points)
        min_area_px = (h * w) * 0.002

        for i, pt in enumerate(prompt_points):
            x, y = pt
            
            # 1. Fast Grid-NMS (Point level)
            if segmented_tracker[y, x] > 0:
                continue

            input_point = np.array([[x, y]])
            input_label = np.array([1])
            
            masks, scores, _ = predictor.predict(
                point_coords=input_point, point_labels=input_label, multimask_output=False
            )

            best_mask = masks[0]
            if best_mask.sum() < min_area_px: 
                continue

            # --- Hole Filling Logic ---
            temp_mask = best_mask.astype(np.uint8) * 255
            tmp_cnts, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not tmp_cnts: continue
            
            cv2.drawContours(temp_mask, tmp_cnts, -1, 255, thickness=cv2.FILLED)
            best_mask_uint8 = temp_mask # Save uint8 version for intersection math
            best_mask = temp_mask > 0
            
            # --- Phase 4: Geometric Prior Validation ---
            cnt_mask = max(tmp_cnts, key=cv2.contourArea)
            pixel_area = cv2.contourArea(cnt_mask)
            pixel_perimeter = cv2.arcLength(cnt_mask, True)
            
            if pixel_perimeter == 0 or pixel_area < min_area_px: 
                continue

            circularity = (4 * np.pi * pixel_area) / (pixel_perimeter ** 2)
            if circularity < 0.70: 
                continue 

            x_b, y_b, w_b, h_b = cv2.boundingRect(cnt_mask)
            aspect_ratio = float(w_b) / h_b
            if aspect_ratio < 0.5 or aspect_ratio > 2.0: 
                continue

            is_duplicate = False
            current_area = np.sum(best_mask_uint8 > 0)
            
            for prev_mask in valid_history_masks:
                # Calculate pixel-wise intersection
                intersection = cv2.bitwise_and(best_mask_uint8, prev_mask)
                intersection_area = np.sum(intersection > 0)
                
                # Check overlap ratio against the SMALLER of the two masks 
                # (This perfectly catches both "Containment" and "Massive Overlap")
                prev_area = np.sum(prev_mask > 0)
                min_compare_area = min(current_area, prev_area)
                
                if min_compare_area > 0 and (intersection_area / min_compare_area) > 0.75:
                    is_duplicate = True
                    # Optional English Debug Log:
                    # print(f"Duplicate rejected: Overlap ratio {(intersection_area / min_compare_area):.2f}")
                    break
                    
            if is_duplicate:
                continue # Discard this redundant prediction
                
            # Add to history for future overlap checks
            valid_history_masks.append(best_mask_uint8.copy())
            # =======================================================

            # Validation Passed! Register the tomato.
            global_mask_small = np.zeros((h, w), dtype=np.uint8)
            global_mask_small[best_mask] = 255
            
            segmented_tracker = cv2.bitwise_or(segmented_tracker, global_mask_small)

            color = np.random.randint(0, 255, (1, 3)).tolist()[0]
            sam_viz_small[best_mask] = color

            mask_orig = cv2.resize(global_mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            final_mask_accumulator_orig = cv2.bitwise_or(final_mask_accumulator_orig, mask_orig)

            cnts_orig, _ = cv2.findContours(mask_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts_orig: continue
            cnt_mask_orig = max(cnts_orig, key=cv2.contourArea)

            real_area = pixel_area / (scale_factor * scale_factor)
            real_perimeter = pixel_perimeter / scale_factor
            equiv_diameter = 2 * np.sqrt(real_area / np.pi)

            radius = equiv_diameter / 2
            volume = (4 / 3) * np.pi * (radius ** 3)

            rect = cv2.minAreaRect(cnt_mask_orig)
            (cx, cy), (box_w, box_h), angle = rect
            length = max(box_w, box_h) / scale_factor
            width = min(box_w, box_h) / scale_factor

            mask_bool = mask_orig > 0
            mean_rgb = raw_image_orig[mask_bool].mean(axis=0) if np.sum(mask_bool) > 0 else np.array([0,0,0])
            real_rgb = (int(mean_rgb[2]), int(mean_rgb[1]), int(mean_rgb[0]))
            color_name = color_namer.get_name(real_rgb)
            hex_color = "#{:02X}{:02X}{:02X}".format(real_rgb[0], real_rgb[1], real_rgb[2])

            cx_text, cy_text = int(cx), int(cy)

            analyzed_data.append({
                "ID": uid_counter,
                "Length": round(length, 1),
                "Width": round(width, 1),
                "Diameter": round(equiv_diameter, 1),
                "Volume": int(volume),
                "Perimeter": round(real_perimeter, 1),
                "Area": int(real_area),
                "Color": color_name,
                "RGB": f"{real_rgb[0]}, {real_rgb[1]}, {real_rgb[2]}",
                "ColorHex": hex_color,
                "Swatch": "",
                "Box": [int(x_b/scale_factor), int(y_b/scale_factor), int((x_b+w_b)/scale_factor), int((y_b+h_b)/scale_factor)],
                "_RGB_tuple": real_rgb,
                "_contour": cnt_mask_orig,
                "_center": (cx_text, cy_text),
                "_thickness_boost": 2
            })

            uid_counter += 1
            
            if i % 10 == 0 or i == total_pts - 1:
                step_progress = 40 + int((i / total_pts) * 50)
                progress_callback.emit(step_progress, f"Scanning valid structures: {uid_counter-1} Tomatoes found...")

        stats = {}
        if analyzed_data:
            avg_vol = int(np.mean([d["Volume"] for d in analyzed_data]))
            stats = {
                "Total Tomatoes": len(analyzed_data),
                "Avg Volume": avg_vol
            }
        else:
            stats = {"Total Tomatoes": 0, "Avg Volume": 0}

        return {
            'analyzed_data': analyzed_data, 'stats': stats,
            'depth_mask_small': depth_mask_small, 'sam_viz_small': sam_viz_small,
            'final_mask_orig': final_mask_accumulator_orig, 'annotated_img_rgb': annotated_img_rgb
        }

class TomatoPlugin(UniversalPhenoEngine):
    def __init__(self):
        super().__init__(TomatoStrategy())