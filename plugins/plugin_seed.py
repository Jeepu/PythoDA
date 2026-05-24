import cv2
import numpy as np
from core.universal_engine import UniversalPhenoEngine

class SeedStrategy:
    """
    Seed phenotyping strategy using SAM + Distance Transform Watershed 
    for Touching Seeds Separation.
    """

    def get_config(self):
        return {
            'max_infer_dim': 1024,
            'sam_mode': 'amg',
            'sam_params': {
                'points_per_side': 64,
                'pred_iou_thresh': 0.86,
                'stability_score_thresh': 0.92,
                'crop_n_layers': 0,
                'min_mask_region_area': 30  # auto-scaled by the engine
            }
        }

    def generate_prompts(self, infer_image, depth_uint8):
        """Extract foreground via adaptive threshold on depth map."""
        dav2_mask = cv2.adaptiveThreshold(
            depth_uint8, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, -10
        )
        contours, _ = cv2.findContours(dav2_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(dav2_mask, contours, -1, 255, cv2.FILLED)
        return dav2_mask

    def extract_features(self, raw_image_orig, infer_image, depth_uint8, sam_masks, prompt_data,
                         scale_factor, color_namer, viz_params, progress_callback):
        """SAM masking followed by Distance Transform Watershed to cut touching seeds."""
        dav2_mask = prompt_data
        h, w = infer_image.shape[:2]
        orig_h, orig_w = raw_image_orig.shape[:2]
        total_pixels = h * w

        raw_fragments_map = np.zeros((h, w), dtype=np.uint8)
        sam_all_viz = np.zeros_like(infer_image)

        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dav2_mask_dilated = cv2.morphologyEx(dav2_mask, cv2.MORPH_DILATE, kernel_dilate)

        # Stage 1: Filter SAM masks by depth overlap
        for ann in sam_masks:
            mask = ann['segmentation']
            area = ann['area']
            if area > total_pixels * 0.2: continue
            
            intersection = np.logical_and(mask, dav2_mask_dilated > 0)
            if np.sum(intersection) / area < 0.2: continue

            raw_fragments_map = cv2.bitwise_or(raw_fragments_map, mask.astype(np.uint8) * 255)
            color = np.random.randint(0, 255, (1, 3)).tolist()[0]
            sam_all_viz[mask] = color

        # =================================================================
        # 🔥 Stage 2: Watershed Algorithm for Touching Seeds Separation
        # =================================================================
        progress_callback.emit(75, "Applying Watershed Segmentation...")
        
        # 1. Fill tiny holes gently (Do NOT use large kernels here)
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_mask = cv2.morphologyEx(raw_fragments_map, cv2.MORPH_CLOSE, clean_kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, clean_kernel)

        # 2. Find sure background
        sure_bg = cv2.dilate(binary_mask, clean_kernel, iterations=2)

        # 3. Find sure foreground (Seed peaks) via Distance Transform
        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        
        # PARAMETER TWEAK: 0.45 controls how aggressive the cut is.
        # Higher (e.g., 0.6) cuts more aggressively. Lower (e.g., 0.3) cuts less.
        _, sure_fg = cv2.threshold(dist_transform, 0.45 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        # 4. Find unknown region (The potential touching boundaries)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # 5. Marker labelling
        num_labels, markers = cv2.connectedComponents(sure_fg)
        
        # Add 1 to all labels so that sure background is not 0, but 1
        markers = markers + 1
        
        # Mark the region of unknown with zero
        markers[unknown == 255] = 0

        # 6. Apply Watershed (Use masked infer_image to guide cuts along actual physical gaps)
        watershed_input = cv2.bitwise_and(infer_image, infer_image, mask=binary_mask)
        markers = cv2.watershed(watershed_input, markers)

        # =================================================================
        # Stage 3: Restore original coordinates and compute geometry
        # =================================================================
        progress_callback.emit(85, "Analyzing Separated Seeds...")
        analyzed_data = []
        annotated_img = raw_image_orig.copy()
        valid_seed_mask_small = np.zeros((h, w), dtype=np.uint8)
        image_rgb = cv2.cvtColor(infer_image, cv2.COLOR_BGR2RGB)

        seed_id = 1
        # Start from 2 because label 1 is background, and label -1 is the watershed boundary
        for marker_id in range(2, num_labels + 1): 
            
            # Extract the specific seed from markers
            current_mask_uint8 = np.zeros((h, w), dtype=np.uint8)
            current_mask_uint8[markers == marker_id] = 255
            current_mask_bool = current_mask_uint8 > 0
            
            # Area noise filter
            area_small = np.sum(current_mask_bool)
            if area_small < 50: 
                continue 

            valid_seed_mask_small = cv2.bitwise_or(valid_seed_mask_small, current_mask_uint8)

            contours, _ = cv2.findContours(current_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: continue
            cnt_small = max(contours, key=cv2.contourArea)

            real_area = area_small / (scale_factor * scale_factor)
            cnt_orig = (cnt_small / scale_factor).astype(np.int32)

            perimeter = cv2.arcLength(cnt_orig, True)
            rect = cv2.minAreaRect(cnt_orig)
            (cx, cy), (box_w, box_h), angle = rect
            length, width = max(box_w, box_h), min(box_w, box_h)

            if area_small > 0:
                mean_rgb = image_rgb[current_mask_bool].mean(axis=0)
            else:
                mean_rgb = np.array([0,0,0])
            
            
            real_rgb = (int(mean_rgb[0]), int(mean_rgb[1]), int(mean_rgb[2]))
            
            color_name = color_namer.get_name(real_rgb)
            hex_color = "#{:02X}{:02X}{:02X}".format(*real_rgb)

            analyzed_data.append({
                "ID": seed_id,
                "Length": round(length, 1),
                "Width": round(width, 1),
                "L/W Ratio": round(length/width if width>0 else 0, 2),
                "Circularity": round((4*np.pi*real_area)/(perimeter**2) if perimeter>0 else 0, 2),
                "Perimeter": round(perimeter, 1), 
                "Area": int(real_area),
                "Color": color_name,
                "RGB": f"{real_rgb[0]}, {real_rgb[1]}, {real_rgb[2]}",
                "ColorHex": hex_color,
                "Swatch": "",
                "_RGB_tuple": real_rgb,
                "_contour": cnt_orig,
                "_center": (int(cx), int(cy))
            })

            cv2.drawContours(annotated_img, [cnt_orig], -1, (34, 139, 34), viz_params['viz_thickness'])
            
            # Inner white solid text with outer black stroke for maximum visibility
            cv2.putText(annotated_img, str(seed_id), (int(cx), int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, viz_params['viz_font_scale'], (0, 0, 0), viz_params['viz_thickness'] + 2, cv2.LINE_AA)
            cv2.putText(annotated_img, str(seed_id), (int(cx), int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, viz_params['viz_font_scale'], (255, 255, 255), viz_params['viz_thickness'], cv2.LINE_AA)
            
            seed_id += 1

        stats = {
            "Total Seeds": len(analyzed_data),
            "Avg Length": round(np.mean([d["Length"] for d in analyzed_data]), 1) if analyzed_data else 0,
            "Avg Area": int(np.mean([d["Area"] for d in analyzed_data])) if analyzed_data else 0
        }

        final_mask_orig = cv2.resize(valid_seed_mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        return {
            'analyzed_data': analyzed_data,
            'stats': stats,
            'depth_mask_small': dav2_mask,
            'sam_viz_small': sam_all_viz,
            'final_mask_orig': final_mask_orig,
            'annotated_img_rgb': annotated_img_rgb
        }

class SeedPlugin(UniversalPhenoEngine):
    def __init__(self):
        super().__init__(SeedStrategy())