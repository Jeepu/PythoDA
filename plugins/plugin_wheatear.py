import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator

from core.universal_engine import UniversalPhenoEngine
from core.model_loader import ModelLoader

class WheatEarStrategy:
    """Wheat ear phenotyping strategy with per-ROI SAM segmentation and dual depth/area adaptive filtering."""

    def get_config(self):
        return {
            'max_infer_dim': 1280,
            'sam_mode': 'custom'  # strategy handles SAM locally within each ear ROI
        }

    def generate_prompts(self, infer_image, depth_uint8):
        """Extract wheat ear regions via Otsu depth threshold and generate ROI boxes."""
        _, depth_bin = cv2.threshold(depth_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        depth_mask = cv2.morphologyEx(depth_bin, cv2.MORPH_CLOSE, kernel, iterations=3)

        contours, _ = cv2.findContours(depth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = infer_image.shape[:2]
        prompt_boxes = []
        min_ear_area = (h * w) * 0.01

        for cnt in contours:
            if cv2.contourArea(cnt) < min_ear_area: continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            pad = 10
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + bw + pad)
            y2 = min(h, y + bh + pad)
            prompt_boxes.append((x1, y1, x2, y2))

        return {
            'boxes': prompt_boxes,
            'depth_mask': depth_mask
        }

    def extract_features(self, raw_image_orig, infer_image, depth_uint8, sam_masks, prompt_data,
                         scale_factor, color_namer, viz_params, progress_callback):
        """Per-ROI fine-grained SAM segmentation, NMS, dual depth/area filtering, coordinate restoration."""
        prompt_boxes = prompt_data['boxes']
        depth_mask = prompt_data['depth_mask']

        orig_h, orig_w = raw_image_orig.shape[:2]
        h, w = infer_image.shape[:2]

        analyzed_data = []
        final_mask_accumulator_orig = np.zeros((orig_h, orig_w), dtype=np.uint8)
        sam_viz_small = np.zeros_like(infer_image)
        annotated_img_rgb = cv2.cvtColor(raw_image_orig, cv2.COLOR_BGR2RGB).copy()

        if not prompt_boxes:
            return {
                'analyzed_data': [],
                'stats': {"Status": "No Ears Found"},
                'depth_mask_small': depth_mask,
                'sam_viz_small': sam_viz_small,
                'final_mask_orig': final_mask_accumulator_orig,
                'annotated_img_rgb': annotated_img_rgb
            }

        # Build a high-density SAM mask generator for fine-grained kernel detection.
        # min_mask_region_area is manually scaled here because the engine's auto-scaling
        # only applies to 'amg' mode, and this plugin uses 'custom'.
        _, sam_model, _ = ModelLoader().get_models()
        mask_generator = SamAutomaticMaskGenerator(
            model=sam_model,
            points_per_side=64,
            pred_iou_thresh=0.75,
            stability_score_thresh=0.80,
            crop_n_layers=0,
            min_mask_region_area=int(20 * scale_factor * scale_factor)
        )

        kernel_id_counter = 1

        for i, box in enumerate(prompt_boxes):
            x1, y1, x2, y2 = box

            roi_img = infer_image[y1:y2, x1:x2]
            roi_depth_mask = depth_mask[y1:y2, x1:x2]
            roi_depth = depth_uint8[y1:y2, x1:x2]
            roi_rgb = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
            roi_h, roi_w = roi_img.shape[:2]
            roi_total = roi_h * roi_w

            roi_masks = mask_generator.generate(roi_rgb)

            # --- Step 1: NMS to get plump kernel candidates ---
            roi_masks.sort(key=lambda x: x['predicted_iou'], reverse=True)
            occupied = np.zeros((roi_h, roi_w), dtype=bool)

            plump_candidates = []

            for ann in roi_masks:
                area = ann['area']
                mask = ann['segmentation']

                if area < (50 * scale_factor * scale_factor): continue
                if area > roi_total * 0.3: continue

                overlap = np.sum(np.logical_and(mask, roi_depth_mask > 0))
                if overlap / area < 0.5: continue

                intersection = np.logical_and(mask, occupied)
                if np.sum(intersection) / area > 0.2: continue

                occupied = np.logical_or(occupied, mask)
                plump_candidates.append(ann)

            if not plump_candidates: continue

            # --- Step 2: 3D depth + area dual adaptive filter ---
            cand_depths = []
            for ann in plump_candidates:
                mean_d = cv2.mean(roi_depth, mask=ann['segmentation'].astype(np.uint8))[0]
                cand_depths.append(mean_d)

            depth_thresh = np.percentile(cand_depths, 30) if cand_depths else 0

            areas_np = np.array([ann['area'] for ann in plump_candidates])
            median_area = np.median(areas_np)
            min_area_thresh = median_area * 0.3
            max_area_thresh = median_area * 3.0

            # --- Step 3: final filtering and data packaging ---
            for idx, ann in enumerate(plump_candidates):
                local_mask = ann['segmentation']
                area = ann['area']
                mean_d = cand_depths[idx]

                if area < min_area_thresh or area > max_area_thresh: continue
                if mean_d < depth_thresh: continue

                mask_u8_local = local_mask.astype(np.uint8) * 255
                cnts_local, _ = cv2.findContours(mask_u8_local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cnts_local: continue
                cnt_local = max(cnts_local, key=cv2.contourArea)

                rect = cv2.minAreaRect(cnt_local)
                (rcx, rcy), (rw, rh), rangle = rect
                length = max(rw, rh)
                width = min(rw, rh)
                aspect_ratio = length / width if width > 0 else 0
                if aspect_ratio > 4.0: continue

                # Restore coordinates to original image space
                cnt_small_global = cnt_local + np.array([x1, y1])
                cnt_orig_global = (cnt_small_global / scale_factor).astype(np.int32)

                real_area = area / (scale_factor * scale_factor)
                real_len = length / scale_factor
                real_wid = width / scale_factor

                cv2.drawContours(final_mask_accumulator_orig, [cnt_orig_global], -1, 255, -1)

                # SAM small-scale visualization
                global_mask_small = np.zeros((h, w), dtype=bool)
                global_mask_small[y1:y2, x1:x2] = local_mask
                color = np.random.randint(50, 255, (1, 3)).tolist()[0]
                sam_viz_small[global_mask_small] = color

                # Color extraction from original image
                mask_u8_global_orig = np.zeros((orig_h, orig_w), dtype=np.uint8)
                cv2.drawContours(mask_u8_global_orig, [cnt_orig_global], -1, 255, -1)
                mask_bool = mask_u8_global_orig > 0

                if np.sum(mask_bool) > 0:
                    mean_color = raw_image_orig[mask_bool].mean(axis=0)
                    real_rgb = (int(mean_color[2]), int(mean_color[1]), int(mean_color[0]))
                else:
                    real_rgb = (0, 0, 0)

                hex_color = "#{:02X}{:02X}{:02X}".format(*real_rgb)
                color_name = color_namer.get_name(real_rgb)

                # Compute geometric center
                M = cv2.moments(cnt_orig_global)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    rx, ry, rbox_w, rbox_h = cv2.boundingRect(cnt_orig_global)
                    cx, cy = rx + rbox_w // 2, ry + rbox_h // 2

                analyzed_data.append({
                    "ID": kernel_id_counter,
                    "Length": round(real_len, 1),
                    "Width": round(real_wid, 1),
                    "Area": int(real_area),
                    "L/W Ratio": round(aspect_ratio, 2),
                    "Color": color_name,
                    "RGB": f"{real_rgb[0]}, {real_rgb[1]}, {real_rgb[2]}",
                    "ColorHex": hex_color,
                    "Swatch": "",
                    "_RGB_tuple": real_rgb,
                    "_contour": cnt_orig_global,
                    "_center": (cx, cy)
                })

                kernel_id_counter += 1

            progress = 30 + int((i + 1) / len(prompt_boxes) * 60)
            progress_callback.emit(progress, f"Ear {i+1}: Processing...")

        stats = {}
        if analyzed_data:
            avg_len = np.mean([d['Length'] for d in analyzed_data])
            avg_area = np.mean([d['Area'] for d in analyzed_data])
            stats = {
                "Total Kernels": len(analyzed_data),
                "Avg Length": round(avg_len, 1),
                "Avg Area": int(avg_area),
                "Method": "Stats Filter"
            }
        else:
            stats = {"Status": "No Kernels Found"}

        return {
            'analyzed_data': analyzed_data,
            'stats': stats,
            'depth_mask_small': depth_mask,
            'sam_viz_small': sam_viz_small,
            'final_mask_orig': final_mask_accumulator_orig,
            'annotated_img_rgb': annotated_img_rgb
        }

class WheatEarPlugin(UniversalPhenoEngine):
    def __init__(self):
        super().__init__(WheatEarStrategy())
