import cv2
import numpy as np

from scipy.spatial.distance import pdist, squareform
from core.universal_engine import UniversalPhenoEngine

class CanopyStrategy:
    """Canopy phenotyping strategy with ExG semantic filtering and multi-metric geometric analysis."""

    def get_config(self):
        return {
            'max_infer_dim': 1280,
            'sam_mode': 'amg',
            'sam_params': {
                'points_per_side': 32,
                'pred_iou_thresh': 0.88,
                'stability_score_thresh': 0.92,
                'crop_n_layers': 0,
                'min_mask_region_area': 100  # auto-scaled by the engine
            }
        }

    def generate_prompts(self, infer_image, depth_uint8):
        """Extract coarse foreground mask via Otsu threshold on depth map."""
        _, depth_mask_infer = cv2.threshold(depth_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        depth_mask_infer = cv2.morphologyEx(depth_mask_infer, cv2.MORPH_CLOSE, kernel)
        return depth_mask_infer

    def extract_features(self, raw_image_orig, infer_image, depth_uint8, sam_masks, prompt_data,
                         scale_factor, color_namer, viz_params, progress_callback):
        """SAM segmentation -> dual depth+ExG filter -> physical unit conversion -> data packaging."""
        depth_mask_infer = prompt_data
        h, w = infer_image.shape[:2]
        orig_h, orig_w = raw_image_orig.shape[:2]

        progress_callback.emit(60, "Filtering Background via Depth & ExG...")

        plant_mask_accum = np.zeros((h, w), dtype=np.uint8)
        sam_viz_small = np.zeros_like(infer_image)

        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        depth_mask_loose = cv2.morphologyEx(depth_mask_infer, cv2.MORPH_DILATE, kernel_dilate)

        img_infer_rgb = cv2.cvtColor(infer_image, cv2.COLOR_BGR2RGB)

        # Stage 1: SAM mask cleaning via depth overlap + ExG color verification
        for ann in sam_masks:
            mask = ann['segmentation']
            area = ann['area']

            # Skip giant background regions
            if area > (h * w) * 0.8: continue

            # Depth verification
            overlap = np.sum(np.logical_and(mask, depth_mask_loose > 0))
            if overlap / area < 0.4: continue

            # ExG color verification to exclude pots, soil, etc.
            if not self._is_green_plant(img_infer_rgb, mask): continue

            plant_mask_accum = cv2.bitwise_or(plant_mask_accum, mask.astype(np.uint8) * 255)
            color = np.random.randint(0, 255, (1, 3)).tolist()[0]
            sam_viz_small[mask] = color

        # Morphological close to fill internal holes in plant regions
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        plant_mask_accum = cv2.morphologyEx(plant_mask_accum, cv2.MORPH_CLOSE, kernel_close)

        progress_callback.emit(80, "Analyzing Geometry & Physiology on Original Scale...")

        # Stage 2: restore to original resolution for high-precision connected component analysis
        final_mask_orig = cv2.resize(plant_mask_accum, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        num_labels, labels, stats_cv, centroids = cv2.connectedComponentsWithStats(final_mask_orig, connectivity=8)

        analyzed_data = []
        annotated_img_rgb = cv2.cvtColor(raw_image_orig, cv2.COLOR_BGR2RGB).copy()
        image_hsv_orig = cv2.cvtColor(raw_image_orig, cv2.COLOR_BGR2HSV)

        total_pixels_orig = orig_h * orig_w
        min_plant_area = total_pixels_orig * 0.005  # filter noise below 0.5%
        plant_id = 1

        # Stage 3: geometric and physiological trait computation
        for i in range(1, num_labels):
            pixel_area = stats_cv[i, cv2.CC_STAT_AREA]
            if pixel_area < min_plant_area: continue

            current_mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: continue
            cnt = max(contours, key=cv2.contourArea)
            real_area = pixel_area 
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w_rect, h_rect), angle = rect
            b_len = max(w_rect, h_rect) 
            b_wid = min(w_rect, h_rect) 
            

            (cir_x, cir_y), cir_radius = cv2.minEnclosingCircle(cnt)
            real_cir_dia = cir_radius * 2 

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = pixel_area / hull_area if hull_area > 0 else 0
            
            real_hull_area = hull_area 
            real_perimeter = cv2.arcLength(cnt, True) 

            hull_pts = hull[:, 0, :]
            if len(hull_pts) >= 2:
                dists = pdist(hull_pts, metric='euclidean')
                dists_mat = squareform(dists)
                max_idx_flat = np.argmax(dists_mat)
                idx_i, idx_j = np.unravel_index(max_idx_flat, dists_mat.shape)
                p1, p2 = hull_pts[idx_i], hull_pts[idx_j]
                true_len = dists_mat[idx_i, idx_j] 

                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                rot_angle = np.degrees(np.arctan2(dy, dx))
                rot_center = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
                M_rot = cv2.getRotationMatrix2D(rot_center, rot_angle, 1.0)

                cnt_pts_float = cnt.astype(np.float32)
                rotated_pts = cv2.transform(cnt_pts_float, M_rot)
                y_coords = rotated_pts[:, 0, 1]
                true_wid = (np.max(y_coords) - np.min(y_coords)) 
            else:
                true_len, true_wid = b_len, b_wid

            plant_pixels_hsv = image_hsv_orig[labels == i]
            if len(plant_pixels_hsv) > 0:
                h_channel = plant_pixels_hsv[:, 0]
                green_pixels = np.sum((h_channel > 35) & (h_channel < 85))
                green_ratio = green_pixels / len(plant_pixels_hsv)
            else:
                green_ratio = 0

            mean_rgb = cv2.mean(annotated_img_rgb, mask=current_mask)[:3]
            real_rgb = (int(mean_rgb[0]), int(mean_rgb[1]), int(mean_rgb[2]))
            hex_color = "#{:02X}{:02X}{:02X}".format(*real_rgb)
            color_name = color_namer.get_name(real_rgb)

            analyzed_data.append({
                "ID": plant_id,
                "Area": int(real_area),
                "Hull Area": int(real_hull_area),
                "Perimeter": round(real_perimeter, 1),
                "Length": round(true_len, 1),
                "Width": round(true_wid, 1),
                "B.Length": round(b_len, 1),
                "B.Width": round(b_wid, 1),
                "Circle Dia": round(real_cir_dia, 1),
                "Solidity": f"{solidity*100:.1f}%",
                "Green": f"{green_ratio*100:.1f}%",
                "Color": color_name,
                "RGB": f"{real_rgb[0]}, {real_rgb[1]}, {real_rgb[2]}",
                "ColorHex": hex_color,
                "Swatch": "",
                "_RGB_tuple": real_rgb,
                "_contour": cnt,
                "_center": (int(cx), int(cy)),
                "_hull": hull,
                "_box": np.int32(cv2.boxPoints(rect)),
                "_circle": ((int(cir_x), int(cir_y)), int(cir_radius))
            })

            plant_id += 1

        stats = {
            "Total Plants": len(analyzed_data),
            "Avg Area": int(np.mean([d["Area"] for d in analyzed_data])) if analyzed_data else 0,
            "Avg Green": f"{np.mean([float(d['Green'][:-1]) for d in analyzed_data]):.1f}%" if analyzed_data else "0%"
        }

        return {
            'analyzed_data': analyzed_data,
            'stats': stats,
            'depth_mask_small': depth_mask_infer,
            'sam_viz_small': sam_viz_small,
            'final_mask_orig': final_mask_orig,
            'annotated_img_rgb': annotated_img_rgb
        }

    def _is_green_plant(self, image_rgb, mask):
        """Check if a masked region is green vegetation using the ExG (Excess Green) index.

        ExG = 2*G - R - B.  Values above 15 indicate green plant material.
        """
        mean_color = cv2.mean(image_rgb, mask=mask.astype(np.uint8))
        r, g, b = mean_color[0], mean_color[1], mean_color[2]
        exg = 2 * g - r - b
        return exg > 15

class CanopyPlugin(UniversalPhenoEngine):
    def __init__(self):
        super().__init__(CanopyStrategy())
