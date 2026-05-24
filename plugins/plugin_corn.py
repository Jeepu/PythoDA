import cv2
import numpy as np
from scipy.spatial import KDTree
from segment_anything import SamAutomaticMaskGenerator

from core.universal_engine import UniversalPhenoEngine
from core.model_loader import ModelLoader

class CornStrategy:
    """Corn ear phenotyping strategy with bilateral filtering, ROI SAM segmentation, and KDTree row tracking."""

    def __init__(self):
        # Row color palette (BGR)
        self.row_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
            (255, 165, 0), (0, 128, 128), (128, 0, 128), (128, 128, 0)
        ]

    def get_config(self):
        return {
            'max_infer_dim': 1280,
            'sam_mode': 'custom'  # strategy handles SAM locally within ROI
        }

    def generate_prompts(self, infer_image, depth_uint8):
        """Locate the corn ear ROI via depth map Otsu thresholding."""
        h, w = infer_image.shape[:2]
        _, depth_mask = cv2.threshold(depth_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Denoise and extract the largest contour as the ear
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        depth_mask_clean = cv2.morphologyEx(depth_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(depth_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        ear_cnt = max(contours, key=cv2.contourArea)
        ear_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(ear_mask, [ear_cnt], -1, 255, -1)

        rx, ry, rw, rh = cv2.boundingRect(ear_cnt)
        pad = 10
        rx, ry = max(0, rx - pad), max(0, ry - pad)
        rw, rh = min(w - rx, rw + 2*pad), min(h - ry, rh + 2*pad)

        return {
            'ear_mask': ear_mask,
            'roi_box': (rx, ry, rw, rh)
        }

    def extract_features(self, raw_image_orig, infer_image, depth_uint8, sam_masks, prompt_data,
                         scale_factor, color_namer, viz_params, progress_callback):
        """Bilateral filter -> local SAM segmentation -> NMS -> KDTree row tracking."""
        if not prompt_data:
            return self._empty_result(infer_image, raw_image_orig)

        ear_mask = prompt_data['ear_mask']
        rx, ry, rw, rh = prompt_data['roi_box']
        h, w = infer_image.shape[:2]
        orig_h, orig_w = raw_image_orig.shape[:2]

        progress_callback.emit(40, "Step 2/4: Segmenting Kernels (Bilateral Filter)...")

        # Stage 1: bilateral filter on ROI to remove silk strands
        roi_img = infer_image[ry:ry+rh, rx:rx+rw]
        roi_img_blur = cv2.bilateralFilter(roi_img, d=9, sigmaColor=75, sigmaSpace=75)

        # Stage 2: high-density SAM segmentation within ROI
        _, sam_model, _ = ModelLoader().get_models()
        mask_generator = SamAutomaticMaskGenerator(
            model=sam_model, points_per_side=64, pred_iou_thresh=0.88,
            stability_score_thresh=0.92, crop_n_layers=0, min_mask_region_area=20
        )
        roi_masks = mask_generator.generate(cv2.cvtColor(roi_img_blur, cv2.COLOR_BGR2RGB))

        progress_callback.emit(60, "Filtering Giant Objects & NMS...")

        # Stage 3: giant-object filter and NMS
        kernel_masks_small = []
        sam_all_viz = np.zeros_like(infer_image)
        occupied = np.zeros((h, w), dtype=bool)
        roi_area = rw * rh

        roi_masks.sort(key=lambda x: x['predicted_iou'], reverse=True)

        for ann in roi_masks:
            mask_roi = ann['segmentation']
            area_roi = ann['area']

            if area_roi > roi_area * 0.03: continue  # skip giant objects

            mask_full = np.zeros((h, w), dtype=bool)
            mask_full[ry:ry+rh, rx:rx+rw] = mask_roi

            y_idx, x_idx = np.where(mask_full)
            if len(y_idx) == 0: continue
            cy, cx = int(np.mean(y_idx)), int(np.mean(x_idx))
            if ear_mask[cy, cx] == 0: continue  # must lie inside ear mask

            intersection = np.logical_and(mask_full, occupied)
            if np.sum(intersection) / np.sum(mask_full) > 0.2: continue  # NMS

            occupied = np.logical_or(occupied, mask_full)
            kernel_masks_small.append(mask_full)
            color = np.random.randint(0, 255, (1, 3)).tolist()[0]
            sam_all_viz[mask_full] = color

        if not kernel_masks_small:
            return self._empty_result(infer_image, raw_image_orig)

        progress_callback.emit(70, "Step 3/4: Geometric Analysis (KDTree)...")

        # Stage 4: map masks back to original coordinates
        kernel_masks_orig = []
        final_mask_accumulator_orig = np.zeros((orig_h, orig_w), dtype=np.uint8)

        for mask_s in kernel_masks_small:
            m_orig = cv2.resize(mask_s.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            kernel_masks_orig.append(m_orig)
            final_mask_accumulator_orig = cv2.bitwise_or(final_mask_accumulator_orig, m_orig * 255)

        # Stage 5: KDTree-based row/column tracking
        analyzed_data, total_cnt, rows_cnt, avg_row, max_row_len = self._analyze_cols_rectified(
            kernel_masks_orig, (0, 0), raw_image_orig, color_namer, scale_factor
        )

        progress_callback.emit(90, "Finalizing Visuals...")

        annotated_img_rgb = cv2.cvtColor(raw_image_orig, cv2.COLOR_BGR2RGB).copy()

        stats = {
            "Total Kernels": total_cnt,
            "Est. Rows": rows_cnt,
            "Kernels/Row": f"{avg_row:.1f}",
            "Max Row Len": f"{max_row_len:.1f}"
        }

        return {
            'analyzed_data': analyzed_data,
            'stats': stats,
            'depth_mask_small': ear_mask,
            'sam_viz_small': sam_all_viz,
            'final_mask_orig': final_mask_accumulator_orig,
            'annotated_img_rgb': annotated_img_rgb
        }

    def _empty_result(self, infer_image, raw_image_orig):
        """Return an empty result when no ear can be segmented."""
        return {
            'analyzed_data': [],
            'stats': {"Status": "Failed to Segment Kernels"},
            'depth_mask_small': np.zeros_like(infer_image[:,:,0]),
            'sam_viz_small': np.zeros_like(infer_image),
            'final_mask_orig': np.zeros_like(raw_image_orig[:,:,0]),
            'annotated_img_rgb': cv2.cvtColor(raw_image_orig, cv2.COLOR_BGR2RGB)
        }

    def _analyze_cols_rectified(self, kernel_masks, offset, raw_image, color_namer, scale_factor):
        """KDTree-based texture orientation correction and vertical row tracking."""
        ox, oy = offset
        kernels = []

        for i, mask in enumerate(kernel_masks):
            moments = cv2.moments(mask.astype(np.uint8))
            if moments["m00"] == 0: continue
            cx = int(moments["m10"] / moments["m00"]) + ox
            cy = int(moments["m01"] / moments["m00"]) + oy

            pixel_area = np.sum(mask)
            real_area = pixel_area / (scale_factor * scale_factor)

            mask_bool = mask.astype(bool)
            mean_rgb = raw_image[mask_bool].mean(axis=0)
            real_rgb = (int(mean_rgb[2]), int(mean_rgb[1]), int(mean_rgb[0]))
            color_name = color_namer.get_name(real_rgb)

            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: continue
            cnt = contours[0]

            rect = cv2.minAreaRect(cnt)
            (rcx, rcy), (rw, rh), rang = rect

            k_len = max(rw, rh) / scale_factor
            k_wid = min(rw, rh) / scale_factor

            kernels.append({
                "center": np.array([cx, cy]), "area": real_area, "pixel_area": pixel_area,
                "length": k_len, "width": k_wid, "contour": cnt,
                "real_rgb": real_rgb, "color_name": color_name
            })

        if len(kernels) < 10: return [], 0, 0, 0, 0

        # Texture orientation correction via KDTree neighbor angle analysis
        points = np.array([k["center"] for k in kernels])
        tree = KDTree(points)
        angles = []
        distances, indices = tree.query(points, k=4)

        for i in range(len(points)):
            p1 = points[i]
            for j in indices[i][1:]:
                p2 = points[j]
                delta = p2 - p1
                if delta[1] < 0: delta = -delta
                angle = np.degrees(np.arctan2(delta[1], delta[0]))
                if 45 <= angle <= 135: angles.append(angle)

        if len(angles) > 10:
            hist, bin_edges = np.histogram(angles, bins=18, range=(0, 180))
            dominant_angle = (bin_edges[np.argmax(hist)] + bin_edges[np.argmax(hist)+1]) / 2
            rotation_correction = 90 - dominant_angle
        else:
            rotation_correction = 0

        theta = np.radians(rotation_correction)
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        for k in kernels:
            x, y = k["center"]
            k["rot_x"] = x * cos_t - y * sin_t
            k["rot_y"] = x * sin_t + y * cos_t

        # Vertical row tracking using pixel-area-based diameter as threshold
        kernels.sort(key=lambda x: x["rot_y"])
        avg_diam = np.median([np.sqrt(k["pixel_area"]/np.pi)*2 for k in kernels])

        columns = []
        for k in kernels:
            best_col_idx = -1
            min_x_dist = float('inf')
            for c_idx, col in enumerate(columns):
                last_k = col[-1]
                dx, dy = abs(k["rot_x"] - last_k["rot_x"]), k["rot_y"] - last_k["rot_y"]
                if dy <= 0 or dy > avg_diam * 2.8: continue
                if dx > avg_diam * 0.7: continue
                if dx < min_x_dist:
                    min_x_dist, best_col_idx = dx, c_idx

            if best_col_idx != -1: columns[best_col_idx].append(k)
            else: columns.append([k])

        # Merge nearby columns
        columns.sort(key=lambda col: np.mean([k["rot_x"] for k in col]))

        valid_cols = []
        col_x_means = []
        max_row_length_pixel = 0

        for col in columns:
            if len(col) < 3: continue
            row_len = np.linalg.norm(col[0]["center"] - col[-1]["center"])
            if row_len > max_row_length_pixel: max_row_length_pixel = row_len
            col_x_means.append(np.mean([k["rot_x"] for k in col]))
            valid_cols.append(col)

        sorted_cols = [valid_cols[i] for i in np.argsort(col_x_means)]

        final_data = []
        id_counter = 1
        total_kernels_count = 0

        max_row_length = max_row_length_pixel / scale_factor

        for c_idx, col in enumerate(sorted_cols):
            for k in col:
                total_kernels_count += 1

                real_rgb = k["real_rgb"]
                hex_color = "#{:02X}{:02X}{:02X}".format(*real_rgb)

                final_data.append({
                    "ID": id_counter, "Row": c_idx + 1, "Area": int(k["area"]),
                    "Length": round(k["length"], 1), "Width": round(k["width"], 1),
                    "Color": k["color_name"],
                    "RGB": f"{real_rgb[0]}, {real_rgb[1]}, {real_rgb[2]}",
                    "ColorHex": hex_color,
                    "Swatch": "",
                    "_RGB_tuple": real_rgb,
                    "_center": (int(k["center"][0]), int(k["center"][1])),
                    "_contour": k["contour"],
                    "_viz_color": self.row_colors[c_idx % len(self.row_colors)]
                })
                id_counter += 1

        return final_data, total_kernels_count, len(sorted_cols), total_kernels_count/len(sorted_cols) if sorted_cols else 0, max_row_length

class CornPlugin(UniversalPhenoEngine):
    def __init__(self):
        super().__init__(CornStrategy())
