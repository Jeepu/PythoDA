import cv2
import numpy as np
from scipy.spatial.distance import pdist, squareform

from core.universal_engine import UniversalPhenoEngine

class LeafStrategy:
    """Leaf phenotyping strategy with contour-isolation and convex-hull major/minor axis computation."""

    def get_config(self):
        return {
            'max_infer_dim': 1024,
            'sam_mode': 'amg',
            'sam_params': {
                'points_per_side': 32,
                'pred_iou_thresh': 0.82,
                'stability_score_thresh': 0.88,
                'crop_n_layers': 0,
                'min_mask_region_area': 500
            }
        }

    def generate_prompts(self, infer_image, depth_uint8):
        """Extract coarse plant foreground via Otsu threshold on depth map."""
        _, dav2_mask = cv2.threshold(depth_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dav2_mask = cv2.morphologyEx(dav2_mask, cv2.MORPH_CLOSE, kernel_close)
        return dav2_mask

    def extract_features(self, raw_image_orig, infer_image, depth_uint8, sam_masks, prompt_data,
                         scale_factor, color_namer, viz_params, progress_callback):
        """Multi-stage filtering (area, stem removal, depth) then NMS and axis fitting."""
        dav2_mask = prompt_data
        orig_h, orig_w = raw_image_orig.shape[:2]
        h, w = infer_image.shape[:2]
        total_pixels_small = h * w

        progress_callback.emit(60, "Filtering & Applying NMS...")

        candidates = []
        sam_viz_small = np.zeros_like(infer_image)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        dav2_mask_dilated = cv2.morphologyEx(dav2_mask, cv2.MORPH_DILATE, kernel_dilate)

        # Stage 1: multi-condition filtering (area, stem removal, depth overlap)
        for ann in sam_masks:
            raw_sam_mask = ann['segmentation']
            
            mask_uint8_raw = raw_sam_mask.astype(np.uint8) * 255
            isolated_contours, _ = cv2.findContours(mask_uint8_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in isolated_contours:
                area = cv2.contourArea(cnt)
                
                if area > total_pixels_small * 0.85: continue
                if area < total_pixels_small * 0.005: continue

                single_mask_uint8 = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(single_mask_uint8, [cnt], -1, 255, -1)
                mask = single_mask_uint8 > 0  
                
                # Morphological opening to detect and reject thin stems
                mask_uint8_check = mask.astype(np.uint8)
                check_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                opened_mask = cv2.morphologyEx(mask_uint8_check, cv2.MORPH_OPEN, check_kernel)
                if np.sum(opened_mask) < area * 0.45: continue

                # Depth foreground verification
                intersection = np.logical_and(mask, dav2_mask_dilated > 0)
                if np.sum(intersection) / area < 0.5: continue

                candidates.append({'mask': mask, 'area': area, 'score': area})
                color = np.random.randint(0, 255, (1, 3)).tolist()[0]
                sam_viz_small[mask] = color

        # Stage 2: non-maximum suppression
        candidates.sort(key=lambda x: x['score'], reverse=True)
        final_masks_small = []
        occupied_mask = np.zeros((h, w), dtype=bool)
        for cand in candidates:
            mask = cand['mask']
            intersection = np.logical_and(mask, occupied_mask)
            if np.sum(intersection) / np.sum(mask) > 0.2: continue
            final_masks_small.append(mask)
            occupied_mask = np.logical_or(occupied_mask, mask)

        # Stage 3: major/minor axis computation from convex hull
        progress_callback.emit(80, "Analyzing Leaf Phenotypes (Geometry)...")
        analyzed_data = []

        annotated_img_rgb = cv2.cvtColor(raw_image_orig, cv2.COLOR_BGR2RGB).copy()
        total_leaf_mask_small = np.zeros((h, w), dtype=np.uint8)

        leaf_id = 1
        for mask in final_masks_small:
            mask_uint8 = mask.astype(np.uint8) * 255
            total_leaf_mask_small = cv2.bitwise_or(total_leaf_mask_small, mask_uint8)
            area_small = np.sum(mask)

            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: continue
            cnt_small = max(contours, key=cv2.contourArea)

            cnt_orig = (cnt_small / scale_factor).astype(np.int32)
            real_area = area_small / (scale_factor * scale_factor)

            perimeter = cv2.arcLength(cnt_orig, True)

            M_orig = cv2.moments(cnt_orig)
            if M_orig["m00"] != 0:
                cx, cy = int(M_orig["m10"] / M_orig["m00"]), int(M_orig["m01"] / M_orig["m00"])
            else:
                rx, ry, rw, rh = cv2.boundingRect(cnt_orig)
                cx, cy = rx + rw//2, ry + rh//2

            hull = cv2.convexHull(cnt_orig)
            hull_pts = hull[:, 0, :]
            if len(hull_pts) < 2: continue

            dists = pdist(hull_pts, metric='euclidean')
            dists_mat = squareform(dists)
            max_idx_flat = np.argmax(dists_mat)
            i, j = np.unravel_index(max_idx_flat, dists_mat.shape)

            p1, p2 = hull_pts[i], hull_pts[j]
            max_len = dists_mat[i, j]

            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            angle = np.degrees(np.arctan2(dy, dx))
            center = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            cnt_pts_float = cnt_orig.astype(np.float32)
            rotated_pts = cv2.transform(cnt_pts_float, M)
            y_coords = rotated_pts[:, 0, 1]
            min_y, max_y = np.min(y_coords), np.max(y_coords)
            max_wid = max_y - min_y

            rot_center_x = (rotated_pts[:, 0, 0].max() + rotated_pts[:, 0, 0].min()) / 2
            p_wid_top_rot = np.array([[[rot_center_x, min_y]]], dtype=np.float32)
            p_wid_bot_rot = np.array([[[rot_center_x, max_y]]], dtype=np.float32)
            M_inv = cv2.invertAffineTransform(M)
            p_wid_top = cv2.transform(p_wid_top_rot, M_inv)[0][0]
            p_wid_bot = cv2.transform(p_wid_bot_rot, M_inv)[0][0]

            aspect_ratio = max_len / max_wid if max_wid > 0 else 0
            
            circularity = (4 * np.pi * real_area) / (perimeter ** 2) if perimeter > 0 else 0

            mean_rgb_bgr = infer_image[mask].mean(axis=0)
            real_rgb = (int(mean_rgb_bgr[2]), int(mean_rgb_bgr[1]), int(mean_rgb_bgr[0]))
            hex_color = "#{:02X}{:02X}{:02X}".format(*real_rgb)
            color_name = color_namer.get_name(real_rgb)

            analyzed_data.append({
                "ID": leaf_id,
                "Length": round(max_len, 1),
                "Width": round(max_wid, 1),
                "L/W Ratio": round(aspect_ratio, 2),
                "Perimeter": round(perimeter, 1),        
                "Circularity": round(circularity, 2),    
                "Area": int(real_area),
                "Color": color_name,
                "RGB": f"{real_rgb[0]}, {real_rgb[1]}, {real_rgb[2]}",
                "ColorHex": hex_color,
                "Swatch": "",
                "_RGB_tuple": real_rgb,
                "_contour": cnt_orig,
                "_center": (int(cx), int(cy)),
                "_major_axis": ((int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))),
                "_minor_axis": ((int(p_wid_top[0]), int(p_wid_top[1])), (int(p_wid_bot[0]), int(p_wid_bot[1]))),
                "_viz_color": (255, 255, 255)  
            })

            cv2.drawContours(annotated_img_rgb, [cnt_orig], -1, (255, 255, 255), viz_params['viz_thickness'])
            cv2.line(annotated_img_rgb, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), 
                     (255, 0, 0), viz_params['viz_thickness'])
            cv2.line(annotated_img_rgb, (int(p_wid_top[0]), int(p_wid_top[1])), (int(p_wid_bot[0]), int(p_wid_bot[1])), 
                     (0, 0, 255), viz_params['viz_thickness'])
            cv2.putText(annotated_img_rgb, str(leaf_id), (int(cx), int(cy)), 
                        cv2.FONT_HERSHEY_SIMPLEX, viz_params['viz_font_scale'], (0, 0, 0), viz_params['viz_thickness'] + 2, cv2.LINE_AA)
            cv2.putText(annotated_img_rgb, str(leaf_id), (int(cx), int(cy)), 
                        cv2.FONT_HERSHEY_SIMPLEX, viz_params['viz_font_scale'], (255, 255, 255), viz_params['viz_thickness'], cv2.LINE_AA)
            leaf_id += 1

        stats = {
            "Total Leaves": len(analyzed_data),
            "Avg Area": int(np.mean([d["Area"] for d in analyzed_data])) if analyzed_data else 0,
            "Avg L/W": round(np.mean([d["L/W Ratio"] for d in analyzed_data]), 2) if analyzed_data else 0
        }

        final_mask_orig = cv2.resize(total_leaf_mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        return {
            'analyzed_data': analyzed_data,
            'stats': stats,
            'depth_mask_small': dav2_mask,
            'sam_viz_small': sam_viz_small,
            'final_mask_orig': final_mask_orig,
            'annotated_img_rgb': annotated_img_rgb
        }

class LeafPlugin(UniversalPhenoEngine):
    def __init__(self):
        super().__init__(LeafStrategy())