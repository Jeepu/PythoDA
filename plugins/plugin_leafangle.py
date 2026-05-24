import cv2
import numpy as np
from skimage.morphology import skeletonize

from core.universal_engine import UniversalPhenoEngine

class WheatLeafAngleStrategy:
    """Wheat flag leaf angle strategy using DAv2 anchor + SAM foreground voting + skeleton fitting."""

    def get_config(self):
        return {
            'max_infer_dim': 1024,
            'sam_mode': 'amg',
            'sam_params': {
                'points_per_side': 32,  # restored to 32 for better leaf tip recall
                'pred_iou_thresh': 0.86,
                'stability_score_thresh': 0.80,
                'min_mask_region_area': 100
            }
        }

    def generate_prompts(self, infer_image, depth_uint8):
        """Extract coarse plant mask from depth map as SAM anchor."""
        _, binary_mask = cv2.threshold(depth_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

        return binary_mask

    def extract_features(self, raw_image_orig, infer_image, depth_uint8, sam_masks, prompt_data,
                         scale_factor, color_namer, viz_params, progress_callback):
        """SAM foreground voting against depth anchor, then skeletonization and line fitting."""
        binary_mask = prompt_data
        h, w = infer_image.shape[:2]
        orig_h, orig_w = raw_image_orig.shape[:2]
        total_pixels = h * w

        sam_viz_small = np.zeros_like(infer_image)
        refined_sam_mask = np.zeros((h, w), dtype=np.uint8)

        # SAM foreground voting: absorb SAM pieces that overlap the depth anchor
        for ann in sam_masks:
            mask = ann['segmentation']
            area = ann['area']

            if area > total_pixels * 0.6: continue  # ignore giant background chunks

            intersection = np.logical_and(mask, binary_mask > 0)
            overlap_ratio = np.sum(intersection) / area

            # Low threshold (0.25) so thin leaf tips that barely touch the anchor are included
            if overlap_ratio > 0.25:
                refined_sam_mask = cv2.bitwise_or(refined_sam_mask, mask.astype(np.uint8) * 255)
                color = np.random.randint(0, 255, (1, 3)).tolist()[0]
                sam_viz_small[mask] = color

        # Fallback if SAM produced almost nothing
        if np.sum(refined_sam_mask) < total_pixels * 0.01:
            refined_sam_mask = binary_mask.copy()

        # Topological welding: close microscopic gaps between SAM fragments
        weld_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        refined_sam_mask = cv2.morphologyEx(refined_sam_mask, cv2.MORPH_CLOSE, weld_kernel)

        # Skeletonization and branch fitting
        progress_callback.emit(60, "Fitting Lines & Extracting Skeleton...")
        skel_mask = (refined_sam_mask > 0).astype(np.uint8)
        skeleton = skeletonize(skel_mask).astype(np.uint8) * 255

        angle_data = self._analyze_via_fitting(skeleton)

        # Package data for UI rendering
        progress_callback.emit(90, "Restoring Coordinates & Finalizing...")
        annotated_img_rgb = cv2.cvtColor(raw_image_orig, cv2.COLOR_BGR2RGB).copy()
        stats = {}
        analyzed_data = []

        if angle_data:
            j_orig = (int(angle_data['junction'][0] / scale_factor), int(angle_data['junction'][1] / scale_factor))
            s_end_orig = (int(angle_data['stem_line'][1][0] / scale_factor), int(angle_data['stem_line'][1][1] / scale_factor))
            l_end_orig = (int(angle_data['leaf_line'][1][0] / scale_factor), int(angle_data['leaf_line'][1][1] / scale_factor))
            angle = angle_data['angle']

            stats = {"Leaf Angle": f"{angle:.2f}°"}

            analyzed_data.append({
                "ID": 1,
                "Feature": "Leaf Angle",
                "Value": f"{angle:.2f}",
                "Unit": "Degree",
                "_center": j_orig,
                "_angle_data": {
                    "junction": j_orig,
                    "stem_end": s_end_orig,
                    "leaf_end": l_end_orig,
                    "angle_val": angle
                }
            })
        else:
            stats = {"Status": "Failed"}

        best_mask_orig = cv2.resize(refined_sam_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        return {
            'analyzed_data': analyzed_data,
            'stats': stats,
            'depth_mask_small': binary_mask,
            'sam_viz_small': sam_viz_small,
            'final_mask_orig': best_mask_orig,
            'annotated_img_rgb': annotated_img_rgb
        }

    def _analyze_via_fitting(self, skeleton):
        """Detect skeleton junction, separate stem and leaf branches, fit lines, compute angle."""
        h, w = skeleton.shape
        kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
        neighbors = cv2.filter2D((skeleton>0).astype(np.uint8), -1, kernel)
        junctions = np.argwhere(neighbors >= 13)

        if len(junctions) == 0: return None

        junctions = [tuple(p[::-1]) for p in junctions]
        junctions.sort(key=lambda p: p[1], reverse=True)
        main_j = junctions[0]

        skel_temp = skeleton.copy()
        cv2.circle(skel_temp, main_j, 5, 0, -1)

        num, labels, stats, centroids = cv2.connectedComponentsWithStats(skel_temp, connectivity=8)
        if num < 2: return None

        branches = []
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] < 20: continue
            pts = np.argwhere(labels == i)
            pts = [p[::-1] for p in pts]
            centroid = centroids[i]
            dy = centroid[1] - main_j[1]
            dx = centroid[0] - main_j[0]
            branches.append({'pts': np.array(pts, dtype=np.int32), 'dy': dy, 'dx': dx, 'mean_y': centroid[1]})

        if len(branches) < 2: return None

        branches.sort(key=lambda b: b['mean_y'])
        stem_branch = branches[0]
        remaining = branches[1:]
        if not remaining: return None
        leaf_branch = max(remaining, key=lambda b: abs(b['dx']))

        line_s = cv2.fitLine(stem_branch['pts'], cv2.DIST_L2, 0, 0.01, 0.01)
        vx_s, vy_s, x_s, y_s = line_s.flatten()
        if vy_s > 0: vx_s, vy_s = -vx_s, -vy_s

        line_l = cv2.fitLine(leaf_branch['pts'], cv2.DIST_L2, 0, 0.01, 0.01)
        vx_l, vy_l, x_l, y_l = line_l.flatten()
        if (leaf_branch['dx'] > 0 and vx_l < 0) or (leaf_branch['dx'] < 0 and vx_l > 0):
            vx_l, vy_l = -vx_l, -vy_l

        dot = vx_s * vx_l + vy_s * vy_l
        angle_rad = np.arccos(np.clip(dot, -1.0, 1.0))
        angle_deg = float(np.degrees(angle_rad))

        len_line = 150
        p_stem_end = (int(main_j[0] + vx_s*len_line), int(main_j[1] + vy_s*len_line))
        p_leaf_end = (int(main_j[0] + vx_l*len_line), int(main_j[1] + vy_l*len_line))

        return {
            'junction': main_j,
            'stem_line': (main_j, p_stem_end),
            'leaf_line': (main_j, p_leaf_end),
            'angle': angle_deg
        }

class WheatLeafAnglePlugin(UniversalPhenoEngine):
    def __init__(self):
        super().__init__(WheatLeafAngleStrategy())
