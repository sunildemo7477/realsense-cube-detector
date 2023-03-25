import cv2
import numpy as np
from .utils import logger

def draw_vertices(image, vertices, color=(0, 255, 0)):
    """Draw 2D/3D projected verts (from P20, P21)."""
    if len(vertices) > 0:
        if vertices.shape[1] == 2:  # 2D
            pts = vertices.astype(np.int32)
            cv2.polylines(image, [pts], True, color, 2)
            for pt in pts:
                cv2.circle(image, tuple(pt), 5, color, -1)
        else:  # 3D - project first (simplified)
            logger.warning("3D verts need deprojection for full draw.")
    return image

def show_frames(color_img, depth_img=None, mask=None, title='Detection'):
    """Side-by-side viz (from P13/P15)."""
    if mask is not None:
        vis = np.hstack([color_img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
    else:
        vis = color_img
    if depth_img is not None:
        depth_colored = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)
        vis = np.hstack([vis, depth_colored])
    cv2.imshow(title, vis)