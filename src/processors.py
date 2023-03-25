import pyrealsense2 as rs
import numpy as np
import open3d as o3d
from .utils import logger

class PointCloudProcessor:
    """Handles 3D point clouds and vertices (merges P46, P34, P35)."""
    
    def __init__(self, config):
        self.config = config
        self.pipeline = rs.pipeline()
        self.config_rs = rs.config()
        self.config_rs.enable_stream(rs.stream.depth, *self.config['window_size'], rs.format.z16, 30)
        self.config_rs.enable_stream(rs.stream.color, *self.config['window_size'], rs.format.bgr8, 30)
        self.pipeline.start(self.config_rs)
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
    
    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        return np.asanyarray(depth.get_data()), np.asanyarray(color.get_data()), depth, color
    
    def mask_pointcloud(self, color_frame, depth_frame, mask_2d):
        """Filter PC with 2D mask."""
        points = rs.pointcloud()
        points.map_to(color_frame)
        pc = points.calculate(depth_frame)
        pc_array = np.asanyarray(pc.get_vertices())
        mask_bool = mask_2d.astype(bool)
        filtered = np.where(mask_bool[..., None], pc_array, 0)
        cube_pts = filtered[filtered[:, 2] != 0]  # Z-filter
        if len(cube_pts) == 0:
            return np.array([])
        
        min_xyz = np.min(cube_pts, axis=0)
        max_xyz = np.max(cube_pts, axis=0)
        # Generate all 8 corners using min/max
        verts = np.array([
            [min_xyz[0], min_xyz[1], min_xyz[2]],
            [min_xyz[0], min_xyz[1], max_xyz[2]],
            [min_xyz[0], max_xyz[1], min_xyz[2]],
            [min_xyz[0], max_xyz[1], max_xyz[2]],
            [max_xyz[0], min_xyz[1], min_xyz[2]],
            [max_xyz[0], min_xyz[1], max_xyz[2]],
            [max_xyz[0], max_xyz[1], min_xyz[2]],
            [max_xyz[0], max_xyz[1], max_xyz[2]]
        ])
        center = (min_xyz + max_xyz) / 2
        dims = max_xyz - min_xyz
        logger.info(f"Cube center: {center}, dims: {dims}")
        return verts
    
    def visualize_pc(self, points):
        if len(points) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            self.vis.clear_geometries()
            self.vis.add_geometry(pcd)
            self.vis.poll_events()
            self.vis.update_renderer()
    
    def __del__(self):
        self.pipeline.stop()
        self.vis.destroy_window()