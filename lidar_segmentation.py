import os
import numpy as np
import cv2
import open3d as o3d
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import matplotlib.cm as cm

class LidarSegmentation:
    def __init__(self):
        # LiDAR specifications for KITTI 
        self.vertical_fov = (-24.9, 2.0)  
        self.horizontal_fov = (-180, 180)  
        self.vertical_resolution = 64  
        self.horizontal_resolution = 2048 
        
        # Ground segmentation parameters
        self.ground_threshold = 0.3  # Height threshold for ground points (meters)
        self.ground_angle_threshold = 10  # Angle threshold (degrees)
        
        # Clustering parameters
        self.clustering_eps = 0.5  
        self.clustering_min_samples = 10 
        self.min_cluster_points = 50  
        self.min_laser_lines = 3  
        
        # Colors for visualization
        self.ground_color = [0.5, 0.5, 0.5] 
        self.noise_color = [0, 0, 0] 
        
    def cartesian_to_spherical(self, points):
        """Convert 3D cartesian coordinates to spherical coordinates"""
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        
        # Calculate range (distance from origin)
        range_vals = np.sqrt(x**2 + y**2 + z**2)
        
        # Calculate azimuth angle (horizontal angle)
        azimuth = np.arctan2(y, x) * 180 / np.pi
        
        # Calculate elevation angle (vertical angle)
        elevation = np.arcsin(z / (range_vals + 1e-8)) * 180 / np.pi
        
        return range_vals, azimuth, elevation
    
    def project_to_range_image(self, points):
        """Project 3D point cloud to 2D range image"""
        range_vals, azimuth, elevation = self.cartesian_to_spherical(points)
        
        # Normalize angles to image coordinates
        # Horizontal: map [-180, 180] to [0, horizontal_resolution-1]
        u = ((azimuth - self.horizontal_fov[0]) / 
             (self.horizontal_fov[1] - self.horizontal_fov[0]) * 
             (self.horizontal_resolution - 1)).astype(int)
        
        # Vertical: map [min_elevation, max_elevation] to [0, vertical_resolution-1]
        v = ((elevation - self.vertical_fov[0]) / 
             (self.vertical_fov[1] - self.vertical_fov[0]) * 
             (self.vertical_resolution - 1)).astype(int)
        
        # Clip to valid range
        u = np.clip(u, 0, self.horizontal_resolution - 1)
        v = np.clip(v, 0, self.vertical_resolution - 1)
        
        # Create range image
        range_image = np.zeros((self.vertical_resolution, self.horizontal_resolution))
        point_indices = np.zeros((self.vertical_resolution, self.horizontal_resolution), dtype=int) - 1
        
        # Fill range image (keep closest point for each pixel)
        for i, (u_i, v_i, r_i) in enumerate(zip(u, v, range_vals)):
            if range_image[v_i, u_i] == 0 or r_i < range_image[v_i, u_i]:
                range_image[v_i, u_i] = r_i
                point_indices[v_i, u_i] = i
        
        return range_image, point_indices, u, v
    
    def segment_ground_fast(self, points, range_image, point_indices, u, v):
        """Fast ground plane segmentation using range image"""
        ground_mask = np.zeros(len(points), dtype=bool)
        
        # Process each column (vertical slice) of the range image
        for col in range(self.horizontal_resolution):
            # Get points in this column from bottom to top
            column_points = []
            column_indices = []
            
            for row in range(self.vertical_resolution):
                if point_indices[row, col] >= 0:
                    idx = point_indices[row, col]
                    column_points.append(points[idx])
                    column_indices.append(idx)
            
            if len(column_points) < 2:
                continue
            
            column_points = np.array(column_points)
            column_indices = np.array(column_indices)
            
            # Start from bottom and check for ground plane continuity
            prev_height = column_points[0, 2]  # z-coordinate is height
            ground_indices = [column_indices[0]]  # First point is likely ground
            
            for i in range(1, len(column_points)):
                current_height = column_points[i, 2]
                height_diff = abs(current_height - prev_height)
                
                # If height change is small, consider it ground
                if height_diff < self.ground_threshold:
                    ground_indices.append(column_indices[i])
                    prev_height = current_height
                else:
                    break  # Stop when we hit a significant height change
            
            # Mark these points as ground
            if ground_indices:
                ground_mask[ground_indices] = True
        
        return ground_mask
    
    def cluster_non_ground_points(self, points, ground_mask, u, v):
        """Cluster non-ground points using image-based clustering"""
        non_ground_indices = np.where(~ground_mask)[0]
        
        if len(non_ground_indices) == 0:
            return np.array([]), np.array([])
        
        non_ground_points = points[non_ground_indices]
        
        # Use DBSCAN clustering in 3D space
        clustering = DBSCAN(eps=self.clustering_eps, 
                           min_samples=self.clustering_min_samples)
        cluster_labels = clustering.fit_predict(non_ground_points[:, :3])
        
        # Map cluster labels back to original point indices
        full_labels = np.full(len(points), -1, dtype=int)
        full_labels[non_ground_indices] = cluster_labels
        
        return cluster_labels, non_ground_indices
    
    def filter_clusters(self, points, cluster_labels, non_ground_indices, u, v):
        """Filter clusters based on size and laser line span"""
        unique_labels = np.unique(cluster_labels)
        valid_clusters = []
        
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
            
            # Get points in this cluster
            cluster_mask = cluster_labels == label
            cluster_indices = non_ground_indices[cluster_mask]
            cluster_points = points[cluster_indices]
            
            # Check cluster size
            if len(cluster_points) < self.min_cluster_points:
                continue
            
            # Check laser line span
            cluster_u = u[cluster_indices]
            cluster_v = v[cluster_indices]
            unique_laser_lines = len(np.unique(cluster_v))
            
            if unique_laser_lines < self.min_laser_lines:
                continue
            
            valid_clusters.append(label)
        
        return valid_clusters
    
    def assign_cluster_labels(self, points, cluster_labels, non_ground_indices, valid_clusters):
        """Assign unique labels to valid clusters"""
        final_labels = np.full(len(points), -1, dtype=int)  # -1 for ground/noise
        
        label_counter = 0
        for old_label in valid_clusters:
            cluster_mask = cluster_labels == old_label
            cluster_indices = non_ground_indices[cluster_mask]
            final_labels[cluster_indices] = label_counter
            label_counter += 1
        
        return final_labels
    
    def process_point_cloud(self, points):
        """Main processing function for point cloud segmentation"""
        # Step 1: Project to range image
        range_image, point_indices, u, v = self.project_to_range_image(points)
        
        # Step 2: Ground segmentation
        ground_mask = self.segment_ground_fast(points, range_image, point_indices, u, v)
        
        # Step 3: Cluster non-ground points
        cluster_labels, non_ground_indices = self.cluster_non_ground_points(points, ground_mask, u, v)
        
        # Step 4: Filter clusters
        valid_clusters = self.filter_clusters(points, cluster_labels, non_ground_indices, u, v)
        
        # Step 5: Assign final labels
        final_labels = self.assign_cluster_labels(points, cluster_labels, non_ground_indices, valid_clusters)
        
        return {
            'range_image': range_image,
            'ground_mask': ground_mask,
            'cluster_labels': final_labels,
            'num_clusters': len(valid_clusters),
            'point_indices': point_indices
        }
    
    def visualize_results(self, points, results):
        """Visualize segmentation results"""
        range_image = results['range_image']
        ground_mask = results['ground_mask']
        cluster_labels = results['cluster_labels']
        num_clusters = results['num_clusters']
        
        # Create point cloud for Open3D visualization
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        
        # Color points based on segmentation
        colors = np.zeros((len(points), 3))
        
        # Color ground points
        colors[ground_mask] = self.ground_color
        
        # Color clusters with different colors
        if num_clusters > 0:
            try:
                # Use new matplotlib API (3.7+)
                colormap = plt.get_cmap('tab20')
            except AttributeError:
                # Fallback for older matplotlib versions
                colormap = cm.get_cmap('tab20')
            
            for i in range(num_clusters):
                cluster_mask = cluster_labels == i
                if np.any(cluster_mask):
                    color = colormap(i % 20)[:3]  # Use tab20 colormap
                    colors[cluster_mask] = color
        
        # Color noise points (unclustered non-ground)
        noise_mask = (cluster_labels == -1) & (~ground_mask)
        colors[noise_mask] = self.noise_color
        
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Visualize range image
        plt.figure(figsize=(12, 4))
        plt.imshow(range_image, cmap='viridis', aspect='auto')
        plt.title('Range Image')
        plt.xlabel('Horizontal Angle')
        plt.ylabel('Laser Ring')
        plt.colorbar(label='Range (meters)')
        plt.tight_layout()
        plt.show()
        
        return pcd

def main():
    # Path to KITTI dataset
    kitti_path = 'Kitti'
    lidar_dir = os.path.join(kitti_path, 'velodyne_points', 'data')

    # Get list of LiDAR files
    lidar_files = sorted([f for f in os.listdir(lidar_dir) if f.endswith('.bin')])

    # Initialize segmentation processor
    segmenter = LidarSegmentation()
    
    # Initialize Open3D visualizer
    vis = o3d.visualization.Visualizer()

    vis.create_window(window_name="KITTI LiDAR Segmentation")
    
    for i, lidar_file in enumerate(lidar_files):
        lidar_path = os.path.join(lidar_dir, lidar_file)
        
        # Load point cloud
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        
        print(f"\n------ Processing frame {i}: {lidar_file} -------")
        
        # Process point cloud
        results = segmenter.process_point_cloud(points)
        
        # Visualize results
        pcd = segmenter.visualize_results(points, results)
        
        # Update Open3D visualization
        if i == 0:
            vis.add_geometry(pcd)
        else:
            vis.update_geometry(pcd)
        
        vis.poll_events()
        vis.update_renderer()
        
        # Wait for user input
        print("Press Enter to continue to next frame, or 'q' to quit...")
        user_input = input()
        if user_input.lower() == 'q':
            break
    
    vis.destroy_window()

if __name__ == "__main__":
    main()