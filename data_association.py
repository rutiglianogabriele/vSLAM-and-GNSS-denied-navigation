import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from kitti_utils import *
import json 
from itertools import combinations
    

# Calibration matrices for KITTI dataset.
R_02 = np.array([
    [9.999758e-01, -5.267463e-03, -4.552439e-03],
    [5.251945e-03, 9.999804e-01, -3.413835e-03],
    [4.570332e-03, 3.389843e-03, 9.999838e-01]
]) # Rotation Matrix for color camera

T_02 = np.array([
    [5.956621e-02, 2.900141e-04, 2.577209e-03]
]) # Translation Matrix for color camera

Tr_02 = np.vstack((np.hstack((R_02, T_02.T)), np.array([0, 0, 0, 1]))) # Transformation Matrix for color camera

# Camera 02 calibration matrices for KITTI dataset
P_rect_02 = np.array([
    [7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],
    [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
    [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]
])

# Rectification matrix
R_rect_00 = np.array([
    [9.999239e-01, 9.837760e-03, -7.445048e-03],
    [-9.869795e-03, 9.999421e-01, -4.278459e-03],
    [7.402527e-03, 4.351614e-03, 9.999631e-01]
])

R_rect_02 = np.array([
    [9.998817e-01, 1.511453e-02, -2.841595e-03],
    [-1.511724e-02, 9.998853e-01, -9.338510e-04],
    [2.827154e-03, 9.766976e-04, 9.999955e-01]
])

# Lidar to Camera calibration matrices
Tr_velo_to_cam_00 = np.array([
    [7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
    [1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02],
    [9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01],
    [0, 0, 0, 1],
])

def select_foreground_points_by_depth(points_3d, labels):
    """
    Selects the foreground points based on minimum average depth.
    """
    # Get unique cluster labels, excluding noise (-1)
    unique_labels = np.unique(labels[labels != -1])
    
    if len(unique_labels) == 0:
        # No clusters, return all non-noise points
        return np.where(labels != -1)[0]
    
    min_avg_depth = float('inf')
    foreground_label = -1
    
    # Find the cluster with minimum average depth (Z coordinate)
    for label in unique_labels:
        cluster_mask = labels == label
        cluster_points = points_3d[cluster_mask]
        avg_depth = np.mean(cluster_points[:, 2])
        
        if avg_depth < min_avg_depth:
            min_avg_depth = avg_depth
            foreground_label = label
    
    return np.where(labels == foreground_label)[0]

def fit_plane_to_three_points(points):
    """
    Fits a plane to points by selecting three points that form a triangle with maximum area.
    """
    max_area = -1
    best_triangle = None
    
    # If we have too many points, randomly sample to avoid excessive computation
    if points.shape[0] > 20:
        indices = np.random.choice(points.shape[0], min(20, points.shape[0]), replace=False)
        sample_points = points[indices]
    else:
        sample_points = points
    
    # Find three points that form triangle with maximum area
    for triangle_points in combinations(range(len(sample_points)), 3):
        p1 = sample_points[triangle_points[0]]
        p2 = sample_points[triangle_points[1]]
        p3 = sample_points[triangle_points[2]]
        
        # Calculate triangle area
        area = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
        
        if area > max_area:
            max_area = area
            best_triangle = (p1, p2, p3)
    
    if best_triangle is None or max_area < 1e-6:  # Minimum area threshold
        return None
    
    p1, p2, p3 = best_triangle
    
    # Calculate plane normal
    normal = np.cross(p2 - p1, p3 - p1)
    a, b, c = normal
    
    # Calculate d using point p1
    d = -np.dot(normal, p1)
    
    return (a, b, c, d)

def estimate_depth_from_plane(plane_params, u_norm, v_norm):
    """
    Estimates depth for a normalized image point given a plane.
    """
    if plane_params is None:
        return None
    
    a, b, c, d = plane_params
    
    # For a normalized point (u_norm, v_norm, 1) with depth Z,
    # the 3D point is (u_norm*Z, v_norm*Z, Z)
    # Substituting into plane equation: a*u_norm*Z + b*v_norm*Z + c*Z + d = 0
    # Solving for Z: Z = -d / (a*u_norm + b*v_norm + c)
    
    denominator = a * u_norm + b * v_norm + c
    
    if np.abs(denominator) < 1e-6:
        return None
    
    depth = -d / denominator
    
    return depth

def validate_depth_estimate(plane_params, u_norm, v_norm, angle_threshold_deg=85):
    """
    Validates a depth estimate based on geometric constraints.
    """
    a, b, c, d = plane_params
    
    # Check angle between plane normal and viewing ray
    normal = np.array([a, b, c])
    viewing_ray = np.array([u_norm, v_norm, 1])
    
    normal_norm = np.linalg.norm(normal)
    if normal_norm == 0:
        return False
    
    # Calculate angle
    cos_angle = np.abs(np.dot(normal, viewing_ray)) / (normal_norm * np.linalg.norm(viewing_ray))
    angle_deg = np.arccos(np.clip(cos_angle, 0.0, 1.0)) * 180 / np.pi
    
    if angle_deg > angle_threshold_deg:
        return False
    
    return True

def get_neighborhood_points(u_feat, v_feat, u_lidar, v_lidar, points_3d, labels, 
                           pixel_radius=20, min_points=3):
    """
    Gets neighborhood points around a visual feature for plane fitting.
    """
    # Find LiDAR points near this visual feature
    pixel_distances = np.sqrt((u_lidar - u_feat)**2 + (v_lidar - v_feat)**2)
    nearby_mask = pixel_distances < pixel_radius
    
    if np.sum(nearby_mask) < min_points:
        return np.array([]).reshape(0, 3), np.array([])
    
    neighborhood_points = points_3d[nearby_mask]
    neighborhood_labels = labels[nearby_mask]
    
    # Filter out noise points
    valid_mask = neighborhood_labels != -1
    if np.sum(valid_mask) < min_points:
        return np.array([]).reshape(0, 3), np.array([])
    
    return neighborhood_points[valid_mask], neighborhood_labels[valid_mask]

def rectify_point_cloud(point_cloud):
    """
    Rectifies the point cloud to the camera coordinate system.
    """
    # Transform to homogeneous coordinates
    point_cloud_hom = np.hstack((point_cloud[:, :3], np.ones((len(point_cloud), 1))))
    print(f"1. Homogeneous points shape: {point_cloud_hom.shape}")
    
    # Apply rectification
    R_rect_homo = np.eye(4)
    R_rect_homo[:3, :3] = R_rect_00

    # Apply the entire transformation sequence (4x4 matrix)
    rectified_coords = ( R_rect_homo @ Tr_velo_to_cam_00 @ point_cloud_hom.T).T
    print(f"2. Rectified coords shape: {rectified_coords.shape}")
    
    # Extract 3D points
    rectified_points_3d = rectified_coords[:, :3]
    print(f"3. 3D points shape: {rectified_points_3d.shape}")
    
    return rectified_points_3d

def pointcloud_to_uvx(lidar_path, image):
    # Apply rectification to R
    R_rect_homo = np.eye(4)
    R_rect_homo[:3, :3] = R_rect_00

    # Combined transformation matrix from Lidar to Camera
    Tr_velo_to_cam2 = P_rect_02 @ R_rect_homo @ Tr_02 @ Tr_velo_to_cam_00
    Tr_velo_to_cam2 = Tr_velo_to_cam2[0:3,:]

    # get LiDAR points and transform them to image/camera space
    velo_uvz = project_velobin2uvz(lidar_path, Tr_velo_to_cam2, image, remove_plane=False)

    return velo_uvz

def associate_and_get_3d_visual_features(visual_features_2d, visual_feature_ids, labeled_point_cloud, rectified_points_3d):
    """
    Associates 2D visual features with 3D LiDAR points and estimates depth.
    Maintains original visual feature IDs.
    """
    if visual_features_2d is None or len(visual_features_2d) == 0:
        return []
    
    # Ensure visual_features_2d has the right shape
    if len(visual_features_2d.shape) == 3:
        visual_features_2d = visual_features_2d.reshape(-1, 2)
    
    # Filter points in front of camera
    points_in_front_mask = rectified_points_3d[:, 2] > 0.5
    valid_3d_points = rectified_points_3d[points_in_front_mask]
    valid_labels = labeled_point_cloud['cluster_labels'][points_in_front_mask]
    
    if len(valid_3d_points) == 0:
        return []
    
    # Extract camera parameters from P_rect_02
    fx = P_rect_02[0, 0]  # 721.5377
    fy = P_rect_02[1, 1]  # 721.5377
    cx = P_rect_02[0, 2]  # 609.5593
    cy = P_rect_02[1, 2]  # 172.8540
    
    # Project to image coordinates
    u_lidar = valid_3d_points[:, 0] * fx / valid_3d_points[:, 2] + cx
    v_lidar = valid_3d_points[:, 1] * fy / valid_3d_points[:, 2] + cy
    
    # For each visual feature, estimate depth using plane fitting
    visual_features_3d = []
    
    for i, (u_feat, v_feat) in enumerate(visual_features_2d):
        # Get ID for this feature
        feature_id = visual_feature_ids[i] if i < len(visual_feature_ids) else -1
        
        # Get neighborhood points
        neighborhood_points, neighborhood_labels = get_neighborhood_points(
            u_feat, v_feat, u_lidar, v_lidar, valid_3d_points, valid_labels,
            pixel_radius= 25
        )
        
        if len(neighborhood_points) < 3:
            # Fall back to simple method if not enough points
            pixel_distances = np.sqrt((u_lidar - u_feat)**2 + (v_lidar - v_feat)**2)
            nearby_mask = pixel_distances < 20
            
            if np.any(nearby_mask):
                nearby_points = valid_3d_points[nearby_mask]
                nearby_labels = valid_labels[nearby_mask]
                
                # Filter out noise and ground
                valid_mask = (nearby_labels != -1) & (nearby_labels > 0)
                if np.any(valid_mask):
                    nearby_points = nearby_points[valid_mask]
                    if len(nearby_points) > 0:
                        # Use median depth
                        depths = nearby_points[:, 2]
                        median_idx = np.argsort(depths)[len(depths)//2]
                        depth = depths[median_idx]
                        
                        if 0.5 < depth < 100:
                            X = (u_feat - cx) * depth / fx
                            Y = (v_feat - cy) * depth / fy
                            Z = depth
                            
                            visual_features_3d.append({'id': feature_id, 'point3d': np.array([X, Y, Z])
                            })
            continue
        
        # Select foreground points
        foreground_indices = select_foreground_points_by_depth( neighborhood_points, neighborhood_labels )
        
        if len(foreground_indices) < 3:
            continue
        
        foreground_points = neighborhood_points[foreground_indices]
        
        # Fit plane to foreground points using robust method
        plane_params = fit_plane_to_three_points(foreground_points)
        
        if plane_params is None:
            continue
        
        # Calculate normalized coordinates
        u_norm = (u_feat - cx) / fx
        v_norm = (v_feat - cy) / fy
        
        # Estimate depth from plane
        depth = estimate_depth_from_plane(plane_params, u_norm, v_norm)
        
        # Validate depth estimate
        if validate_depth_estimate(plane_params, u_norm, v_norm):
            # Back-project to 3D
            X = u_norm * depth
            Y = v_norm * depth
            Z = depth
            
            visual_features_3d.append({'id': feature_id, 'point3d': np.array([X, Y, Z])})
    
    return visual_features_3d
  
def visualize_3d_points(points_3d, title="3D Visual Features"):
    """
    Visualizes 3D points using matplotlib.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='r', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.show()

def perform_association(image_path, lidar_path, save_path=None, frame_id=0):
    # Load data
    image = cv2.imread(image_path)
    point_cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    print(f"Loaded image: {image.shape}")
    print(f"Loaded point cloud: {point_cloud.shape}")
    
    # Get visual features
    from visual_features import VisualFeatureTracker
    tracker = VisualFeatureTracker()
    visual_features_2d, visual_ids = tracker.process_frame(image)
    
    if visual_features_2d is not None:
        visual_features_2d = visual_features_2d.reshape(-1, 2)
    
    # Get LiDAR segmentation
    from lidar_segmentation import LidarSegmentation
    segmenter = LidarSegmentation()
    labeled_point_cloud = segmenter.process_point_cloud(point_cloud)

    # Rectify point cloud
    rectified_points_3d = rectify_point_cloud(point_cloud)
    
    # Visual-3D association
    visual_3d_list = associate_and_get_3d_visual_features(visual_features_2d, visual_ids, labeled_point_cloud, rectified_points_3d)
    
    print(f"Associated {len(visual_3d_list)}/{len(visual_features_2d)} features")
    if len(visual_3d_list) > 0:
        # For printing depth range, convert to numpy array temporarily
        temp_visual_3d_array = np.array([item['point3d'] for item in visual_3d_list])
        print(f"3D features depth range: [{temp_visual_3d_array[:, 2].min():.2f}, {temp_visual_3d_array[:, 2].max():.2f}] meters")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image with 2D features
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].scatter(visual_features_2d[:, 0], visual_features_2d[:, 1],
                   c='red', s=10, alpha=0.7)
    axes[0].set_title(f'2D Features ({len(visual_features_2d)})')
    axes[0].axis('off')
    
    # Projected LiDAR points to uvz
    velo_uvz = pointcloud_to_uvx(lidar_path, image)

    velo_image = draw_velo_on_image(velo_uvz, image.copy())
    
    axes[1].imshow(cv2.cvtColor(velo_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Projected LiDAR')
    axes[1].axis('off')
    
    # Associated 3D features projected back
    axes[2].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[2].scatter(visual_features_2d[:, 0], visual_features_2d[:, 1],
                   c='red', s=20, alpha=0.5, label='2D')
    
    # Extract camera parameters from P_rect_02
    fx = P_rect_02[0, 0]  # 721.5377
    fy = P_rect_02[1, 1]  # 721.5377
    cx = P_rect_02[0, 2]  # 609.5593
    cy = P_rect_02[1, 2]  # 172.8540
    
    if len(visual_3d_list) > 0:
        # Project 3D points from visual_3d_list for visualization
        visual_3d_array_for_projection = np.array([item['point3d'] for item in visual_3d_list])
        u_3d = visual_3d_array_for_projection[:, 0] * fx / visual_3d_array_for_projection[:, 2] + cx
        v_3d = visual_3d_array_for_projection[:, 1] * fy / visual_3d_array_for_projection[:, 2] + cy
        axes[2].scatter(u_3d, v_3d, c='green', s=10, alpha=0.7, label='3D projected')
    
    axes[2].set_title(f'Associated ({len(visual_3d_list)}/{len(visual_features_2d)})')
    axes[2].axis('off')
    
    plt.tight_layout()
    if save_path:
        # Ensure the directory exists
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'frame_{frame_id:06d}_association.png'))
        plt.close(fig) # Close the figure to free memory
    else:
        plt.show(block=False) # Display the plot without blocking
        plt.pause(0.1) # Pause for 0.1 seconds to show the plot
        plt.clf() # Clear the current figure
        plt.close() # Close the figure window
    
    return visual_3d_list

def main():
    """Process entire dataset and save 3D visual features."""
    kitti_path = 'Kitti'
    image_dir = os.path.join(kitti_path, 'image_02', 'data')
    lidar_dir = os.path.join(kitti_path, 'velodyne_points', 'data')
    output_dir = 'output/visual_3d_features' # Output directory for 3D features
    
    os.makedirs(output_dir, exist_ok=True) # Create output directory if it doesn't exist
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    lidar_files = sorted([f for f in os.listdir(lidar_dir) if f.endswith('.bin')])
    
    # Assume both directories have the same number of frames
    num_frames = len(image_files)

    for i in range(num_frames):
        image_path = os.path.join(image_dir, image_files[i])
        lidar_path = os.path.join(lidar_dir, lidar_files[i])
        
        print(f"\nProcessing frame {i+1}/{num_frames}:")
        print(f"  Image: {image_path}")
        print(f"  LiDAR: {lidar_path}")
        
        visual_3d_list = perform_association(image_path, lidar_path, save_path=output_dir, frame_id=i)
        
        if len(visual_3d_list) > 0:
            # Save visual_3d_list to a JSON file
            output_filename = os.path.join(output_dir, f'frame_{i:06d}_visual_3d.json')
            with open(output_filename, 'w') as f:
                serializable_list = []
                for item in visual_3d_list:
                    serializable_list.append({
                        'id': int(item['id']), 
                        'point3d': item['point3d'].tolist() 
                    })
                json.dump(serializable_list, f, indent=4)
            print(f"Saved {len(visual_3d_list)} 3D visual features to {output_filename}")
        else:
            print(f"No 3D visual features were successfully associated for frame {i:06d}.")
    
    # if len(visual_3d_list) > 0:
    #     # Convert list of dicts to numpy array for visualization
    #     points_to_visualize = np.array([item['point3d'] for item in visual_3d_list])
    #     visualize_3d_points(points_to_visualize, "Associated 3D Visual Features")
    # else:
    #     print("No 3D visual features were successfully associated.")

if __name__ == "__main__":
    main()