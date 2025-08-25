# vSLAM-and-GNSS-denied-navigation (Sensor Fusion Navigation Module)

This repository contains a personal implementation of a navigation module for Unmanned Ground Vehicles (UGVs), with potential applicability to Unmanned Aerial Vehicles (UAVs), specifically designed for GNSS-denied environments. The goal is to build a custom sensor fusion and SLAM pipeline from scratch to gain a deeper understanding of the underlying mechanisms and present a significant personal challenge.

The module focuses on fusing data from a camera and a LiDAR sensor to generate a robust and detailed 3D environmental representation. Cameras inherently lack direct depth information and are susceptible to varying environmental conditions (e.g., lighting, weather), while LiDAR provides sparse point clouds and lacks semantic understanding of objects. By integrating these two modalities, we can generate a more robust and detailed 3D environmental representation.

## Core Modules

This project is structured around several key processing stages, with a current focus on:

1.  **LiDAR Segmentation (`lidar_segmentation.py`)**
    This module processes raw LiDAR point clouds to delineate distinct objects from the ground plane. It uses techniques like range image projection and DBSCAN clustering to identify individual entities within the 3D scene. The output is a segmented point cloud where non-ground points are assigned to specific object clusters, providing a structured 3D scene representation.

2.  **LiDAR Feature Extraction (`lidar_features.py`)**
    This module extracts salient 3D features from the segmented LiDAR point clouds. It identifies "edge" features (high curvature points) and "planar" features (low curvature points) by analyzing the local geometry of the point cloud. These features are crucial for robust odometry and mapping, providing stable geometric primitives for tracking and localization. Features are assigned unique IDs for tracking across frames.

3.  **Visual Feature Tracking**
    This component focuses on extracting and tracking 2D visual features (e.g., corners, edges, blobs) from camera images. These features act like unique visual landmarks that the system can easily spot and follow over time, even if the camera's perspective shifts or the lighting conditions change. They serve as critical visual anchors for subsequent data association.

4.  **Multimodal Data Integration / Data Association (`data_association.py`)**
    This module is responsible for combining the 2D visual landmarks from the camera with the 3D segmented and featured LiDAR data to estimate precise 3D coordinates for each visual landmark. The process involves:
    *   **Aligning Sensors:** Projecting 3D LiDAR points onto the 2D camera image using the calibration information to establish spatial correspondence with visual features.
    *   **Finding Context:** Performing a localized neighborhood search around each 2D visual landmark within the projected LiDAR points to understand the local 3D shape (e.g., edge, flat surface).
    *   **Fitting a Surface:** Fitting a small, flat surface (a "plane") to the most reliable LiDAR points in the neighborhood to create a stable and accurate local surface representation, mitigating noise from individual LiDAR measurements.
    *   **Calculating Depth:** Determining the depth for the visual landmark as the perpendicular distance from its 2D projected 3D location to this fitted plane, providing a robust depth measurement.

The 3D features position information is crucial for evaluating the vehicle's pose at each time step, a piece of information which will be crucial in next steps for visual odometry, which I look forward to implement in this repository soon enough.

## Dataset Setup (KITTI)

This project utilizes the [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/raw_data.php) for sensor data and leverages sinchronised data. To run the scripts, you need to download the sinchronised version of the data sequence (e.g., `2011_09_26_drive_0001_sync`) and place the relevant directories within a `Kitti/` folder in the root of this repository.

The following directory structure is expected:

```
Sensor Fusion/
├── Kitti/
│   ├── calib_cam_to_cam.txt
│   ├── calib_imu_to_velo.txt
│   ├── calib_velo_to_cam.txt
│   ├── image_00/
│   │   └── data/
│   ├── image_01/
│   │   └── data/
│   ├── image_02/
│   │   └── data/  (Contains camera images, e.g., `0000000000.png`)
│   ├── image_03/
│   │   └── data/
│   ├── oxts/
│   │   └── data/
│   └── velodyne_points/
│       └── data/  (Contains LiDAR point clouds, e.g., `0000000000.bin`)
├── data_association.py
├── kitti_utils.py
├── lidar_features.py
├── lidar_segmentation.py
└── ... (other project files)
```

Ensure that the `image_02/data/` and `velodyne_points/data/` directories contain the synchronized image and LiDAR `.bin` files, respectively. The calibration files (`calib_cam_to_cam.txt`, `calib_imu_to_velo.txt`, `calib_velo_to_cam.txt`) should be directly under the `Kitti/` directory.


https://github.com/user-attachments/assets/33ba2e52-42a5-41b5-b199-0a8a515c388c

