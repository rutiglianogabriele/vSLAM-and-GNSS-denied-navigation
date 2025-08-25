import cv2
import numpy as np
import argparse
import os

class VisualFeatureTracker:
    def __init__(self):
        # Settings for finding corners
        self.corner_params = {
            'maxCorners': 300,      # Maximum number of corners to find
            'qualityLevel': 0.01,     # Quality of corners (0.01 is good)
            'minDistance': 10,       # Minimum distance between corners
            'blockSize': 3         # Size of neighborhood for corner detection
        }
        
        # Settings for optical flow tracking
        self.flow_params = {
            'winSize': (15, 15),    # Size of search window
            'maxLevel': 2,      # Number of pyramid levels
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        }
        
        # Settings for RANSAC outlier removal
        self.ransac_threshold = 1.0  # Pixel distance threshold
        
        # Storage for previous frame and features
        self.prev_frame = None
        self.prev_features = None
        self.feature_ids = None
        self.next_feature_id = 0
        self.tracked_feature_id = None
        
        # Create CLAHE object for image enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def enhance_image(self, frame):
        """Make the image clearer using CLAHE"""
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Apply CLAHE to improve contrast
        enhanced = self.clahe.apply(gray)
        return enhanced
    
    def find_new_corners(self, frame, existing_features=None):
        """Find good corners in the image"""
        # Create a mask to avoid finding corners too close to existing ones
        mask = np.ones(frame.shape, dtype=np.uint8) * 255
        
        if existing_features is not None and len(existing_features) > 0:
            for feature in existing_features:
                x, y = int(feature[0][0]), int(feature[0][1])
                cv2.circle(mask, (x, y), 30, 0, -1)  # Block area around existing features
        
        # Find corners using Shi-Tomasi method
        corners = cv2.goodFeaturesToTrack(frame, mask=mask, **self.corner_params)
        return corners
    
    def track_features(self, prev_frame, current_frame, prev_features):
        """Track features from previous frame to current frame"""
        if prev_features is None or len(prev_features) == 0:
            return None, None, None
        
        # Track features using Lucas-Kanade optical flow
        new_features, status, error = cv2.calcOpticalFlowPyrLK(
            prev_frame, current_frame, prev_features, None, **self.flow_params
        )
        
        # Keep only successfully tracked features
        good_new = new_features[status == 1]
        good_old = prev_features[status == 1]
        
        return good_new, good_old, status
    
    def remove_outliers(self, prev_features, current_features, inlier_mask):
        """Remove bad feature matches using RANSAC"""
        if len(prev_features) < 8 or len(current_features) < 8:
            inlier_mask[:] = True
            return current_features  # Need at least 8 points for fundamental matrix
        
        # Find fundamental matrix and remove outliers
        _, mask = cv2.findFundamentalMat(
            prev_features, current_features,
            cv2.FM_RANSAC, self.ransac_threshold
        )
        
        if mask is not None:
            # Keep only inlier features
            inlier_mask[:] = mask.ravel() == 1
            good_features = current_features[inlier_mask]
            return good_features.reshape(-1, 1, 2)
        
        inlier_mask[:] = True
        return current_features
    
    def process_frame(self, frame):
        """Main function to process each frame"""
        # Step 1: Enhance the image
        enhanced_frame = self.enhance_image(frame)
        
        # Step 2: If this is the first frame, just find corners
        if self.prev_frame is None:
            self.prev_features = self.find_new_corners(enhanced_frame)
            if self.prev_features is not None:
                self.feature_ids = np.arange(self.next_feature_id, self.next_feature_id + len(self.prev_features))
                self.next_feature_id += len(self.prev_features)
            else:
                self.feature_ids = np.array([])
            self.prev_frame = enhanced_frame
            return self.prev_features, self.feature_ids
        
        # Step 3: Track existing features
        tracked_features, prev_good, status = self.track_features(
            self.prev_frame, enhanced_frame, self.prev_features
        )
        
        # Update feature IDs
        self.feature_ids = self.feature_ids[status.ravel() == 1]
        
        if tracked_features is not None and len(tracked_features) > 0:
            # Step 4: Remove outliers
            if len(tracked_features) >= 8:
                # Keep track of which features are kept
                inlier_mask = np.zeros(len(tracked_features), dtype=bool)
                clean_features = self.remove_outliers(prev_good, tracked_features, inlier_mask)
                self.feature_ids = self.feature_ids[inlier_mask]
            else:
                clean_features = tracked_features
        else:
            clean_features = np.array([]).reshape(-1, 1, 2)
        
        # Step 5: Add new features if we don't have enough
        if len(clean_features) < 200: 
            new_corners = self.find_new_corners(enhanced_frame, clean_features)
            if new_corners is not None:
                new_ids = np.arange(self.next_feature_id, self.next_feature_id + len(new_corners))
                self.next_feature_id += len(new_corners)
                if len(clean_features) > 0:
                    all_features = np.vstack((clean_features, new_corners))
                    self.feature_ids = np.concatenate((self.feature_ids, new_ids))
                else:
                    all_features = new_corners
                    self.feature_ids = new_ids
            else:
                all_features = clean_features
        else:
            all_features = clean_features
        
        self.prev_frame = enhanced_frame
        self.prev_features = all_features
        
        return all_features, self.feature_ids
    
    def draw_features(self, frame, features):
        """Draw features on the image for visualization"""
        display_frame = frame.copy()
        
        if features is not None and len(features) > 0:
            for feature in features:
                x, y = map(int, feature.ravel())
                cv2.circle(display_frame, (x, y), 3, (0, 255, 0), -1)
        
        # Show feature count
        text = f"Features: {len(features) if features is not None else 0}"
        cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return display_frame

# Example usage
def main():
    kitti_path = 'Kitti'
    image_dir = os.path.join(kitti_path, 'image_02', 'data')
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    # Create the tracker
    tracker = VisualFeatureTracker()
    
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_dir, image_file)
        frame = cv2.imread(image_path)
        # Process the frame
        features, feature_ids = tracker.process_frame(frame)

        print(f"The following feature id: {feature_ids[10]}, has the following position: {features[10]}")
        
        # Draw features on frame
        display_frame = tracker.draw_features(frame, features)
        
        # Show the result
        cv2.imshow('Feature Tracker', display_frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()