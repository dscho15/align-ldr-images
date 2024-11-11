import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union

@dataclass
class MatchingParameters:
    """Parameters for feature detection and matching."""
    max_corners_pct: float = 0.1
    quality_level: float = 0.01
    min_distance_l2_pct: float = 1
    template_size_pct: float = 2
    roi_size_pct: float = 5
    matching_threshold: float = 0.90
    ransac_max_trials: int = 10000
    ransac_residual_threshold: float = 2.0

class AffineTransform:
    """Handles affine transformation computations."""
    
    @staticmethod
    def compute_matrix(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Computes the affine transformation matrix from p1 to p2."""
        A = []
        B = []
        for i in range(3):
            x, y = p1[i]
            xp, yp = p2[i]
            A.extend([[x, y, 1, 0, 0, 0], [0, 0, 0, x, y, 1]])
            B.extend([xp, yp])
        
        A = np.array(A)
        B = np.array(B)
        x = np.linalg.lstsq(A, B, rcond=None)[0]
        
        return np.array([[x[0], x[1], x[2]],
                        [x[3], x[4], x[5]],
                        [0, 0, 1]])

    @staticmethod
    def ransac(src_points: np.ndarray, 
              dst_points: np.ndarray, 
              params: MatchingParameters) -> Tuple[np.ndarray, np.ndarray]:
        """Estimates affine transformation using RANSAC."""
        num_points = src_points.shape[0]
        best_inliers = None
        best_affine = None
        best_num_inliers = 0
        
        for _ in range(params.ransac_max_trials):
            indices = np.random.choice(num_points, 3, replace=False)
            affine_matrix = AffineTransform.compute_matrix(src_points[indices], 
                                                         dst_points[indices])
            
            src_points_hom = np.hstack([src_points, np.ones((num_points, 1))])
            transformed_points = src_points_hom @ affine_matrix.T
            
            residuals = np.linalg.norm(transformed_points[:, :2] - dst_points, axis=1)
            inliers = residuals < params.ransac_residual_threshold
            num_inliers = np.sum(inliers)
            
            if num_inliers > best_num_inliers:
                best_num_inliers = num_inliers
                best_inliers = inliers
                best_affine = affine_matrix
                
        return best_affine, best_inliers

class ImageProcessor:
    """Handles image loading and preprocessing operations."""
    
    @staticmethod
    def load_images(path: Union[str, Path]) -> List[np.ndarray]:
        """Loads images from the specified path."""
        if isinstance(path, str):
            path = Path(path)
        return [cv2.imread(str(img)) for img in path.rglob("*")]
    
    @staticmethod
    def convert_to_grayscale(images: List[np.ndarray]) -> List[np.ndarray]:
        """Converts BGR images to grayscale."""
        return [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
    
    @staticmethod
    def sort_by_intensity(images: List[np.ndarray]) -> List[np.ndarray]:
        """Sorts images by mean intensity."""
        return sorted(images, key=lambda x: np.mean(x))

class FeatureDetector:
    """Handles feature detection and matching operations."""
    
    def __init__(self, params: MatchingParameters):
        self.params = params
    
    def detect_features(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Detects good features to track in the images."""
        h, w = images[0].shape
        min_distance = int(self.params.min_distance_l2_pct / 100 * w)
        max_corners = int(h * w * self.params.max_corners_pct / 100.)
        
        return [cv2.goodFeaturesToTrack(img, max_corners, 
                                      self.params.quality_level, 
                                      min_distance) for img in images]
    
    def extract_patch(self, 
                     image: np.ndarray, 
                     corner: Tuple[int, int],
                     size_pct: float) -> Tuple[Optional[np.ndarray], Tuple[int, int, int]]:
        """Extracts a patch around a corner point."""
        x, y = corner
        h, w = image.shape
        
        template_w = int(size_pct / 100 * w)
        template_w = template_w + 1 if template_w % 2 == 0 else template_w
        
        x_start = max(0, x - template_w // 2)
        y_start = max(0, y - template_w // 2)
        x_end = min(w, x + template_w // 2)
        y_end = min(h, y + template_w // 2)
        
        if x_end - x_start != y_end - y_start:
            return None, (x_start, y_start, template_w)
        
        return image[y_start:y_end, x_start:x_end], (x_start, y_start, template_w)

class FeatureMatcher:
    """Handles feature matching between image pairs."""
    
    def __init__(self, params: MatchingParameters):
        self.params = params
        self.detector = FeatureDetector(params)
    
    def find_matches(self, 
                    corners: np.ndarray, 
                    img0: np.ndarray, 
                    img1: np.ndarray) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Finds matches between two images using template matching."""
        matches = []
        
        for corner in corners:
            x_query, y_query = map(int, corner.ravel())
            
            template, (_, _, template_w) = self.detector.extract_patch(
                img0, (x_query, y_query), self.params.template_size_pct)
            roi, (x_start_roi, y_start_roi, _) = self.detector.extract_patch(
                img1, (x_query, y_query), self.params.roi_size_pct)
            
            if template is None or roi is None:
                continue
            
            result = cv2.matchTemplate(roi, template, cv2.TM_CCORR_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            x_match = x_start_roi + max_loc[0] + template_w // 2
            y_match = y_start_roi + max_loc[1] + template_w // 2
            
            if max_val >= self.params.matching_threshold:
                matches.append(((x_query, y_query), (x_match, y_match)))
        
        return matches

class ImageMatchingPipeline:
    """Main pipeline for image matching process."""
    
    def __init__(self, params: MatchingParameters = MatchingParameters()):
        self.params = params
        self.matcher = FeatureMatcher(params)
        self.detector = FeatureDetector(params)
    
    def process(self, image_path: Union[str, Path]) -> List[np.ndarray]:
        """Runs the complete image matching pipeline."""
        # Load and preprocess images
        images = ImageProcessor.load_images(image_path)
        images = ImageProcessor.convert_to_grayscale(images)
        images = ImageProcessor.sort_by_intensity(images)
        
        # Detect features
        corners = self.detector.detect_features(images)
        
        # Match features between consecutive images
        transformations = []
        for i in range(len(images) - 1):
            # Find matches in both directions
            matches_forward = self.matcher.find_matches(corners[i], images[i], images[i + 1])
            matches_backward = self.matcher.find_matches(corners[i + 1], images[i + 1], images[i])
            
            # Combine matches
            src_pts = ([match[0] for match in matches_forward] + 
                      [match[1] for match in matches_backward])
            dst_pts = ([match[1] for match in matches_forward] + 
                      [match[0] for match in matches_backward])
            
            if src_pts and dst_pts:
                # Estimate transformation
                affine_mat, _ = AffineTransform.ransac(
                    np.array(src_pts), 
                    np.array(dst_pts), 
                    self.params
                )
                transformations.append(affine_mat)
        
        print(f"Found {len(transformations)} transformations.")
        print(transformations)
        
        return transformations

# Usage example
if __name__ == "__main__":

    params = MatchingParameters(
        max_corners_pct=0.1,
        quality_level=0.01,
        min_distance_l2_pct=1,
        matching_threshold=0.90
    )
    
    pipeline = ImageMatchingPipeline(params)
    transformations = pipeline.process("images/")