from abc import ABC, abstractmethod
import logging
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from itertools import permutations


# --- Utilities ---
def get_background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def extract_components(grid: np.ndarray, background_color: int):
    mask = (grid != background_color).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    components = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        component_mask = np.zeros_like(mask)
        cv2.drawContours(component_mask, [contour], -1, 255, thickness=cv2.FILLED)
        centroid = (x + w // 2, y + h // 2)

        components.append({
            "bbox": (x, y, w, h),
            "area": area,
            "mask": component_mask.astype(bool),
            "centroid": centroid
        })
    return components


# --- Abstract base classes ---
class AbstractProblemAnalyzer(ABC):
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    def analyze(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict:
        pass


class AbstractFeatureExtractor(AbstractProblemAnalyzer):
    @abstractmethod
    def get_shape_features(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict:
        pass

    @abstractmethod
    def get_value_features(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict:
        pass


class AbstractPatternDetector(AbstractProblemAnalyzer):
    @abstractmethod
    def detect_repetition(self, grid: np.ndarray) -> Dict:
        pass

    @abstractmethod
    def detect_symmetry(self, grid: np.ndarray) -> Dict:
        pass


class AbstractTransformationDetector(AbstractProblemAnalyzer):
    """Abstract base class for transformation detection"""
    @abstractmethod
    def detect_transformations(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict:
        pass


# --- New Transformation Detector ---
class GenericTransformationDetector(AbstractTransformationDetector):
    """Detects various types of transformations between input and output grids"""
    
    def __init__(self, logger=None, max_permutation_size=6):
        super().__init__(logger)
        self.max_permutation_size = max_permutation_size
    
    def analyze(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict:
        return {
            'transformations': self.detect_transformations(input_grid, output_grid)
        }
    
    def _compute_binary_difference(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict:
        """
        Compute a boolean matrix indicating where the two grids differ.
        Only valid if both grids have the same shape.
        """
        assert input_grid.shape == output_grid.shape, "Grids must have the same shape"

        diff = input_grid != output_grid
        return {
            "binary_diff": diff.tolist(),
            "change_count": int(np.sum(diff)),
            "changed_rows": [int(i) for i, row in enumerate(diff) if any(row)],
            "changed_columns": [int(j) for j, col in enumerate(diff.T) if any(col)]
        }

    def _compute_grid_difference(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict:
        """
        Compare two grids cell-by-cell and return a structured diff.
        Only valid for grids of the same shape.
        """
        assert input_grid.shape == output_grid.shape, "Grids must have the same shape"

        diff = {
            "changes": [],
            "change_count": 0,
            "value_mapping": {},
            "row_changes": {},  # row index → list of changed column indices
            "value_substitutions": []  # list of (old_value, new_value)
        }

        for i in range(input_grid.shape[0]):
            for j in range(input_grid.shape[1]):
                in_val = input_grid[i, j]
                out_val = output_grid[i, j]
                if in_val != out_val:
                    diff["changes"].append({
                        "position": (i, j),
                        "before": int(in_val),
                        "after": int(out_val)
                    })
                    diff["change_count"] += 1
                    diff["row_changes"].setdefault(i, []).append(j)
                    diff["value_substitutions"].append((int(in_val), int(out_val)))
                    diff["value_mapping"][int(in_val)] = int(out_val)

        return diff

    def detect_transformations(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict:
        """Main method to detect all types of transformations"""
        transformations = {}
        
        # Always check size transformations first
        transformations.update(self._detect_size_transformations(input_grid, output_grid))
        
        # Structural transformations - check if grids have same shape OR if one is contained in the other
        if input_grid.shape == output_grid.shape:
            # Compute detailed grid difference
            transformations.update(self._compute_binary_difference(input_grid, output_grid))
            transformations["grid_diff"] = self._compute_grid_difference(input_grid, output_grid)
            # Same shape - check all structural transformations
            transformations.update(self._detect_row_transformations(input_grid, output_grid))
            transformations.update(self._detect_column_transformations(input_grid, output_grid))
            transformations.update(self._detect_geometric_transformations(input_grid, output_grid))
        else:
            # Different shapes - check if transformations apply to cropped/padded regions
            transformations.update(self._detect_cross_shape_transformations(input_grid, output_grid))
        
        # Color transformations work regardless of shape
        transformations.update(self._detect_color_transformations(input_grid, output_grid))
        
        # Pattern extraction transformations (e.g., extracting rows/columns)
        transformations.update(self._detect_extraction_transformations(input_grid, output_grid))
        
        return transformations
    
    def _detect_row_transformations(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict:
        """Detect row-based transformations like swaps, reversals, shifts"""
        results = {
            'row_swap': False,
            'row_swap_indices': None,
            'row_reverse': False,
            'row_shift': False,
            'row_shift_amount': 0
        }
        
        rows, cols = input_grid.shape
        
        # Check for row reversal (entire grid flipped vertically)
        if np.array_equal(output_grid, input_grid[::-1, :]):
            results['row_reverse'] = True
            return results
        
        # Check for row shifts (circular shift)
        for shift in range(1, rows):
            shifted = np.roll(input_grid, shift, axis=0)
            if np.array_equal(output_grid, shifted):
                results['row_shift'] = True
                results['row_shift_amount'] = shift
                return results
        
        # Check for row swaps/permutations (only for small grids to avoid performance issues)
        if rows <= self.max_permutation_size:
            input_rows = [input_grid[i, :] for i in range(rows)]
            output_rows = [output_grid[i, :] for i in range(rows)]
            
            # Find if output rows are a permutation of input rows
            row_mapping = self._find_row_permutation(input_rows, output_rows)
            if row_mapping and row_mapping != list(range(rows)):
                results['row_swap'] = True
                results['row_swap_indices'] = row_mapping
        
        return results
    
    def _detect_column_transformations(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict:
        """Detect column-based transformations"""
        results = {
            'col_swap': False,
            'col_swap_indices': None,
            'col_reverse': False,
            'col_shift': False,
            'col_shift_amount': 0
        }
        
        rows, cols = input_grid.shape
        
        # Check for column reversal
        if np.array_equal(output_grid, input_grid[:, ::-1]):
            results['col_reverse'] = True
            return results
        
        # Check for column shifts
        for shift in range(1, cols):
            shifted = np.roll(input_grid, shift, axis=1)
            if np.array_equal(output_grid, shifted):
                results['col_shift'] = True
                results['col_shift_amount'] = shift
                return results
        
        # Check for column swaps
        if cols <= self.max_permutation_size:
            input_cols = [input_grid[:, i] for i in range(cols)]
            output_cols = [output_grid[:, i] for i in range(cols)]
            
            col_mapping = self._find_row_permutation(input_cols, output_cols)
            if col_mapping and col_mapping != list(range(cols)):
                results['col_swap'] = True
                results['col_swap_indices'] = col_mapping
        
        return results
    
    def _detect_geometric_transformations(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict:
        """Detect geometric transformations like rotation, transposition"""
        results = {
            'rotation_90': False,
            'rotation_180': False,
            'rotation_270': False,
            'transpose': False,
            'anti_transpose': False
        }
        
        # Only check rotations if grid is square
        if input_grid.shape[0] == input_grid.shape[1]:
            # 90-degree rotation
            if np.array_equal(output_grid, np.rot90(input_grid, k=1)):
                results['rotation_90'] = True
            # 180-degree rotation
            elif np.array_equal(output_grid, np.rot90(input_grid, k=2)):
                results['rotation_180'] = True
            # 270-degree rotation
            elif np.array_equal(output_grid, np.rot90(input_grid, k=3)):
                results['rotation_270'] = True
        
        # Check for transpose (only if dimensions allow)
        if input_grid.shape == output_grid.shape[::-1]:
            if np.array_equal(output_grid, input_grid.T):
                results['transpose'] = True
            elif np.array_equal(output_grid, np.rot90(input_grid.T, k=2)):
                results['anti_transpose'] = True
        
        return results
    
    def _detect_color_transformations(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict:
        """Detect color-based transformations - works with different shapes"""
        results = {
            'color_mapping': {},
            'color_inversion': False,
            'color_shift': False,
            'identical_structure': False,
            'color_filtering': False,
            'filtered_colors': []
        }
        
        # Check if structure is identical (same shape with potential color changes)
        if input_grid.shape == output_grid.shape:
            # Find color mapping
            unique_input = np.unique(input_grid)
            unique_output = np.unique(output_grid)
            
            # Check if it's a simple color mapping
            color_map = {}
            for i_val in unique_input:
                input_positions = (input_grid == i_val)
                output_values_at_positions = output_grid[input_positions]
                unique_output_vals = np.unique(output_values_at_positions)
                
                if len(unique_output_vals) == 1:
                    color_map[int(i_val)] = int(unique_output_vals[0])
                else:
                    # Not a simple mapping
                    color_map = {}
                    break
            
            if color_map and len(color_map) == len(unique_input):
                results['color_mapping'] = color_map
                
                # Check if structure is preserved
                reconstructed = np.zeros_like(input_grid)
                for old_color, new_color in color_map.items():
                    reconstructed[input_grid == old_color] = new_color
                
                if np.array_equal(reconstructed, output_grid):
                    results['identical_structure'] = True
        
        # Check for color filtering (output contains only certain colors from input)
        else:
            input_colors = set(np.unique(input_grid))
            output_colors = set(np.unique(output_grid))
            
            if output_colors.issubset(input_colors) and len(output_colors) < len(input_colors):
                results['color_filtering'] = True
                results['filtered_colors'] = list(output_colors)
        
        return results
    
    def _detect_size_transformations(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict:
        """Detect size-related transformations"""
        results = {
            'size_change': input_grid.shape != output_grid.shape,
            'scale_factor_row': output_grid.shape[0] / input_grid.shape[0] if input_grid.shape[0] > 0 else 0,
            'scale_factor_col': output_grid.shape[1] / input_grid.shape[1] if input_grid.shape[1] > 0 else 0,
            'cropping': False,
            'padding': False,
            'tiling': False
        }
        
        if results['size_change']:
            # Check for cropping (output smaller than input)
            if output_grid.shape[0] <= input_grid.shape[0] and output_grid.shape[1] <= input_grid.shape[1]:
                results['cropping'] = self._check_cropping(input_grid, output_grid)
            
            # Check for padding (output larger than input)
            elif output_grid.shape[0] >= input_grid.shape[0] and output_grid.shape[1] >= input_grid.shape[1]:
                results['padding'] = self._check_padding(input_grid, output_grid)
            
            # Check for tiling (output is multiple of input)
            if (output_grid.shape[0] % input_grid.shape[0] == 0 and 
                output_grid.shape[1] % input_grid.shape[1] == 0):
                results['tiling'] = self._check_tiling(input_grid, output_grid)
        
        return results
    
    def _find_row_permutation(self, input_rows: List[np.ndarray], output_rows: List[np.ndarray]) -> Optional[List[int]]:
        """Find if output_rows is a permutation of input_rows"""
        if len(input_rows) != len(output_rows):
            return None
        
        # Try to find mapping
        mapping = []
        used_indices = set()
        
        for out_row in output_rows:
            found = False
            for i, in_row in enumerate(input_rows):
                if i not in used_indices and np.array_equal(out_row, in_row):
                    mapping.append(i)
                    used_indices.add(i)
                    found = True
                    break
            if not found:
                return None
        
        return mapping if len(mapping) == len(input_rows) else None
    
    def _check_cropping(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Check if output is a crop of input"""
        in_h, in_w = input_grid.shape
        out_h, out_w = output_grid.shape
        
        # If output is larger than input, it can't be a crop
        if out_h > in_h or out_w > in_w:
            return False
        
        for start_row in range(in_h - out_h + 1):
            for start_col in range(in_w - out_w + 1):
                crop = input_grid[start_row:start_row + out_h, start_col:start_col + out_w]
                if np.array_equal(crop, output_grid):
                    return True
        
        return False
    
    def _detect_cross_shape_transformations(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict:
        """Detect transformations between grids of different shapes"""
        results = {
            'cross_shape_row_extraction': False,
            'cross_shape_col_extraction': False,
            'cross_shape_diagonal_extraction': False,
            'cross_shape_subgrid_transform': False,
            'extracted_indices': None,
            'subgrid_location': None
        }
        
        # Check if output is a single row from input
        if output_grid.shape[0] == 1 and output_grid.shape[1] == input_grid.shape[1]:
            for i in range(input_grid.shape[0]):
                if np.array_equal(output_grid[0, :], input_grid[i, :]):
                    results['cross_shape_row_extraction'] = True
                    results['extracted_indices'] = i
                    return results
        
        # Check if output is a single column from input
        if output_grid.shape[1] == 1 and output_grid.shape[0] == input_grid.shape[0]:
            for j in range(input_grid.shape[1]):
                if np.array_equal(output_grid[:, 0], input_grid[:, j]):
                    results['cross_shape_col_extraction'] = True
                    results['extracted_indices'] = j
                    return results
        
        # Check if output is multiple rows from input
        if output_grid.shape[1] == input_grid.shape[1] and output_grid.shape[0] < input_grid.shape[0]:
            # Try to find consecutive rows
            for start_row in range(input_grid.shape[0] - output_grid.shape[0] + 1):
                if np.array_equal(output_grid, input_grid[start_row:start_row + output_grid.shape[0], :]):
                    results['cross_shape_row_extraction'] = True
                    results['extracted_indices'] = list(range(start_row, start_row + output_grid.shape[0]))
                    return results
            
            # Try to find non-consecutive rows
            row_indices = self._find_matching_rows(input_grid, output_grid)
            if row_indices:
                results['cross_shape_row_extraction'] = True
                results['extracted_indices'] = row_indices
                return results
        
        # Check if output is multiple columns from input
        if output_grid.shape[0] == input_grid.shape[0] and output_grid.shape[1] < input_grid.shape[1]:
            # Try consecutive columns
            for start_col in range(input_grid.shape[1] - output_grid.shape[1] + 1):
                if np.array_equal(output_grid, input_grid[:, start_col:start_col + output_grid.shape[1]]):
                    results['cross_shape_col_extraction'] = True
                    results['extracted_indices'] = list(range(start_col, start_col + output_grid.shape[1]))
                    return results
            
            # Try non-consecutive columns
            col_indices = self._find_matching_columns(input_grid, output_grid)
            if col_indices:
                results['cross_shape_col_extraction'] = True
                results['extracted_indices'] = col_indices
                return results
        
        # Check if output is a transformed subgrid of input
        subgrid_result = self._check_subgrid_transformations(input_grid, output_grid)
        if subgrid_result['found']:
            results['cross_shape_subgrid_transform'] = True
            results['subgrid_location'] = subgrid_result
        
        return results
    
    def _detect_extraction_transformations(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict:
        """Detect pattern extraction transformations"""
        results = {
            'diagonal_extraction': False,
            'border_extraction': False,
            'center_extraction': False,
            'corners_extraction': False,
            'checkerboard_extraction': False,
            'pattern_type': None
        }
        
        # Check diagonal extraction
        if self._check_diagonal_extraction(input_grid, output_grid):
            results['diagonal_extraction'] = True
            results['pattern_type'] = 'diagonal'
        
        # Check border extraction
        elif self._check_border_extraction(input_grid, output_grid):
            results['border_extraction'] = True
            results['pattern_type'] = 'border'
        
        # Check center extraction
        elif self._check_center_extraction(input_grid, output_grid):
            results['center_extraction'] = True
            results['pattern_type'] = 'center'
        
        # Check corners extraction
        elif self._check_corners_extraction(input_grid, output_grid):
            results['corners_extraction'] = True
            results['pattern_type'] = 'corners'
        
        # Check checkerboard pattern
        elif self._check_checkerboard_extraction(input_grid, output_grid):
            results['checkerboard_extraction'] = True
            results['pattern_type'] = 'checkerboard'
        
        return results
    
    def _find_matching_rows(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[List[int]]:
        """Find which input rows match the output rows in order"""
        matching_indices = []
        used_rows = set()
        
        for out_row_idx in range(output_grid.shape[0]):
            found = False
            for in_row_idx in range(input_grid.shape[0]):
                if in_row_idx not in used_rows and np.array_equal(output_grid[out_row_idx, :], input_grid[in_row_idx, :]):
                    matching_indices.append(in_row_idx)
                    used_rows.add(in_row_idx)
                    found = True
                    break
            if not found:
                return None
        
        return matching_indices if len(matching_indices) == output_grid.shape[0] else None
    
    def _find_matching_columns(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[List[int]]:
        """Find which input columns match the output columns in order"""
        matching_indices = []
        used_cols = set()
        
        for out_col_idx in range(output_grid.shape[1]):
            found = False
            for in_col_idx in range(input_grid.shape[1]):
                if in_col_idx not in used_cols and np.array_equal(output_grid[:, out_col_idx], input_grid[:, in_col_idx]):
                    matching_indices.append(in_col_idx)
                    used_cols.add(in_col_idx)
                    found = True
                    break
            if not found:
                return None
        
        return matching_indices if len(matching_indices) == output_grid.shape[1] else None
    
    def _check_subgrid_transformations(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict:
        """Check if output is a transformed version of a subgrid from input"""
        result = {'found': False, 'location': None, 'transformation': None}
        
        in_h, in_w = input_grid.shape
        out_h, out_w = output_grid.shape
        
        # Try all possible subgrid locations
        for start_row in range(in_h - out_h + 1):
            for start_col in range(in_w - out_w + 1):
                subgrid = input_grid[start_row:start_row + out_h, start_col:start_col + out_w]
                
                # Check direct match
                if np.array_equal(subgrid, output_grid):
                    result = {
                        'found': True,
                        'location': (start_row, start_col, out_h, out_w),
                        'transformation': 'identity'
                    }
                    return result
                
                # Check rotations (if square)
                if out_h == out_w:
                    for k in range(1, 4):
                        rotated = np.rot90(subgrid, k)
                        if np.array_equal(rotated, output_grid):
                            result = {
                                'found': True,
                                'location': (start_row, start_col, out_h, out_w),
                                'transformation': f'rotation_{90*k}'
                            }
                            return result
                
                # Check flips
                if np.array_equal(subgrid[::-1, :], output_grid):
                    result = {
                        'found': True,
                        'location': (start_row, start_col, out_h, out_w),
                        'transformation': 'vertical_flip'
                    }
                    return result
                
                if np.array_equal(subgrid[:, ::-1], output_grid):
                    result = {
                        'found': True,
                        'location': (start_row, start_col, out_h, out_w),
                        'transformation': 'horizontal_flip'
                    }
                    return result
        
        return result
    
    def _check_diagonal_extraction(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Check if output is diagonal elements from input"""
        min_dim = min(input_grid.shape)
        
        # Main diagonal
        if output_grid.size == min_dim:
            main_diag = np.array([input_grid[i, i] for i in range(min_dim)])
            if np.array_equal(output_grid.flatten(), main_diag):
                return True
        
        # Anti-diagonal
        if output_grid.size == min_dim:
            anti_diag = np.array([input_grid[i, input_grid.shape[1]-1-i] for i in range(min_dim)])
            if np.array_equal(output_grid.flatten(), anti_diag):
                return True
        
        return False
    
    def _check_border_extraction(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Check if output contains border elements from input"""
        h, w = input_grid.shape
        border_elements = []
        
        # Top and bottom rows
        border_elements.extend(input_grid[0, :])
        if h > 1:
            border_elements.extend(input_grid[-1, :])
        
        # Left and right columns (excluding corners already added)
        for i in range(1, h-1):
            border_elements.append(input_grid[i, 0])
            if w > 1:
                border_elements.append(input_grid[i, -1])
        
        border_array = np.array(border_elements)
        return np.array_equal(output_grid.flatten(), border_array) or np.array_equal(output_grid.flatten(), border_array[:output_grid.size])
    
    def _check_center_extraction(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Check if output is center region of input"""
        h, w = input_grid.shape
        out_h, out_w = output_grid.shape
        
        if out_h >= h or out_w >= w:
            return False
        
        start_row = (h - out_h) // 2
        start_col = (w - out_w) // 2
        center_region = input_grid[start_row:start_row + out_h, start_col:start_col + out_w]
        
        return np.array_equal(center_region, output_grid)
    
    def _check_corners_extraction(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Check if output contains corner elements from input"""
        h, w = input_grid.shape
        corners = [
            input_grid[0, 0],      # top-left
            input_grid[0, -1],     # top-right
            input_grid[-1, 0],     # bottom-left
            input_grid[-1, -1]     # bottom-right
        ]
        
        corners_array = np.array(corners)
        return np.array_equal(output_grid.flatten(), corners_array) or np.array_equal(output_grid.flatten(), corners_array[:output_grid.size])
    
    def _check_checkerboard_extraction(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Check if output follows a checkerboard pattern from input"""
        h, w = input_grid.shape
        
        # Extract checkerboard pattern starting with (0,0)
        pattern1 = []
        for i in range(h):
            for j in range(w):
                if (i + j) % 2 == 0:
                    pattern1.append(input_grid[i, j])
        
        # Extract checkerboard pattern starting with (0,1)
        pattern2 = []
        for i in range(h):
            for j in range(w):
                if (i + j) % 2 == 1:
                    pattern2.append(input_grid[i, j])
        
        pattern1_array = np.array(pattern1)
        pattern2_array = np.array(pattern2)
        output_flat = output_grid.flatten()
        
        return (np.array_equal(output_flat, pattern1_array[:output_flat.size]) or 
                np.array_equal(output_flat, pattern2_array[:output_flat.size]))
        return False
    
    def _check_padding(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Check if input is contained within output (with padding)"""
        in_h, in_w = input_grid.shape
        out_h, out_w = output_grid.shape
        
        for start_row in range(out_h - in_h + 1):
            for start_col in range(out_w - in_w + 1):
                region = output_grid[start_row:start_row + in_h, start_col:start_col + in_w]
                if np.array_equal(region, input_grid):
                    return True
        return False
    
    def _check_tiling(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Check if output is a tiled version of input"""
        in_h, in_w = input_grid.shape
        out_h, out_w = output_grid.shape
        
        tile_rows = out_h // in_h
        tile_cols = out_w // in_w
        
        for row in range(tile_rows):
            for col in range(tile_cols):
                start_row = row * in_h
                start_col = col * in_w
                tile = output_grid[start_row:start_row + in_h, start_col:start_col + in_w]
                if not np.array_equal(tile, input_grid):
                    return False
        return True


# --- Existing analyzers (unchanged) ---
class BasicShapeAnalyzer(AbstractFeatureExtractor):
    def analyze(self, input_grid, output_grid):
        return {
            'shapefeature': self.get_shape_features(input_grid, output_grid),
            'densityfeature': self.get_value_features(input_grid, output_grid)
        }

    def get_shape_features(self, input_grid, output_grid):
        return {
            'rows': input_grid.shape[0],
            'cols': input_grid.shape[1],
            'aspect_ratio': input_grid.shape[1] / input_grid.shape[0] if input_grid.shape[0] != 0 else 0,
            'total_cells': input_grid.size
        }

    def get_value_features(self, input_grid, output_grid):
        bg_color = get_background_color(input_grid)
        fg_count = np.sum(input_grid != bg_color)
        fg_ratio = fg_count / input_grid.size if input_grid.size else 0
        return {
            'foreground_ratio': fg_ratio
        }


class BasicPatternDetector(AbstractPatternDetector):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.min_repeats = 2

    def analyze(self, input_grid, output_grid):
        return {
            'patterns': {
                'repetition': self.detect_repetition(output_grid),
                'symmetry': self.detect_symmetry(output_grid)
            }
        }

    def detect_repetition(self, grid):
        return {
            'horizontal': self._check_horizontal_repetition(grid),
            'vertical': self._check_vertical_repetition(grid)
        }

    def _check_horizontal_repetition(self, grid):
        if grid.shape[1] < self.min_repeats:
            return False
        pattern_width = grid.shape[1] // self.min_repeats
        first_block = grid[:, :pattern_width]
        for i in range(1, self.min_repeats):
            block = grid[:, i*pattern_width:(i+1)*pattern_width]
            if not np.array_equal(first_block, block):
                return False
        return True

    def _check_vertical_repetition(self, grid):
        if grid.shape[0] < self.min_repeats:
            return False
        pattern_height = grid.shape[0] // self.min_repeats
        first_block = grid[:pattern_height, :]
        for i in range(1, self.min_repeats):
            block = grid[i*pattern_height:(i+1)*pattern_height, :]
            if not np.array_equal(first_block, block):
                return False
        return True

    def detect_symmetry(self, grid):
        return {
            'horizontal': np.array_equal(grid, grid[::-1, :]),
            'vertical': np.array_equal(grid, grid[:, ::-1]),
            'diagonal': np.array_equal(grid, grid.T)
        }


class ComponentFeatureExtractor(AbstractFeatureExtractor):
    def analyze(self, input_grid, output_grid):
        return {
            'componentstats': {
                'input': self.get_shape_features(input_grid, None),
                'output': self.get_shape_features(output_grid, None)
            }
        }

    def get_shape_features(self, grid, _):
        bg_color = get_background_color(grid)
        components = extract_components(grid, bg_color)

        if not components:
            return {
                'num_components': 0,
                'avg_area': 0,
                'avg_width': 0,
                'avg_height': 0
            }

        areas = [c['area'] for c in components]
        widths = [c['bbox'][2] for c in components]
        heights = [c['bbox'][3] for c in components]

        return {
            'num_components': len(components),
            'avg_area': float(np.mean(areas)),
            'avg_width': float(np.mean(widths)),
            'avg_height': float(np.mean(heights))
        }

    def get_value_features(self, input_grid, output_grid):
        return {}


class RelationalFeatureExtractor(AbstractFeatureExtractor):
    def analyze(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict:
        return {
            'relational_shape': self.get_relational_shape_features(input_grid, output_grid),
            'relational_color': self.get_relational_color_features(input_grid, output_grid),
            'relational_density': self.get_relational_density_features(input_grid, output_grid),
        }

    def get_shape_features(self, input_grid, output_grid):
        return self.get_relational_shape_features(input_grid, output_grid)

    def get_value_features(self, input_grid, output_grid):
        color_features = self.get_relational_color_features(input_grid, output_grid)
        density_features = self.get_relational_density_features(input_grid, output_grid)
        return {**color_features, **density_features}

    def get_relational_shape_features(self, input_grid, output_grid):
        row_ratio = output_grid.shape[0] / input_grid.shape[0] if input_grid.shape[0] != 0 else 0
        col_ratio = output_grid.shape[1] / input_grid.shape[1] if input_grid.shape[1] != 0 else 0
        return {
            'row_ratio': row_ratio,
            'col_ratio': col_ratio,
            'area_ratio': (output_grid.shape[0] * output_grid.shape[1]) / (input_grid.shape[0] * input_grid.shape[1]) if input_grid.size else 0
        }

    def get_relational_color_features(self, input_grid, output_grid):
        input_colors = set(np.unique(input_grid))
        output_colors = set(np.unique(output_grid))
        is_subset = output_colors.issubset(input_colors)
        is_superset = output_colors.issuperset(input_colors) and output_colors != input_colors
        intersection_size = len(input_colors.intersection(output_colors))
        return {
            'output_subset_of_input': is_subset,
            'output_superset_of_input': is_superset,
            'color_intersection_size': intersection_size,
            'input_color_count': len(input_colors),
            'output_color_count': len(output_colors)
        }

    def get_relational_density_features(self, input_grid, output_grid):
        input_bg = get_background_color(input_grid)
        output_bg = get_background_color(output_grid)
        input_foreground = np.sum(input_grid != input_bg)
        output_foreground = np.sum(output_grid != output_bg)
        input_size = input_grid.size
        output_size = output_grid.size
        input_density = input_foreground / input_size if input_size else 0
        output_density = output_foreground / output_size if output_size else 0
        ratio = output_density / input_density if input_density else 0
        return {
            'foreground_ratio_change': ratio
        }


# --- Enhanced Result container ---
class AnalysisResult:
    def __init__(self):
        self.features = {
            'shapefeature': {},
            'densityfeature': {},
            'patterns': {},
            'componentstats': {},
            'relational_shape': {},
            'relational_color': {},
            'relational_density': {},
            'transformations': {}  # New category for transformations
        }

    def add_features(self, feature_type: str, features: Dict):
        if feature_type not in self.features:
            self.features[feature_type] = {}
        self.features[feature_type].update(features)

    def to_dict(self) -> Dict:
        return {
            'features': self.features,
        }


# --- Enhanced Coordinator ---
class ProblemAnalyzer:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.analyzers = [
            BasicShapeAnalyzer(logger),
            BasicPatternDetector(logger),
            ComponentFeatureExtractor(logger),
            RelationalFeatureExtractor(logger),
            GenericTransformationDetector(logger),  # New analyzer
        ]
        self.result = AnalysisResult()

    def analyze(self, input_grid: np.ndarray, output_grid: np.ndarray) -> AnalysisResult:
        try:
            for analyzer in self.analyzers:
                self.logger.debug(f"Running analyzer: {analyzer.__class__.__name__}")
                features = analyzer.analyze(input_grid, output_grid)
                for k, v in features.items():
                    self.result.add_features(k, v)

            return self.result
        except Exception as e:
            self.logger.error(f"Error during problem analysis: {e}")
            raise
