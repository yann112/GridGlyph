from abc import ABC, abstractmethod
import logging
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from itertools import permutations


# --- Utilities ---
def get_background_color(grid: np.ndarray,
                                         dominant_threshold_ratio: float = 0.60,
                                         min_grid_size_for_dominant: int = 16) -> int:
    """
    Determines the background color of an ARC-like grid using a refined heuristic.

    Args:
        grid (np.ndarray): The input grid (2D numpy array of integers).
        dominant_threshold_ratio (float): The minimum proportion (e.g., 0.60 for 60%)
                                          a color must represent to be considered dominant.
        min_grid_size_for_dominant (int): The minimum total number of cells in the grid
                                           for the dominant color heuristic to apply
                                           (e.g., 36 for a 6x6 grid).

    Returns:
        int: The identified background color. Returns -1 if no clear background
             is identified by the heuristics.
    """
    values, counts = np.unique(grid, return_counts=True)
    total_cells = grid.size

    if total_cells == 0:
        return None

    # Iapply dominant color heuristic for larger grids
    if total_cells >= min_grid_size_for_dominant:
        max_count_idx = np.argmax(counts)
        dominant_color = int(values[max_count_idx])
        dominant_proportion = counts[max_count_idx] / total_cells

        if dominant_proportion >= dominant_threshold_ratio:
            return dominant_color

    # 3. If neither rule applies, no clear background identified by this function
    print(f"Warning: No clear background color identified for grid with size {grid.shape}. "
          f"Values: {values}, Counts: {counts}")
    return None


def extract_components(grid: np.ndarray, background_color: int | None): # Add None to type hint
    """
    Extracts connected components from a grid, excluding the background color.

    Args:
        grid (np.ndarray): The input grid (2D numpy array of integers).
        background_color (int | None): The identified background color. If None,
                                        the entire grid is considered a single component
                                        (assuming no explicit background).

    Returns:
        list: A list of dictionaries, each describing a component.
    """
    if background_color is None:
        # If no specific background color is found, treat the entire grid as one component
        # This assumes all cells are effectively foreground or part of the "main" component.
        h, w = grid.shape
        full_mask = np.ones_like(grid, dtype=bool) # All cells are part of the component
        
        # Create a single component representing the whole grid
        components = [{
            "bbox": (0, 0, w, h), # Bounding box covers the whole grid
            "area": w * h,        # Area is total cells
            "mask": full_mask,
            "centroid": (w // 2, h // 2)
        }]
        return components

    # Original logic for when a background_color is provided
    # Ensure mask is created with 255 for cv2 functions
    mask = (grid != background_color).astype(np.uint8) * 255

    # Check if the mask is entirely zero (no foreground elements)
    if not np.any(mask):
        return [] # No foreground components found

    # Ensure mask is contiguous for findContours if it's not already
    mask = np.ascontiguousarray(mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    components = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # Ensure area is non-zero (empty contours can sometimes be found)
        if area == 0 and w == 0 and h == 0:
            continue

        component_mask = np.zeros_like(mask, dtype=np.uint8) # Use uint8 for cv2.drawContours
        cv2.drawContours(component_mask, [contour], -1, 255, thickness=cv2.FILLED)
        centroid = (x + w // 2, y + h // 2)

        components.append({
            "bbox": (x, y, w, h),
            "area": area,
            "mask": component_mask.astype(bool), # Convert back to bool if preferred for mask
            "centroid": centroid
        })
    return components


# --- Abstract base classes ---
class AbstractProblemAnalyzer(ABC):
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
    
    @abstractmethod
    def analyze(self, reference_grid: np.ndarray, compared_grid: np.ndarray) -> Dict:
        pass

    @abstractmethod
    def describe(self) -> Dict[str, str]:
        """Returns a dictionary mapping feature keys to descriptions."""
        pass


class AbstractFeatureExtractor(AbstractProblemAnalyzer):
    @abstractmethod
    def get_shape_features(self, reference_grid: np.ndarray, compared_grid: np.ndarray) -> Dict:
        pass

    @abstractmethod
    def get_value_features(self, reference_grid: np.ndarray, compared_grid: np.ndarray) -> Dict:
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
    def detect_transformations(self, reference_grid: np.ndarray, compared_grid: np.ndarray) -> Dict:
        pass


# --- New Transformation Detector ---
class GenericTransformationDetector(AbstractTransformationDetector):
    """Detects various types of transformations between referencegrid and compared grids"""
    def describe(self) -> Dict[str, Dict[str, str]]:
        return {
            'transformations': {
                # Size transformations
                'size_change': "True if referencegrid and compared grid have different dimensions",
                'scale_factor_row': "compared grid rows / referencegrid rows",
                'scale_factor_col': "compared grid cols / referencegrid cols",
                'cropping': "True if compared grid is cropped from referencegrid",
                'padding': "True if referencegrid is embedded within compared grid",
                'tiling': "True if compared grid is tiled version of referencegrid",

                # Row/column transformations
                'row_swap': "True if rows are swapped",
                'row_swap_indices': "List showing which row indices were swapped",
                'row_reverse': "True if rows are reversed",
                'row_shift': "True if rows are circularly shifted",
                'row_shift_amount': "Number of positions rows were shifted",
                'col_swap': "True if columns are swapped",
                'col_swap_indices': "List showing which column indices were swapped",
                'col_reverse': "True if columns are reversed",
                'col_shift': "True if columns are circularly shifted",
                'col_shift_amount': "Number of positions columns were shifted",

                # Geometric transformations
                'rotation_90': "True if compared grid is rotated 90 degrees",
                'rotation_180': "True if compared grid is rotated 180 degrees",
                'rotation_270': "True if compared grid is rotated 270 degrees",
                'transpose': "True if matrix is transposed",
                'anti_transpose': "True if matrix is anti-transposed",

                # Color transformations
                'color_mapping': "Dictionary mapping referencegrid color → compared grid color",
                'identical_structure': "True if only color changed, not structure",
                'color_filtering': "True if some colors were removed in compared grid",
                'filtered_colors': "List of colors present in compared grid but not in referencegrid",

                # Cross-shape transformations
                'cross_shape_row_extraction': "True if compared grid is extracted row(s) from referencegrid",
                'cross_shape_col_extraction': "True if compared grid is extracted column(s) from referencegrid",
                'cross_shape_diagonal_extraction': "True if compared grid is diagonal of referencegrid",
                'cross_shape_subgrid_transform': "True if compared grid matches subgrid of referencegrid (with transformation)",

                # Pattern extractions
                'diagonal_extraction': "True if compared grid contains diagonal elements of referencegrid",
                'border_extraction': "True if compared grid contains border elements of referencegrid",
                'center_extraction': "True if compared grid contains center region of referencegrid",
                'corners_extraction': "True if compared grid contains corner elements of referencegrid",
                'checkerboard_extraction': "True if compared grid follows checkerboard pattern from referencegrid",

                # Grid difference
                'binary_diff': "Boolean matrix showing where referencegrid and compared grid differ",
                'change_count': "Total number of differing cells",
                'changed_rows': "List of rows where changes occurred",
                'changed_columns': "List of columns where changes occurred",
                'value_mapping': "Map of referencegrid values to compared grid values",
                'value_substitutions': "List of (old_value, new_value) substitutions"
            }
        }

    def __init__(self, logger=None, max_permutation_size=6):
        super().__init__(logger)
        self.max_permutation_size = max_permutation_size
    
    def analyze(self, reference_grid: np.ndarray, compared_grid: np.ndarray) -> Dict:
        return {
            'transformations': self.detect_transformations(reference_grid, compared_grid)
        }
    
    def _compute_binary_difference(self, reference_grid: np.ndarray, compared_grid: np.ndarray) -> Dict:
        """
        Compute a boolean matrix indicating where the two grids differ.
        Only valid if both grids have the same shape.
        """
        assert reference_grid.shape == compared_grid.shape, "Grids must have the same shape"

        diff = reference_grid != compared_grid
        return {
            "binary_diff": diff.tolist(),
            "change_count": int(np.sum(diff)),
            "changed_rows": [int(i) for i, row in enumerate(diff) if any(row)],
            "changed_columns": [int(j) for j, col in enumerate(diff.T) if any(col)]
        }

    def _compute_grid_difference(self, reference_grid: np.ndarray, compared_grid: np.ndarray) -> Dict:
        """
        Compare two grids cell-by-cell and return a structured diff.
        Only valid for grids of the same shape.
        """
        assert reference_grid.shape == compared_grid.shape, "Grids must have the same shape"

        diff = {
            "changes": [],
            "change_count": 0,
            "value_mapping": {},
            "row_changes": {},  # row index → list of changed column indices
            "value_substitutions": []  # list of (old_value, new_value)
        }

        for i in range(reference_grid.shape[0]):
            for j in range(reference_grid.shape[1]):
                in_val = reference_grid[i, j]
                out_val = compared_grid[i, j]
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

    def detect_transformations(self, reference_grid: np.ndarray, compared_grid: np.ndarray) -> Dict:
        """Main method to detect all types of transformations"""
        transformations = {}
        
        # Always check size transformations first
        transformations.update(self._detect_size_transformations(reference_grid, compared_grid))
        
        # Structural transformations - check if grids have same shape OR if one is contained in the other
        if reference_grid.shape == compared_grid.shape:
            # Compute detailed grid difference
            transformations.update(self._compute_binary_difference(reference_grid, compared_grid))
            transformations["grid_diff"] = self._compute_grid_difference(reference_grid, compared_grid)
            # Same shape - check all structural transformations
            transformations.update(self._detect_row_transformations(reference_grid, compared_grid))
            transformations.update(self._detect_column_transformations(reference_grid, compared_grid))
            transformations.update(self._detect_geometric_transformations(reference_grid, compared_grid))
        else:
            # Different shapes - check if transformations apply to cropped/padded regions
            transformations.update(self._detect_cross_shape_transformations(reference_grid, compared_grid))
        
        # Color transformations work regardless of shape
        transformations.update(self._detect_color_transformations(reference_grid, compared_grid))
        
        # Pattern extraction transformations (e.g., extracting rows/columns)
        transformations.update(self._detect_extraction_transformations(reference_grid, compared_grid))
        
        return transformations
    
    def _detect_row_transformations(self, reference_grid: np.ndarray, compared_grid: np.ndarray) -> Dict:
        """Detect row-based transformations like swaps, reversals, shifts"""
        results = {
            'row_swap': False,
            'row_swap_indices': None,
            'row_reverse': False,
            'row_shift': False,
            'row_shift_amount': 0
        }
        
        rows, cols = reference_grid.shape
        
        # Check for row reversal (entire grid flipped vertically)
        if np.array_equal(compared_grid, reference_grid[::-1, :]):
            results['row_reverse'] = True
            return results
        
        # Check for row shifts (circular shift)
        for shift in range(1, rows):
            shifted = np.roll(reference_grid, shift, axis=0)
            if np.array_equal(compared_grid, shifted):
                results['row_shift'] = True
                results['row_shift_amount'] = shift
                return results
        
        # Check for row swaps/permutations (only for small grids to avoid performance issues)
        if rows <= self.max_permutation_size:
            referencegrid_rows = [reference_grid[i, :] for i in range(rows)]
            compared_grid_rows = [compared_grid[i, :] for i in range(rows)]
            
            # Find if compared grid rows are a permutation of referencegrid rows
            row_mapping = self._find_row_permutation(referencegrid_rows, compared_grid_rows)
            if row_mapping and row_mapping != list(range(rows)):
                results['row_swap'] = True
                results['row_swap_indices'] = row_mapping
        
        return results
    
    def _detect_column_transformations(self, reference_grid: np.ndarray, compared_grid: np.ndarray) -> Dict:
        """Detect column-based transformations"""
        results = {
            'col_swap': False,
            'col_swap_indices': None,
            'col_reverse': False,
            'col_shift': False,
            'col_shift_amount': 0
        }
        
        rows, cols = reference_grid.shape
        
        # Check for column reversal
        if np.array_equal(compared_grid, reference_grid[:, ::-1]):
            results['col_reverse'] = True
            return results
        
        # Check for column shifts
        for shift in range(1, cols):
            shifted = np.roll(reference_grid, shift, axis=1)
            if np.array_equal(compared_grid, shifted):
                results['col_shift'] = True
                results['col_shift_amount'] = shift
                return results
        
        # Check for column swaps
        if cols <= self.max_permutation_size:
            referencegrid_cols = [reference_grid[:, i] for i in range(cols)]
            comparedgrid_cols = [compared_grid[:, i] for i in range(cols)]
            
            col_mapping = self._find_row_permutation(referencegrid_cols, comparedgrid_cols)
            if col_mapping and col_mapping != list(range(cols)):
                results['col_swap'] = True
                results['col_swap_indices'] = col_mapping
        
        return results
    
    def _detect_geometric_transformations(self, reference_grid: np.ndarray, compared_grid: np.ndarray) -> Dict:
        """Detect geometric transformations like rotation, transposition"""
        results = {
            'rotation_90': False,
            'rotation_180': False,
            'rotation_270': False,
            'transpose': False,
            'anti_transpose': False
        }
        
        # Only check rotations if grid is square
        if reference_grid.shape[0] == reference_grid.shape[1]:
            # 90-degree rotation
            if np.array_equal(compared_grid, np.rot90(reference_grid, k=1)):
                results['rotation_90'] = True
            # 180-degree rotation
            elif np.array_equal(compared_grid, np.rot90(reference_grid, k=2)):
                results['rotation_180'] = True
            # 270-degree rotation
            elif np.array_equal(compared_grid, np.rot90(reference_grid, k=3)):
                results['rotation_270'] = True
        
        # Check for transpose (only if dimensions allow)
        if reference_grid.shape == compared_grid.shape[::-1]:
            if np.array_equal(compared_grid, reference_grid.T):
                results['transpose'] = True
            elif np.array_equal(compared_grid, np.rot90(reference_grid.T, k=2)):
                results['anti_transpose'] = True
        
        return results
    
    def _detect_color_transformations(self, reference_grid: np.ndarray, compared_grid: np.ndarray) -> Dict:
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
        if reference_grid.shape == compared_grid.shape:
            # Find color mapping
            unique_referencegrid = np.unique(reference_grid)
            unique_comparedgrid = np.unique(compared_grid)
            
            # Check if it's a simple color mapping
            color_map = {}
            for i_val in unique_referencegrid:
                referencegrid_positions = (reference_grid == i_val)
                comparedgrid_values_at_positions = compared_grid[referencegrid_positions]
                unique_comparedgrid_vals = np.unique(comparedgrid_values_at_positions)
                
                if len(unique_comparedgrid_vals) == 1:
                    color_map[int(i_val)] = int(unique_comparedgrid_vals[0])
                else:
                    # Not a simple mapping
                    color_map = {}
                    break
            
            if color_map and len(color_map) == len(unique_referencegrid):
                results['color_mapping'] = color_map
                
                # Check if structure is preserved
                reconstructed = np.zeros_like(reference_grid)
                for old_color, new_color in color_map.items():
                    reconstructed[reference_grid == old_color] = new_color
                
                if np.array_equal(reconstructed, compared_grid):
                    results['identical_structure'] = True
        
        # Check for color filtering (comparedgrid contains only certain colors from referencegrid)
        else:
            referencegrid_colors = set(np.unique(reference_grid))
            comparedgrid_colors = set(np.unique(compared_grid))
            
            if comparedgrid_colors.issubset(referencegrid_colors) and len(comparedgrid_colors) < len(referencegrid_colors):
                results['color_filtering'] = True
                results['filtered_colors'] = list(comparedgrid_colors)
        
        return results
    
    def _detect_size_transformations(self, reference_grid: np.ndarray, compared_grid: np.ndarray) -> Dict:
        """Detect size-related transformations"""
        results = {
            'size_change': reference_grid.shape != compared_grid.shape,
            'scale_factor_row': compared_grid.shape[0] / reference_grid.shape[0] if reference_grid.shape[0] > 0 else 0,
            'scale_factor_col': compared_grid.shape[1] / reference_grid.shape[1] if reference_grid.shape[1] > 0 else 0,
            'cropping': False,
            'padding': False,
            'tiling': False
        }
        
        if results['size_change']:
            # Check for cropping (comparedgrid smaller than referencegrid)
            if compared_grid.shape[0] <= reference_grid.shape[0] and compared_grid.shape[1] <= reference_grid.shape[1]:
                results['cropping'] = self._check_cropping(reference_grid, compared_grid)
            
            # Check for padding (comparedgrid larger than referencegrid)
            elif compared_grid.shape[0] >= reference_grid.shape[0] and compared_grid.shape[1] >= reference_grid.shape[1]:
                results['padding'] = self._check_padding(reference_grid, compared_grid)
            
            # Check for tiling (comparedgrid is multiple of referencegrid)
            if (compared_grid.shape[0] % reference_grid.shape[0] == 0 and 
                compared_grid.shape[1] % reference_grid.shape[1] == 0):
                results['tiling'] = self._check_tiling(reference_grid, compared_grid)
        
        return results
    
    def _find_row_permutation(self, referencegrid_rows: List[np.ndarray], comparedgrid_rows: List[np.ndarray]) -> Optional[List[int]]:
        """Find if comparedgrid_rows is a permutation of referencegrid_rows"""
        if len(referencegrid_rows) != len(comparedgrid_rows):
            return None
        
        # Try to find mapping
        mapping = []
        used_indices = set()
        
        for out_row in comparedgrid_rows:
            found = False
            for i, in_row in enumerate(referencegrid_rows):
                if i not in used_indices and np.array_equal(out_row, in_row):
                    mapping.append(i)
                    used_indices.add(i)
                    found = True
                    break
            if not found:
                return None
        
        return mapping if len(mapping) == len(referencegrid_rows) else None
    
    def _check_cropping(self, reference_grid: np.ndarray, compared_grid: np.ndarray) -> bool:
        """Check if comparedgrid is a crop of referencegrid"""
        in_h, in_w = reference_grid.shape
        out_h, out_w = compared_grid.shape
        
        # If comparedgrid is larger than referencegrid, it can't be a crop
        if out_h > in_h or out_w > in_w:
            return False
        
        for start_row in range(in_h - out_h + 1):
            for start_col in range(in_w - out_w + 1):
                crop = reference_grid[start_row:start_row + out_h, start_col:start_col + out_w]
                if np.array_equal(crop, compared_grid):
                    return True
        
        return False
    
    def _detect_cross_shape_transformations(self, reference_grid: np.ndarray, compared_grid: np.ndarray) -> Dict:
        """Detect transformations between grids of different shapes"""
        results = {
            'cross_shape_row_extraction': False,
            'cross_shape_col_extraction': False,
            'cross_shape_diagonal_extraction': False,
            'cross_shape_subgrid_transform': False,
            'extracted_indices': None,
            'subgrid_location': None
        }
        
        # Check if comparedgrid is a single row from referencegrid
        if compared_grid.shape[0] == 1 and compared_grid.shape[1] == reference_grid.shape[1]:
            for i in range(reference_grid.shape[0]):
                if np.array_equal(compared_grid[0, :], reference_grid[i, :]):
                    results['cross_shape_row_extraction'] = True
                    results['extracted_indices'] = i
                    return results
        
        # Check if comparedgrid is a single column from referencegrid
        if compared_grid.shape[1] == 1 and compared_grid.shape[0] == reference_grid.shape[0]:
            for j in range(reference_grid.shape[1]):
                if np.array_equal(compared_grid[:, 0], reference_grid[:, j]):
                    results['cross_shape_col_extraction'] = True
                    results['extracted_indices'] = j
                    return results
        
        # Check if comparedgrid is multiple rows from referencegrid
        if compared_grid.shape[1] == reference_grid.shape[1] and compared_grid.shape[0] < reference_grid.shape[0]:
            # Try to find consecutive rows
            for start_row in range(reference_grid.shape[0] - compared_grid.shape[0] + 1):
                if np.array_equal(compared_grid, reference_grid[start_row:start_row + compared_grid.shape[0], :]):
                    results['cross_shape_row_extraction'] = True
                    results['extracted_indices'] = list(range(start_row, start_row + compared_grid.shape[0]))
                    return results
            
            # Try to find non-consecutive rows
            row_indices = self._find_matching_rows(reference_grid, compared_grid)
            if row_indices:
                results['cross_shape_row_extraction'] = True
                results['extracted_indices'] = row_indices
                return results
        
        # Check if comparedgrid is multiple columns from referencegrid
        if compared_grid.shape[0] == reference_grid.shape[0] and compared_grid.shape[1] < reference_grid.shape[1]:
            # Try consecutive columns
            for start_col in range(reference_grid.shape[1] - compared_grid.shape[1] + 1):
                if np.array_equal(compared_grid, reference_grid[:, start_col:start_col + compared_grid.shape[1]]):
                    results['cross_shape_col_extraction'] = True
                    results['extracted_indices'] = list(range(start_col, start_col + compared_grid.shape[1]))
                    return results
            
            # Try non-consecutive columns
            col_indices = self._find_matching_columns(reference_grid, compared_grid)
            if col_indices:
                results['cross_shape_col_extraction'] = True
                results['extracted_indices'] = col_indices
                return results
        
        # Check if comparedgrid is a transformed subgrid of referencegrid
        subgrid_result = self._check_subgrid_transformations(reference_grid, compared_grid)
        if subgrid_result['found']:
            results['cross_shape_subgrid_transform'] = True
            results['subgrid_location'] = subgrid_result
        
        return results
    
    def _detect_extraction_transformations(self, reference_grid: np.ndarray, compared_grid: np.ndarray) -> Dict:
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
        if self._check_diagonal_extraction(reference_grid, compared_grid):
            results['diagonal_extraction'] = True
            results['pattern_type'] = 'diagonal'
        
        # Check border extraction
        elif self._check_border_extraction(reference_grid, compared_grid):
            results['border_extraction'] = True
            results['pattern_type'] = 'border'
        
        # Check center extraction
        elif self._check_center_extraction(reference_grid, compared_grid):
            results['center_extraction'] = True
            results['pattern_type'] = 'center'
        
        # Check corners extraction
        elif self._check_corners_extraction(reference_grid, compared_grid):
            results['corners_extraction'] = True
            results['pattern_type'] = 'corners'
        
        # Check checkerboard pattern
        elif self._check_checkerboard_extraction(reference_grid, compared_grid):
            results['checkerboard_extraction'] = True
            results['pattern_type'] = 'checkerboard'
        
        return results
    
    def _find_matching_rows(self, reference_grid: np.ndarray, compared_grid: np.ndarray) -> Optional[List[int]]:
        """Find which referencegrid rows match the comparedgrid rows in order"""
        matching_indices = []
        used_rows = set()
        
        for out_row_idx in range(compared_grid.shape[0]):
            found = False
            for in_row_idx in range(reference_grid.shape[0]):
                if in_row_idx not in used_rows and np.array_equal(compared_grid[out_row_idx, :], reference_grid[in_row_idx, :]):
                    matching_indices.append(in_row_idx)
                    used_rows.add(in_row_idx)
                    found = True
                    break
            if not found:
                return None
        
        return matching_indices if len(matching_indices) == compared_grid.shape[0] else None
    
    def _find_matching_columns(self, reference_grid: np.ndarray, compared_grid: np.ndarray) -> Optional[List[int]]:
        """Find which referencegrid columns match the comparedgrid columns in order"""
        matching_indices = []
        used_cols = set()
        
        for out_col_idx in range(compared_grid.shape[1]):
            found = False
            for in_col_idx in range(reference_grid.shape[1]):
                if in_col_idx not in used_cols and np.array_equal(compared_grid[:, out_col_idx], reference_grid[:, in_col_idx]):
                    matching_indices.append(in_col_idx)
                    used_cols.add(in_col_idx)
                    found = True
                    break
            if not found:
                return None
        
        return matching_indices if len(matching_indices) == compared_grid.shape[1] else None
    
    def _check_subgrid_transformations(self, reference_grid: np.ndarray, compared_grid: np.ndarray) -> Dict:
        """Check if comparedgrid is a transformed version of a subgrid from referencegrid"""
        result = {'found': False, 'location': None, 'transformation': None}
        
        in_h, in_w = reference_grid.shape
        out_h, out_w = compared_grid.shape
        
        # Try all possible subgrid locations
        for start_row in range(in_h - out_h + 1):
            for start_col in range(in_w - out_w + 1):
                subgrid = reference_grid[start_row:start_row + out_h, start_col:start_col + out_w]
                
                # Check direct match
                if np.array_equal(subgrid, compared_grid):
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
                        if np.array_equal(rotated, compared_grid):
                            result = {
                                'found': True,
                                'location': (start_row, start_col, out_h, out_w),
                                'transformation': f'rotation_{90*k}'
                            }
                            return result
                
                # Check flips
                if np.array_equal(subgrid[::-1, :], compared_grid):
                    result = {
                        'found': True,
                        'location': (start_row, start_col, out_h, out_w),
                        'transformation': 'vertical_flip'
                    }
                    return result
                
                if np.array_equal(subgrid[:, ::-1], compared_grid):
                    result = {
                        'found': True,
                        'location': (start_row, start_col, out_h, out_w),
                        'transformation': 'horizontal_flip'
                    }
                    return result
        
        return result
    
    def _check_diagonal_extraction(self, reference_grid: np.ndarray, compared_grid: np.ndarray) -> bool:
        """Check if comparedgrid is diagonal elements from referencegrid"""
        min_dim = min(reference_grid.shape)
        
        # Main diagonal
        if compared_grid.size == min_dim:
            main_diag = np.array([reference_grid[i, i] for i in range(min_dim)])
            if np.array_equal(compared_grid.flatten(), main_diag):
                return True
        
        # Anti-diagonal
        if compared_grid.size == min_dim:
            anti_diag = np.array([reference_grid[i, reference_grid.shape[1]-1-i] for i in range(min_dim)])
            if np.array_equal(compared_grid.flatten(), anti_diag):
                return True
        
        return False
    
    def _check_border_extraction(self, reference_grid: np.ndarray, compared_grid: np.ndarray) -> bool:
        """Check if comparedgrid contains border elements from referencegrid"""
        h, w = reference_grid.shape
        border_elements = []
        
        # Top and bottom rows
        border_elements.extend(reference_grid[0, :])
        if h > 1:
            border_elements.extend(reference_grid[-1, :])
        
        # Left and right columns (excluding corners already added)
        for i in range(1, h-1):
            border_elements.append(reference_grid[i, 0])
            if w > 1:
                border_elements.append(reference_grid[i, -1])
        
        border_array = np.array(border_elements)
        return np.array_equal(compared_grid.flatten(), border_array) or np.array_equal(compared_grid.flatten(), border_array[:compared_grid.size])
    
    def _check_center_extraction(self, reference_grid: np.ndarray, compared_grid: np.ndarray) -> bool:
        """Check if comparedgrid is center region of referencegrid"""
        h, w = reference_grid.shape
        out_h, out_w = compared_grid.shape
        
        if out_h >= h or out_w >= w:
            return False
        
        start_row = (h - out_h) // 2
        start_col = (w - out_w) // 2
        center_region = reference_grid[start_row:start_row + out_h, start_col:start_col + out_w]
        
        return np.array_equal(center_region, compared_grid)
    
    def _check_corners_extraction(self, reference_grid: np.ndarray, compared_grid: np.ndarray) -> bool:
        """Check if comparedgrid contains corner elements from referencegrid"""
        h, w = reference_grid.shape
        corners = [
            reference_grid[0, 0],      # top-left
            reference_grid[0, -1],     # top-right
            reference_grid[-1, 0],     # bottom-left
            reference_grid[-1, -1]     # bottom-right
        ]
        
        corners_array = np.array(corners)
        return np.array_equal(compared_grid.flatten(), corners_array) or np.array_equal(compared_grid.flatten(), corners_array[:compared_grid.size])
    
    def _check_checkerboard_extraction(self, reference_grid: np.ndarray, compared_grid: np.ndarray) -> bool:
        """Check if comparedgrid follows a checkerboard pattern from referencegrid"""
        h, w = reference_grid.shape
        
        # Extract checkerboard pattern starting with (0,0)
        pattern1 = []
        for i in range(h):
            for j in range(w):
                if (i + j) % 2 == 0:
                    pattern1.append(reference_grid[i, j])
        
        # Extract checkerboard pattern starting with (0,1)
        pattern2 = []
        for i in range(h):
            for j in range(w):
                if (i + j) % 2 == 1:
                    pattern2.append(reference_grid[i, j])
        
        pattern1_array = np.array(pattern1)
        pattern2_array = np.array(pattern2)
        comparedgrid_flat = compared_grid.flatten()
        
        return (np.array_equal(comparedgrid_flat, pattern1_array[:comparedgrid_flat.size]) or 
                np.array_equal(comparedgrid_flat, pattern2_array[:comparedgrid_flat.size]))
        return False
    
    def _check_padding(self, reference_grid: np.ndarray, compared_grid: np.ndarray) -> bool:
        """Check if referencegrid is contained within comparedgrid (with padding)"""
        in_h, in_w = reference_grid.shape
        out_h, out_w = compared_grid.shape
        
        for start_row in range(out_h - in_h + 1):
            for start_col in range(out_w - in_w + 1):
                region = compared_grid[start_row:start_row + in_h, start_col:start_col + in_w]
                if np.array_equal(region, reference_grid):
                    return True
        return False
    
    def _check_tiling(self, reference_grid: np.ndarray, compared_grid: np.ndarray) -> bool:
        """Check if comparedgrid is a tiled version of referencegrid"""
        in_h, in_w = reference_grid.shape
        out_h, out_w = compared_grid.shape
        
        tile_rows = out_h // in_h
        tile_cols = out_w // in_w
        
        for row in range(tile_rows):
            for col in range(tile_cols):
                start_row = row * in_h
                start_col = col * in_w
                tile = compared_grid[start_row:start_row + in_h, start_col:start_col + in_w]
                if not np.array_equal(tile, reference_grid):
                    return False
        return True


# --- Existing analyzers (unchanged) ---
class BasicShapeAnalyzer(AbstractFeatureExtractor):
    def analyze(self, reference_grid, compared_grid):
        return {
            'shapefeature': self.get_shape_features(reference_grid, compared_grid),
            'densityfeature': self.get_value_features(reference_grid)
        }

    def get_shape_features(self, reference_grid, compared_grid):
        return {
            'rows': reference_grid.shape[0],
            'cols': reference_grid.shape[1],
            'aspect_ratio': reference_grid.shape[1] / reference_grid.shape[0] if reference_grid.shape[0] != 0 else 0,
            'total_cells': reference_grid.size
        }

    def get_value_features(self, reference_grid):
        bg_color = get_background_color(reference_grid)
        if not bg_color:
            return {
            'foreground_ratio': 1
        }
        fg_count = np.sum(reference_grid != bg_color)
        fg_ratio = fg_count / reference_grid.size if reference_grid.size else 0
        return {
            'foreground_ratio': fg_ratio
        }

    def describe(self) -> Dict[str, str]:
        return {
            'shapefeature': {
                'rows': "Number of rows in the reference grid.",
                'cols': "Number of columns in the reference grid.",
                'aspect_ratio': "Ratio of width to height (cols/rows).",
                'total_cells': "Total number of cells in the reference grid."
            },
            'densityfeature': {
                'foreground_ratio': "Proportion of non-background pixels in the reference grid."
            }
        }

class BasicPatternDetector(AbstractPatternDetector):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.min_repeats = 2

    def analyze(self, reference_grid, compared_grid):
        return {
            'patterns': {
                'repetition': self.detect_repetition(compared_grid),
                'symmetry': self.detect_symmetry(compared_grid)
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

    def describe(self) -> Dict[str, Dict[str, str]]:
        return {
            'patterns': {
                'repetition': {
                    'horizontal': "True if comparedgrid contains repeated horizontal pattern",
                    'vertical': "True if comparedgrid contains repeated vertical pattern"
                },
                'symmetry': {
                    'horizontal': "True if comparedgrid is horizontally symmetric",
                    'vertical': "True if comparedgrid is vertically symmetric",
                    'diagonal': "True if comparedgrid is diagonally symmetric"
                }
            }
        }
         
class ComponentFeatureExtractor(AbstractFeatureExtractor):
    def describe(self) -> Dict[str, Dict[str, str]]:
        return {
            'componentstats': {
                'num_components': "Number of distinct foreground components in the grid",
                'avg_area': "Average area (in pixels) of all components",
                'avg_width': "Average width of bounding boxes of components",
                'avg_height': "Average height of bounding boxes of components"
            }
        }

    def analyze(self, reference_grid, compared_grid):
        return {
            'componentstats': {
                'referencegrid': self.get_shape_features(reference_grid, None),
                'comparedgrid': self.get_shape_features(compared_grid, None)
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

    def get_value_features(self, reference_grid, compared_grid):
        return {}


class RelationalFeatureExtractor(AbstractFeatureExtractor):

    def describe(self) -> Dict[str, Dict[str, str]]:
        return {
            'relational_shape': {
                'row_ratio': "Ratio of comparedgrid rows to referencegrid rows",
                'col_ratio': "Ratio of comparedgrid columns to referencegrid columns",
                'area_ratio': "Ratio of total cell count (comparedgrid/referencegrid)"
            },
            'relational_color': {
                'comparedgrid_subset_of_referencegrid': "True if comparedgrid uses only colors from referencegrid",
                'comparedgrid_superset_of_referencegrid': "True if comparedgrid introduces new colors",
                'color_intersection_size': "Number of shared colors between referencegrid and comparedgrid",
                'referencegrid_color_count': "Number of unique colors in referencegrid",
                'comparedgrid_color_count': "Number of unique colors in comparedgrid"
            },
            'relational_density': {
                'foreground_ratio_change': "Ratio of foreground density change (comparedgrid/referencegrid)"
            }
        }

    def analyze(self, reference_grid: np.ndarray, compared_grid: np.ndarray) -> Dict:
        return {
            'relational_shape': self.get_relational_shape_features(reference_grid, compared_grid),
            'relational_color': self.get_relational_color_features(reference_grid, compared_grid),
            'relational_density': self.get_relational_density_features(reference_grid, compared_grid),
        }

    def get_shape_features(self, reference_grid, compared_grid):
        return self.get_relational_shape_features(reference_grid, compared_grid)

    def get_value_features(self, reference_grid, compared_grid):
        color_features = self.get_relational_color_features(reference_grid, compared_grid)
        density_features = self.get_relational_density_features(reference_grid, compared_grid)
        return {**color_features, **density_features}

    def get_relational_shape_features(self, reference_grid, compared_grid):
        row_ratio = compared_grid.shape[0] / reference_grid.shape[0] if reference_grid.shape[0] != 0 else 0
        col_ratio = compared_grid.shape[1] / reference_grid.shape[1] if reference_grid.shape[1] != 0 else 0
        return {
            'row_ratio': row_ratio,
            'col_ratio': col_ratio,
            'area_ratio': (compared_grid.shape[0] * compared_grid.shape[1]) / (reference_grid.shape[0] * reference_grid.shape[1]) if reference_grid.size else 0
        }

    def get_relational_color_features(self, reference_grid, compared_grid):
        referencegrid_colors = set(np.unique(reference_grid))
        comparedgrid_colors = set(np.unique(compared_grid))
        is_subset = comparedgrid_colors.issubset(referencegrid_colors)
        is_superset = comparedgrid_colors.issuperset(referencegrid_colors) and comparedgrid_colors != referencegrid_colors
        intersection_size = len(referencegrid_colors.intersection(comparedgrid_colors))
        return {
            'comparedgrid_subset_of_referencegrid': is_subset,
            'comparedgrid_superset_of_referencegrid': is_superset,
            'color_intersection_size': intersection_size,
            'referencegrid_color_count': len(referencegrid_colors),
            'comparedgrid_color_count': len(comparedgrid_colors)
        }

    def get_relational_density_features(self, reference_grid, compared_grid):
        referencegrid_bg = get_background_color(reference_grid)
        comparedgrid_bg = get_background_color(compared_grid)
        referencegrid_foreground = np.sum(reference_grid != referencegrid_bg)
        comparedgrid_foreground = np.sum(compared_grid != comparedgrid_bg)
        referencegrid_size = reference_grid.size
        comparedgrid_size = compared_grid.size
        referencegrid_density = referencegrid_foreground / referencegrid_size if referencegrid_size else 0
        comparedgrid_density = comparedgrid_foreground / comparedgrid_size if comparedgrid_size else 0
        ratio = comparedgrid_density / referencegrid_density if referencegrid_density else 0
        return {
            'foreground_ratio_change': ratio
        }


# --- Enhanced Result container ---
class AnalysisResult:
    def __init__(self):
        self.features = {
            'description': {},
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
    # Define which analyzers are used in each mode
    ANALYZER_MAP = {
        'io': [
            BasicShapeAnalyzer,
            BasicPatternDetector,
            ComponentFeatureExtractor,
            RelationalFeatureExtractor,
            GenericTransformationDetector,
        ],
        'ii': [
            BasicShapeAnalyzer,
            BasicPatternDetector,
            ComponentFeatureExtractor,
            RelationalFeatureExtractor,
        ],
        'oo': [
            BasicShapeAnalyzer,
            BasicPatternDetector,
            ComponentFeatureExtractor,
            RelationalFeatureExtractor,
        ]
    }

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.result = AnalysisResult()

    def analyze(self, grid1: np.ndarray, grid2: np.ndarray, mode: str) -> AnalysisResult:
        """Analyze two grids according to the specified mode."""
        assert mode in self.ANALYZER_MAP, f"Invalid mode: {mode}. Valid modes: {list(self.ANALYZER_MAP.keys())}"

        try:
            # Reset result for fresh analysis
            self.result = AnalysisResult()

            # Instantiate analyzers based on current mode
            analyzers = [cls(self.logger) for cls in self.ANALYZER_MAP[mode]]

            for analyzer in analyzers:
                self.logger.debug(f"Running analyzer: {analyzer.__class__.__name__}")
                features = analyzer.analyze(grid1, grid2)
                descriptions = analyzer.describe()
                self.result.features['description'].update(descriptions)
                for feature_type, feature_data in features.items():
                    
                    self.result.add_features(feature_type, feature_data)
                    
            return self.result

        except Exception as e:
            self.logger.error(f"Error during problem analysis: {e}")
            raise