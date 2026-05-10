import numpy as np
import random
import re
import numpy as np
import random
import cv2

def int_to_roman(num: int) -> str:
    mapping = {
        0: "∅", 1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V',
        6: 'VI', 7: 'VII', 8: 'VIII', 9: 'IX', 10: 'X',
        11: 'XI', 12: 'XII', 13: 'XIII', 14: 'XIV', 15: 'XV',
        16: 'XVI', 17: 'XVII', 18: 'XVIII', 19: 'XIX', 20: 'XX',
        21: 'XXI', 22: 'XXII', 23: 'XXIII', 24: 'XXIV', 25: 'XXV',
        26: 'XXVI', 27: 'XXVII', 28: 'XXVIII', 29: 'XXIX', 30: 'XXX'
    }
    return mapping.get(num, "∅")

def roman_to_int(roman_numeral: str) -> int:
    mapping = {
        '∅': 0, 'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
        'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10,
        'XI': 11, 'XII': 12, 'XIII': 13, 'XIV': 14, 'XV': 15,
        'XVI': 16, 'XVII': 17, 'XVIII': 18, 'XIX': 19, 'XX': 20,
        'XXI': 21, 'XXII': 22, 'XXIII': 23, 'XXIV': 24, 'XXV': 25,
        'XXVI': 26, 'XXVII': 27, 'XXVIII': 28, 'XXIX': 29, 'XXX': 30
    }
    return mapping.get(roman_numeral.upper())

def generate_single_random_grid(shape=(10, 10), max_dim=30, num_range=9):
    rows, cols = shape
    rows = min(max(1, rows), max_dim)
    cols = min(max(1, cols), max_dim)
    return np.random.randint(0, num_range + 1, size=(rows, cols), dtype=np.uint8)

def is_rule_compatible_with_grid(dsl_rule_str: str, grid_shape: tuple) -> bool:
    rows, cols = grid_shape
    if rows < 1 or cols < 1: return False
    if dsl_rule_str.startswith('⇅('):
        match = re.match(r'⇅\(([IVX]+),([IVX]+)\)', dsl_rule_str)
        if not match: return False
        row_str1, row_str2 = match.groups()
        idx1, idx2 = roman_to_int(row_str1), roman_to_int(row_str2)
        return idx1 is not None and idx2 is not None and idx1 <= rows and idx2 <= rows
    return True

def create_base_grid(height: int, width: int, background_value: int = 0) -> np.ndarray:
    """Initializes a grid with a uniform background color."""
    return np.full((height, width), background_value, dtype=np.uint8)

def draw_random_shapes(grid: np.ndarray, num_shapes: int = 3, value_range: tuple[int, int] = (1, 9), fill_probability: float = 0.7) -> np.ndarray:
    """
    Uses OpenCV to draw circles, rectangles, or triangles on the grid.
    This provides the visual 'objects' for symbolic rules to act upon.
    """
    height, width = grid.shape
    for _ in range(num_shapes):
        # Safety for very small grids where shapes might not fit
        if height < 3 or width < 3:
            if random.random() < 0.3:
                 r = random.randint(0, height - 1)
                 c = random.randint(0, width - 1)
                 grid[r, c] = random.randint(*value_range)
            continue

        color = random.randint(*value_range)
        shape_type = random.choice(['circle', 'rectangle', 'triangle'])
        is_filled = random.random() < fill_probability
        
        # In OpenCV, thickness -1 means the shape is filled
        thickness = -1 if is_filled else random.randint(1, 2)
        
        try:
            if shape_type == 'circle':
                radius = random.randint(1, max(1, min(height, width) // 4))
                center_x = random.randint(radius, max(radius, width - radius - 1))
                center_y = random.randint(radius, max(radius, height - radius - 1))
                cv2.circle(grid, (center_x, center_y), radius, color, thickness)
                
            elif shape_type == 'rectangle':
                x1 = random.randint(0, max(0, width - 3))
                y1 = random.randint(0, max(0, height - 3))
                x2 = random.randint(x1 + 2, width - 1)
                y2 = random.randint(y1 + 2, height - 1)
                cv2.rectangle(grid, (x1, y1), (x2, y2), color, thickness)
                
            elif shape_type == 'triangle':
                pts = np.array([
                    [random.randint(0, width - 1), random.randint(0, height - 1)],
                    [random.randint(0, width - 1), random.randint(0, height - 1)],
                    [random.randint(0, width - 1), random.randint(0, height - 1)]
                ], np.int32).reshape((-1, 1, 2))
                
                if is_filled:
                    cv2.fillPoly(grid, [pts], color)
                else:
                    cv2.polylines(grid, [pts], isClosed=True, color=color, thickness=max(1, thickness))
        except Exception:
            # Silently skip if coordinates go out of bounds
            pass
            
    return grid

def generate_random_shape_grid(min_dim: int = 5, max_dim: int = 30, value_range: tuple[int, int] = (1, 9), 
                               num_shapes: int = 5, fill_prob: float = 0.6, background_value: int = 0) -> np.ndarray:
    """
    Main entry point to generate a complete visual scene.
    """
    height = random.randint(min_dim, max_dim)
    width = random.randint(min_dim, max_dim)
    
    grid = create_base_grid(height, width, background_value)
    grid = draw_random_shapes(grid, num_shapes=num_shapes, value_range=value_range, fill_probability=fill_prob)
    
    return grid