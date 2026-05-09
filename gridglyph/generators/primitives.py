import numpy as np
import cv2
import random

def create_safe_base_grid(width: int, height: int, background: int = 0) -> np.ndarray:
    return np.full((height, width), background, dtype=np.uint8)

def add_random_shapes(grid: np.ndarray, num_shapes: int = 3):
    height, width = grid.shape
    for _ in range(num_shapes):
        color = random.randint(1, 9)
        shape_type = random.choice(['rect', 'triangle'])
        
        # Blinding the randint logic to prevent Error 500/Crash
        x1 = random.randint(0, max(0, width - 3))
        y1 = random.randint(0, max(0, height - 3))
        
        if shape_type == 'rect':
            x2 = random.randint(min(x1 + 2, width - 1), width - 1)
            y2 = random.randint(min(y1 + 2, height - 1), height - 1)
            cv2.rectangle(grid, (x1, y1), (x2, y2), color, -1)
        else:
            pts = np.array([
                [x1, y1],
                [random.randint(0, width-1), random.randint(0, height-1)],
                [random.randint(0, width-1), random.randint(0, height-1)]
            ], np.int32)
            cv2.fillPoly(grid, [pts.reshape((-1, 1, 2))], color)
    return grid