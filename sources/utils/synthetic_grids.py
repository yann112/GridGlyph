import random
import cv2
import numpy as np


def generate_single_random_grid(original_grid_shape: tuple, max_dim: int = 30, num_range: int = 9) -> np.ndarray:
    from utils.synthetic_grids import generate_random_shape_grid

    rows_orig, cols_orig = original_grid_shape if original_grid_shape and len(original_grid_shape) == 2 else (0, 0)

    use_shapes = random.random() < 0.3

    if use_shapes:
        try:
            if rows_orig > 0 and cols_orig > 0 and random.random() < 0.5:
                 new_rows = max(1, rows_orig + random.randint(-2, 2))
                 new_cols = max(1, cols_orig + random.randint(-2, 2))
            else:
                 new_rows = random.randint(1, max_dim + 1)
                 new_cols = random.randint(1, max_dim + 1)

            new_grid = generate_random_shape_grid(
                min_dim=min(new_rows, new_cols, 3),
                max_dim=max(new_rows, new_cols),
                value_range=(0, max(0, num_range)),
                num_shapes=random.randint(1, max(1, min(new_rows, new_cols) // 2)),
                fill_prob=random.uniform(0.3, 0.7),
                background_value=random.randint(0, num_range)
            )
            
            if new_grid.shape[0] > new_rows:
                new_grid = new_grid[:new_rows, :]
            if new_grid.shape[1] > new_cols:
                new_grid = new_grid[:, :new_cols]
                
        except Exception:
            new_grid = _generate_pure_random_grid(new_rows, new_cols, num_range)

    else:
        if random.random() < 0.7:
            new_rows = random.randint(1, max_dim + 1)
            new_cols = random.randint(1, max_dim + 1)
            new_values = np.random.choice(range(0, num_range + 1), size=new_rows * new_cols, replace=True)
            new_grid = new_values.reshape(new_rows, new_cols)
        else:
            if rows_orig > 0 and cols_orig > 0:
                max_val = num_range
                attempts = 0
                while attempts < 10:
                    new_values = np.random.choice(range(0, num_range + 1), size=rows_orig * cols_orig, replace=True)
                    temp_grid = new_values.reshape(rows_orig, cols_orig)
                    if not np.array_equal(temp_grid, np.zeros((rows_orig, cols_orig))) or not np.array_equal(temp_grid, np.full((rows_orig, cols_orig), max_val)):
                        new_grid = temp_grid
                        break
                    attempts += 1
                else:
                    new_grid = temp_grid if 'temp_grid' in locals() else np.array([[random.randint(0, num_range)]])
            else:
                new_grid = np.array([[random.randint(0, num_range)]])

    return new_grid


def _generate_pure_random_grid(rows: int, cols: int, num_range: int) -> np.ndarray:
    new_values = np.random.choice(range(0, num_range + 1), size=rows * cols, replace=True)
    return new_values.reshape(rows, cols)


def generate_single_random_grid(original_grid_shape: tuple, max_dim: int = 30, num_range: int = 9) -> np.ndarray:
    rows_orig, cols_orig = original_grid_shape if original_grid_shape and len(original_grid_shape) == 2 else (0,0)
    if random.random() < 0.7:
        new_rows = random.randint(1, max_dim + 1)
        new_cols = random.randint(1, max_dim + 1)
        new_values = np.random.choice(range(1, num_range + 1), size=new_rows * new_cols, replace=True)
        new_grid = new_values.reshape(new_rows, new_cols)
    else:
        if rows_orig > 0 and cols_orig > 0:
            while True:
                new_values = np.random.choice(range(1, num_range + 1), size=rows_orig * cols_orig, replace=True)
                temp_grid = new_values.reshape(rows_orig, cols_orig)
                if not np.array_equal(temp_grid, np.zeros((rows_orig, cols_orig))) or np.array_equal(temp_grid, np.full((rows_orig, cols_orig), num_range)):
                    new_grid = temp_grid
                    break
        else:
            new_grid = np.array([[random.randint(1, num_range)]])
    return new_grid

def create_base_grid(height: int, width: int, background_value: int = 0) -> np.ndarray:
    return np.full((height, width), background_value, dtype=np.uint8)

def draw_random_shapes(grid: np.ndarray, num_shapes: int = 3, value_range: tuple[int, int] = (1, 9), fill_probability: float = 0.7) -> np.ndarray:
    height, width = grid.shape
    for _ in range(num_shapes):
        if height < 3 or width < 3:
            if random.random() < 0.3:
                 r = random.randint(0, height - 1)
                 c = random.randint(0, width - 1)
                 grid[r, c] = random.randint(*value_range)
            continue
        color = random.randint(*value_range)
        shape_type = random.choice(['circle', 'rectangle', 'triangle'])
        is_filled = random.random() < fill_probability
        thickness = -1 if is_filled else random.randint(1, 2)
        try:
            if shape_type == 'circle':
                radius = random.randint(1, max(1, min(height, width) // 4))
                center_x = random.randint(radius, max(radius, width - radius - 1))
                center_y = random.randint(radius, max(radius, height - radius - 1))
                center = (center_x, center_y)
                cv2.circle(grid, center, radius, color, thickness)
            elif shape_type == 'rectangle':
                x1 = random.randint(0, max(0, width - 3))
                y1 = random.randint(0, max(0, height - 3))
                x2 = random.randint(min(x1 + 2, width - 1), width - 1)
                y2 = random.randint(min(y1 + 2, height - 1), height - 1)
                if x1 < x2 and y1 < y2:
                    pt1 = (x1, y1)
                    pt2 = (x2, y2)
                    cv2.rectangle(grid, pt1, pt2, color, thickness)
            elif shape_type == 'triangle':
                pts = np.array([
                    [random.randint(0, width - 1), random.randint(0, height - 1)],
                    [random.randint(0, width - 1), random.randint(0, height - 1)],
                    [random.randint(0, width - 1), random.randint(0, height - 1)]
                ], np.int32)
                pts = pts.reshape((-1, 1, 2))
                if is_filled:
                    cv2.fillPoly(grid, [pts], color)
                else:
                    line_thickness = max(1, min(thickness, 2))
                    cv2.polylines(grid, [pts], isClosed=True, color=color, thickness=line_thickness)
        except Exception:
            pass
    return grid

def generate_random_shape_grid(min_dim: int = 5, max_dim: int = 20, value_range: tuple[int, int] = (1, 9), 
                        num_shapes: int = 5, fill_prob: float = 0.6, background_value: int = 0) -> np.ndarray:
    height = random.randint(min_dim, max_dim)
    width = random.randint(min_dim, max_dim)
    grid = create_base_grid(height, width, background_value)
    grid = draw_random_shapes(grid, num_shapes=num_shapes, value_range=value_range, fill_probability=fill_prob)
    return grid