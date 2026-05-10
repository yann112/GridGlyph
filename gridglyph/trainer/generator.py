import json
import random

# NO imports from .dataset or .train allowed here!

KANJI_MAP = {0: "零", 1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九"}

def apply_isomorphism(grid, mapping):
    return [[mapping.get(cell, cell) for cell in row] for row in grid]

class GridAlchemist:
    def __init__(self, seeds):
        self.seeds = seeds

    def get_sample(self):
        seed = random.choice(self.seeds)
        dice = random.random()
        if dice < 0.3: return seed
        elif dice < 0.6:
            return {
                "input_grid": apply_isomorphism(seed["input_grid"], KANJI_MAP),
                "output_grid": apply_isomorphism(seed["output_grid"], KANJI_MAP),
                "dsl_rule": seed["dsl_rule"]
            }
        else:
            digits = list(range(10)); shuffled = digits.copy(); random.shuffle(shuffled)
            perm_map = dict(zip(digits, shuffled))
            return {
                "input_grid": apply_isomorphism(seed["input_grid"], perm_map),
                "output_grid": apply_isomorphism(seed["output_grid"], perm_map),
                "dsl_rule": seed["dsl_rule"]
            }