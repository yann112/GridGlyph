import random

class GridAlchemist:
    def __init__(self, seeds, tokenizer, kanji_limit=2000):
        self.seeds = seeds
        self.tokenizer = tokenizer
        # Safe pool built once at startup using injected tokenizer
        self.kanji_pool = self._generate_safe_pool(kanji_limit)

    def _generate_safe_pool(self, limit):
        safe_chars = []
        # CJK Block: 0x4E00 to 0x9FFF
        for i in range(0x4E00, 0x9FFF):
            char = chr(i)
            # Ensure the injected tokenizer sees this as exactly 1 token
            ids = self.tokenizer.encode(char, add_special_tokens=False)
            if len(ids) == 1:
                safe_chars.append(char)
            if len(safe_chars) >= limit:
                break
        return safe_chars

    def get_sample(self):
        seed = random.choice(self.seeds)
        # Fresh symbolic mapping every single time
        symbols = random.sample(self.kanji_pool, 10)
        perm_map = dict(zip(range(10), symbols))
        
        return {
            "input_grid": self._apply_iso(seed["input_grid"], perm_map),
            "output_grid": self._apply_iso(seed["output_grid"], perm_map),
            "dsl_rule": seed["dsl_rule"]
        }

    def _apply_iso(self, grid, mapping):
        return [[mapping.get(cell, cell) for cell in row] for row in grid]