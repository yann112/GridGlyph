import random

class GridAlchemist:
    def __init__(self, seeds, tokenizer, kanji_limit=2000):
        """
        Injects dependencies from the client.
        :param seeds: List of raw numeric grids.
        :param tokenizer: The pre-instantiated tokenizer from the training client.
        :param kanji_limit: How many unique safe symbols to pool.
        """
        self.seeds = seeds
        self.tokenizer = tokenizer
        # Build the safe pool immediately using the injected tokenizer
        self.kanji_pool = self._generate_safe_pool(kanji_limit)

    def _generate_safe_pool(self, limit):
        """Filters characters to ensure they are single-token for the injected model."""
        safe_chars = []
        # CJK Unified Ideographs block
        for i in range(0x4E00, 0x9FFF):
            char = chr(i)
            # Check if this specific tokenizer sees this as exactly 1 token
            ids = self.tokenizer.encode(char, add_special_tokens=False)
            if len(ids) == 1:
                safe_chars.append(char)
            if len(safe_chars) >= limit:
                break
        return safe_chars

    def apply_isomorphism(self, grid, mapping):
        return [[mapping.get(cell, cell) for cell in row] for row in grid]

    def get_sample(self):
        """Returns a logically consistent but symbolically unique grid sample."""
        seed = random.choice(self.seeds)
        
        # Fresh mapping for every call to prevent the 0.15 plateau
        symbols = random.sample(self.kanji_pool, 10)
        perm_map = dict(zip(range(10), symbols))
        
        return {
            "input_grid": self.apply_isomorphism(seed["input_grid"], perm_map),
            "output_grid": self.apply_isomorphism(seed["output_grid"], perm_map),
            "dsl_rule": seed["dsl_rule"]
        }