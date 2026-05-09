def int_to_roman(n: int) -> str:
    """Converts an integer to a Roman numeral string."""
    if not 0 <= n <= 30: # Safety for ARC grid sizes
        return str(n)
    val = [10, 9, 5, 4, 1]
    syb = ["X", "IX", "V", "IV", "I"]
    roman_num = ''
    i = 0
    while n > 0:
        for _ in range(n // val[i]):
            roman_num += syb[i]
            n -= val[i]
        i += 1
    return roman_num or "0"

def get_sigil_alias(sigil: str) -> str:
    """Mapping complex unicodes to safe strings if needed."""
    mapping = {
        '↻': 'ROTATE',
        '↔': 'HFLIP',
        '↕': 'VFLIP',
        '⤨': 'SWAP_VAL',
        '✂': 'CROP'
    }
    return mapping.get(sigil, sigil)