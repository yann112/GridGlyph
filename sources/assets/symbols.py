"""
symbol_data.py
Embedded symbolic glyph sets grouped by function: grid_glyphs (data), operation_glyphs (logic/actions).
Useful for environments where file access is restricted or packaging is required.
"""
ROM_VAL_MAP = {
        '∅':0, 'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9,
        'X': 10, 'XI': 11, 'XII': 12, 'XIII': 13, 'XIV': 14, 'XV': 15, 'XVI': 16, 'XVII': 17, 'XVIII': 18, 'XIX': 19,
        'XX': 20, 'XXI': 21, 'XXII': 22, 'XXIII': 23, 'XXIV': 24, 'XXV': 25, 'XXVI': 26, 'XXVII': 27, 'XXVIII': 28, 'XXIX': 29,
        'XXX': 30
    }

SYMBOL_SETS_JSON = {
    "grid_glyphs": [
        {
            "id": "katakana",
            "description": "Clean Japanese syllabary; monospaced, structured, and ideal for pattern recognition.",
            "symbols": ["ツ", "レ", "ハ", "ア", "ヤ", "ユ", "ヨ", "キ", "ク", "ケ"]
        },
        {
        "id": "katakana_final_refined_1",
        "description": "The premier set for abstract visual puzzles; characters chosen for unparalleled visual uniqueness and zero linguistic bias.",
        "symbols": ["コ", "ム", "メ", "ホ", "ロ", "ア", "レ", "ノ", "フ", "ヲ"]
        },
        {
        "id": "katakana_final_refined_2",
        "description": "Second highly refined set, offering strong visual differentiation with distinct shapes and balanced complexity.",
        "symbols": ["エ", "ケ", "セ", "タ", "ワ", "ネ", "ヘ", "モ", "ユ", "ソ"]
        },
        {
        "id": "katakana_final_refined_3",
        "description": "Third refined set, providing excellent variety in form and maintaining strict neutrality for puzzle grid use.",
        "symbols": ["ウ", "ク", "ス", "テ", "ナ", "ハ", "マ", "ヌ", "ラ", "ル"]
        },
        
        {
            "id": "emoji",
            "description": "Visually rich and distinct; great for attention-grabbing patterns but may distract in complex logic.",
            "symbols": ["💥", "⚡", "🟥", "🟦", "🌀", "✦", "⚠️", "✅", "✈", "🛞"]
        },
        {
            "id": "dice_dots",
            "description": "Represents values as dice faces; good for state-based logic but risks numeric interpretation.",
            "symbols": ["⚀", "⚁", "⚂", "⚃", "⚄", "⚅", "⚁", "⚂", "⚃", "⚄"]
        },
        {
            "id": "braille",
            "description": "Tactile writing system; great for spatial orientation and physical logic puzzles.",
            "symbols": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠶", "⠧", "⠇", "⠿", "⠟"]
        },
        {
            "id": "old_italic",
            "description": "Etruscan/Oscan roots; simple shapes, left-to-right logic.",
            "symbols": ["𐌖", "𐌋", "𐌂", "𐌄", "𐌉", "𐌏", "𐌑", "𐌓", "𐌘", "𐌙"]
        },
        {
            "id": "gothic_script",
            "description": "Extinct Germanic language; unique glyphs force abstraction.",
            "symbols": ["𐌰", "𐌱", "𐌲", "𐌳", "𐌴", "𐌵", "𐌶", "𐌷", "𐌸", " 𐌸"]
        },
        {
            "id": "ancient_indus_script",
            "description": "Undeciphered script; forces pure pattern recognition.Extremly neutral",
            "symbols": ["𐦀", "𐦁", "𐦂", "𐦃", "𐦄", "𐦅", "𐦆", "𐦇", "𐦈", "𐦉"]
        },
        {
            "id": "phoenician",
            "description": "First true alphabet; minimalistic, high contrast.",
            "symbols": ["𐤀", "𐤁", "𐤂", "Ɣ", "𐤃", "𐤄", "𐤅", "𐤆", "𐤇", "𐤈"]
        },
        {
            "id": "old_armenian",
            "description": "Armenian script; unique baseline and structure.",
            "symbols": ["Ա", "Բ", "Գ", "Դ", "Ե", "Զ", "Է", "Ը", "Թ", "Ժ"]
        },
        {
            "id": "tibetan",
            "description": "High contrast, unique baseline; good for shape variation.",
            "symbols": ["ཤ", "ཇ", "ང", "ར", "ལ", "འ", "བ", "མ", "ཙ", "ཚ"]
        },
        {
            "id": "coptic_script",
            "description": "Used in liturgy; still used today, supports clear structure.",
            "symbols": ["Ⲁ", "ⲁ", "Ⲃ", "ⲃ", "Ⲅ", "ⲅ", "Ⲇ", "ⲇ", "Ⲉ", "ⲉ"]
        },
        {
            "id": "cypriot_syllabary",
            "description": "Ancient Greek-like script; excellent for syllabic pattern detection.",
            "symbols": ["𐠃", "𐠄", "𐠅", "𐠆", "𐠇", "𐠈", "𐠉", "𐠊", "𐠋", "𐠌"]
        },
        {
            "id": "linear_b",
            "description": "Used in Mycenaean Greece; pictographic and abstract.",
            "symbols": ["𐀀", "𐀁", "𐀂", "𐀃", "𐀄", "𐀅", "𐀆", "𐀇", "𐀈", "𐀉"]
        },
        {
            "id": "indus_valley_seals",
            "description": "Undeciphered script; perfect for forcing pure pattern inference.",
            "symbols": ["𐻁", "𐻂", "𐻃", "𐻄", "𐻅", "𐻆", "𐻇", "𐻈", "𐻉", "𐻊"]
        },
        {
            "id": "cypro_minoan",
            "description": "Cypro-Minoan script; rare and pictographic, good for pattern-only learning.",
            "symbols": ["𐘀", "𐘁", "𐘂", "𐘃", "𐘄", "𐘅", "𐘆", "𐘇", "𐘈", "𐘉"]
        },
        {
            "id": "chinese_zodiac",
            "description": "Animal signs; helps with classification and object tracking.",
            "symbols": ["🐭", "🐮", "🐯", "🐰", "🐲", "🐍", "🐴", "🐑", "🐵", "🐔"]
        },
        {
            "id": "weather_emoticons",
            "description": "Weather-based emoji; helps infer state changes over time.",
            "symbols": ["☀", "☁", "🌧", "❄", "🌪", "🌀", "🌬", "💨", "🌞", "⛄"]
        },
        {
            "id": "currency_symbols",
            "description": "Money-related glyphs; good for classification and value shift puzzles.",
            "symbols": ["$", "€", "£", "¥", "₣", "₡", "₮", "₩", "₦", "₭"]
        }
    ],
    "sigils": [
      {
            "id": "runic",
            "description": "Ancient Germanic script with geometric clarity; excellent for mirroring, tiling, and symmetry tasks.",
            "symbols": ["ᚠ", "ᚢ", "ᚣ", "ᚤ", "ᚥ", "ᚦ", "ᚨ", "ᛇ", "ᛈ", "ᛉ"]
        },
        {
            "id": "box_drawing",
            "description": "Structured characters used for drawing boxes; ideal for spatial transformations like rotation or reflection.",
            "symbols": ["┌", "─", "┐", "│", "└", "┘", "├", "┤", "┬", "┴"]
        },
        {
            "id": "ogham",
            "description": "Irish tree alphabet; vertical lines help with symmetry tasks.",
            "symbols": ["Ⳁ", "ⳁ", "Ⳃ", "ⳃ", "Ⳅ", "ⳅ", "Ⳇ", "ⳇ", "Ⳉ", "ⳉ"]
        },
        {
            "id": "geometric_shapes",
            "description": "Basic geometric forms; great for rotation, symmetry, and spatial logic.",
            "symbols": ["△", "▲", "◆", "◇", "■", "□", "⬟", "⬠", "⬡", "⬢"]
        },
        {
            "id": "runes_with_direction",
            "description": "Runic variants with directional cues; helps with flip/mirror reasoning.",
            "symbols": ["ᛈ", "ᛉ", "ᛊ", "ᛏ", "ᛦ", "⫯", "⫰", "⫱", "⫲", "⫳"]
        },
        {
            "id": "directional_unicode",
            "description": "Unicode arrow variants; ideal for movement and physics puzzles.",
            "symbols": ["⇑", "⇓", "⇐", "⇒", "⇖", "⇗", "⇘", "⇙", "⇚", "⇛"]
        },
        {
            "id": "roman_numerals_extended",
            "description": "Extended Roman numerals with alternate glyphs; good for basic transformations.",
            "symbols": ["Ⅰ", "Ⅱ", "Ⅲ", "Ⅳ", "Ⅴ", "Ⅵ", "Ⅶ", "Ⅷ", "Ⅸ", "Ⅹ"]
        },
        {
            "id": "arrows",
            "description": "Directional glyphs that imply motion or transformation direction; useful for physics-based puzzles.",
            "symbols": ["←", "→", "↑", "↓", "↰", "↱", "↲", "↳", "🔄", "↷"]
        },
        {
            "id": "math_symbols",
            "description": "Abstract logic symbols; ideal for formal rule learning but may confuse small models.",
            "symbols": ["¬", "∨", "∧", "⇒", "⇔", "⊻", "≡", "≠", "⊨", "∉"]
        },
        {
            "id": "boolean_logic",
            "description": "Formal logic operators; good for parity and binary transformations.",
            "symbols": ["¬", "∨", "∧", "⇒", "⇔", "⊻", "≡", "≠", "⊨", "∉"]
        },
        {
            "id": "chess_notation",
            "description": "Chess piece notation; helps detect movement and swapping.",
            "symbols": ["♙", "♖", "♘", "♗", "♕", "♔", "♟", "♜", "♞", "♝"]
        },
        {
            "id": "cardinal_points",
            "description": "Directional indicators; useful for flow, pathfinding, or trajectory puzzles.",
            "symbols": ["⬆", "⬇", "⬅", "➡", "⬈", "⬊", "⬉", "⬋", "↗", "↘"]
        },
        {
            "id": "chinese_radicals_movement",
            "description": "Radicals that suggest movement or change; good for dynamic puzzles.",
            "symbols": ["行", "气", "水", "风", "雨", "雷", "电", "火", "光", "影"]
        },
        {
            "id": "runic_with_motion",
            "description": "Old Futhark runes with directional overlays; helps combine motion + mirroring logic.",
            "symbols": ["ᚠ", "ᚢ", "ᚣ", "ᚤ", "ᚥ", "ᚦ", "ᚨ", "ᚧ", "ᚨ", "ᚩ"]
        },
        {
            "id": "ogham_with_direction",
            "description": "Vertical Irish script with directional clues; good for spatial reasoning.",
            "symbols": ["Ⳁ", "ⳁ", "Ⳃ", "ⳃ", "Ⳅ", "ⳅ", "Ⳇ", "ⳇ", "Ⳉ", "ⳉ"]
        },
        {
            "id": "chinese_stroke_order",
            "description": "Symbols represent stroke order; excellent for sequential logic.",
            "symbols": ["㇀", "㇁", "㇂", "㇃", "㇄", "㇅", "㇆", "㇇", "㇈", "㇉"]
        }
    ],
    "hybrid_glyphs": [
        {
            "id": "greek_letters",
            "description": "Classical Greek alphabet; good for logic and repetition.",
            "symbols": ["α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "ι", "κ"]
        },
        {
            "id": "tarot_major_arcana",
            "description": "Tarot card symbols; good for abstract state transitions.",
            "symbols": ["🠀", "🠁", "▂", "▃", "▄", "▆", "▇", "█", "▎", "▍"]
        },
        {
            "id": "musical_notes",
            "description": "Symbolic representation of rhythm; great for repetition and sequence logic.",
            "symbols": ["♩", "♪", "♫", "𝅘𝅥", "𝅗𝅥", "♭", "♯", "♮", "𝄪", "𝄫"]
        }
    ]
}