import pytest
import numpy as np
import logging
from typing import Optional

from core.dsl_symbolic_interpreter import SymbolicRuleParser, roman_to_int
from core.dsl_symbolic_executor import DSLExecutor
from core.dsl_nodes import InputGridReference


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


common_input_grid = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
], dtype=int)

asymmetric_grid = np.array([
    [1, 2, 0],
    [3, 4, 0],
    [0, 0, 0]
], dtype=int)

tiny_grid_1x1_val1 = np.array([[1]], dtype=int)
tiny_grid_1x1_val2 = np.array([[2]], dtype=int)




TEST_CASES = [
    ("Ⳁ", tiny_grid_1x1_val1, tiny_grid_1x1_val1),
    ("Ⳁ", np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]])),
    ("Ⳁ", np.array([1, 2, 3]), np.array([1, 2, 3])),

    ("⇒(II, I)", np.array([[1, 2, 3], [2, 0, 2]]), np.array([[1, 1, 3], [1, 0, 1]])),
    ("⇒(∅, X)", np.array([[0, 1], [0, 0]]), np.array([[10, 1], [10, 10]])),
    ("⇒(I, I)", np.array([[1, 2], [3, 1]]), np.array([[1, 2], [3, 1]])),

    ("↔", tiny_grid_1x1_val1, tiny_grid_1x1_val1),
    ("↔", np.array([[1, 2, 3]]), np.array([[3, 2, 1]])),
    ("↔", np.array([[1], [2], [3]]), np.array([[1], [2], [3]])),

    ("→(I, Ⳁ)", np.array([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]])),
    ("→(II, Ⳁ)", np.array([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]])),
    ("→(I, ⇒(I, ∅))", np.array([[1, 1], [1, 1]]), np.array([[0, 0], [1, 1]])),

    ("⟹(↔, ↕)", np.array([[1, 2], [3, 4]]), np.array([[4, 3], [2, 1]])),
    ("⟹(↢, ↢)", np.array([[1, 2, 3]]), np.array([[1, 2, 3]])),

    ("⬒(I)", np.array([[1, 2]]), np.array([[1, 2]])),
    ("⬒(III)", tiny_grid_1x1_val1, np.array([[1], [1], [1]])),

    ("⇄(II,I)", np.array([[1, 2], [3, 4]]), np.array([[3, 4], [1, 2]])),
    ("⇄(II,III)", np.array([[1, 2], [3, 4], [5, 6]]), np.array([[1, 2], [5, 6], [3, 4]])),

    ("⮝(I,I)", np.array([[1, 2, 3, 4]]), np.array([[4, 1, 2, 3]])),
    ("⮝(I,∅)", np.array([[1, 2, 3]]), np.array([[1, 2, 3]])),
    ("⮝(V,I)", np.array([[1],[2],[3],[4],[5]]), np.array([[1],[2],[3],[4],[5]])),

    ("↕", tiny_grid_1x1_val1, tiny_grid_1x1_val1),
    ("↕", np.array([[1, 2]]), np.array([[1, 2]])),
    ("↕", np.array([[1], [2], [3]]), np.array([[3], [2], [1]])),

    ("⮞(I,I)", np.array([[1, 2], [3, 4]]), np.array([[3, 2], [1, 4]])),
    ("⮞(I,∅)", np.array([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]])),
    ("⮞(III,II)", np.array([[1,2,3,4],[5,6,7,8]]), np.array([[1,2,3,4],[5,6,7,8]])),

    ("◨(I)", np.array([[1], [2]]), np.array([[1], [2]])),
    ("◨(IV)", tiny_grid_1x1_val1, np.array([[1, 1, 1, 1]])),

    ("⊕(I,V,I)", None, np.array([[1, 1, 1, 1, 1]], dtype=int)), # Added dtype
    ("⊕(V,I,II)", None, np.array([[2], [2], [2], [2], [2]], dtype=int)), # Added dtype

    ("⤨(I)", np.array([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]])),
    ("⤨(IV)", np.array([[0, 1]]), np.array([[0,0,0,0,1,1,1,1],[0,0,0,0,1,1,1,1],[0,0,0,0,1,1,1,1],[0,0,0,0,1,1,1,1]])),

    ("⧈", np.array([[1, 1, 1]]), np.array([[1, 1, 1]])),
    ("⧈", np.array([[0, 0, 0], [0, 5, 0], [0, 0, 0]]), np.array([[5]])),
    ("⧈", np.array([[0, 0], [1, 0]]), np.array([[1]])),

    ("⧀", tiny_grid_1x1_val1, np.array([1])),
    ("⧀", np.array([[]]).reshape(0,0), np.array([]).reshape(0,)),

    ("⊡(I,I)", tiny_grid_1x1_val1, np.array([[1]])),
    ("⊡(I,III)", np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), np.array([[3]])),
    ("⊡(II,I)", np.array([[1, 2], [3, 4]]), np.array([[3]])),

    ("↱I", None, np.array([[1]], dtype=int)), # Added dtype
    ("↱∅", None, np.array([[0]], dtype=int)), # Added dtype
    ("↱X", None, np.array([[10]], dtype=int)), # Added dtype

    ("≡(⊡(II,II), ⊡(I,I))", np.array([[1, 2], [1, 1]]), np.array([[1]])),
    ("≡(⊡(II,II), ⊡(I,I))", np.array([[1, 2], [3, 4]]), np.array([[0]])),
    ("≡(↱I, ⊡(I,I))", tiny_grid_1x1_val1, np.array([[1]])),

    ("≗(↔, ↢)", np.array([[1, 2], [3, 4]]), np.array([[1]])),
    ("≗(⬒(II), ◨(II))", tiny_grid_1x1_val1, np.array([[0]])),
    ("≗(⬒(II), ⬒(II))", np.array([[1, 2]]), np.array([[1]])),

    ("⍰(≡(↱V,↱V), ↔, ↕)", np.array([[1, 2], [3, 4]]), np.array([[2, 1], [4, 3]])),
    ("⍰(≡(↱V,↱X), ↔, ↕)", np.array([[1, 2], [3, 4]]), np.array([[3, 4], [1, 2]])),
    ("⍰(≡(↱∅, ↱I), Ⳁ, ⤨(II))", tiny_grid_1x1_val1, np.array([[1, 1], [1, 1]])),

    # --- Corrected ▦ test cases ---
    ("▦(I,I,[[I]])", None, np.array([[1]], dtype=int)),
    ("▦(I,I,[[∅]])", None, np.array([[0]], dtype=int)),
    ("▦(I,II,[[I,∅]])", None, np.array([[1, 0]], dtype=int)),
    ("▦(II,I,[[I],[I]])", None, np.array([[1],[1]], dtype=int)),
    ("▦(II,II,[[I,∅],[∅,I]])", None, np.array([[1,0],[0,1]], dtype=int)),
    ("▦(II,III,[[I,∅,I],[∅,I,∅]])", None, np.array([[1,0,1],[0,1,0]], dtype=int)),
    ("▦(III,III,[[∅,∅,I],[∅,II,I],[I,∅,∅]])", None, np.array([[0, 0, 1], [0, 2, 1], [1, 0, 0]], dtype=int)),
    ("▦(IV,IV,[[I,∅,I,∅],[∅,I,∅,I],[I,∅,I,∅],[∅,I,∅,I]])", None, np.array([
        [1,0,1,0],
        [0,1,0,1],
        [1,0,1,0],
        [0,1,0,1]
    ], dtype=int)),
    ("▦(V,V,[[I,∅,I,∅,I],[∅,I,∅,I,∅],[I,∅,I,∅,I],[∅,I,∅,I,∅],[I,∅,I,∅,I]])", None, np.array([
        [1,0,1,0,1],
        [0,1,0,1,0],
        [1,0,1,0,1],
        [0,1,0,1,0],
        [1,0,1,0,1]
    ], dtype=int)),
    # --- End of corrected ▦ test cases ---

    ("⧎(↱I, ↱I, ↱∅)", None, np.array([[1]])),
    ("⧎(⊕(II,II,V), ▦(II,II,[[I,∅],[∅,I]]), ⊕(II,II,I))", None, np.array([[5, 1], [1, 5]])),
    ("⧎(Ⳁ, ⊕(II,II,∅), ⊕(II,II,X))", np.array([[1, 2], [3, 4]]), np.array([[10, 10], [10, 10]])),

    ("¿(↕, ≡(⊡(I,I), ↱V))", np.array([[1,2],[5,6]]), np.array([[1,2],[5,6]])),
    ("¿(↕, ≡(⊡(II,I), ↱V))", np.array([[1,2],[5,6]]), np.array([[5,6],[1,2]])),
    ("¿(⤨(II), ≗(Ⳁ,Ⳁ))", tiny_grid_1x1_val1, np.array([[1,1],[1,1]])),

    ("◎(III)", np.array([[1,2,3],[3,2,1]]), np.array([[0,0,3],[3,0,0]])),
    ("◎(I)", np.array([[1,1],[1,1]]), np.array([[1,1],[1,1]])),
    ("◎(V)", np.array([[1,2],[3,4]]), np.array([[0,0],[0,0]])),
    
    ("Ⳁ", np.array([[1, 0], [0, 1]], dtype=int), np.array([[1, 0], [0, 1]], dtype=int)),
    ("⇒(I,∅)", np.array([[1, 8], [0, 1]], dtype=int), np.array([[0, 8], [0, 0]], dtype=int)),
    ("↔", np.array([[1, 2], [3, 4]], dtype=int), np.array([[2, 1], [4, 3]], dtype=int)),
    ("→(I, ↢)", np.array([[1, 2], [3, 4]], dtype=int), np.array([[2, 1], [3, 4]], dtype=int)),
    ("⟹(→(I,↢),↔)", np.array([[1, 2], [3, 4]], dtype=int), np.array([[1, 2], [4, 3]], dtype=int)),
    ("⬒(II)",  np.array([[1, 0], [0, 1]], dtype=int), np.tile( np.array([[1, 0], [0, 1]], dtype=int), (2, 1))),
    ("⇄(I,II)", np.array([[1, 2], [3, 4]], dtype=int), np.array([[3, 4], [1, 2]], dtype=int)),
    ("⮝(I,II)", np.array([[1, 2, 3], [4, 5, 6]], dtype=int), np.array([[2, 3, 1], [4, 5, 6]], dtype=int)),
    ("↕", np.array([[1, 2], [3, 4]], dtype=int), np.array([[3, 4], [1, 2]], dtype=int)),
    ("↕", np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=int), np.array([[7, 8, 9], [4, 5, 6], [1, 2, 3]], dtype=int)),
    ("⮞(I,II)", np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=int), np.array([[4, 2, 3], [7, 5, 6], [1, 8, 9]], dtype=int)),
    ("⮞(II,I)", np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=int), np.array([[1, 8, 3], [4, 2, 6], [7, 5, 9]], dtype=int)),
    ("◨(II)", np.array([[1, 0], [0, 1]], dtype=int), np.tile(np.array([[1, 0], [0, 1]], dtype=int), (1, 2))),
    ("◨(III)", np.array([[1, 2]], dtype=int), np.array([[1, 2, 1, 2, 1, 2]], dtype=int)),
    ("→(II, ↢)", np.array([[1, 2], [3, 4]], dtype=int), np.array([[1, 2], [4, 3]], dtype=int)),
    ("→(I, Ⳁ)", np.array([[1, 2], [3, 4]], dtype=int), np.array([[1, 2], [3, 4]], dtype=int)),
    ("⇄(I,III)", np.array([[1, 2], [3, 4], [5, 6]], dtype=int), np.array([[5, 6], [3, 4], [1, 2]], dtype=int)),
    ("⇄(I,I)", np.array([[1, 2], [3, 4]], dtype=int), np.array([[1, 2], [3, 4]], dtype=int)),
    ("⮝(I,I)", np.array([[1, 2, 3], [4, 5, 6]], dtype=int), np.array([[3, 1, 2], [4, 5, 6]], dtype=int)),
    ("⮝(II,I)", np.array([[1, 2, 3], [4, 5, 6]], dtype=int), np.array([[1, 2, 3], [6, 4, 5]], dtype=int)),
    ("⟹(↔)", np.array([[1, 2], [3, 4]], dtype=int), np.array([[2, 1], [4, 3]], dtype=int)),
    ("⟹(⮝(I,I), ↔, ⬒(II))", np.array([[1, 2], [3, 4]], dtype=int), np.tile(np.array([[1, 2], [4, 3]], dtype=int), (2, 1))),
    ("⊕(IV,IV,∅)", np.zeros((4,4), dtype=int), np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]], dtype=int)),
    ("⟹(◨(II), ⬒(II))", np.array([[0, 1], [1, 0]], dtype=int), np.array([[0,1,0,1],[1,0,1,0],[0,1,0,1],[1,0,1,0]], dtype=int)),
    ("⧎(⟹(◨(II), ⬒(II)), ⤨(II), ⊕(IV,IV,∅))", np.array([[0, 1], [1, 0]], dtype=int), np.array([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]], dtype=int)),
    ("⌂", np.array([[5, 6], [7, 8]], dtype=int), np.array([[5, 6], [7, 8]], dtype=int)),
    ("⌂", np.array([[1, 1, 1], [0, 0, 0]], dtype=int), np.array([[1, 1, 1], [0, 0, 0]], dtype=int)),
    ("↔(⌂)", np.array([[1, 2], [3, 4]], dtype=int), np.array([[2, 1], [4, 3]], dtype=int)),
    ("↔(⌂)", np.array([[1, 2], [3, 4]], dtype=int), np.array([[2, 1], [4, 3]], dtype=int)),
    ("↔(⌂)", np.array([[1, 2, 3], [4, 5, 6]], dtype=int), np.array([[3, 2, 1], [6, 5, 4]], dtype=int)),
    ("↕(⌂)", np.array([[1, 2], [3, 4]], dtype=int), np.array([[3, 4], [1, 2]], dtype=int)),
    ("↕(⌂)", np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=int), np.array([[7, 8, 9], [4, 5, 6], [1, 2, 3]], dtype=int)),
    ("↢(⌂)", np.array([[1, 2, 3], [4, 5, 6]], dtype=int), np.array([[3, 2, 1], [6, 5, 4]], dtype=int)),
    ("↢(⌂)", np.array([[1, 2]], dtype=int), np.array([[2, 1]], dtype=int)),
    ("⧈(⌂)", np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=int), np.array([[1, 1], [1, 0]], dtype=int)),
    ("⧈(⌂)", np.array([[0, 0], [0, 0]], dtype=int), np.array([[0]], dtype=int)),
    ("⧀(⌂)", np.array([[1, 2], [3, 4]], dtype=int), np.array([1, 2, 3, 4], dtype=int)),
    ("⧀(⌂)", np.array([[5], [6], [7]], dtype=int), np.array([5, 6, 7], dtype=int)),
    ("⟹(⌂, ⤨(II))", np.array([[1, 2], [3, 4]], dtype=int), np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]], dtype=int)),
    ("⟹(⌂, ⤨(III))", np.array([[1]], dtype=int), np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=int)),
    ("⟹(⌂, ⇒(I,V))", np.array([[1, 2], [3, 1]], dtype=int), np.array([[5, 2], [3, 5]], dtype=int)),
    ("⟹(⌂, ⇒(V,∅))", np.array([[5, 5], [0, 0]], dtype=int), np.array([[0, 0], [0, 0]], dtype=int)),
    ("⇌(↔, ↕)", common_input_grid, np.array([[3, 2, 1], [4, 5, 6], [9, 8, 7]], dtype=int)),
    ("⇌(↢, Ⳁ)", common_input_grid, np.array([[3, 2, 1], [4, 5, 6], [9, 8, 7]], dtype=int)),
    ("⇌(⇒(I,∅), ↢)", asymmetric_grid, np.array([[0, 2, 0], [0, 4, 3], [0, 0, 0]], dtype=int)),
    ("¿(↔, ≗(↱I, ↱I))", common_input_grid, np.fliplr(common_input_grid)),
    ("¿(↔, ≗(⊡(I,I), ↱I))", common_input_grid, np.fliplr(common_input_grid)), 
    ("¿(↔, ≗(⊡(I,I), ↱V))", common_input_grid, common_input_grid),
    ("¿(↢, ≗(⌂, Ⳁ))", common_input_grid, np.flip(common_input_grid, axis=1)),
    ("¿(↢, ≗(⌂, ↔(⌂)))", asymmetric_grid, asymmetric_grid),
    ("◫(⌂, [(⊕(I,I,I), ↢)], Ⳁ)", tiny_grid_1x1_val1, tiny_grid_1x1_val1),
    ("◫(⌂, [(⊕(I,I,II), ↢)], Ⳁ)", tiny_grid_1x1_val1, tiny_grid_1x1_val1),
    ("◫(⌂, [(⊕(II,II,I), ↔)], ↕)", np.array([[1,1],[1,1]], dtype=int), np.array([[1,1],[1,1]], dtype=int)), 
    ("◫(⌂, [(⊕(II,II,II), ↔)], ↕)", asymmetric_grid, np.flipud(asymmetric_grid)),
    ("⟹(⇒(VIII, VII), ⇒(I, ∅))", np.array([[8, 8, 0, 0], [8, 0, 0, 1], [0, 0, 1, 1], [0, 0, 0, 0]], dtype=int), np.array([[7, 7, 0, 0], [7, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=int)),
    ("◎(I)", common_input_grid, np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=int)),
    ("◎(V)", common_input_grid, np.array([[0, 0, 0], [0, 5, 0], [0, 0, 0]], dtype=int)),
    ("◎(∅)", common_input_grid, np.zeros_like(common_input_grid, dtype=int)),
    ("◎(X)", common_input_grid, np.zeros_like(common_input_grid, dtype=int)),
    ("◎(I)", tiny_grid_1x1_val1, tiny_grid_1x1_val1),
    ("◎(II)", tiny_grid_1x1_val1, np.array([[0]], dtype=int)),
    ("⧎(⊕(II,II,∅), ▦(II,II,[[I,∅],[∅,I]]), ↱I)", None, np.array([[0, 1], [1, 0]], dtype=int)),
    ("⧎(⊕(III,III,VII), ▦(III,III,[[I,∅,I],[∅,I,∅],[I,∅,I]]), ⊕(III,III,∅))", None, np.array([[7, 0, 7], [0, 7, 0], [7, 0, 7]], dtype=int)),
    
    ("◫(⌂, [(▦(IV,IV,[[I,∅,I,∅],[∅,I,∅,I],[I,∅,I,∅],[∅,I,∅,I]]), ⇒(I, VII))], ⇒(I, V))", np.array([[1,0,1,0],[0,1,0,1],[1,0,1,0],[0,1,0,1]], dtype=int), np.array([[7,0,7,0],[0,7,0,7],[7,0,7,0],[0,7,0,7]], dtype=int)),
    ("◫(⌂, [(▦(IV,IV,[[I,∅,I,∅],[∅,I,∅,I],[I,∅,I,∅],[∅,I,∅,I]]), ⇒(I, VII))], ⇒(I, V))", np.array([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]], dtype=int), np.array([[5,5,5,5],[5,5,5,5],[5,5,5,5],[5,5,5,5]], dtype=int)),
    ("⧈(◎(I))", np.array([[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]], dtype=int), np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)),
    ("◫(⧈(◎(I)), [(▦(III,III,[[∅,I,∅],[I,I,I],[∅,I,∅]]), ⇒(VIII, III))], ⇒(∅, V))", np.array([[8, 0, 0, 0, 8], [0, 0, 1, 0, 0], [8, 1, 1, 1, 8], [0, 0, 1, 0, 0], [8, 0, 0, 0, 8]], dtype=int), np.array([[3, 0, 0, 0, 3], [0, 0, 1, 0, 0], [3, 1, 1, 1, 3], [0, 0, 1, 0, 0], [3, 0, 0, 0, 3]], dtype=int)),
    ("◫(⧈(◎(I)), [(▦(III,III,[[∅,I,∅],[I,I,I],[∅,I,∅]]), ⇒(VIII, III))], ⇒(∅, V))", np.array([[8, 0, 0, 0, 8], [0, 1, 0, 0, 0], [8, 1, 1, 1, 8], [0, 0, 1, 0, 0], [8, 0, 0, 0, 8]], dtype=int), np.array([[8, 5, 5, 5, 8], [5, 1, 5, 5, 5], [8, 1, 1, 1, 8], [5, 5, 1, 5, 5], [8, 5, 5, 5, 8]], dtype=int)),
    ("◫(⧈(◎(I)), [(▦(III,III,[[∅,I,∅],[I,I,I],[∅,I,∅]]), ⟹(⇒(VIII, III), ⇒(I, ∅)))], ⟹(⇒(∅, V), ⇒(I, ∅)))", np.array([[8, 0, 0, 0, 8], [0, 0, 1, 0, 0], [8, 1, 1, 1, 8], [0, 0, 1, 0, 0], [8, 0, 0, 0, 8]], dtype=int), np.array([[3, 0, 0, 0, 3], [0, 0, 0, 0, 0], [3, 0, 0, 0, 3], [0, 0, 0, 0, 0], [3, 0, 0, 0, 3]], dtype=int)),
    ("◫(⧈(◎(I)), [(▦(III,III,[[∅,I,∅],[I,I,I],[∅,I,∅]]), ⟹(⇒(VIII, III), ⇒(I, ∅)))], ⟹(⇒(∅, V), ⇒(I, ∅)))", np.array([[8, 0, 0, 0, 8], [0, 1, 0, 0, 0], [8, 1, 1, 1, 8], [0, 0, 1, 0, 0], [8, 0, 0, 0, 8]], dtype=int), np.array([[8, 5, 5, 5, 8], [5, 0, 5, 5, 5], [8, 0, 0, 0, 8], [5, 5, 0, 5, 5], [8, 5, 5, 5, 8]], dtype=int)),
    ("⏚(∅)", np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])),
    ("⏚(∅)", np.array([[0,0],[0,0]]), np.array([[1,1],[1,1]])),
    ("⏚(∅)", np.array([[0,1,0],[1,1,1],[0,1,0]]), np.array([[1,0,1],[0,0,0],[1,0,1]])),
    ("⏚(V)", np.array([[5,1,5],[1,2,1],[5,1,5]]), np.array([[1,0,1],[0,0,0],[1,0,1]])),
    ("⏚(∅)", np.array([[0,0,0],[0,7,0],[0,0,0]]), np.array([[1,1,1],[1,0,1],[1,1,1]])),
    ("⏚(∅)", np.array([
        [0,0,0,0,0,0],
        [0,1,1,1,1,0],
        [0,1,1,1,1,0],
        [0,1,1,1,1,0],
        [0,0,0,0,0,0]
    ]), np.array([
        [1,1,1,1,1,1],
        [1,0,0,0,0,1],
        [1,0,0,0,0,1],
        [1,0,0,0,0,1],
        [1,1,1,1,1,1]
    ])),
    ("⟹(⏚(∅), ⟹(⟹(⇒(∅, X), ⇒(I, ∅)), ⇒(X, I)))", np.array([[0,0,0,0,0],[0,5,5,5,0],[0,5,0,5,0],[0,5,5,5,0],[0,0,0,0,0]], dtype=int), np.array([[0,0,0,0,0],[0,1,1,1,0],[0,1,1,1,0],[0,1,1,1,0],[0,0,0,0,0]], dtype=int)),
    ("¬(↱I)", None, np.array([[0]], dtype=int)),
    ("¬(↱∅)", None, np.array([[1]], dtype=int)),
    ("¬(⊕(II,II,I))", None, np.array([[0, 0], [0, 0]], dtype=int)),
    ("¬(⊕(II,II,∅))", None, np.array([[1, 1], [1, 1]], dtype=int)),
    ("¬(▦(II,II,[[I,∅],[∅,I]]))", None, np.array([[0, 1], [1, 0]], dtype=int)),
    ("¬(⟹(◎(V), ⇒(V,I)))", np.array([[1, 5], [5, 1]], dtype=int), np.array([[1, 0], [0, 1]], dtype=int)),
    ("¬(⟹(◎(I), ⇒(I,I)))", np.array([[1, 5], [5, 1]], dtype=int), np.array([[0, 1], [1, 0]], dtype=int)),
    ("∧(↱I, ↱I)", None, np.array([[1]], dtype=int)),
    ("∧(↱I, ↱∅)", None, np.array([[0]], dtype=int)),
    ("∧(↱∅, ↱∅)", None, np.array([[0]], dtype=int)),
    ("∧(⊕(II,II,I), ⊕(II,II,∅))", None, np.array([[0, 0], [0, 0]], dtype=int)),
    ("∧(⊕(II,II,I), ▦(II,II,[[I,∅],[∅,I]]))", None, np.array([[1, 0], [0, 1]], dtype=int)),
    ("∧(▦(II,II,[[I,∅],[∅,I]]), ▦(II,II,[[I,∅],[I,∅]]))", None, np.array([[1, 0], [0, 0]], dtype=int)),
    ("∧(⟹(◎(I), ⇒(I,I)), ⟹(◎(V), ⇒(V,I)))", np.array([[1, 5], [5, 1]], dtype=int), np.array([[0, 0], [0, 0]], dtype=int)),
    ("∧(⟹(◎(I), ⇒(I,I)), ⟹(◎(I), ⇒(I,I)))", np.array([[1, 5], [5, 1]], dtype=int), np.array([[1, 0], [0, 1]], dtype=int)),
    ("∨(↱I, ↱I)", None, np.array([[1]], dtype=int)),
    ("∨(↱I, ↱∅)", None, np.array([[1]], dtype=int)),
    ("∨(↱∅, ↱∅)", None, np.array([[0]], dtype=int)),
    ("∨(⊕(II,II,I), ⊕(II,II,∅))", None, np.array([[1, 1], [1, 1]], dtype=int)),
    ("∨(⊕(II,II,∅), ▦(II,II,[[I,∅],[∅,I]]))", None, np.array([[1, 0], [0, 1]], dtype=int)),
    ("∨(▦(II,II,[[I,∅],[∅,I]]), ▦(II,II,[[I,∅],[I,∅]]))", None, np.array([[1, 0], [1, 1]], dtype=int)),
    ("∨(⟹(◎(I), ⇒(I,I)), ⟹(◎(V), ⇒(V,I)))", np.array([[1, 5], [5, 1]], dtype=int), np.array([[1, 1], [1, 1]], dtype=int)),
    ("∨(⟹(◎(I), ⇒(I,I)), ⟹(◎(II), ⇒(II,I)))", np.array([[1, 2], [3, 4]], dtype=int), np.array([[1, 1], [0, 0]], dtype=int)),
    
    ("⇒(∅, IV)", np.array([[0, 1, 0], [2, 0, 3]], dtype=int), np.array([[4, 1, 4], [2, 4, 3]], dtype=int)),
    ("⇒(∅, IV)", np.array([[0]], dtype=int), np.array([[4]], dtype=int)),
    ("⇒(∅, IV)", np.array([[1]], dtype=int), np.array([[1]], dtype=int)),
    ("◎(∅)", np.array([[3,3,3,3,3],[3,0,0,0,3],[3,0,3,0,3],[3,0,0,0,3],[3,3,3,3,3]], dtype=int), np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]], dtype=int)),
    ("◎(∅)", np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=int), np.array([[0,0,0],[0,0,0],[0,0,0]], dtype=int)),
    ("◎(∅)", np.array([[1,2],[3,4]], dtype=int), np.array([[0,0],[0,0]], dtype=int)),
    ("⏚(∅)", np.array([[3,3,3,3,3],[3,0,0,0,3],[3,0,3,0,3],[3,0,0,0,3],[3,3,3,3,3]], dtype=int), np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]], dtype=int)),
    ("⏚(∅)", np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=int), np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=int)),
    ("⏚(∅)", np.array([[0,0],[0,0]], dtype=int), np.array([[1,1],[1,1]], dtype=int)),
    ("¬(⏚(∅))", np.array([[3,3,3,3,3],[3,0,0,0,3],[3,0,3,0,3],[3,0,0,0,3],[3,3,3,3,3]], dtype=int), np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]], dtype=int)),
    ("¬(⏚(∅))", np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=int), np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=int)),
    ("¬(Ⳁ)", np.array([[1,0],[0,1]], dtype=int), np.array([[0,1],[1,0]], dtype=int)),
    ("¬(Ⳁ)", np.array([[1,0],[5,0]], dtype=int), np.array([[0,1],[0,1]], dtype=int)),
    ("¬(◎(I))", np.array([[1,0],[5,1]], dtype=int), np.array([[0,1],[1,0]], dtype=int)),
    ("∧(◎(∅), ¬(⏚(∅)))", np.array([[3,3,3,3,3],[3,0,0,0,3],[3,0,3,0,3],[3,0,0,0,3],[3,3,3,3,3]], dtype=int), np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]], dtype=int)),
    ("∧(◎(∅), ¬(⏚(∅)))", np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=int), np.array([[0,0,0],[0,0,0],[0,0,0]], dtype=int)),
    ("∧(Ⳁ, ¬(Ⳁ))", np.array([[1,0],[2,3]], dtype=int), np.array([[0,0],[0,0]], dtype=int)),
    ("∧(⊕(II,II,I), ▦(II,II,[[I,∅],[∅,I]]))", None, np.array([[1,0],[0,1]], dtype=int)),
    ("∧(Ⳁ, Ⳁ)", np.array([[1,0],[2,3]], dtype=int), np.array([[1,0],[1,1]], dtype=int)),
    ("⧎(⇒(∅, IV), ∧(◎(∅), ¬(⏚(∅))), Ⳁ)", np.array([[3,3,3,3,3],[3,0,0,0,3],[3,0,3,0,3],[3,0,0,0,3],[3,3,3,3,3]], dtype=int), np.array([[3,3,3,3,3],[3,0,0,0,3],[3,0,3,0,3],[3,0,0,0,3],[3,3,3,3,3]], dtype=int)),
    ("⧎(⇒(∅, IV), ∧(◎(∅), ¬(⏚(∅))), Ⳁ)", np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=int), np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=int)),
    ("⧎(⇒(∅, IV), ∧(◎(∅), ¬(⏚(∅))), Ⳁ)", np.array([[0,0,0,0,0],[0,3,3,3,0],[0,3,0,3,0],[0,3,3,3,0],[0,0,0,0,0]], dtype=int), np.array([[0,0,0,0,0],[0,3,3,3,0],[0,3,0,3,0],[0,3,3,3,0],[0,0,0,0,0]], dtype=int)),
    ("ⓑ(Ⳁ)", np.array([[0,0],[0,0]], dtype=int), np.array([[0,0],[0,0]], dtype=int)),
    ("ⓑ(Ⳁ)", np.array([[1,1],[1,1]], dtype=int), np.array([[1,1],[1,1]], dtype=int)),
    ("ⓑ(Ⳁ)", np.array([[7,7],[7,7]], dtype=int), np.array([[1,1],[1,1]], dtype=int)),
    ("⏚(∅)", np.array([[0,0,0,0,0],[0,3,3,3,0],[0,3,0,3,0],[0,3,3,3,0],[0,0,0,0,0]], dtype=int), np.array([[1,1,1,1,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,1,1,1,1]], dtype=int)),
    ("∨(⏚(∅), ⓑ(◎(III)))", np.array([[0,0,0,0,0],[0,3,3,3,0],[0,3,0,3,0],[0,3,3,3,0],[0,0,0,0,0]], dtype=int), np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,0,1,1],[1,1,1,1,1],[1,1,1,1,1]], dtype=int)),
    ("⟹(∨(⏚(∅), ⓑ(◎(III))), ⟹(⇒(∅, IV), ⇒(I, ∅)))",  np.array([[0,0,0,0,0],[0,3,3,3,0],[0,3,0,3,0],[0,3,3,3,0],[0,0,0,0,0]], dtype=int), np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,4,0,0],[0,0,0,0,0],[0,0,0,0,0]], dtype=int)),

    ("⏚(∅)",
     np.array([[0,0,0,0,0],[0,3,3,3,0],[0,3,0,3,0],[0,3,3,3,0],[0,0,0,0,0]], dtype=int),
     np.array([[1,1,1,1,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,1,1,1,1]], dtype=int)),
    ("ⓑ(◎(III))",
     np.array([[0,0,0,0,0],[0,3,3,3,0],[0,3,0,3,0],[0,3,3,3,0],[0,0,0,0,0]], dtype=int),
     np.array([[0,0,0,0,0],[0,1,1,1,0],[0,1,0,1,0],[0,1,1,1,0],[0,0,0,0,0]], dtype=int)),
    ("∨(⏚(∅), ⓑ(◎(III)))",
     np.array([[0,0,0,0,0],[0,3,3,3,0],[0,3,0,3,0],[0,3,3,3,0],[0,0,0,0,0]], dtype=int),
     np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,0,1,1],[1,1,1,1,1],[1,1,1,1,1]], dtype=int)),
    ("⟹(⌂, ⟹(⇒(∅, IV), ⇒(I, ∅)))",
     np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=int),
     np.array([[4,0,4],[0,4,0],[4,0,4]], dtype=int)),
    ("⟹(∨(⏚(∅), ⓑ(◎(III))), ⟹(⇒(∅, IV), ⇒(I, ∅)))",
     np.array([[0,0,0,0,0],[0,3,3,3,0],[0,3,0,3,0],[0,3,3,3,0],[0,0,0,0,0]], dtype=int),
     np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,4,0,0],[0,0,0,0,0],[0,0,0,0,0]], dtype=int)),
    ("⧎(⌂,∨(⏚(∅),ⓑ(◎(III))),⟹(∨(⏚(∅),ⓑ(◎(III))),⟹(⇒(∅,IV),⇒(I,∅))))",
     np.array([[0,0,0,0,0],[0,3,3,3,0],[0,3,0,3,0],[0,3,3,3,0],[0,0,0,0,0]], dtype=int),
     np.array([[0,0,0,0,0],[0,3,3,3,0],[0,3,4,3,0],[0,3,3,3,0],[0,0,0,0,0]], dtype=int)),
    ("⌖(↱V, ↱V)", None, np.array([[1]], dtype=int)),
    ("⌖(⊕(II,II,I), ↱I)", None, np.array([[1,1],[1,1]], dtype=int)),
    ("⌖(⊕(II,II,I), ↱V)", None, np.array([[0,0],[0,0]], dtype=int)),
    ("⌖(⊕(III,III,I), ▦(II,II,[[I,I],[I,I]]))", None, np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=int)),
    ("⌖(⊕(III,III,I), ▦(II,II,[[?,?],[?,?]]))", None, np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=int)),
    ("⌖(⊕(III,III,I), ▦(II,II,[[I,?],[?,I]]))", None, np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=int)),
    ("⌖(⊕(IV,IV,I), ▦(II,II,[[?,I],[I,?]]))", None, np.array([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]], dtype=int)),
    ("⌖(⊕(III,III,I), ▦(III,III,[[?,I,?],[I,?,I],[?,I,?]]))", np.array([[0,1,0],[1,2,1],[0,1,0]], dtype=int), np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=int)),
    ("⌖(⊕(III,III,I), ▦(II,II,[[I,I],[I,I]]))", np.array([[1,1,0],[1,1,0],[0,0,0]], dtype=int), np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=int)),
    ("⌖(⊕(V,V,I), ▦(II,II,[[?,?],[?,?]]))", None, np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]], dtype=int)),
    ("⌖(⊕(III,III,I), ▦(II,II,[[∅,∅],[∅,∅]]))", None, np.array([[0,0,0],[0,0,0],[0,0,0]], dtype=int)),
    ("⌖(⊕(II,II,V), ▦(II,II,[[V,V],[V,V]]))", None, np.array([[1,1],[1,1]], dtype=int)),
    ("⌖(⊕(II,II,V), ▦(II,II,[[I,I],[I,I]]))", None, np.array([[0,0],[0,0]], dtype=int)),
    ("⌖(⊕(II,II,V), ▦(III,III,[[V,V,V],[V,V,V],[V,V,V]]))", None, np.array([[0,0],[0,0]], dtype=int)),
    ("⌖(⊕(II,II,V), ▦(I,I,[[?]]))", None, np.array([[1,1],[1,1]], dtype=int)),
    ("⌖(⌂, ▦(II,II,[[?,?],[?,?]]))", np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=int), np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=int)),
    ("⌖(⌂, ▦(I,I,[[5]]))", np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=int), np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=int)),
    ("⌖(⌂, ▦(II,II,[[5,?],[?,9]]))", np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=int), np.array([[0,0,0],[0,1,1],[0,1,1]], dtype=int)),
    ("⌖(⌂, ▦(II,II,[[?,?],[?,?]]))", np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=int), np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=int)), 
    ("⌖(⌂, ▦(I,I,[[1]]))", tiny_grid_1x1_val1, tiny_grid_1x1_val1),
    ("⌖(⌂, ▦(I,I,[[2]]))", tiny_grid_1x1_val1, np.array([[0]], dtype=int)),
    ("⌖(⌂, ▦(II,II,[[?,?],[?,?]]))", tiny_grid_1x1_val1, np.array([[0]], dtype=int)),
]

@pytest.fixture
def parser() -> SymbolicRuleParser:
    return SymbolicRuleParser()

@pytest.fixture
def test_logger() -> logging.Logger:
    return logging.getLogger(__name__)


@pytest.mark.parametrize("rule, initial_input_grid, expected_output_grid", TEST_CASES)
def test_dsl_executor_execution(
    parser: SymbolicRuleParser,
    test_logger: logging.Logger,
    rule: str,
    initial_input_grid: Optional[np.ndarray],
    expected_output_grid: np.ndarray
):
    try:
        parsed_command_tree = parser.parse_rule(rule)
        test_logger.info(f"Successfully parsed rule: '{rule}'")

        if initial_input_grid is None:
            executor_input_grid = np.array([[0]], dtype=int)
        else:
            executor_input_grid = np.array(initial_input_grid, dtype=int)

        executor = DSLExecutor(
            root_command=parsed_command_tree,
            initial_puzzle_input=executor_input_grid,
            logger=test_logger
        )
        test_logger.info("Executor instantiated and initialized commands.")

        result_grid = executor.execute_program()
        test_logger.info(f"Execution complete for rule: '{rule}'")

        assert result_grid.shape == expected_output_grid.shape, \
            f"Shape mismatch for rule '{rule}': {result_grid.shape} vs {expected_output_grid.shape}"

        if np.issubdtype(expected_output_grid.dtype, np.number):
            assert np.array_equal(result_grid, expected_output_grid), \
                f"Output mismatch for rule '{rule}'\nExpected:\n{expected_output_grid}\nGot:\n{result_grid}"
        else:
            assert all(
                np.array_equal(r, e) if isinstance(r, np.ndarray) else r == e
                for r, e in zip(result_grid.flatten(), expected_output_grid.flatten())
            ), f"Output mismatch for rule '{rule}'\nExpected:\n{expected_output_grid}\nGot:\n{result_grid}"

    except Exception as e:
        pytest.fail(f"Test failed for rule '{rule}': {str(e)}")