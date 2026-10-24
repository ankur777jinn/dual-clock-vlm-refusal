"""
Unit tests for configuration, grid geometry, and quadrant partitions.
"""

from src.config import Config, build_grid, count_quadrants


def test_grid_size():
    """Verify that the factorial grid has exactly 55 valid cells."""
    grid = build_grid()
    assert len(grid) == 55
    # Verify every cell satisfies src > dst
    for src, dst in grid:
        assert src > dst, f"Violated src > dst: src={src}, dst={dst}"


def test_quadrant_counts_midpoints():
    """Verify quadrant counts at prior midpoints L_read=12, L_harm=18.

    Expected:
    - Bridge (dst <= 12, src >= 18): 15 cells
    - TooEarly (dst <= 12, src < 18): 27 cells
    - TooLate (dst > 12, src >= 18): 6 cells
    - Neither (dst > 12, src < 18): 7 cells
    Total = 55 cells
    """
    grid = build_grid()
    quadrants = count_quadrants(L_read=12, L_harm=18, grid=grid)

    assert quadrants["Bridge"] == 15, f"Expected 15, got {quadrants['Bridge']}"
    assert quadrants["TooEarly"] == 27, f"Expected 27, got {quadrants['TooEarly']}"
    assert quadrants["TooLate"] == 6, f"Expected 6, got {quadrants['TooLate']}"
    assert quadrants["Neither"] == 7, f"Expected 7, got {quadrants['Neither']}"
    assert sum(quadrants.values()) == 55


def test_quadrant_counts_across_prior_range():
    """Verify that all quadrants are non-empty across the full prior range,
    and verify quadrant counts under anticipated ranges.

    Range: L_read in [10, 13], L_harm in [16, 20]
    At extreme corners (e.g. L_read=13, L_harm=16 or L_read=13, L_harm=20),
    sparse quadrants have 1-2 cells, triggering the pre-registered ridge fallback.
    """
    grid = build_grid()
    for l_read in [10, 11, 12, 13]:
        for l_harm in [16, 17, 18, 19, 20]:
            quads = count_quadrants(L_read=l_read, L_harm=l_harm, grid=grid)
            assert sum(quads.values()) == 55
            assert quads["Bridge"] >= 1, f"Bridge empty at L_read={l_read}, L_harm={l_harm}: {quads}"
            assert quads["TooEarly"] >= 1, f"TooEarly empty at L_read={l_read}, L_harm={l_harm}: {quads}"
            assert quads["TooLate"] >= 1, f"TooLate empty at L_read={l_read}, L_harm={l_harm}: {quads}"
            assert quads["Neither"] >= 1, f"Neither empty at L_read={l_read}, L_harm={l_harm}: {quads}"


def test_anchor_cell():
    """Verify anchor cell (11, 7) is external to the main grid."""
    grid = build_grid()
    # In config, L_SRC does not include 11, so (11, 7) is not in the 55 cells
    assert Config.ANCHOR == (11, 7)
    assert Config.ANCHOR not in grid
