def test_grid_analyzer_tool():
    from tools.grid_analyzer_tool import GridAnalyzerTool

    tool = GridAnalyzerTool()
    input_grid = [[0, 0], [1, 1]]
    output_grid = [[1, 1], [0, 0]]
    
    result = tool.run({
        "input_grid": input_grid,
        "output_grid": output_grid
    })

    assert "swap" in result.lower() or "reverse" in result.lower()
