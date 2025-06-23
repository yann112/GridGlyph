from fixtures import train_data_0, stable_llm_client
from tools.gridglyphs_analyzer_tool import GridGlyphsAnalyzerTool


def test_quality_control_tool_smoke(
    stable_llm_client,
    train_data_0
):

    tool = GridGlyphsAnalyzerTool(llm=stable_llm_client)


    raw_output = tool._run(
        puzzle_data=train_data_0,
        input_grid=None,
        output_grid=None,
        test_grid=None,
        glyphset_ids=[
            # "katakana",
            "runic",
            "box_drawing",
            # "sumerian_cuneiform",
            # "geometric_shapes",
            # "runes_with_direction",
            # "directional_unicode"
                      ],
        n_variants_per_set=30,
        mode=None
        )
    print(raw_output)
    assert raw_output