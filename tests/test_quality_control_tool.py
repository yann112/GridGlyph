from fixtures import train_data_0, stable_llm_client
from tools.quality_control_tool import QcControllerTool


def test_quality_control_tool_smoke(
    stable_llm_client,
    train_data_0
):

    synthesizer_result = {
        'success': True,
        'result_grid': [[1, 2, 1, 2], [3, 4, 3, 4]],
        'program': "<core.dsl_nodes.RepeatGrid object at 0x7feda892b860>",
        'score': 1.0,
        'program_str': '{"operation": "repeat_grid", "parameters": {"inner_command": {"operation": "identity", "parameters": {}}, "vertical_repeats": 1, "horizontal_repeats": 2}}',
        'explanation': 'The program duplicates the input grid horizontally, creating a mirrored version of the grid side by side.',
        'alternatives': [
            {
                'program': "<core.dsl_nodes.RepeatGrid object at 0x7feda892bec0>",
                'score': 1.0,
                'program_str': '{"operation": "repeat_grid", "parameters": {"inner_command": {"operation": "alternate", "parameters": {"first": {"operation": "identity", "parameters": {}}, "second": {"operation": "identity", "parameters": {}}}}, "vertical_repeats": 1, "horizontal_repeats": 2}}',
                'explanation': 'The program duplicates the input grid horizontally, creating a mirrored version of the grid side by side.'
                },
            {
                'program': "<core.dsl_nodes.Sequence object at 0x7feda8929040>",
                'score': 1.0,
                'program_str': '{"operation": "sequence", "parameters": {"commands": [{"operation": "repeat_grid", "parameters": {"inner_command": {"operation": "identity", "parameters": {}}, "vertical_repeats": 1, "horizontal_repeats": 2}}, {"operation": "alternate", "parameters": {"first": {"operation": "identity", "parameters": {}}, "second": {"operation": "identity", "parameters": {}}}}]}}',
                'explanation': 'The program duplicates the input grid horizontally and then alternates between applying no transformation to each row.'}
            ]
        }

    analysis = "The pattern shows horizontal repetition"
    tool = QcControllerTool(llm=stable_llm_client)


    raw_output = tool._run(
        analysis_description = analysis,
        generated_program = synthesizer_result,
        input_grid = None,
        output_grid = None,
        prompt_hint = None
        )
    
    assert raw_output