import re
import inspect
import pprint

from core.dsl_nodes import *

from core.dsl_symbolic_interpreter import SYMBOL_RULES
from core.transformation_factory import TransformationFactory

def  generate_symbolic_dsl_reference_markdown() -> str:
    sections = []
    operation_map = TransformationFactory.OPERATION_MAP 

    for cmd_name, sym_info in sorted(SYMBOL_RULES.items()):
        
        op_entry = operation_map.get(cmd_name)
        
        cmd_cls = op_entry
        if inspect.isfunction(op_entry):
            try:
                class_name_match = re.search(r'\b(\w+)\(', inspect.getsource(op_entry))
                if class_name_match:
                    cmd_cls = globals().get(class_name_match.group(1))
                else:
                    cmd_cls = None
            except OSError:
                cmd_cls = None
        
        human_name = ' '.join(re.findall('[A-Z][a-z0-9]*', cmd_cls.__name__)).replace('Command', '').replace('Grid', 'Grid').strip() if cmd_cls else cmd_name
        if not human_name: human_name = cmd_name.replace('_', ' ').title()

        symbolic_rules_str = pprint.pformat({cmd_name: sym_info}, indent=4, width=80)
        
        explanation_text = "No explanation available."
        if cmd_cls and inspect.isclass(cmd_cls) and hasattr(cmd_cls, 'describe') and inspect.ismethod(cmd_cls.describe):
            explanation_text = cmd_cls.describe().strip()
        else:
            sections.append(f"--- Error: Command '{cmd_name}' class '{cmd_cls.__name__ if cmd_cls else 'N/A'}' or its 'describe' method is missing/invalid. ---")
            continue
        synthesis_rules_str = "No synthesis rules defined."
        if cmd_cls and hasattr(cmd_cls, 'synthesis_rules') and isinstance(cmd_cls.synthesis_rules, dict):
            synthesis_rules_str = pprint.pformat(cmd_cls.synthesis_rules, indent=4, width=80)
        
        markdown_block = f"""
                ### {human_name}

                * **Symbolic Rules:**
                ```python
                {symbolic_rules_str}
                    Synthesis Rules:

                {synthesis_rules_str}

                    Explanation:

                {explanation_text}

        """
        sections.append(markdown_block.strip())

    return "\n\n---\n\n".join(sections)