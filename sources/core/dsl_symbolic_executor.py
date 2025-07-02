import numpy as np
import logging
from typing import Any, Dict, Optional, Iterator
from core.dsl_nodes import *


class DSLExecutor:
    def __init__(self, root_command: AbstractTransformationCommand, initial_puzzle_input: np.ndarray, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.root_command = root_command
        self.initial_puzzle_input = initial_puzzle_input
        self.variables: Dict[str, Any] = {}

        self.logger.info("Executor initialization started.")
        self._initialize_command_tree(self.root_command)
        self._inject_executor_context(self.root_command)
        self.logger.info("Executor initialization complete.")

    def _initialize_command_tree(self, command: AbstractTransformationCommand):
        if isinstance(command, InputGridReference):
            command.set_initial_puzzle_input(self.initial_puzzle_input)
            self.logger.debug(f"Initialized InputGridReference with puzzle input.")

        for child_cmd in command.get_children_commands():
            self._initialize_command_tree(child_cmd)

    def _inject_executor_context(self, command: AbstractTransformationCommand):
        if hasattr(command, 'set_executor_context'):
            command.set_executor_context(self)
            self.logger.debug(f"Injected Executor context into {command.__class__.__name__}.")

        for child_cmd in command.get_children_commands():
            self._inject_executor_context(child_cmd)

    def execute_program(self) -> np.ndarray:
        self.logger.info("Starting DSL program execution.")
        try:
            final_grid = self.root_command.execute(self.initial_puzzle_input)
            self.logger.info("DSL program execution completed successfully.")
            return final_grid
        except Exception as e:
            self.logger.error(f"Error during DSL program execution: {e}", exc_info=True)
            raise