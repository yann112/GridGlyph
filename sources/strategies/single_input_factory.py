from typing import Dict, Type, Any, List
from strategies.single_input_strategies import SingleGridStrategy

class SingleInputStrategyFactory:
    STRATEGY_REGISTRY: Dict[str, Type[SingleGridStrategy]] = {}

    @classmethod
    def register_strategy(cls, name: str):
        """
        Decorator to register a strategy class.
        Usage:
            @register_strategy("greedy")
            class GreedySynthesisStrategy(SingleGridStrategy): ...
        """
        def decorator(strategy_class: Type[SingleGridStrategy]):
            cls.STRATEGY_REGISTRY[name] = strategy_class
            return strategy_class
        return decorator

    @classmethod
    def list_strategies(cls) -> List[str]:
        """List all registered strategy names"""
        return list(cls.STRATEGY_REGISTRY.keys())

    @classmethod
    def get_metadata(cls, name: str) -> Dict[str, Any]:
        """Get strategy metadata without instantiating it"""
        if name not in cls.STRATEGY_REGISTRY:
            raise ValueError(f"Unknown strategy '{name}'")
        strategy_class = cls.STRATEGY_REGISTRY[name]
        return strategy_class.get_metadata()

    @classmethod
    def create_strategy(cls, name: str, **kwargs) -> SingleGridStrategy:
        """Create an instance of the named strategy with given kwargs"""
        if name not in cls.STRATEGY_REGISTRY:
            raise ValueError(f"Unknown strategy '{name}'")
        strategy_class = cls.STRATEGY_REGISTRY[name]
        return strategy_class(**kwargs)