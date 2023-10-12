from ..decorators import *
from . import Plugin


class CalculatorPlugin(Plugin):
    @function
    def evaluate(
        self,
        expression: str = param(
            "The math expression to evaluate. Must be an valid python expression."
        ),
    ):
        """Execute a math expression and return the result. The expression must be an valid python expression that can be execuated by `eval()`."""
        result = eval(expression)
        return result
