from __future__ import annotations


class ChemPerceptionError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class SchemaError(ChemPerceptionError):
    pass
