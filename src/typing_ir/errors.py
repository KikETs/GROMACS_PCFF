from __future__ import annotations


class TypingIRError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        path: str | None = None,
        line: int | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.path = path
        self.line = line
        super().__init__(self.__str__())

    def __str__(self) -> str:
        location = ""
        if self.path is not None and self.line is not None:
            location = f" ({self.path}:{self.line})"
        elif self.path is not None:
            location = f" ({self.path})"
        return f"{self.code}: {self.message}{location}"


class ParseError(TypingIRError):
    pass


class SchemaError(TypingIRError):
    pass
