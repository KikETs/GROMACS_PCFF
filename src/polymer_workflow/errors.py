from __future__ import annotations


class PolymerWorkflowError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class PolymerWorkflowSpecError(PolymerWorkflowError):
    pass


class PolymerWorkflowReportError(PolymerWorkflowError):
    pass
