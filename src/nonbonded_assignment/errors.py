from __future__ import annotations


class NonbondedAssignmentError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class RuleSchemaError(NonbondedAssignmentError):
    pass


class AssignmentReportError(NonbondedAssignmentError):
    pass
