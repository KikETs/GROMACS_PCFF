from .errors import ParseError, SchemaError, TypingIRError
from .ir import (
    CANONICALIZATION_ALGORITHM,
    IR_STAGE,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    dumps_ir,
    load_ir,
    loads_ir,
    validate_ir,
    write_ir,
)
from .parser import parse_file

__all__ = [
    "CANONICALIZATION_ALGORITHM",
    "IR_STAGE",
    "ParseError",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "SchemaError",
    "TypingIRError",
    "dump_ir",
    "dumps_ir",
    "load_ir",
    "loads_ir",
    "parse_file",
    "validate_ir",
    "write_ir",
]


def dump_ir(path, ir):
    write_ir(path, ir)
