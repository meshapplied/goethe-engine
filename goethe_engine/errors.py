"""The engine's error taxonomy (spec/errors.json): every refusal a caller sees carries one code. Stdlib only."""


class EngineError(Exception):
    code = "error"

    def __init__(self, msg="", param=None):
        super().__init__(msg)
        self.param = param                 # the request parameter at fault (parameter errors)


class Refused(EngineError):
    """The pack, its weights or the device cannot serve the request; the message names the reason."""
    code = "refused"


class OutOfMemory(Refused):
    code = "oom"


class UnsupportedParameter(EngineError, ValueError):
    code = "unsupported_parameter"


class InvalidParameter(EngineError, ValueError):
    code = "invalid_parameter"


class NoContract(Refused):
    code = "no_contract"
