class LoadOptions:
    def __init__(self, ctx_length: int = 4096, **kwargs) -> None:
        self.ctx_length = ctx_length
        for name, value in kwargs.items():
            setattr(self, name, value)
