from functools import partial


class OperationWrap:
    def __init__(
        self,
        executor,
        operations,
        operation_chance,
        operations_probs,
        domain,
        postproc_func,
        postprocess_rules,
        attempts,
    ):
        self.executor = executor
        self.operations = operations
        self.operation_chance = operation_chance
        self.operations_probs = operations_probs
        self.domain = domain
        self.postproc_func = postproc_func
        self.postprocess_rules = postprocess_rules
        self.attempts = attempts

    def __repr__(self):
        return f'{self.operations[0].__name__}'

    def __call__(self, *args, **kwargs):
        executor = partial(
            self.executor,
            operations=self.operations,
            operation_chance=self.operation_chance,
            operations_probs=self.operations_probs,
            domain=self.domain,
            **kwargs,
        )
        operation_result = executor(*args)
        corrected = self.postproc_func(
            operation_result,
        )
        if executor.func.__name__ == 'mutate_structure':
            corrected = corrected[0]
        return corrected
