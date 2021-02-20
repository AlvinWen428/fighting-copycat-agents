class GanModule:
    def __init__(self, input_dims, output_dims):
        self._input_dims = input_dims
        self._output_dims = output_dims

    @property
    def input_dims(self):
        return self._input_dims

    @property
    def output_dims(self):
        return self._output_dims
