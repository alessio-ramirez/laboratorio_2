import numpy as np

class MeasurementArray:
    def __init__(self, values, errors, name=None):
        self.values = np.asarray(values)
        self.errors = np.asarray(errors)
        self.name = name
        if self.values.shape != self.errors.shape:
            raise ValueError("Values and errors must have the same shape.")
    
    def __repr__(self):
        return f"MeasurementArray(values={self.values}, errors={self.errors})"
    
    _derivatives = {
        np.add: lambda a, b: (1, 1),
        np.subtract: lambda a, b: (1, -1),
        np.multiply: lambda a, b: (b, a),
        np.divide: lambda a, b: (1 / b, -a / (b ** 2)),
        np.true_divide: lambda a, b: (1 / b, -a / (b ** 2)),
        np.sin: lambda x: (np.cos(x),),
        np.cos: lambda x: (-np.sin(x),),
        np.tan: lambda x: (1 / (np.cos(x) ** 2 + 1e-12),),
        np.arcsin: lambda x: (1 / np.sqrt(1 - x ** 2 + 1e-12),),
        np.arccos: lambda x: (-1 / np.sqrt(1 - x ** 2 + 1e-12),),
        np.arctan: lambda x: (1 / (1 + x ** 2),),
        np.exp: lambda x: (np.exp(x),),
        np.log: lambda x: (1 / (x + 1e-12),),
        np.log10: lambda x: (1 / ((x + 1e-12) * np.log(10)),),
        np.sqrt: lambda x: (1 / (2 * np.sqrt(x + 1e-12)),),
        np.power: lambda x, y: (y * x ** (y - 1), x ** y * np.log(np.where(x != 0, x, 1e-12))),
        np.negative: lambda x: (-1,),
        np.reciprocal: lambda x: (-1 / (x ** 2 + 1e-12),),
    }
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != '__call__':
            return NotImplemented
        
        if ufunc not in self._derivatives:
            return NotImplemented
        
        input_values = []
        input_errors = []
        for inp in inputs:
            if isinstance(inp, MeasurementArray):
                input_values.append(inp.values)
                input_errors.append(inp.errors)
            else:
                input_values.append(inp)
                input_errors.append(np.zeros_like(inp) if np.isscalar(inp) else np.zeros_like(inp, dtype=float))
        
        try:
            output_values = ufunc(*input_values, **kwargs)
        except Exception as e:
            return NotImplemented
        
        try:
            derivatives = self._derivatives[ufunc](*input_values)
        except Exception as e:
            return NotImplemented
        
        if not isinstance(derivatives, tuple):
            derivatives = (derivatives,)
        
        error_squared = np.zeros_like(output_values)
        for deriv, error in zip(derivatives, input_errors):
            term = deriv * error
            error_squared += term ** 2
        
        output_errors = np.sqrt(error_squared)
        
        return MeasurementArray(output_values, output_errors)
    
    def __add__(self, other):
        return np.add(self, other)
    
    def __radd__(self, other):
        return np.add(other, self)
    
    def __sub__(self, other):
        return np.subtract(self, other)
    
    def __rsub__(self, other):
        return np.subtract(other, self)
    
    def __mul__(self, other):
        return np.multiply(self, other)
    
    def __rmul__(self, other):
        return np.multiply(other, self)
    
    def __truediv__(self, other):
        return np.divide(self, other)
    
    def __rtruediv__(self, other):
        return np.divide(other, self)
    
    def __pow__(self, other):
        return np.power(self, other)
    
    def __rpow__(self, other):
        return np.power(other, self)
    
    def to_dict(self):
        return {'value': self.values, 'error': self.errors}
    
    @classmethod
    def from_dict(cls, data):
        return cls(data['value'], data['error'])