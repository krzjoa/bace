# This piece of code is got from http://stackoverflow.com/questions/8100166/inheriting-methods-docstrings-in-python


import types

def inherit_docstring(cls):
    for name, func in vars(cls).items():
        if isinstance(func, types.FunctionType) and not func.__doc__:
            #print func, 'needs doc'
            for parent in cls.__bases__:
                parfunc = getattr(parent, name, None)
                if parfunc and getattr(parfunc, '__doc__', None):
                    func.__doc__ = parfunc.__doc__
                    break
    return cls