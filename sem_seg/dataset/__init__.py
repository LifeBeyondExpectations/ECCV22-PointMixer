from importlib import import_module

def get(name):
    module_name = 'dataset.' + name.lower()
    module = import_module(module_name)
    # return getattr(module, name)
    return module