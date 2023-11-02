import inspect
import pdb

def class2dic(config):
    assert inspect.isclass(config)
    config_dic = dict(config.__dict__)
    del_key_list = []
    for key in config_dic:
        if key.startswith('__') and key.endswith('__'):
            del_key_list.append(key)
    
    for key in del_key_list:
        config_dic.pop(key)
    return config_dic


def class2dic_iterative(config):
    assert inspect.isclass(config)
    config_dic = dict(config.__dict__)
    del_key_list = []
    for key in config_dic:
        if key.startswith('__') and key.endswith('__'):
            del_key_list.append(key)
    
    for key in del_key_list:
        config_dic.pop(key)
    
    for key in config_dic:
        if inspect.isclass(config_dic[key]):
            config_dic[key] = class2dic_iterative(config_dic[key])
    return config_dic


def get_module(config=None, *args, **kwargs):
    import models
    import datasets
    if config != None:
        if type(config) != dict:
            config = class2dic(config)
        
        for key in config:
            kwargs[key] = config[key]
    
    assert 'type' in kwargs
    method_code = eval(kwargs['type'])

    args_count = method_code.__init__.__code__.co_argcount
    input_params = method_code.__init__.__code__.co_varnames[1:args_count]

    new_kwargs = {}
    for i, value in enumerate(args):
        new_kwargs[input_params[i]] = value
    
    for key in kwargs:
        if key in input_params:
            new_kwargs[key] = kwargs[key]
    
    result_module = method_code(**new_kwargs)
    return result_module