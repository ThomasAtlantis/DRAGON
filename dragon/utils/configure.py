from argparse import ArgumentParser
from rich_argparse import RichHelpFormatter
from ast import literal_eval
from functools import partial
import typing


def get_attr_dict(clas):
    attr_dict = clas.__dict__
    if object not in clas.__bases__:
        for base_class in clas.__bases__:
            attr_dict = get_attr_dict(base_class) | attr_dict
    return attr_dict


def get_classes(clas):
    classes = [clas]
    if object not in clas.__bases__:
        for base_class in clas.__bases__:
            classes.extend(get_classes(base_class))
    return classes


def instantiate_nested_class(instance, clas):
    for attr_name, field in clas.__dict__.items():
        if isinstance(field, type):
            nested_instance = field()
            setattr(instance, attr_name, nested_instance)
            instantiate_nested_class(nested_instance, nested_instance.__class__)
        elif isinstance(field, Field):
            setattr(instance, attr_name, field.data)


def add_arguments_recursive(instance, parser, prefix=""):
    group = parser.add_argument_group(":" + prefix[:-1]) if prefix else parser
    group.description = instance.__doc__
    cls__dict__ = get_attr_dict(instance.__class__)
    obj__dict__ = instance.__dict__
    for attr_name, value in cls__dict__.items():
        if isinstance(value, type):
            add_arguments_recursive(obj__dict__[attr_name], parser, prefix + attr_name + ".")
        elif isinstance(value, Field) and not value.freeze:
            kwargs = dict( 
                default=value.data, 
                required=value.required, 
                help=value.help
            )
            if value.type == bool:                        
                kwargs["action"] = "store_true" if value.data is False else "store_false"
            else:
                kwargs["type"] = value.type
                kwargs["metavar"] = value.type.__name__
            group.add_argument(f"--{prefix + attr_name}", **kwargs)


def get_all_items(instance):
    dict_ = {}
    cls__dict__ = get_attr_dict(instance.__class__)
    obj__dict__ = instance.__dict__
    for attr_name, value in cls__dict__.items():
        if isinstance(value, type):
            dict_[attr_name] = get_all_items(obj__dict__[attr_name])
        elif isinstance(value, Field):
            dict_[attr_name] = obj__dict__[attr_name]
    return dict_

class Field:

    def __init__(self, type_, default=None, required=False, help="", freeze=False):
        self.data = default
        self.type = type_
        self.help = help
        self.required = required
        self.freeze = freeze

        if self.type in [typing.Tuple, typing.List, tuple, list]:
            self.type = lambda x: literal_eval(x)
    
class Configure:
        
    def __init__(self):    
        for clas in get_classes(self.__class__):
            instantiate_nested_class(self, clas)
    
    def as_dict(self):
        return get_all_items(self)

    def _add_arguments(self, parser: ArgumentParser):
        add_arguments_recursive(self, parser)
    
    def _parse_args(self, parser: ArgumentParser):
        args = parser.parse_args()
        for key, value in vars(args).items():
            *namespaces, attr_name = key.split(".")
            cls = self
            for cls_name in namespaces:
                cls = getattr(cls, cls_name)
            cls.__setattr__(attr_name, value)
        
    def parse_sys_args(self):
        parser = ArgumentParser(formatter_class=partial(RichHelpFormatter, max_help_position=80))
        self._add_arguments(parser)
        self._parse_args(parser)
