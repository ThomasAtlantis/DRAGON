from .singleton import SingletonType
from argparse import ArgumentParser
from rich_argparse import RichHelpFormatter
from ast import literal_eval
from functools import partial
import typing


class Field:

    def __init__(self, type_, default=None, required=False, help="", freeze=False):
        self.data = default
        self.type = type_
        self.help = help
        self.required = required
        self.freeze = freeze

        if self.type in [typing.Tuple, typing.List, tuple, list]:
            self.type = lambda x: literal_eval(x)
    
class Configure(metaclass=SingletonType):
        
    def __init__(self):
        
        def instantiate_nested_class(instance):
            nested_class = instance.__class__
            for attr_name, field in nested_class.__dict__.items():
                if isinstance(field, type):
                    nested_instance = field()
                    setattr(instance, attr_name, nested_instance)
                    instantiate_nested_class(nested_instance)
                elif isinstance(field, Field):
                    setattr(instance, attr_name, field.data)
        instantiate_nested_class(self)
    
    def as_dict(self):
        def get_all_items(instance):
            dict_ = {}
            cls__dict__ = instance.__class__.__dict__
            obj__dict__ = instance.__dict__
            for attr_name, value in cls__dict__.items():
                if isinstance(value, type):
                    dict_[attr_name] = get_all_items(obj__dict__[attr_name])
                elif isinstance(value, Field):
                    dict_[attr_name] = obj__dict__[attr_name]
            return dict_
        items = get_all_items(self)
        return items

    def _add_arguments(self, parser: ArgumentParser):
        def add_arguments_recursive(instance, parser, prefix=""):
            group = parser.add_argument_group(":" + prefix[:-1]) if prefix else parser
            group.description = instance.__doc__
            cls__dict__ = instance.__class__.__dict__
            obj__dict__ = instance.__dict__
            for attr_name, value in cls__dict__.items():
                if isinstance(value, type):
                    add_arguments_recursive(obj__dict__[attr_name], parser, prefix + attr_name + ".")
                elif isinstance(value, Field) and not value.freeze:
                    group.add_argument(
                        f"--{prefix + attr_name}", type=value.type, 
                        default=value.data, required=value.required, 
                        help=value.help, metavar=value.type.__name__)
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
