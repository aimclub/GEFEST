from importlib import import_module
from inspect import isclass, isfunction, ismethod, signature
from json import JSONDecoder, JSONEncoder
from typing import Any, Callable, Dict, Type, TypeVar, Union

MODULE_X_NAME_DELIMITER = '/'
INSTANCE_OR_CALLABLE = TypeVar('INSTANCE_OR_CALLABLE', object, Callable)
CLASS_PATH_KEY = '_class_path'


class Serializer(JSONEncoder, JSONDecoder):
    _to_json = 'to_json'
    _from_json = 'from_json'

    CODERS_BY_TYPE = {}

    def __init__(self, *args, **kwargs):
        for base_class, coder_name in [(JSONEncoder, 'default'), (JSONDecoder, 'object_hook')]:
            base_kwargs = {k: kwargs[k] for k in kwargs.keys() & signature(base_class.__init__).parameters}
            base_kwargs[coder_name] = getattr(self, coder_name)
            base_class.__init__(self, **base_kwargs)

        if not Serializer.CODERS_BY_TYPE:
            from gefest.core.opt.result import Result
            from gefest.core.structure.structure import Structure
            from gefest.core.structure.point import Point
            from gefest.core.structure.polygon import Polygon

            from .any import (
                any_from_json,
                any_to_json,
            )

            _to_json = Serializer._to_json
            _from_json = Serializer._from_json
            basic_serialization = {_to_json: any_to_json, _from_json: any_from_json}
            Serializer.CODERS_BY_TYPE = {
                Result: basic_serialization,
                Structure: basic_serialization,
                Polygon: basic_serialization,
                Point: basic_serialization

            }

    @staticmethod
    def _get_field_checker(obj: Union[INSTANCE_OR_CALLABLE, Type[INSTANCE_OR_CALLABLE]]) -> Callable[..., bool]:
        if isclass(obj):
            return issubclass
        return isinstance

    @staticmethod
    def _get_base_type(obj: Union[INSTANCE_OR_CALLABLE, Type[INSTANCE_OR_CALLABLE]]) -> int:
        contains = Serializer._get_field_checker(obj)
        for k_type in Serializer.CODERS_BY_TYPE:
            if contains(obj, k_type):
                return k_type
        return None

    @staticmethod
    def _get_coder_by_type(coder_type: Type, coder_aim: str):
        return Serializer.CODERS_BY_TYPE[coder_type][coder_aim]

    @staticmethod
    def dump_path_to_obj(obj: INSTANCE_OR_CALLABLE) -> Dict[str, str]:
        """Dumps the full path (module + name) to the input object into the dict

        Args:
            obj: object which path should be resolved (class, function or method)

        Returns:
            dictionary with path to the object
        """

        if isclass(obj) or isfunction(obj) or ismethod(obj):
            obj_name = obj.__qualname__
        else:
            obj_name = obj.__class__.__qualname__

        if getattr(obj, '__module__', None) is not None:
            obj_module = obj.__module__
        else:
            obj_module = obj.__class__.__module__
        return {
            CLASS_PATH_KEY: f'{obj_module}{MODULE_X_NAME_DELIMITER}{obj_name}'
        }

    def default(self, obj: INSTANCE_OR_CALLABLE) -> Dict[str, Any]:
        """Tries to encode objects that are not simply json-encodable to JSON-object

        Args:
            obj: object to be encoded (class, function or method)

        Returns:
            json object
        """
        if isfunction(obj) or ismethod(obj):
            return Serializer.dump_path_to_obj(obj)
        base_type = Serializer._get_base_type(obj)
        if base_type is not None:
            return Serializer._get_coder_by_type(base_type, Serializer._to_json)(obj)

        return JSONEncoder.default(self, obj)

    @staticmethod
    def _get_class(class_path: str) -> Type[INSTANCE_OR_CALLABLE]:
        """Gets the object type from the class_path

        Args:
            class_path: full path (module + name) of the class

        Returns:
            class, function or method type
        """

        module_name, class_name = class_path.split(MODULE_X_NAME_DELIMITER)
        obj_cls = import_module(module_name)
        for sub in class_name.split('.'):
            obj_cls = getattr(obj_cls, sub)
        return obj_cls

    def object_hook(self, json_obj: Dict[str, Any]) -> Union[INSTANCE_OR_CALLABLE, dict]:
        """Decodes every JSON-object to python class/func object or just returns dict

        Args:
            json_obj: dict to be decoded into Python class, function or
            method object only if it has some special fields

        Returns:
            Python class, function or method object OR input if it's just a regular dict
        """

        if CLASS_PATH_KEY in json_obj:
            obj_cls = Serializer._get_class(json_obj[CLASS_PATH_KEY])
            del json_obj[CLASS_PATH_KEY]
            base_type = Serializer._get_base_type(obj_cls)
            if isclass(obj_cls) and base_type is not None:
                return Serializer._get_coder_by_type(base_type, Serializer._from_json)(obj_cls, json_obj)
            elif isfunction(obj_cls) or ismethod(obj_cls):
                return obj_cls
            raise TypeError(f'Parsed obj_cls={obj_cls} is not serializable, but should be')
        return json_obj
