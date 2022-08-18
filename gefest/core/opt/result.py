import json
import os
from dataclasses import dataclass
from typing import Union, List, Optional

from gefest.core.serialization.serializer import Serializer
from gefest.core.structure.structure import Structure


@dataclass
class Result:
    name: str
    best_structure: Union[Structure, List[Structure]]
    fitness: float
    metadata: dict

    def save(self, json_file_path: os.PathLike = None) -> Optional[str]:
        if json_file_path is None:
            return json.dumps(self, indent=4, cls=Serializer)
        with open(json_file_path, mode='w') as json_fp:
            json.dump(self, json_fp, indent=4, cls=Serializer)

    @staticmethod
    def load(json_str_or_file_path: Union[str, os.PathLike] = None) -> 'Result':
        try:
            return json.loads(json_str_or_file_path, cls=Serializer)
        except json.JSONDecodeError as exc:
            with open(json_str_or_file_path, mode='r') as json_fp:
                return json.load(json_fp, cls=Serializer)
