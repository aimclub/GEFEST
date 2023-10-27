from core.simulation_case.case import Case
from core.utils.files import get_project_root
import os

os.chdir(f'{get_project_root()}/cases/ob2')
case = Case('./ob_example.json')
case.prepare_case()
case.run()
