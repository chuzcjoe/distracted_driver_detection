from .augmentations import SSDAugmentation
from lib.utils.evaluate_utils import EvalVOC, EvalCOCO, EvalDEEPV

eval_solver_map = {'FACE': EvalDEEPV,
                   'TEMP_LP': EvalDEEPV,
                   'LMDBDet': EvalDEEPV,
                   'VOC0712': EvalVOC,
                   'COCO2014': EvalCOCO}


def eval_solver_factory(loader, cfg):
    Solver = eval_solver_map[cfg.DATASET.NAME]
    eval_solver = Solver(loader, cfg)
    return eval_solver
