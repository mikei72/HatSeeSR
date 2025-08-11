# flake8: noqa
import os.path as osp

import archs
import data
import models
from basicsr.test import test_pipeline

from models import hat_model

if __name__ == '__main__':
    from basicsr.models import MODEL_REGISTRY

    print("Registered models: ", list(MODEL_REGISTRY.keys()))

    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
