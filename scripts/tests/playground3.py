import os
import subprocess
import sys
import time
import datetime

import pytest
import mxnet as mx

@pytest.mark.serial
@pytest.mark.gpu
@pytest.mark.remote_required
@pytest.mark.integration
@pytest.mark.parametrize('dataset', ['MRPC'])
def test_xlnet_finetune_glue(dataset):
    arguments = ['--batch_size', '12', 'task_name', dataset,
                 '--gpu', '0', '--epochs', '1', '--debug']
    process = subprocess.check_call([sys.executable, './scripts/language_model/run_glue.py']
                                    + arguments)
    time.sleep(5)