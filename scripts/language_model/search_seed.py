import subprocess
import sys
import time
import argparse

parser = argparse.ArgumentParser(
    description='seed',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--task',
    type=str,
    default='RTE',
    help='task name')
parser.add_argument('--training_steps',
                    type=int,
                    default=800,
                    help='If specified, epochs will be ignored.')
args = parser.parse_args()

def test_xlnet_finetune_glue(dataset, training_steps):
            arguments = ['--batch_size', '32', '--task_name', dataset, '--optimizer', 'Adam',
                                                     '--gpu', '2', '--training_steps', str(training_steps), '--lr', '3e-5', '--warmup_ratio', '0.25']
            for i in range(0, 100):
                arguments_tmp = arguments + ['--seed', str(i)]
                process = subprocess.call([sys.executable, sys.path[0]+'/run_glue.py']+ arguments_tmp)
                time.sleep(5)

test_xlnet_finetune_glue(args.task, args.training_steps)
