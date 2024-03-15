"""Training and evaluation"""

import run_lib
from absl import app, flags
from ml_collections.config_flags import config_flags
import logging
import os

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config', None, 'Training configuration.', lock_config=True
)
config_flags.DEFINE_config_file(
    'classifier_config_nf', None, 'Training configuration.', lock_config=True
)
flags.DEFINE_string('workdir', None, 'Work directory.')
flags.DEFINE_enum('mode', None, ['train', 'eval'],
                  'Running mode: train or eval')
flags.DEFINE_string('eval_folder', 'eval', 'The folder name for storing evaluation results')
flags.mark_flags_as_required(['config', 'mode'])


def main(argv):
    # Set random seed
    run_lib.set_random_seed(FLAGS.config)

    if FLAGS.mode == 'train':
        logger = logging.getLogger()
        logger.setLevel('INFO')
        # Run the training pipeline
        run_lib.train(FLAGS.config)
    elif FLAGS.mode == 'eval':
        run_lib.evaluate(FLAGS.config)
    else:
        raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == '__main__':
    app.run(main)
