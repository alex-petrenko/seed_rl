# coding=utf-8
# Copyright 2019 The SEED Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""VTrace (IMPALA) binary for DeepMind Lab.

Actor and learner are in the same binary so that all flags are shared.
"""

from absl import app
from absl import flags
from seed_rl.agents.vtrace import learner
from seed_rl.common import actor
from seed_rl.atari import networks_vtrace
from seed_rl.envs.atari.atari_utils import ATARI_W, ATARI_H
from seed_rl.algorithms.utils.arguments import default_cfg
from seed_rl.envs.create_env import create_env

import tensorflow as tf

import logging
from logging import handlers

class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }

    def __init__(self,filename,level='info',when='D',backCount=3,
                 fmt='%(asctime)s : %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')
        th.setFormatter(format_str)
        self.logger.addHandler(sh)
        self.logger.addHandler(th)

FLAGS = flags.FLAGS

# Optimizer settings.
flags.DEFINE_float('learning_rate', 0.00048, 'Learning rate.')
flags.DEFINE_float('adam_epsilon', 3.125e-7, 'Adam epsilon.')
flags.DEFINE_integer('stack_size', 4, 'Number of frames to stack.')
flags.DEFINE_integer('num_action_repeats', 4, 'Number of action repeats.')

def create_agent(action_space, unused_env_observation_space,
                 unused_parametric_action_distribution):
  return networks_vtrace.AtariConvnet(action_space.n)


def create_optimizer(final_iteration):
  learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
      FLAGS.learning_rate, final_iteration, 0)
  # optimizer = tf.keras.optimizers.Adam(learning_rate_fn, beta_1=0,
  #                                      epsilon=FLAGS.adam_epsilon)

  optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate, epsilon=FLAGS.adam_epsilon)
  return optimizer, learning_rate_fn

def create_atari_env(x):
    env_name = 'atari_breakout'
    cfg = default_cfg(env=env_name, algo=None)
    cfg.pixel_format = 'HWC'
    cfg.res_w = ATARI_W
    cfg.res_h = ATARI_H
    cfg.env_framestack = 4
    return create_env(env_name, cfg=cfg)


def main(argv):
  fps_log = Logger('fps.log', level='info')
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  if FLAGS.run_mode == 'actor':
    actor.actor_loop(create_atari_env)
  elif FLAGS.run_mode == 'learner':
    learner.learner_loop(create_atari_env,
                         create_agent,
                         create_optimizer,
                         fps_log)
  else:
    raise ValueError('Unsupported run mode {}'.format(FLAGS.run_mode))


if __name__ == '__main__':
  app.run(main)
