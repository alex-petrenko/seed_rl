import collections
from seed_rl.common import utils
import tensorflow as tf

AgentOutput = collections.namedtuple('AgentOutput',
                                     'action policy_logits baseline')


class AtariConvnet(tf.Module):
  def __init__(self, num_actions):
    super(AtariConvnet, self).__init__(name='AtariConvnet')

    # Parameters and layers for unroll.
    self._num_actions = num_actions
    self._core = tf.keras.layers.LSTMCell(512)

    # Parameters and layers for _torso.
    #(32, 8, 4), (64, 4, 2), (128, 3, 2
    self._stacks = [
      tf.keras.layers.Conv2D(32, [8, 8], 4, padding='valid', activation='relu'),
      tf.keras.layers.Conv2D(64, [4, 4], 2, padding='valid', activation='relu'),
      tf.keras.layers.Conv2D(64, [3, 3], 1, padding='valid', activation='relu'),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(512, activation='relu'),
    ]
    self._conv_to_linear = tf.keras.layers.Dense(512)

    # Layers for _head.
    self._policy_logits = tf.keras.layers.Dense(
        self._num_actions, name='policy_logits')
    self._baseline = tf.keras.layers.Dense(1, name='baseline')

  @tf.function
  def initial_state(self, batch_size):
    return self._core.get_initial_state(batch_size=batch_size, dtype=tf.float32)

  def _torso(self, prev_action, env_output):
    reward, _, frame = env_output

    # Convert to floats.
    frame = tf.cast(frame, tf.float32)

    frame /= 255
    conv_out = frame
    for stack in self._stacks:
      conv_out = stack(conv_out)

    conv_out = tf.nn.relu(conv_out)
    conv_out = tf.keras.layers.Flatten()(conv_out)

    conv_out = self._conv_to_linear(conv_out)
    conv_out = tf.nn.relu(conv_out)

    return conv_out

  def _head(self, core_output):
    policy_logits = self._policy_logits(core_output)
    baseline = tf.squeeze(self._baseline(core_output), axis=-1)

    # Sample an action from the policy.
    new_action = tf.random.categorical(policy_logits, 1, dtype=tf.int64)
    new_action = tf.squeeze(new_action, 1, name='action')

    return AgentOutput(new_action, policy_logits, baseline)

  @tf.function
  def __call__(self, input_, core_state, unroll=False,
               is_training=False):
    if not unroll:
      # Add time dimension.
      input_ = tf.nest.map_structure(lambda t: tf.expand_dims(t, 0), input_)
    prev_actions, env_outputs = input_
    outputs, core_state = self._unroll(prev_actions, env_outputs, core_state)
    if not unroll:
      # Remove time dimension.
      outputs = tf.nest.map_structure(lambda t: tf.squeeze(t, 0), outputs)

    return outputs, core_state

  def _unroll(self, prev_actions, env_outputs, core_state):
    unused_reward, done, unused_observation = env_outputs

    torso_outputs = utils.batch_apply(self._torso, (prev_actions, env_outputs))

    initial_core_state = self._core.get_initial_state(
        batch_size=tf.shape(prev_actions)[1], dtype=tf.float32)
    core_output_list = []
    for input_, d in zip(tf.unstack(torso_outputs), tf.unstack(done)):
      # If the episode ended, the core state should be reset before the next.
      core_state = tf.nest.map_structure(
          lambda x, y, d=d: tf.where(
              tf.reshape(d, [d.shape[0]] + [1] * (x.shape.rank - 1)), x, y),
          initial_core_state,
          core_state)
      core_output, core_state = self._core(input_, core_state)
      core_output_list.append(core_output)
    core_outputs = tf.stack(core_output_list)

    return utils.batch_apply(self._head, (core_outputs,)), core_state
