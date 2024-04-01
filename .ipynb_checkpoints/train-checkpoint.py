# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
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

r"""Run off-policy evaluation training loop."""
import sys
import gc
import json
import os
import pickle
from absl import app
from absl import flags
from absl import logging
import gym
from gym.wrappers import time_limit
import numpy as np
import tensorflow as tf
from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment
from tensorflow.keras.models import load_model
import tqdm
import pdb
import csv
from policy_eval import utils
from policy_eval.actor import Actor
from policy_eval.behavior_cloning import BehaviorCloning
from policy_eval.dataset import D4rlDataset
from policy_eval.dataset import Dataset
from policy_eval.dual_dice import DualDICE
from policy_eval.model_based import ModelBased
from policy_eval.q_fitter import QFitter
from policy_eval.dataset import get_pendulum


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

EPS = np.finfo(np.float32).eps
FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'Reacher-v2',
                    'Environment for training/evaluation.')
flags.DEFINE_bool('d4rl', True, 'Whether to use D4RL envs and datasets.')
flags.DEFINE_string('save_dir', "/home/featurize/checkpoints",
                    'Directory to save behavior dataset and pretrained behavior or dynamics model.')
flags.DEFINE_integer('seed', 0, 'Fixed random seed for training.')
flags.DEFINE_float('lr', 3e-4, 'Critic learning rate.')
flags.DEFINE_float('lr_decay', 1, 'Weight decay.')
flags.DEFINE_float('weight_decay', 1e-5, 'Weight decay.')
flags.DEFINE_float('behavior_policy_std', None,
                   'Noise scale of behavior policy.')
flags.DEFINE_float('target_policy_std', 0.1, 'Noise scale of target policy.')
flags.DEFINE_integer('num_trajectories', 1000, 'Number of trajectories.')
flags.DEFINE_integer('sample_batch_size', 256, 'Batch size.')
flags.DEFINE_integer(
    'num_mc_episodes', 256,
    'Number of episodes to unroll to estimate Monte Carlo returns.')
flags.DEFINE_integer('num_updates', 500000, 'Number of updates.')
flags.DEFINE_integer('eval_interval', 10_000, 'Logging interval.')
flags.DEFINE_integer('log_interval', 10_000, 'Logging interval.')
flags.DEFINE_float('discount', 0.995, 'Discount used for returns.')
flags.DEFINE_float('tau', 0.005,
                   'Soft update coefficient for the target network.')
flags.DEFINE_string(
    'data_dir',
    '/tmp/policy_eval/trajectory_datasets/',
    'Directory with data for evaluation.')
flags.DEFINE_boolean('normalize_states', True, 'Whether to normalize states.')
flags.DEFINE_boolean('normalize_rewards', True, 'Whether to normalize rewards.')
flags.DEFINE_boolean('bootstrap', True,
                     'Whether to generated bootstrap weights.')
flags.DEFINE_enum('algo', 'fqe', ['fqe', 'dual_dice', 'mb', 'iw', 'dr'],
                  'Algorithm for policy evaluation.')
flags.DEFINE_float('noise_scale', 0.25, 'Noise scaling for data augmentation.')


def make_hparam_string(json_parameters=None, **hparam_str_dict):
  if json_parameters:
    for key, value in json.loads(json_parameters).items():
      if key not in hparam_str_dict:
        hparam_str_dict[key] = value
  return ','.join([
      '%s=%s' % (k, str(hparam_str_dict[k]))
      for k in sorted(hparam_str_dict.keys())
  ])


def main(_):
  tf.random.set_seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  hparam_str = make_hparam_string(
      seed=FLAGS.seed, env_name=FLAGS.env_name)
  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.save_dir, 'tb', hparam_str))
  summary_writer.set_as_default()

  # D4RL environments
  if "Pendulum" not in FLAGS.env_name:
      d4rl_env = gym.make(FLAGS.env_name)
      gym_spec = gym.spec(FLAGS.env_name)
      d4rl_env.seed(FLAGS.seed)
      d4rl_dataset = d4rl_env.get_dataset()
  # Other control environments (e.g. Pendulum)
  else:
      d4rl_env = gym.make("Pendulum-v1")
      gym_spec = gym.spec("Pendulum-v1")
      d4rl_env.seed(FLAGS.seed)
      d4rl_dataset = get_pendulum(dataset_type=FLAGS.env_name.split("-")[1])

  if gym_spec.max_episode_steps in [0, None]:  # Add TimeLimit wrapper.
    gym_env = time_limit.TimeLimit(d4rl_env, max_episode_steps=1000)
  else:
    gym_env = d4rl_env

  
  env = tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(gym_env))
  behavior_dataset = D4rlDataset(
      d4rl_dataset,
      normalize_states=FLAGS.normalize_states,
      normalize_rewards=FLAGS.normalize_rewards,
      noise_scale=FLAGS.noise_scale,
      bootstrap=FLAGS.bootstrap)

  # Save behavior dataset into pkl to avoid repreated preprocessing
  with open("{}/behavior_dataset_{}.pkl".format(FLAGS.save_dir, FLAGS.env_name),"wb") as f:
    pickle.dump(behavior_dataset,f)

  tf_dataset = behavior_dataset.with_uniform_sampling(FLAGS.sample_batch_size)
  tf_dataset_iter = iter(tf_dataset)

  # Load actor from pkl
  # with tf.io.gfile.GFile(FLAGS.d4rl_policy_filename, 'rb') as f:
  #   policy_weights = pickle.load(f)
  # actor = utils.D4rlActor(env, policy_weights, is_dapg=False)  

  # Create estimator
  if 'fqe' in FLAGS.algo or 'dr' in FLAGS.algo:
    model = QFitter(env.observation_spec().shape[0],
                    env.action_spec().shape[0], FLAGS.lr, FLAGS.lr_decay,
                    FLAGS.weight_decay, FLAGS.eval_interval, FLAGS.tau)
  elif 'mb' in FLAGS.algo:
    model = ModelBased(env.observation_spec().shape[0],
                       env.action_spec().shape[0], lr=FLAGS.lr, lr_decay=FLAGS.lr_decay,
                       weight_decay=FLAGS.weight_decay, eval_interval=FLAGS.eval_interval)
  elif 'dual_dice' in FLAGS.algo:
    model = DualDICE(env.observation_spec().shape[0],
                     env.action_spec().shape[0], FLAGS.weight_decay)
  if 'iw' in FLAGS.algo or 'dr' in FLAGS.algo:
    behavior = BehaviorCloning(env.observation_spec().shape[0],
                               env.action_spec(), FLAGS.lr, FLAGS.lr_decay, 
                               FLAGS.weight_decay, FLAGS.eval_interval)

  # @tf.function
  # def get_target_actions(states):
  #   return actor(
  #       tf.cast(behavior_dataset.unnormalize_states(states),
  #               env.observation_spec().dtype),
  #       std=FLAGS.target_policy_std)[1]

  # @tf.function
  # def get_target_logprobs(states, actions):
  #   log_probs = actor(
  #       tf.cast(behavior_dataset.unnormalize_states(states),
  #               env.observation_spec().dtype),
  #       actions=actions,
  #       std=FLAGS.target_policy_std)[2]
  #   if tf.rank(log_probs) > 1:
  #     log_probs = tf.reduce_sum(log_probs, -1)
  #   return log_probs

  min_reward = tf.reduce_min(behavior_dataset.rewards)
  max_reward = tf.reduce_max(behavior_dataset.rewards)
  min_state = tf.reduce_min(behavior_dataset.states, 0)
  max_state = tf.reduce_max(behavior_dataset.states, 0)


  @tf.function
  def update_step():
    (states, actions, next_states, rewards, masks, weights,
     _) = next(tf_dataset_iter)
    # initial_actions = get_target_actions(behavior_dataset.initial_states)
    # next_actions = get_target_actions(next_states)

    # if 'fqe' in FLAGS.algo or 'dr' in FLAGS.algo:
    #   loss = model.update(states, actions, next_states, next_actions, rewards, masks,
    #                weights, FLAGS.discount, min_reward, max_reward)
    if 'mb' in FLAGS.algo:
      loss = model.update(states, actions, next_states, rewards, masks,
                   weights)
    # elif 'dual_dice' in FLAGS.algo:
    #   model.update(behavior_dataset.initial_states, initial_actions,
    #                behavior_dataset.initial_weights, states, actions,
    #                next_states, next_actions, masks, weights, FLAGS.discount)

    if 'iw' in FLAGS.algo or 'dr' in FLAGS.algo:
      loss = behavior.update(states, actions, weights)
    
    return loss
  gc.collect()


  # Store pretrained Dynamics Model and Behavior
  pretrained_dir_path = "{}/{}/{}/checkpoint_{}_{}".format(FLAGS.save_dir,
                                                FLAGS.algo, 
                                                FLAGS.env_name, 
                                                FLAGS.lr, 
                                                FLAGS.lr_decay)

  if not os.path.exists(pretrained_dir_path):
    os.makedirs(pretrained_dir_path)

  # Dynamics Model
  if FLAGS.algo == "mb":
    with open("{}/ope.csv".format(pretrained_dir_path), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "dyn_loss", "reward_loss", "done_loss", "Estimate"])  # 写入标题
    epoch_dyn_loss = 0
    epoch_reward_loss = 0
    epoch_done_loss = 0

  # Importance Sampling
  elif FLAGS.algo == "iw":
    with open("{}/ope.csv".format(pretrained_dir_path), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "training_loss", "Estimate"])  # 写入标题
    epoch_loss = 0


  # Training
  for i in tqdm.tqdm(range(1,FLAGS.num_updates+1), desc='Running Training'):
    loss = update_step()

    # Model-Based
    if FLAGS.algo == "mb":
      dyn_loss, reward_loss, done_loss = loss
      epoch_dyn_loss += dyn_loss
      epoch_reward_loss += reward_loss
      epoch_done_loss += done_loss
    # Importance Sampling
    elif FLAGS.algo == "iw":
      epoch_loss += loss

    if i % FLAGS.eval_interval == 0:
      if FLAGS.algo == "mb":
        with open("{}/ope.csv".format(pretrained_dir_path), 'a', newline='') as file:
          writer = csv.writer(file)
          writer.writerow([i // FLAGS.eval_interval, epoch_dyn_loss.numpy() / FLAGS.eval_interval, 
                          epoch_reward_loss.numpy() / FLAGS.eval_interval,
                          epoch_done_loss.numpy() / FLAGS.eval_interval])
      elif FLAGS.algo == "iw":
        with open("{}/ope.csv".format(pretrained_dir_path), 'a', newline='') as file:
          writer = csv.writer(file)
          writer.writerow([i // FLAGS.eval_interval, epoch_loss.numpy() / FLAGS.eval_interval])


      if FLAGS.algo == "mb":
        epoch_dyn_loss = 0
        epoch_reward_loss = 0
        epoch_done_loss = 0
        ckpt = tf.train.Checkpoint(dynamics_net=model.dynamics_net,
                                 rewards_net=model.rewards_net,
                                 done_net=model.done_net)
        
      elif FLAGS.algo == "iw":
        epoch_loss = 0
        ckpt = tf.train.Checkpoint(actor=behavior.actor)

      manager = tf.train.CheckpointManager(ckpt, "{}/".format(pretrained_dir_path), max_to_keep=3)
      save_path = manager.save()
    

if __name__ == '__main__':
  app.run(main)
