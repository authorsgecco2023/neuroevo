# Copyright 2022 The Neuroevo Authors.
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

import argparse
import glob
import os
import pickle
import random
import sys
import warnings

from mpi4py import MPI
import numpy as np
import torch


sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')

warnings.filterwarnings("ignore", category=UserWarning)
sys.setrecursionlimit(2**31-1)

parser = argparse.ArgumentParser()

parser.add_argument('--states_path', '-s', type=str, required=True,
                    help="Path to the saved states <=> "
                         "data/states/<env_path>/<additional_arguments>/"
                         "<bots_path>/<population_size>/")

parser.add_argument('--nb_tests', '-t', type=int, default=10,
                    help="Number of tests to evaluate the agents on.")

parser.add_argument('--nb_obs_per_test', '-o', type=int, default=2**31-1,
                    help="Number of observations per test.")

args = parser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

MAX_INT = 2**31-1

# Backward Compatibility for Control Task experiments
class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "bots.static.rnn.control":
            renamed_module = "bots.network.static.rnn.control"
        elif module == "bots.dynamic.rnn.control":
            renamed_module = "bots.network.dynamic.rnn.control"

        return super(RenameUnpickler, self).find_class(renamed_module, name)

"""
Process arguments
"""

if args.states_path[-1] == '/':
    args.states_path = args.states_path[:-1]

split_path = args.states_path.split('/')

env_path = split_path[-4]
additional_arguments = split_path[-3]
bots_path = split_path[-2]
pop_size = int(split_path[-1])

split_additional_arguments = additional_arguments.split('~')

if 'score' in env_path:

    steps = split_additional_arguments[0].split('.')[1]
    task = split_additional_arguments[1].split('.')[1]
    transfer = split_additional_arguments[2].split('.')[1]
    trials = split_additional_arguments[3].split('.')[1]

else: # 'imitate' in env_path:

    merge = split_additional_arguments[0].split('.')[1]
    steps = split_additional_arguments[1].split('.')[1]
    task = split_additional_arguments[2].split('.')[1]
    transfer = split_additional_arguments[3].split('.')[1]

"""
Initialize environment
"""


import gym
from utils.functions.control import get_task_name

emulator = gym.make(get_task_name(task))

hide_score = lambda x : x


"""
Import bots
"""

if 'dynamic' in bots_path:
    from bots.network.dynamic.rnn.control import Bot
else: # 'static' in bots_path:
    from bots.network.static.rnn.control import Bot

"""
Distribute workload
"""

files = [os.path.basename(x) for x in glob.glob(args.states_path + '/*')]

gens = []

for file in files:
    if file.isdigit() and os.path.isdir(args.states_path + '/' + file):
        gens.append(int(file))

gens.sort()

process_gens = []

for i in range(len(gens)):
    if i % size == rank:
        process_gens.append(gens[i])

for gen in process_gens:

    print('Gen : ' + str(gen))

    path = args.states_path + '/' + str(gen) + '/'

    if os.path.isfile(path + 'scores.npy'):
        continue

    pkl_files = [os.path.basename(x) for x in glob.glob(path + '*.pkl')]

    state_files = []

    for pkl_file in pkl_files:

        if pkl_file[:-4].isdigit():

            state_files.append(pkl_file)

    if len(state_files) == 0:
        raise Exception("Directory '" + path + "' empty.")

    try:

        with open(path + '0.pkl', 'rb') as f:
            state = RenameUnpickler(f).load()

    except Exception:

        print("File '" + path + "0.pkl' doesn't exist / is corrupted.")

    if len(state) == 3:

        full_seed_list, _, _ = state

    else: # len(state) == 4:

        _, _, latest_fitnesses_and_bot_sizes, bots = state

        for i in range(1, len(state_files)):

            try:

                with open(path + str(i) + '.pkl', 'rb') as f:
                    bots += RenameUnpickler(f).load()[0]

            except Exception:

                print("File '" + path + str(i) + \
                      ".pkl' doesn't exist / is corrupted.")

        fitnesses_sorting_indices = \
            latest_fitnesses_and_bot_sizes[:, :, 0].argsort(axis=0)
        fitnesses_rankings = fitnesses_sorting_indices.argsort(axis=0)
        selected = np.greater_equal(fitnesses_rankings, pop_size//2)
        selected_indices = np.where(selected[:,0] == True)[0]

    scores = np.zeros((pop_size//2, args.nb_tests))

    for i in range(pop_size//2):

        if len(state) == 3:

            bot = Bot(0)
            bot.build(full_seed_list[i][0])

        else: # len(state) == 4:

            bot = bots[selected_indices[i]][0]
        
        bot.setup_to_run()

        for j in range(args.nb_tests):

            np.random.seed(MAX_INT-j)
            torch.manual_seed(MAX_INT-j)
            random.seed(MAX_INT-j)
            bot.reset()

            emulator.seed(MAX_INT-j)
            obs = emulator.reset()
            done = False
            
            for k in range(args.nb_obs_per_test):

                if 'imitate' in env_path:
                    obs = hide_score(obs)

                obs, rew, done, _ = emulator.step(bot(obs))

                scores[i][j] += rew

                if done:
                    break

    np.save(path + 'scores.npy', scores)