from __future__ import division, print_function, absolute_import
from scipy.spatial.distance import cdist
import matplotlib
# must select appropriate backend before importing any matplotlib functions
# matplotlib.use("TkAgg")

import math
import os
import logging
import errno
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
import tarfile
from matplotlib.image import imread
from matplotlib.lines import Line2D
import scipy.stats
import time
from tabulate import tabulate
from collections import OrderedDict

plt.interactive(False)

EPS = 1e-10

''' https://github.com/cianeastwood/bufr/blob/79acbb3ed7456366ae03dc3ccaf6001d6d4d1b54/lib/utils.py#L31 '''
class GOATLogger:

    def __init__(self, mode, save_root, log_freq, base_name, n_iterations, n_eval_iterations, *argv):
        self.mode = mode
        self.save_root = save_root
        self.log_freq = log_freq
        self.n_iterations = n_iterations
        self.n_eval_iterations = n_eval_iterations

        if self.mode == 'train' or self.mode == 'eval_psnr':
            filename = os.path.join(self.save_root, f'{base_name}.log')
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            logging.basicConfig(level=logging.INFO,  # DEBUG causes: https://github.com/camptocamp/pytest-odoo/issues/15
                                format='{} | %(asctime)s.%(msecs)03d - %(message)s'.format(mode),
                                datefmt='%b-%d %H:%M:%S',
                                filename=filename,
                                filemode='a' if os.path.exists(filename) else 'w')
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            console.setFormatter(logging.Formatter('%(message)s'))
            self.log = logging.getLogger('')
            self.log.addHandler(console)
        else:
            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s.%(msecs)03d - %(message)s',
                                datefmt='%b-%d %H:%M:%S')

        self.stats = {}
        self.reset_stats()
        self.time = time.time()

    def reset_stats(self):
        if self.mode == 'train':
            self.stats = {'train': {'loss': [], 'acc': []},
                          'eval': {'loss': [], 'acc': []},
                          'eval-per-task': {}}
        else:
            self.stats = {'eval': {'loss': [], 'acc': []},
                          'eval-per-task': {}}

    def batch_info(self, **kwargs):
        if kwargs['phase'] == 'train':
            self.stats['train']['loss'].append(kwargs['loss'])
            self.stats['train']['acc'].append(kwargs['acc'])

            if kwargs['iteration'] % self.log_freq == 0:
                # mean on the last n samples, where n=self.log_freq
                batch_loss = np.mean(self.stats['train']['loss'][-self.log_freq:])
                batch_acc = np.mean(self.stats['train']['acc'][-self.log_freq:])

                self.loginfo("[{0}/{1}] Loss:{2:.3f}, Acc:{3:.2f}".format(kwargs['iteration'], self.n_iterations,
                                                                          batch_loss, batch_acc))
                self.loginfo("Time for {0} iteration(s): {1}s".format(self.log_freq, int(time.time() - self.time)))
                self.time = time.time()

        elif kwargs['phase'] == 'eval':
            self.stats['eval']['loss'].append(kwargs['loss'])
            self.stats['eval']['acc'].append(kwargs['acc'])

        elif kwargs['phase'] == 'eval-done':
            loss_mean = np.mean(self.stats['eval']['loss'])
            loss_std = np.std(self.stats['eval']['loss'])
            acc_mean = np.mean(self.stats['eval']['acc'])
            acc_std = np.std(self.stats['eval']['acc'])

            self.loginfo("[{:5d}] Eval ({:3d} episode) - "
                         "Loss: {:6.4f} +- {:6.4f},"
                         "Acc: {:6.3f} +- {:5.3f}%. "
                         "Low CI {:6.3f}, "
                         "High CI {:6.3f}.".format(kwargs['iteration'], self.n_eval_iterations, loss_mean, loss_std,
                                                   acc_mean, acc_std,
                                                   acc_mean - 1.96 * (acc_std/np.sqrt(self.n_eval_iterations)),
                                                   acc_mean + 1.96 * (acc_std/np.sqrt(self.n_eval_iterations))))
            self.reset_stats()
            return acc_mean

        elif kwargs['phase'] == 'eval-per-task':
            attrs = ['loss', 'acc', 'per_step_accs', 'layer_distances', 'c_thlds', 'p_thlds', 'c_ss', 'p_ss']
            if kwargs['task_name'] in self.stats['eval-per-task']:      # append to list
                for attr in attrs:
                    append_to_dict_entry(self.stats, kwargs, attr)
            else:                                                       # create list
                self.stats['eval-per-task'][kwargs['task_name']] = {}
                for attr in attrs:
                    add_new_dict_entry(self.stats, kwargs, attr)

        elif kwargs['phase'] == 'eval-per-task-done':
            #  Gather info -- lists used over arrays to allow different batches sizes for different tasks (slower)
            task_names = list(self.stats['eval-per-task'].keys())
            task_losses = np.array([self.stats['eval-per-task'][task_name]['loss'] for task_name in task_names])
            task_accs = np.array([self.stats['eval-per-task'][task_name]['acc'] for task_name in task_names])

            #  Calc mean loss/acc *per task* -- could print within-task std over batches, but not so insightful
            mean_loss_per_task = task_losses.mean(1)
            mean_acc_per_task = task_accs.mean(1)

            #  Log per-task results
            tabular_results = [(t_name, "{0:.3f}".format(t_loss), "{0:.2f}".format(t_acc))
                               for t_name, t_loss, t_acc in zip(task_names, mean_loss_per_task, mean_acc_per_task)]
            tabular_results.sort(key=lambda r: r[0])             # sort by task name
            results_table = tabulate(tabular_results, headers=["Task", "Loss", "Accuracy"], tablefmt="rst")
            self.loginfo("[{0}/{1}] Meta-validation".format(kwargs['iteration'], self.n_iterations) + "\n"
                         "Per-task results (query samples):\n" + results_table + "\n")

            #  Log/print per-step, per-task results (if available)
            if 'per_step_accs' in self.stats['eval-per-task'][task_names[0]]:
                res_table = tabulate_mean_results(self.stats, 'per_step_accs', task_names,
                                                  'Per-step accuracies [s_1, s_1, ...]')
                self.loginfo("Per-task, per-step results (avg. across support samples):\n" + res_table + "\n")

            #  Log/print layer distances, if available
            if 'layer_distances' in self.stats['eval-per-task'][task_names[0]]:
                res_table = tabulate_mean_results(self.stats, 'layer_distances', task_names,
                                                  'Per-layer dist. moved [l_1, l_2, ...]')
                self.loginfo("Distances moved by each layer after n inner steps:\n" + res_table + "\n")

            #  Log/print current thresholds, if available
            if 'c_thlds' in self.stats['eval-per-task'][task_names[0]]:
                res_table = tabulate_mean_results(self.stats, 'c_thlds', task_names,
                                                  'Current Thresholds [Conv, Linear] / [l_1, l_2, ...]')
                self.loginfo("Thresholds for current surprises:\n" + res_table + "\n")

            #  Log/print parent thresholds, if available
            if 'p_thlds' in self.stats['eval-per-task'][task_names[0]]:
                res_table = tabulate_mean_results(self.stats, 'p_thlds', task_names,
                                                  'Parent Thresholds [Conv, Linear] / [l_1, l_2, ...]')
                self.loginfo("Thresholds for parent surprises:\n" + res_table + "\n")

            #  Log/print current surprises, if available
            if 'c_ss' in self.stats['eval-per-task'][task_names[0]]:
                res_table = tabulate_mean_results_per_layer(self.stats, 'c_ss', task_names,
                                                            'Current Surprises [s_1, s_2, ...]')
                self.loginfo("Current surprises (avg. across support samples):\n" + res_table + "\n")

            #  Log/print current surprises, if available
            if 'p_ss' in self.stats['eval-per-task'][task_names[0]]:
                res_table = tabulate_mean_results_per_layer(self.stats, 'p_ss', task_names,
                                                            'Parent Surprises [s_1, s_2, ...]')
                self.loginfo("Parent surprises (avg. across support samples):\n" + res_table + "\n")

            #  Calc average and std *across tasks*
            loss_mean = mean_loss_per_task.mean()
            loss_std = mean_loss_per_task.std()
            acc_mean = mean_acc_per_task.mean()
            acc_std = mean_acc_per_task.std()

            #  Log avg results
            self.loginfo("Avg task loss: {0:.3f} +- {1:.3f}, "
                         "Avg task acc: {2:.2f} +- {3:.2f}.\n".format(loss_mean, loss_std, acc_mean, acc_std))

            self.reset_stats()
            return acc_mean

        else:
            raise ValueError("phase {} not supported".format(kwargs['phase']))

    def logdebug(self, strout):
        logging.debug(strout)

    def loginfo(self, strout):
        logging.info(strout)

    def shutdown(self):
        handlers = self.log.handlers[:]
        for handler in handlers:
            handler.close()
            self.log.removeHandler(handler)