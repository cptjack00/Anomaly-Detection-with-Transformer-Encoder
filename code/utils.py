import argparse
import json
import os
from datetime import datetime

import torch
from torch.autograd import Variable

def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param: json_file
    :return: config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
        config_file.close()
    return config_dict


def save_config(config):
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y-%H-%M")
    filename = config['result_dir'] + \
        'training_config_{}.txt'.format(timestampStr)
    config_to_save = json.dumps(config)
    f = open(filename, "w")
    f.write(config_to_save)
    f.close()


def process_config(json_file):
    config = get_config_from_json(json_file)

    # create directories to save experiment results and trained models
    if config['load_dir'] == "default":
        save_dir = "../experiments/local-results/{}/{}/batch-{}".format(
            config['exp_name'], config['dataset'], config['batch_size'])
    else:
        save_dir = config['load_dir']
    # specify the saving folder name for this experiment
    if config['TRAIN_sigma'] == 1:
        save_name = '{}-{}-{}-{}-{}-trainSigma'.format(config['exp_name'],
                                                       config['dataset'],
                                                       config['l_win'],
                                                       config['l_seq'],
                                                       config['code_size'])
    else:
        save_name = '{}-{}-{}-{}-{}-fixedSigma-{}'.format(config['exp_name'],
                                                          config['dataset'],
                                                          config['l_win'],
                                                          config['l_seq'],
                                                          config['code_size'],
                                                          config['sigma'])
    config['summary_dir'] = os.path.join(save_dir, save_name, "summary/")
    config['result_dir'] = os.path.join(save_dir, save_name, "result/")
    config['checkpoint_dir'] = os.path.join(save_dir, save_name, "checkpoint/")
    return config

def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)   

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    # argparser.add_argument(
    #     '-n', '--num-client',
    #     default=1,
    #     help='The number of clients participating in Federated Learning')
    argparser.add_argument(
        '-d', '--dataset',
        default='None',
        help="Dataset")
    args = argparser.parse_args()
    return args

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup**(-1.5)))
        
    def get_std_opt(model):
        return NoamOpt(model.src_embed[0].d_model, 2, 4000,
        torch.optim.Adam(model.parameters(),
            lr=0, betas=(0.9, 0.98), eps=1e-9))


def loss_backprop(generator, criterion, out, targets, normalize, bp=True):
    """
    Memory optmization. Compute each timestep separately and sum grads.
    """
    assert out.size(1) == targets.size(1)
    total = 0.0
    out_grad = []
    for i in range(out.size(1)):
        out_column = Variable(out[:, i].data, requires_grad=True)
        gen = generator(out_column)
        loss = criterion(gen, targets[:, i]) / normalize
        total += loss.data[0]
        loss.backward()
        out_grad.append(out_column.grad.data.clone())
    if bp:
        out_grad = torch.stack(out_grad, dim=1)
        out.backward(gradient=out_grad)
    return total