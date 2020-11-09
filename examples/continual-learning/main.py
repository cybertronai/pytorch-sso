import os
import argparse
from importlib import import_module
import shutil
import json

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models

import torchsso
from torchsso.optim import VIOptimizer, VOGN
from torchsso.utils import Logger
from data_generators import PermutedMnistGenerator


DATASET_permMNIST = 'permuted_MNIST'


def main():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--root', type=str, default='./data',
                        help='root of dataset (mnist.pkl.gz)')
    parser.add_argument('--nr_tasks', type=int, default=10,
                        help='number of tasks')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=128,
                        help='input batch size for valing')
    # Training Settings
    parser.add_argument('--arch_file', type=str, default=None,
                        help='name of file which defines the architecture')
    parser.add_argument('--arch_name', type=str, default='LeNet5',
                        help='name of the architecture')
    parser.add_argument('--arch_args', type=json.loads, default=None,
                        help='[JSON] arguments for the architecture')
    parser.add_argument('--optim_name', type=str, default=VIOptimizer.__name__,
                        help='name of the optimizer')
    parser.add_argument('--optim_args', type=json.loads, default=None,
                        help='[JSON] arguments for the optimizer')
    parser.add_argument('--curv_args', type=json.loads, default=None,
                        help='[JSON] arguments for the curvature')
    # Options
    parser.add_argument('--create_graph', action='store_true', default=False,
                        help='create graph of the derivative')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of sub processes for data loading')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--log_file_name', type=str, default='log',
                        help='log file name')
    parser.add_argument('--checkpoint_interval', type=int, default=50,
                        help='how many epochs to wait before logging training status')
    parser.add_argument('--out', type=str, default='result',
                        help='dir to save output files')
    parser.add_argument('--config', default=None,
                        help='config file path')

    args = parser.parse_args()
    dict_args = vars(args)

    if args.optim_args is None:
        args.optim_args = {}

    # Set device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    seeds = [12, 55, 189, 321, 65, 194, 309, 385, 64, 172,
             41, 176, 215, 312, 24, 360, 84, 147, 336, 156]

    all_accuracies = []
    for seed_id, seed in enumerate(seeds):
        # Load config file
        if args.config is not None:
            with open(args.config) as f:
                config = json.load(f)
            dict_args.update(config)

        print('')
        print('Run {}: seed {}'.format(seed_id + 1, seed))
        print('')

        # Set random seed
        np.random.seed(1)
        torch.manual_seed(seed)

        # Setup data generator
        nr_tasks = args.nr_tasks
        data_gen = PermutedMnistGenerator(nr_tasks, root=args.root)
        in_dim, output_size = data_gen.get_dims()

        # Setup model
        if args.arch_file is None:
            arch_class = getattr(models, args.arch_name)
        else:
            _, ext = os.path.splitext(args.arch_file)
            dirname = os.path.dirname(args.arch_file)

            if dirname == '':
                module_path = args.arch_file.replace(ext, '')
            elif dirname == '.':
                module_path = os.path.basename(args.arch_file).replace(ext, '')
            else:
                module_path = '.'.join(os.path.split(args.arch_file)).replace(ext, '')

            module = import_module(module_path)
            arch_class = getattr(module, args.arch_name)

        arch_kwargs = {} if args.arch_args is None else args.arch_args
        arch_kwargs['input_size'] = in_dim
        arch_kwargs['output_size'] = output_size

        model = arch_class(**arch_kwargs)
        setattr(model, 'output_size', output_size)
        model = model.to(device).double()

        # Set initial prior
        prior_mean = args.optim_args['prior_mean']
        prior_precision = args.optim_args['prior_precision']

        val_loaders = []
        all_accuracy = np.array([])
        for task_id in range(nr_tasks):
            train_x, train_y, val_x, val_y = data_gen.next_task()

            # Create train DataLoader for current task
            train_x_tensor = torch.from_numpy(train_x).double()
            train_y_tensor = torch.from_numpy(train_y).long()
            train_set = TensorDataset(train_x_tensor, train_y_tensor)
            batch_size = train_x.shape[0] if (args.batch_size is None) else args.batch_size
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                      num_workers=args.num_workers)

            # Create validation DataLoader for current task
            val_x_tensor = torch.from_numpy(val_x).double()
            val_y_tensor = torch.from_numpy(val_y).long()
            val = TensorDataset(val_x_tensor, val_y_tensor)
            val_batch_size = val_x.shape[0] if (args.val_batch_size is None) else args.val_batch_size
            val_loader = DataLoader(val, batch_size=val_batch_size, shuffle=False,
                                    num_workers=args.num_workers)
            val_loaders.append(val_loader)

            # Set new prior (initial prior for first task, then posterior of previous task)
            args.optim_args['prior_mean'] = prior_mean
            args.optim_args['prior_precision'] = prior_precision

            # Reset weights, initial precision is still the same
            model.apply(weight_reset)

            # Setup optimizer
            assert args.optim_name in [VIOptimizer.__name__, VOGN.__name__], 'You need to be Bayesian to do this.'
            optim_class = getattr(torchsso.optim, args.optim_name)
            optimizer = optim_class(model, dataset_size=len(train_loader.dataset),
                                    seed=seed, **args.optim_args, curv_kwargs=args.curv_args)

            start_epoch = 1

            # All config
            print('===========================')
            print('Run {}, Task {}'.format(seed_id + 1, task_id+1))
            print(f'Dataset: {DATASET_permMNIST}')
            print('train data size: {}'.format(len(train_loader.dataset)))
            print('val data size: {}'.format(len(val_loader.dataset)))
            for key, val in vars(args).items():
                if key == 'optim_args':
                    items = {}
                    for k, v in val.items():
                        if isinstance(v, list):
                            continue
                        else:
                            items[k] = v
                    print('{}: {}'.format(key, items))
                else:
                    print('{}: {}'.format(key, val))
            print('===========================')

            # Copy this file & config to args.out
            if task_id == 0:
                if not os.path.isdir(args.out):
                    os.makedirs(args.out)
                shutil.copy(os.path.realpath(__file__), args.out)

                if args.config is not None:
                    shutil.copy(args.config, args.out)
                if args.arch_file is not None:
                    shutil.copy(args.arch_file, args.out)

            # Setup logger
            logger = Logger(args.out, args.log_file_name)
            logger.start()

            # Run training on current task
            for epoch in range(start_epoch, args.epochs + 1):
                scheduler = None
                # train
                accuracy, loss, confidence = train(model, device, train_loader, optimizer,
                                                   scheduler, epoch, args, logger)

                # val
                val_accuracy, val_loss = validate(model, device, val_loader, optimizer)

                # save log
                task = task_id + 1
                iteration = epoch * len(train_loader)
                log = {'run': seed_id + 1, 'seed': seed, 'task': task, 'epoch': epoch, 'iteration': iteration,
                       'accuracy': accuracy, 'loss': loss, 'confidence': confidence,
                       'val_accuracy': val_accuracy, 'val_loss': val_loss,
                       'lr': optimizer.param_groups[0]['lr'],
                       'momentum': optimizer.param_groups[0].get('momentum', 0)}
                logger.write(log)

                # save checkpoint
                if epoch % args.checkpoint_interval == 0 or epoch == args.epochs:
                    path = os.path.join(args.out, 'run{}_task{}_epoch{}.ckpt'.format(
                        seed_id + 1, task, epoch))
                    data = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch
                    }
                    torch.save(data, path)

            # Set prior for next task to posterior of current model
            prior_mean, prior_precision = get_posterior(optimizer)

            # Validate current model on all previous tasks
            val_accuracies = []
            for loader in val_loaders:
                val_accuracies.append(validate(model, device, loader, optimizer, print_result=False)[0])
            all_accuracy = concatenate_results(val_accuracies, all_accuracy)

        all_accuracies.append(all_accuracy)

        print('Validation accuracies of all tasks:')
        print('')
        print(all_accuracy)
        print('')
        print('Average validation accuracy on all tasks: {:.2f}%'.format(np.mean(all_accuracy[-1])))

    stacked_result = np.stack(all_accuracies)
    mean_accuracy = np.nanmean(stacked_result, 2)
    mean_run = np.mean(mean_accuracy, 0)
    std_run = np.std(mean_accuracy, 0)

    print('')
    print('')
    print(stacked_result)
    print(stacked_result.shape)
    print(mean_accuracy)
    print(mean_accuracy.shape)
    print(mean_run)
    print(std_run)


def weight_reset(m):
    if isinstance(m, torch.nn.Linear):
        m.reset_parameters()


def concatenate_results(score, all_score):
    if all_score.size == 0:
        all_score = np.reshape(score, (1,-1))
    else:
        new_arr = np.empty((all_score.shape[0], all_score.shape[1]+1))
        new_arr[:] = np.nan
        new_arr[:,:-1] = all_score
        all_score = np.vstack((new_arr, score))
    return all_score


def get_posterior(optimizer):

    mean = []
    precision = []
    for group in optimizer.param_groups:
        mean.append(group['mean'])
        scale = group['std_scale'] ** 2
        precision.append([1/var.mul(scale) for var in group['curv'].inv])

    return mean, precision


def train(model, device, train_loader, optimizer, scheduler, epoch, args, logger):

    def scheduler_type(_scheduler):
        if _scheduler is None:
            return 'none'
        return getattr(_scheduler, 'scheduler_type', 'epoch')

    if scheduler_type(scheduler) == 'epoch':
        scheduler.step(epoch - 1)

    model.train()

    total_correct = 0
    loss = None
    confidence = {'top1': 0, 'top1_true': 0, 'top1_false': 0, 'true': 0, 'false': 0}
    total_data_size = 0
    epoch_size = len(train_loader.dataset)
    num_iters_in_epoch = len(train_loader)
    base_num_iter = (epoch - 1) * num_iters_in_epoch

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        if scheduler_type(scheduler) == 'iter':
            scheduler.step()

        for name, param in model.named_parameters():
            attr = 'p_pre_{}'.format(name)
            setattr(model, attr, param.detach().clone())

        # update params
        def closure():
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target.nonzero()[:, 1])
            loss.backward(create_graph=args.create_graph)

            return loss, output

        loss, output = optimizer.step(closure=closure)

        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.nonzero()[:, 1].view_as(pred)).sum().item()

        loss = loss.item()
        total_correct += correct

        prob = F.softmax(output, dim=1)
        for p, idx in zip(prob, target.nonzero()[:, 1]):
            confidence['top1'] += torch.max(p).item()
            top1 = torch.argmax(p).item()
            if top1 == idx:
                confidence['top1_true'] += p[top1].item()
            else:
                confidence['top1_false'] += p[top1].item()
            confidence['true'] += p[idx].item()
            confidence['false'] += (1 - p[idx].item())

        iteration = base_num_iter + batch_idx + 1
        total_data_size += len(data)

        if batch_idx % args.log_interval == 0:
            accuracy = 100. * total_correct / total_data_size
            elapsed_time = logger.elapsed_time
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, '
                  'Accuracy: {:.0f}/{} ({:.2f}%), '
                  'Elapsed Time: {:.1f}s'.format(
                epoch, total_data_size, epoch_size, 100. * (batch_idx + 1) / num_iters_in_epoch,
                loss, total_correct, total_data_size, accuracy, elapsed_time))

            # save log
            lr = optimizer.param_groups[0]['lr']
            log = {'epoch': epoch, 'iteration': iteration, 'elapsed_time': elapsed_time,
                   'accuracy': accuracy, 'loss': loss, 'lr': lr}

            for name, param in model.named_parameters():
                attr = 'p_pre_{}'.format(name)
                p_pre = getattr(model, attr)
                p_norm = param.norm().item()
                p_shape = list(param.size())
                p_pre_norm = p_pre.norm().item()
                g_norm = param.grad.norm().item()
                upd_norm = param.sub(p_pre).norm().item()
                noise_scale = getattr(param, 'noise_scale', 0)

                p_log = {'p_shape': p_shape, 'p_norm': p_norm, 'p_pre_norm': p_pre_norm,
                         'g_norm': g_norm, 'upd_norm': upd_norm, 'noise_scale': noise_scale}
                log[name] = p_log

            logger.write(log)

    accuracy = 100. * total_correct / epoch_size
    confidence['top1'] /= epoch_size
    confidence['top1_true'] /= total_correct
    confidence['top1_false'] /= (epoch_size - total_correct)
    confidence['true'] /= epoch_size
    confidence['false'] /= (epoch_size * (model.output_size - 1))

    return accuracy, loss, confidence


def validate(model, device, val_loader, optimizer, print_result=True):
    model.eval()
    val_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in val_loader:

            data, target = data.to(device), target.to(device)

            if isinstance(optimizer, VIOptimizer):
                output = optimizer.prediction(data)
            else:
                output = model(data)

            val_loss += F.cross_entropy(output, target.nonzero()[:, 1], reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.nonzero()[:, 1].view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    val_accuracy = 100. * correct / len(val_loader.dataset)

    if print_result:
        print('\nEval: Average loss: {:.4f}, Accuracy: {:.0f}/{} ({:.2f}%)\n'.format(
            val_loss, correct, len(val_loader.dataset), val_accuracy))

    return val_accuracy, val_loss


if __name__ == '__main__':
    main()
