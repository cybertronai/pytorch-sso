import os
import argparse
from importlib import import_module
import shutil
import json

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchsso
from torchsso.optim import SecondOrderOptimizer, VIOptimizer
from torchsso.utils import Logger

DATASET_CIFAR10 = 'CIFAR-10'
DATASET_CIFAR100 = 'CIFAR-100'
DATASET_MNIST = 'MNIST'


def main():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--dataset', type=str,
                        choices=[DATASET_CIFAR10, DATASET_CIFAR100, DATASET_MNIST], default=DATASET_CIFAR10,
                        help='name of dataset')
    parser.add_argument('--root', type=str, default='./data',
                        help='root of dataset')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=128,
                        help='input batch size for valing')
    parser.add_argument('--normalizing_data', action='store_true',
                        help='[data pre processing] normalizing data')
    parser.add_argument('--random_crop', action='store_true',
                        help='[data augmentation] random crop')
    parser.add_argument('--random_horizontal_flip', action='store_true',
                        help='[data augmentation] random horizontal flip')
    # Training Settings
    parser.add_argument('--arch_file', type=str, default=None,
                        help='name of file which defines the architecture')
    parser.add_argument('--arch_name', type=str, default='LeNet5',
                        help='name of the architecture')
    parser.add_argument('--arch_args', type=json.loads, default=None,
                        help='[JSON] arguments for the architecture')
    parser.add_argument('--optim_name', type=str, default=SecondOrderOptimizer.__name__,
                        help='name of the optimizer')
    parser.add_argument('--optim_args', type=json.loads, default=None,
                        help='[JSON] arguments for the optimizer')
    parser.add_argument('--curv_args', type=json.loads, default=dict(),
                        help='[JSON] arguments for the curvature')
    parser.add_argument('--fisher_args', type=json.loads, default=dict(),
                        help='[JSON] arguments for the fisher')
    parser.add_argument('--scheduler_name', type=str, default=None,
                        help='name of the learning rate scheduler')
    parser.add_argument('--scheduler_args', type=json.loads, default=None,
                        help='[JSON] arguments for the scheduler')
    parser.add_argument('--metainit', action='store_true', default=False,
                        help='if True, apply MetaInit')
    # Options
    parser.add_argument('--download', action='store_true', default=False,
                        help='if True, downloads the dataset (CIFAR-10 or 100) from the internet')
    parser.add_argument('--create_graph', action='store_true', default=False,
                        help='create graph of the derivative')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of sub processes for data loading')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_param_log', action='store_true', default=False,
                        help='if True, save parameters status log')
    parser.add_argument('--param_log_interval', type=int, default=50,
                        help='how many batches to wait before logging parameters status')
    parser.add_argument('--save_grad_stats_log', action='store_true', default=False,
                        help='if True, save grads stats log')
    parser.add_argument('--grad_stats_log_interval', type=int, default=50,
                        help='how many iterations to wait before logging grad stats')
    parser.add_argument('--save_grad_noise_log', action='store_true', default=False,
                        help='if True, save grads noise log')
    parser.add_argument('--grad_noise_log_interval', type=int, default=50,
                        help='how many iterations to wait before logging grad noise')
    parser.add_argument('--grad_noise_num_samples', type=int, default=10,
                        help='number of samples for estimating minibatch gradient noise')
    parser.add_argument('--save_curv_eigvals', action='store_true', default=False,
                        help='if True, save curvature eigenvalues')
    parser.add_argument('--curv_eigvals_interval', type=int, default=50,
                        help='how many batches to wait before logging curvature eigenvalues')
    parser.add_argument('--log_file_name', type=str, default='log',
                        help='log file name')
    parser.add_argument('--checkpoint_interval', type=int, default=50,
                        help='how many epochs to wait before logging training status')
    parser.add_argument('--resume', type=str, default=None,
                        help='checkpoint path for resume training')
    parser.add_argument('--out', type=str, default='result',
                        help='dir to save output files')
    parser.add_argument('--config', default='configs/cifar10/lenet_kfac.json',
                        help='config file path')

    args = parser.parse_args()
    dict_args = vars(args)

    # Load config file
    if args.config is not None:
        with open(args.config) as f:
            config = json.load(f)
        dict_args.update(config)

    # Set device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Set random seed
    torch.manual_seed(args.seed)

    # Setup data augmentation & data pre processing
    train_transforms, val_transforms = [], []
    if args.random_crop:
        train_transforms.append(transforms.RandomCrop(32, padding=4))

    if args.random_horizontal_flip:
        train_transforms.append(transforms.RandomHorizontalFlip())

    train_transforms.append(transforms.ToTensor())
    val_transforms.append(transforms.ToTensor())

    if args.normalizing_data:
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_transforms.append(normalize)
        val_transforms.append(normalize)

    train_transform = transforms.Compose(train_transforms)
    val_transform = transforms.Compose(val_transforms)

    # Setup data loader
    if args.dataset == DATASET_CIFAR10:
        # CIFAR-10
        num_classes = 10
        dataset_class = datasets.CIFAR10
    elif args.dataset == DATASET_CIFAR100:
        # CIFAR-100
        num_classes = 100
        dataset_class = datasets.CIFAR100
    elif args.dataset == DATASET_MNIST:
        num_classes = 10
        dataset_class = datasets.MNIST
    else:
        assert False, f'unknown dataset {args.dataset}'

    train_dataset = dataset_class(
        root=args.root, train=True, download=args.download, transform=train_transform)
    val_dataset = dataset_class(
        root=args.root, train=False, download=args.download, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers)

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
    arch_kwargs['num_classes'] = num_classes

    model = arch_class(**arch_kwargs)
    setattr(model, 'num_classes', num_classes)
    model = model.to(device)

    optim_kwargs = {} if args.optim_args is None else args.optim_args

    # Setup optimizer
    if args.optim_name == SecondOrderOptimizer.__name__:
        optimizer = SecondOrderOptimizer(model, **optim_kwargs, curv_kwargs=args.curv_args)
    elif args.optim_name == VIOptimizer.__name__:
        optimizer = VIOptimizer(model, dataset_size=len(train_loader.dataset), seed=args.seed,
                                **optim_kwargs, curv_kwargs=args.curv_args)
    else:
        optim_class = getattr(torch.optim, args.optim_name)
        optimizer = optim_class(model.parameters(), **optim_kwargs)

    # Setup lr scheduler
    if args.scheduler_name is None:
        scheduler = None
    else:
        scheduler_class = getattr(torchsso.optim.lr_scheduler, args.scheduler_name, None)
        if scheduler_class is None:
            scheduler_class = getattr(torch.optim.lr_scheduler, args.scheduler_name)
        scheduler_kwargs = {} if args.scheduler_args is None else args.scheduler_args
        scheduler = scheduler_class(optimizer, **scheduler_kwargs)

    start_epoch = 1

    # Load checkpoint
    if args.resume is not None:
        print('==> Resuming from checkpoint..')
        assert os.path.exists(args.resume), 'Error: no checkpoint file found'
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']

    # All config
    print('===========================')
    for key, val in vars(args).items():
        if key == 'dataset':
            print('{}: {}'.format(key, val))
            print('train data size: {}'.format(len(train_loader.dataset)))
            print('val data size: {}'.format(len(val_loader.dataset)))
        else:
            print('{}: {}'.format(key, val))
    print('===========================')

    # Copy this file & config to args.out
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

    if args.metainit:
        x, _ = next(iter(train_loader))
        torchsso.utils.metainit(model, F.cross_entropy, x.size(), num_classes, steps=200)

    # Run training
    for epoch in range(start_epoch, args.epochs + 1):

        # train
        accuracy, loss, confidence = train(model, device, train_loader, optimizer, scheduler, epoch, args, logger)

        # val
        val_accuracy, val_loss = validate(model, device, val_loader, optimizer)

        # save log
        iteration = epoch * len(train_loader)
        log = {'epoch': epoch, 'iteration': iteration,
               'accuracy': accuracy, 'loss': loss, 'confidence': confidence,
               'val_accuracy': val_accuracy, 'val_loss': val_loss,
               'lr': optimizer.param_groups[0]['lr'],
               'momentum': optimizer.param_groups[0].get('momentum', 0)}
        logger.write(log)

        # save checkpoint
        if epoch % args.checkpoint_interval == 0 or epoch == args.epochs:
            path = os.path.join(args.out, 'epoch{}.ckpt'.format(epoch))
            data = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(data, path)


def train(model, device, train_loader, optimizer, scheduler, epoch, args, logger):

    def scheduler_type(_scheduler):
        if _scheduler is None:
            return 'none'
        return getattr(_scheduler, 'scheduler_type', 'epoch')

    model.train()

    total_correct = 0
    loss = None
    confidence = {'top1': 0, 'top1_true': 0, 'top1_false': 0, 'true': 0, 'false': 0}
    total_data_size = 0
    epoch_size = len(train_loader.dataset)
    num_iters_in_epoch = len(train_loader)
    base_num_iter = (epoch - 1) * num_iters_in_epoch

    iteration = base_num_iter + 1
    for batch_idx, (data, target) in enumerate(train_loader):

        if args.save_grad_stats_log and (iteration % args.grad_stats_log_interval == 0 or iteration == 1):
            observe_percase_grad_stats(model, optimizer, device, train_loader, epoch, iteration, args)

        if args.save_grad_noise_log and (iteration % args.grad_noise_log_interval == 0 or iteration == 1):
            calculate_batch_grad(model, optimizer, device, train_loader)
            observe_minibatch_grad_noise(model, optimizer, device, train_loader, epoch, iteration, args)

        data, target = data.to(device), target.to(device)

        for name, param in model.named_parameters():
            attr = 'p_pre_{}'.format(name)
            setattr(model, attr, param.detach().clone())

        # update params
        def closure():
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward(create_graph=args.create_graph)

            return loss, output

        if isinstance(optimizer, SecondOrderOptimizer) and optimizer.curv_type == 'Fisher':
            closure = torchsso.get_closure_for_fisher(optimizer, model, data, target, **args.fisher_args)

        loss, output = optimizer.step(closure=closure)

        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()

        loss = loss.item()
        total_correct += correct

        total_data_size += len(data)

        prob = F.softmax(output, dim=1)
        for p, idx in zip(prob, target):
            confidence['top1'] += torch.max(p).item()
            top1 = torch.argmax(p).item()
            if top1 == idx:
                confidence['top1_true'] += p[top1].item()
            else:
                confidence['top1_false'] += p[top1].item()
            confidence['true'] += p[idx].item()
            confidence['false'] += (1 - p[idx].item())

        if iteration == 1 or iteration % args.log_interval == 0:
            accuracy = 100. * total_correct / total_data_size
            elapsed_time = logger.elapsed_time
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, '
                  'Accuracy: {:.0f}/{} ({:.2f}%), '
                  'Elapsed Time: {:.1f}s'.format(
                epoch, total_data_size, epoch_size, 100. * (batch_idx + 1) / num_iters_in_epoch,
                loss, total_correct, total_data_size, accuracy, elapsed_time), flush=True)

            # save log
            lr = optimizer.param_groups[0]['lr']
            log = {'epoch': epoch, 'iteration': iteration, 'elapsed_time': elapsed_time,
                   'accuracy': accuracy, 'loss': loss, 'lr': lr}

            logger.write(log)

        if args.save_param_log and \
                (iteration == 1 or iteration % args.param_log_interval == 0):
            log = {'epoch': epoch, 'iteration': iteration}

            # save param norm log
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

            logdir = os.path.join(args.out, 'param_stats')
            if not os.path.isdir(logdir):
                os.makedirs(logdir)
            logpath = os.path.join(logdir, f'iter{iteration}.log')
            with open(logpath, 'w') as f:
                json.dump(log, f, indent=4)

        if args.save_curv_eigvals and \
                (iteration == 1 or iteration % args.curv_eigvals_interval == 0):
            assert isinstance(optimizer, torchsso.optim.SecondOrderOptimizer)

            log = {'epoch': epoch, 'iteration': iteration, 'batch_size': args.batch_size, 'eigvals': {}}

            for group in optimizer.local_param_groups:
                name = group['name']
                curv = group['curv']
                eigvals = curv.get_eigenvalues()
                log['eigvals'][name] = eigvals.tolist()

            logdir = os.path.join(args.out, 'eigvals')
            if not os.path.isdir(logdir):
                os.makedirs(logdir)
            logpath = os.path.join(logdir, f'iter{iteration}.log')
            with open(logpath, 'w') as f:
                json.dump(log, f, indent=4)

        iteration += 1

        if scheduler_type(scheduler) == 'iter':
            scheduler.step()

    if scheduler_type(scheduler) == 'epoch':
        scheduler.step(epoch - 1)

    accuracy = 100. * total_correct / epoch_size
    confidence['top1'] /= epoch_size
    confidence['top1_true'] /= total_correct
    confidence['top1_false'] /= (epoch_size - total_correct)
    confidence['true'] /= epoch_size
    confidence['false'] /= (epoch_size * (model.num_classes - 1))

    return accuracy, loss, confidence


def validate(model, device, val_loader, optimizer):
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

            val_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    val_accuracy = 100. * correct / len(val_loader.dataset)

    print('\nEval: Average loss: {:.4f}, Accuracy: {:.0f}/{} ({:.2f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset), val_accuracy), )

    return val_accuracy, val_loss


def calculate_batch_grad(model, optimizer, device, train_loader):

    model.train()
    num_iters_in_epoch = len(train_loader)

    # initialize data for each param
    grads = {}
    for name, param in model.named_parameters():
        grads[name] = torchsso.utils.TensorAccumulator()

    print('calculating batch gradient...')
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logit = model(data)
        loss = F.cross_entropy(logit, target)
        loss.backward()

        # accumulate grads
        for name, param in model.named_parameters():
            grads[name].update(param.grad, scale=1/num_iters_in_epoch)

    # set batch grad to each param
    for name, param in model.named_parameters():
        setattr(param, 'batch_grad', grads[name].get())


def observe_minibatch_grad_noise(model, optimizer, device, loader, epoch, iteration, args):

    model.train()

    batch_size = args.batch_size
    num_samples = args.grad_noise_num_samples

    log = {'epoch': epoch, 'iteration': iteration,
           'batch_size': batch_size, 'num_samples': num_samples}

    batch_g_sqnorm = 0
    for name, param in model.named_parameters():
        assert hasattr(param, 'batch_grad'), 'batch gradient needs to be calculated in advance.'
        batch_g = param.batch_grad
        batch_g_sqnorm += torch.pow(batch_g, 2).sum()

    batch_g_norm = batch_g_sqnorm.sqrt()
    avg_err = torchsso.utils.TensorAccumulator()

    log['errors'] = []
    log['g_norms'] = []
    print('estimating minibatch gradient noise...')
    for i in range(num_samples):

        iterator = iter(loader)
        data, target = next(iterator)
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        logit = model(data)
        loss = F.cross_entropy(logit, target)
        loss.backward()

        g_sqnorm = 0
        err_sqnorm = 0
        for name, param in model.named_parameters():
            g_sqnorm += torch.pow(param.grad, 2).sum()
            e = param.grad - param.batch_grad
            err_sqnorm += torch.pow(e, 2).sum()

        err = err_sqnorm.sqrt().div(batch_g_norm)
        log['errors'].append(err.item())
        avg_err.update(err, scale=1/num_samples)
        log['g_norms'].append(g_sqnorm.sqrt().item())

    log['error'] = avg_err.get().item()

    logdir = os.path.join(args.out, 'grad_noise')
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    logpath = os.path.join(logdir, f'bs{batch_size}_iter{iteration}.log')
    with open(logpath, 'w') as f:
        json.dump(log, f, indent=4)


def observe_percase_grad_stats(model, optimizer, device, train_loader, epoch, iteration, args):

    model.train()

    num_iters_in_epoch = len(train_loader)

    log = {'epoch': epoch, 'iteration': iteration,
           'entropy': [], 'cross_entropy': [], 'grads': {}}

    # initialize data for each param
    grads = {'total': {'g_norms': []},
             'logit': {'g_norms': []}}
    for name, param in model.named_parameters():
        grads[name] = {'avg_g': torchsso.utils.TensorAccumulator(),
                       'square_avg_g': torchsso.utils.TensorAccumulator(),
                       'avg_square_g': torchsso.utils.TensorAccumulator(),
                       'g_norms': []
                       }

    print('collecting gradient stats ...')
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        with torchsso.autograd.save_batched_grads(model):
            optimizer.zero_grad()
            logit = model(data)
            loss = F.cross_entropy(logit, target)
            loss.backward()

        batch_size = data.size(0)
        prob = F.softmax(logit)
        log_prob = F.log_softmax(logit)
        entropy = (-prob * log_prob).sum(axis=1)
        cross_entropy = torch.zeros(batch_size)
        logit_grads = prob
        for i in range(batch_size):
            cross_entropy[i] = -log_prob[i][target[i]]
            logit_grads[i][target[i]] -= 1.

        log['entropy'].extend(entropy.tolist())
        log['cross_entropy'].extend(cross_entropy.tolist())

        batched_logit_g_norm = (logit_grads * logit_grads).sum(axis=1).sqrt()
        grads['logit']['g_norms'].extend(batched_logit_g_norm.tolist())

        # accumulate grads
        batched_g_sqnorm_total = torch.zeros(batch_size).to(device)
        for name, param in model.named_parameters():
            grads[name]['avg_g'].update(param.grad, scale=1/num_iters_in_epoch)
            batched_g = param.grads
            batched_square_g = batched_g * batched_g
            avg_square_g = batched_square_g.mean(axis=0)
            batched_g_sqnorm = batched_square_g.flatten(start_dim=1).sum(axis=1)
            batched_g_sqnorm_total += batched_g_sqnorm
            batched_g_norm = batched_g_sqnorm.sqrt()

            grads[name]['avg_square_g'].update(avg_square_g, scale=1/num_iters_in_epoch)
            grads[name]['g_norms'].extend(batched_g_norm.tolist())

        batched_g_norm_total = batched_g_sqnorm_total.sqrt()
        grads['total']['g_norms'].extend(batched_g_norm_total.tolist())

    # save log
    log['grads']['total'] = grads['total']
    log['grads']['logit'] = grads['logit']

    for name, param in model.named_parameters():
        avg_g = grads[name]['avg_g'].get()
        square_avg_g = avg_g * avg_g
        avg_square_g = grads[name]['avg_square_g'].get()
        avg_square_g_norm = avg_square_g.norm().item()

        _log = {'g_shape': list(avg_g.size()),
                'avg_g_norm': avg_g.norm().item(),
                'square_avg_g_norm': square_avg_g.norm().item(),
                'avg_square_g_norm': avg_square_g_norm,
                'square_avg_error': (square_avg_g - avg_square_g).norm().item() / avg_square_g_norm,
                'g_norms': grads[name]['g_norms']
                }

        log['grads'][name] = _log

    logdir = os.path.join(args.out, 'grad_stats')
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    logpath = os.path.join(logdir, f'epoch{epoch}.log')
    with open(logpath, 'w') as f:
        json.dump(log, f, indent=4)


if __name__ == '__main__':
    main()
