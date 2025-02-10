# Copyright (C) 2025 Björn A. Lindqvist <bjourne@gmail.com>
from mlstuff import (
    load_cifar,
    loader_sample_figure,
    propagate_epoch,
    seed_all
)
from mlstuff.networks import QCFS, load_net
from os import environ
from pathlib import Path
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

import click
import torch

__version__ = "0.0.1"

def is_primary(dev):
    return dev in {0, "cuda", "cpu"}

def is_distributed(dev):
    return type(dev) == int

def get_device():
    lo_rank = environ.get("LOCAL_RANK")
    if lo_rank is not None:
        return int(lo_rank)
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def print_device(dev):
    if is_distributed(dev):
        n_devs = torch.cuda.device_count()
        print("Training process %d of %d" % (dev, n_devs))
    else:
        print("Running on %s device" % dev)

def write_thetas(writer, net, epoch):
    kvs = {}
    for name, m in net.named_modules():
        if isinstance(m, QCFS):
            kvs[name] = m.theta
    writer.add_scalars('thetas', kvs, epoch)

def write_epoch_stats(
    net, sched, writer, epoch, names,
    l_tr, l_te,
    tr_loss, te_loss, tr_acc, te_acc, max_te_acc
):
    fmt = "losses %5.3f/%5.3f acc %5.3f/%5.3f, (best %5.3f)"
    print(fmt % (tr_loss, te_loss, tr_acc, te_acc, max_te_acc))

    write_thetas(writer, net, epoch)
    writer.add_scalars('acc', {'train' : tr_acc, 'test' : te_acc}, epoch)
    writer.add_scalars('loss', {'train' : tr_loss, 'test' : te_loss}, epoch)
    writer.add_scalar('lr', sched.get_last_lr()[0], epoch)
    for label, loader in [('training', l_tr), ('testing', l_te)]:
        fig = loader_sample_figure(loader, names, net)
        writer.add_figure(label, fig, epoch)
    writer.flush()


@click.group(
    invoke_without_command = True,
    no_args_is_help=True,
    context_settings={'show_default': True}
)
@click.pass_context
@click.version_option(__version__)
@click.argument(
    "network"
)
@click.argument(
    "dataset"
)
@click.option(
    "--seed",
    default = 1001,
    help = "Random number seed"
)
@click.option(
    "--batch-size",
    default = 128,
    help = "Batch size"
)
@click.option(
    "--data-dir",
    default = "/tmp/data",
    type = click.Path(dir_okay = True),
    help = "Dataset directory"
)
@click.option(
    "--print-interval",
    default = 10,
    help = "Print interval"
)
def cli(ctx, seed, network, dataset, batch_size, data_dir, print_interval):
    dev = get_device()

    seed_all(seed)

    # Load network
    n_cls = 10 if dataset == "cifar10" else 100
    net = load_net(network, n_cls).to(dev)
    if is_distributed(dev):
        net = DistributedDataParallel(
            net,
            device_ids = [dev],
            output_device = dev
        )

    # Load dataset
    data_path = Path(data_dir)
    data_path.mkdir(parents = True, exist_ok = True)
    data = load_cifar(data_path, batch_size, n_cls, dev)
    ctx.obj = dict(
        net_name = network,
        net = net,
        is_distributed = is_distributed,
        dev = dev,
        data = data,
        batch_size = batch_size,
        print_interval = print_interval
    )

@cli.command(
    context_settings={'show_default': True},
    help="Trains NETWORK using DATASET"
)
@click.option(
    "--log-dir",
    default = "/tmp/logs",
    type = click.Path(dir_okay = True),
    help = "Logging directory"
)
@click.option(
    "--weight-decay",
    default = 0.0005,
    help = "Weight decay"
)
@click.option(
    "--learning-rate",
    default = 0.1,
    help = "Learning rate"
)
@click.option(
    "--sgd-momentum",
    default = 0.9,
    help = "SGD momentum"
)
@click.option(
    "--t-max",
    default = 600,
    help = "T max"
)
@click.option(
    "--n-epochs",
    default = 600,
    help = "Number of epochs"
)
@click.pass_context
def train(
    ctx, log_dir,
    weight_decay: float,
    learning_rate: float,
    sgd_momentum: float,
    t_max: int,
    n_epochs: int
):
    obj = ctx.obj
    net_name = obj["net_name"]
    net = obj["net"]
    dev = obj["dev"]
    batch_size = obj["batch_size"]
    l_tr, l_te, names = obj["data"]
    print_interval = obj["print_interval"]
    if is_distributed(dev):
        init_process_group(backend="nccl")
        torch.cuda.set_device(dev)
    print_device(dev)

    dir_name = 'runs_%s_%03d_%.4f' % (net_name, batch_size, weight_decay)
    log_path = Path(log_dir)
    log_path.mkdir(parents = True, exist_ok = True)
    out_dir = log_path / dir_name
    writer = SummaryWriter(out_dir)
    opt = SGD(
        net.parameters(),
        learning_rate,
        sgd_momentum,
        weight_decay = weight_decay
    )
    sched = CosineAnnealingLR(opt, T_max=t_max)
    max_te_acc = 0
    for i in range(n_epochs):
        net.train()
        tr_loss, tr_acc = propagate_epoch(
            net, opt, l_tr, i, n_epochs, print_interval
        )
        if is_distributed(dev):
            torch.cuda.synchronize()
        if is_primary(dev):
            net.eval()
            with torch.no_grad():
                te_loss, te_acc = propagate_epoch(
                    net, opt, l_te, i, n_epochs, print_interval
                )
            if te_acc > max_te_acc:
                torch.save(net.state_dict(), out_dir / 'net.pth')

            max_te_acc = max(max_te_acc, te_acc)
            write_epoch_stats(
                net, sched, writer, i, names,
                l_tr, l_te,
                tr_loss, te_loss, tr_acc, te_acc, max_te_acc
            )
        sched.step()

    if is_distributed:
        torch.cuda.synchronize()
        barrier()
        destroy_process_group()

@cli.command(
    context_settings={'show_default': True},
    help="Test NETWORK using DATASET and WEIGHTS"
)
@click.argument(
    "weights",
    type = click.Path(exists = True)
)
@click.option(
    "--time-steps",
    default = None,
    help = "Number of simulation time steps (for qcfs networks)"
)
@click.pass_context
def test(ctx, weights: str, time_steps):
    obj = ctx.obj
    _, l_te, _ = obj["data"]
    net = obj["net"]
    net_name = obj["net_name"]
    print_interval = obj["print_interval"]

    d = torch.load(weights, weights_only = True)
    if net_name.endswith('qcfs'):
        d = rename_bu2023(net_name, d)
        net.set_snn_mode(time_steps)
        net.net.load_state_dict(d)
    else:
        net.load_state_dict(d)

    net.eval()
    with torch.no_grad():
        te_loss, te_acc = propagate_epoch(
            net, None, l_te, 0, 1, print_interval
        )
    print("loss %5.3f, acc %5.3f" % (te_loss, te_acc))

if __name__ == "__main__":
    cli()
