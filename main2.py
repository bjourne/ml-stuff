# Copyright (C) 2025 Björn A. Lindqvist <bjourne@gmail.com>
#
# DenseNet, VGG16, ResNet, etc. trained on CIFAR10/100.
from mlstuff import (
    get_device,
    is_distributed,
    is_primary,
    load_cifar,
    loader_sample_figure,
    propagate_epoch,
    seed_all,
    synchronize
)
from mlstuff.networks import QCFS, load_net
from os import environ
from pathlib import Path
from torch.distributed import (
    barrier, destroy_process_group, init_process_group
)
from torch.nn.parallel import DistributedDataParallel
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

import click
import platform
import torch

__version__ = "0.0.1"

def print_device(dev):
    if is_distributed(dev):
        n_devs = torch.cuda.device_count()
        print("Process %d of %d" % (dev, n_devs))
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
    tr_stats, te_stats, max_te_acc
):
    fmt = "losses %5.3f/%5.3f acc %5.3f/%5.3f, (best %5.3f)"
    args = (
        tr_stats.loss,
        te_stats.loss,
        tr_stats.acc,
        te_stats.acc,
        max_te_acc
    )
    print(fmt % args)

    acc_scalars = {'train' : tr_stats.acc, 'test' : te_stats.acc}
    loss_scalars = {'train' : tr_stats.loss, 'test' : te_stats.loss}
    dur_scalars = {"train" : tr_stats.dur, "test" : te_stats.dur}
    scalar_sets = {
        "acc" : acc_scalars,
        "loss" : loss_scalars,
        "dur" : dur_scalars,
    }
    for sname, v in scalar_sets.items():
        writer.add_scalars(sname, v, epoch)
    writer.add_scalar("lr", sched.get_last_lr()[0], epoch)
    write_thetas(writer, net, epoch)
    for label, loader in [('training', l_tr), ('testing', l_te)]:
        fig = loader_sample_figure(loader, names, net)
        writer.add_figure(label, fig, epoch)
    writer.flush()

def log_dir_name(ds_name, net_name, seed, n_epochs, bs, wd, tm, lr):
    fmts = [
        ("%s", platform.node()),
        ("%s", ds_name),
        ("%s", net_name),
        ("%04d", seed),
        ("%04d", n_epochs),
        ("%04d", bs),
        ("%.5f", wd),
        ("%04d", tm),
        ("%.3f", lr)
    ]
    return "_".join(f % v for (f, v) in fmts)

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
    default = 256,
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
@click.option(
    "--time-steps",
    default = 0,
    help = "Number of simulation time steps (for qcfs networks)"
)
def cli(
    ctx,
    seed,
    network,
    dataset,
    batch_size,
    data_dir,
    print_interval,
    time_steps
):
    dev = get_device()
    print_device(dev)
    seed_all(seed)

    # Load network
    n_cls = 10 if dataset == "cifar10" else 100
    net = load_net(network, n_cls, time_steps)
    net = net.to(dev)
    if is_distributed(dev):
        init_process_group(backend="nccl")
        torch.cuda.set_device(dev)
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
        batch_size = batch_size,
        data = data,
        dev = dev,
        ds_name = dataset,
        is_distributed = is_distributed,
        net = net,
        net_name = network,
        print_interval = print_interval,
        seed = seed
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
    default = 1000,
    help = "T max"
)
@click.option(
    "--n-epochs",
    default = 1000,
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
    ds_name = obj["ds_name"]
    net = obj["net"]
    dev = obj["dev"]
    batch_size = obj["batch_size"]
    l_tr, l_te, names = obj["data"]
    print_interval = obj["print_interval"]
    seed = obj["seed"]

    dir_name = log_dir_name(
        ds_name, net_name,
        seed, n_epochs,
        batch_size, weight_decay,
        t_max, learning_rate
    )
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
        if is_distributed(dev):
            l_tr.sampler.set_epoch(i)
        tr_stats = propagate_epoch(
            dev, net, opt, l_tr, i, n_epochs, print_interval
        )
        synchronize(dev)
        if is_primary(dev):
            # Ensure local network is used.
            lnet = net.module if is_distributed(dev) else net
            lnet.eval()
            with torch.no_grad():
                te_stats = propagate_epoch(
                    dev, lnet, opt, l_te, i, n_epochs, print_interval
                )
            if te_stats.acc > max_te_acc:
                torch.save(lnet.state_dict(), out_dir / 'net.pth')
            max_te_acc = max(max_te_acc, te_stats.acc)
            write_epoch_stats(
                lnet, sched, writer, i, names,
                l_tr, l_te,
                tr_stats, te_stats, max_te_acc
            )
        sched.step()

    if is_distributed(dev):
        synchronize(dev)
        destroy_process_group()

@cli.command(
    context_settings={'show_default': True},
    help="Test NETWORK using DATASET and WEIGHTS"
)
@click.argument(
    "weights",
    type = click.Path(exists = True)
)
@click.pass_context
def test(ctx, weights: str):
    obj = ctx.obj
    _, l_te, _ = obj["data"]
    net = obj["net"]
    dev = obj["dev"]
    print_interval = obj["print_interval"]

    d = torch.load(weights, weights_only = True)
    net.load_state_dict(d)
    net.eval()
    with torch.no_grad():
        te_stats = propagate_epoch(
            dev, net, None, l_te, 0, 1, print_interval
        )
    print("loss %5.3f, acc %5.3f" % (te_stats.loss, te_stats.acc))

if __name__ == "__main__":
    cli()
