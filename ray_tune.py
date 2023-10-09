import os
import ray
import argparse
import collections
import torch
import numpy as np
from functools import partial
from torch.utils.data import DataLoader, random_split
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
from model.model import MnistModel
from parse_config import ConfigParser
from utils import prepare_device


def train_tune(config, logger, args, data_loader):
    # model
    model = MnistModel(config['l1'], config['l2'])
    
    # get device
    device, device_ids = prepare_device(args['n_gpu'], logger)
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    # get function handles of loss and metrics
    criterion = getattr(module_loss, args['loss'])
    metrics = [getattr(module_metric, met) for met in args['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(trainable_params, lr=config['lr'], momentum=0.9)
    lr_scheduler = args.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    # checkpoint
    # To restore a checkpoint, use `train.get_checkpoint()`.
    loaded_checkpoint = train.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
           model_state, optimizer_state = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # dataset
    trainset, _ = data_loader.get_dataset(data_dir=args['data_loader']['args']['data_dir'])
    test_abs = int(len(trainset) * 0.2)
    train_subset, val_subset = random_split(trainset, [len(trainset) - test_abs, test_abs])
    train_loader = DataLoader(train_subset, 
                              batch_size=args['data_loader']['args']['batch_size'], 
                              shuffle=args['data_loader']['args']['shuffle'], 
                              num_workers=args['data_loader']['args']['num_workers'])
    val_loader = DataLoader(val_subset,
                            batch_size=args['data_loader']['args']['batch_size'], 
                            shuffle=args['data_loader']['args']['shuffle'], 
                            num_workers=args['data_loader']['args']['num_workers'])
    
    # train
    for epoch in range(args['trainer']['epochs']):
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()    
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:    # print every 2000 mini-batches
                logger.info('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / epoch_steps))
                running_loss = 0.0
        
        if lr_scheduler is not None:
            lr_scheduler.step()
    
        # validation loss
        val_loss = 0.0
        val_steps = 0
        y_pred, y_true = [], []
        for i, data in enumerate(val_loader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_steps += 1
                
                y_pred.append(outputs.cpu().detach().numpy())
                y_true.append(labels.cpu().detach().numpy())
        
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        report = {'loss': val_loss / val_steps}
        for met in metrics:
            report[met.__name__] = met(y_pred, y_true)

        # checkpoint
        os.makedirs("my_model", exist_ok=True)
        torch.save(
            (model.state_dict(), optimizer.state_dict()), "my_model/checkpoint.pt")
        checkpoint = Checkpoint.from_directory("my_model")

        train.report(report, checkpoint=checkpoint)
    
    logger.info('Finished Training')

def test(args, model, testset, device):
    testloader = DataLoader(testset, 
                            batch_size=args['data_loader']['args']['batch_size'], 
                            shuffle=args['data_loader']['args']['shuffle'], 
                            num_workers=args['data_loader']['args']['num_workers'])

    metrics = [getattr(module_metric, met) for met in args['metrics']]
    test_metrics = {}
    y_pred, y_true = [], []
    for i, data in enumerate(testloader, 0):
        with torch.no_grad():
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            y_pred.append(outputs.cpu().detach().numpy())
            y_true.append(labels.cpu().detach().numpy())

    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    for met in metrics:
        test_metrics[met.__name__] = met(y_pred, y_true)

    return test_metrics


def main(args):
    logger = args.get_logger('Tune')

    # set the temp directory of Ray, usually broken when the path is too long
    ray_tmp_dir = '/home/jiannan/ray_temp_log'
    if not os.path.exists(ray_tmp_dir):
        os.makedirs(ray_tmp_dir)
    ray.init(_temp_dir=str(ray_tmp_dir))

    args['data_loader']['args']['logger'] = logger
    data_loader = args.init_obj('data_loader', module_data)
    _, testset = data_loader.get_dataset(data_dir=args['data_loader']['args']['data_dir'])

    config = {
        "l1": tune.grid_search([2**i for i in range(9)]),
        "l2": tune.grid_search([2**i for i in range(9)]),
        "lr": tune.grid_search([1e-4, 1e-3, 1e-2, 1e-1]),
        "batch_size": tune.grid_search([2, 4, 8, 16]),
    }
    scheduler = ASHAScheduler(
        max_t=args['trainer']['epochs'],
        grace_period=1,
        reduction_factor=2,
        metric='loss',
        mode='min')
    result = tune.run(
        partial(train_tune, logger=logger, args=args, data_loader=data_loader),
        resources_per_trial={"cpu": args['trainer']['cpus_per_trial'], 
                             "gpu": args['trainer']['gpus_per_trial']},
        config=config,
        num_samples=args['trainer']['num_samples'],
        scheduler=scheduler,
        local_dir=str(args.log_dir.absolute()))
    
    best_trial = result.get_best_trial("loss", "min", "last")
    logger.info("Best trial config: {}".format(best_trial.config))
    logger.info("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    logger.info("Best trial final validation metrics:")
    for key, value in best_trial.last_result.items():
        if key != "loss":
            logger.info("\t{}: {}".format(key, value))
    
    best_trained_model = MnistModel(best_trial.config["l1"], best_trial.config["l2"])
    device, device_ids = prepare_device(args['n_gpu'], logger)
    best_trained_model = best_trained_model.to(device)
    if len(device_ids) > 1:
        best_trained_model = torch.nn.DataParallel(best_trained_model, device_ids=device_ids)
    # load best checkpoint
    checkpoint_path = os.path.join(best_trial.checkpoint.to_directory(), "checkpoint.pt")
    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    # test
    test_metrics = test(args, best_trained_model, testset, device)
    value_format = ''.join(['{:15s}: {:.2f}\t'.format(k, v) for k, v in test_metrics.items()])
    logger.info('    {:15s}: {}'.format('test', value_format))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    args = ConfigParser.from_args(args, options)
    main(args)
