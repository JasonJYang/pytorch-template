import argparse
import collections
import torch
import numpy as np
from functools import partial
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
from model.model import MnistModel
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device

from torch.utils.data import DataLoader, random_split
from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler

def train_tune(config, logger, parameter_config, data_loader):
    model = MnistModel(parameter_config['l1'], parameter_config['l2'])
    # get device
    device, device_ids = prepare_device(config['n_gpu'], logger)
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(trainable_params, lr=parameter_config['lr'], momentum=0.9)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    # checkpoint
    checkpoint = session.get_checkpoint()
    if checkpoint:
        checkpoint_state = checkpoint.to_dict()
        start_epoch = checkpoint_state["epoch"]
        model.load_state_dict(checkpoint_state["net_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    # dataset
    trainset, _ = data_loader.get_dataset(data_dir=config['data_loader']['args']['data_dir'])
    test_abs = int(len(trainset) * 0.2)
    train_subset, val_subset = random_split(trainset, [len(trainset) - test_abs, test_abs])
    train_loader = DataLoader(train_subset, 
                              batch_size=config['data_loader']['args']['batch_size'], 
                              shuffle=config['data_loader']['args']['shuffle'], 
                              num_workers=config['data_loader']['args']['num_workers'])
    val_loader = DataLoader(val_subset,
                            batch_size=config['data_loader']['args']['batch_size'], 
                            shuffle=config['data_loader']['args']['shuffle'], 
                            num_workers=config['data_loader']['args']['num_workers'])
    
    # train
    for epoch in range(start_epoch, config['trainer']['epoch']):
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
        val_metrics = {}
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
        checkpoint = Checkpoint(epoch=epoch,
                                net_state_dict=model.state_dict(),
                                optimizer_state_dict=optimizer.state_dict())
        session.report(**report, checkpoint=checkpoint)
    
    logger.info('Finished Training')

def test(config, model, testset, device):
    testloader = DataLoader(testset, 
                            batch_size=config['data_loader']['args']['batch_size'], 
                            shuffle=config['data_loader']['args']['shuffle'], 
                            num_workers=config['data_loader']['args']['num_workers'])

    metrics = [getattr(module_metric, met) for met in config['metrics']]
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


def main(config):
    logger = config.get_logger('Tune')

    config['data_loader']['args']['logger'] = logger
    data_loader = config.init_obj('data_loader', module_data)

    parameter_config = {
        "l1": tune.choice([2**i for i in range(9)]),
        "l2": tune.choice([2**i for i in range(9)]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16]),
    }
    scheduler = ASHAScheduler(
        max_t=config['trainer']['epochs'],
        grace_period=1,
        reduction_factor=2,
        metric='loss',
        mode='min')
    result = tune.run(
        partial(train_tune, config, logger, parameter_config, data_loader),
        resources_per_trial={"cpu": config['trainer']['cpus_per_trial'], 
                             "gpu": config['trainer']['gpus_per_trial']},
        config=parameter_config,
        num_samples=config['trainer']['num_samples'],
        scheduler=scheduler,
        config_dir=config.log_dir)
    
    best_trial = result.get_best_trial("loss", "min", "last")
    logger.info("Best trial config: {}".format(best_trial.config))
    logger.info("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    logger.info("Best trial final validation metrics:")
    for key, value in best_trial.last_result.items():
        if key != "loss":
            logger.info("\t{}: {}".format(key, value))
    
    best_trained_model = MnistModel(best_trial.config["l1"], best_trial.config["l2"])
    device, device_ids = prepare_device(config['n_gpu'], logger)
    best_trained_model = best_trained_model.to(device)
    if len(device_ids) > 1:
        best_trained_model = torch.nn.DataParallel(best_trained_model, device_ids=device_ids)
    
    # load best checkpoint
    best_checkpoint = best_trial.checkpoint.to_air_checkpoint()
    best_checkpoint_data = best_checkpoint.to_dict()

    best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])

    # test
    _, testset = data_loader.get_dataset(data_dir=config['data_loader']['args']['data_dir'])
    test_metrics = test(config, best_trained_model, testset, device)
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
    config = ConfigParser.from_args(args, options)
    main(config)
