import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
import argparse
import logging
import os
import copy
from math import *
import random

import datetime
#from torch.utils.tensorboard import SummaryWriter

from model import *
from utils import *
from vggmodel import *
from resnetcifar import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mlp', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=2,  help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg', help='fl algorithms: fedavg/scaffold/moon')
    parser.add_argument('--use_projection_head', type=bool, default=False, help='whether add an additional header to model or not (see MOON)')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--loss', type=str, default='contrastive', help='for moon')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--is_same_initial', type=int, default=1, help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox')
    parser.add_argument('--noise', type=float, default=0, help='how much noise we add to some party')
    parser.add_argument('--noise_type', type=str, default='level', help='Different level of noise or different space of noise')
    parser.add_argument('--rho', type=float, default=0, help='Parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=1, help='Sample ratio for each communication round')
    parser.add_argument('--rootdir', type=str, required=False, default="./result/bench/", help='root log directory path')
    args = parser.parse_args()
    return args

def init_nets(net_configs, dropout_p, n_parties, args):

    nets = {net_i: None for net_i in range(n_parties)}
    
    if args.dataset.lower() in ("cifar10", "fmnist"):
        num_classes = 10
    elif args.dataset.lower() == "cifar100":
        num_classes = 100
    else:
        raise ValueError("Choose dataset from: CIFAR10/CIFAR100/FMNIST")
    
    if args.dataset.lower() in ("cifar10", "cifar100"):
        channel, im_size = 3, (32, 32)
    elif args.dataset.lower() == "fmnist":
        channel, im_size = 1, (28, 28)
    else:
        raise ValueError("Choose dataset from: CIFAR10/CIFAR100/FMNIST")
                
    for net_i in range(n_parties):
        if args.model.lower() == 'resnet18':
            net = torchvision.models.resnet18(num_classes=num_classes).to(args.device)
        elif args.model.lower() == 'convnet':
            net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'batchnorm', 'avgpooling'
            net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='batchnorm', net_pooling=net_pooling, im_size=im_size)
            net.to(args.device)
        nets[net_i] = net    

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)
    return nets, model_meta_data, layer_type



def train_net_scaffold(net_id, net, global_model, c_local, c_global, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    # train_acc = compute_accuracy(net, train_dataloader, device=device)
    # test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    # logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    # logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    #writer = SummaryWriter()

    c_global_para = c_global.state_dict()
    c_local_para = c_local.state_dict()

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                _, out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                net_para = net.state_dict()
                for key in net_para:
                    net_para[key] = net_para[key] - args.lr * (c_global_para[key] - c_local_para[key])
                net.load_state_dict(net_para)

                cnt += 1
                epoch_loss_collector.append(loss.item())


        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    c_new_para = c_local.state_dict()
    c_delta_para = copy.deepcopy(c_local.state_dict())
    global_model_para = global_model.state_dict()
    net_para = net.state_dict()
    for key in net_para:
        c_new_para[key] = c_new_para[key] - c_global_para[key] + (global_model_para[key] - net_para[key]) / (cnt * args.lr)
        c_delta_para[key] = c_new_para[key] - c_local_para[key]
    c_local.load_state_dict(c_new_para)


    # train_acc = compute_accuracy(net, train_dataloader, device=device)
    # test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    # logger.info('>> Training accuracy: %f' % train_acc)
    # logger.info('>> Test accuracy: %f' % test_acc)

    # logger.info(' ** Training complete **')
    # return train_acc, test_acc, c_delta_para

    test_acc, test_loss = evaluation(args, net, test_dataloader)
    logger.info('>> Client {:2d} Test loss: {:.4f}'.format(net_id, loss))
    logger.info('>> Client {:2d} Test acc: {:.2f}%'.format(net_id, acc*100))
    logger.info(' ** Training complete **')
    net.to('cpu')
    return test_loss, test_acc, c_delta_para


def train_net_moon(net_id, net, global_net, previous_nets, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, temperature, args,
                      round, device="cpu"):

    logger.info('Training network %s' % str(net_id))

    # train_acc = compute_accuracy(net, train_dataloader, device=device)
    # test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    # logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    # logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    # conloss = ContrastiveLoss(temperature)

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9, weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)
    # global_net.to(device)

    if args.loss != 'l2norm':
        for previous_net in previous_nets:
            previous_net.to(device)
    global_w = global_net.state_dict()

    cnt = 0
    cos=torch.nn.CosineSimilarity(dim=-1).to(device)
    # mu = 0.001

    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            pro1, out = net(x)
            pro2, _ = global_net(x)
            if args.loss == 'l2norm':
                loss2 = mu * torch.mean(torch.norm(pro2-pro1, dim=1))

            elif args.loss == 'only_contrastive' or args.loss == 'contrastive':
                posi = cos(pro1, pro2)
                logits = posi.reshape(-1,1)

                for previous_net in previous_nets:
                    previous_net.to(device)
                    pro3, _ = previous_net(x)
                    nega = cos(pro1, pro3)
                    logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)

                    # previous_net.to('cpu')

                logits /= temperature
                labels = torch.zeros(x.size(0)).to(device).long()

                # loss = criterion(out, target) + mu * ContraLoss(pro1, pro2, pro3)

                loss2 = mu * criterion(logits, labels)

            if args.loss == 'only_contrastive':
                loss = loss2
            else:
                loss1 = criterion(out, target)
                loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f' % (epoch, epoch_loss, epoch_loss1, epoch_loss2))


    if args.loss != 'l2norm':
        for previous_net in previous_nets:
            previous_net.to('cpu')

    # train_acc = compute_accuracy(net, train_dataloader, device=device)
    # test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    # logger.info('>> Training accuracy: %f' % train_acc)
    # logger.info('>> Test accuracy: %f' % test_acc)
    test_acc, test_loss = evaluation(args, net, test_dataloader)
    logger.info('>> Client {:2d} Test loss: {:.4f}'.format(net_id, loss))
    logger.info('>> Client {:2d} Test acc: {:.2f}%'.format(net_id, acc*100))
    logger.info(' ** Training complete **')
    net.to('cpu')
    return test_loss, test_acc



def local_train_net_scaffold(nets, selected, global_model, c_nets, c_global, args, train_dl_local, test_dl = None, device="cpu"):
    avg_acc = 0.0

    total_delta = copy.deepcopy(global_model.state_dict())
    for key in total_delta:
        total_delta[key] = 0.0
    c_global.to(device)
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        logger.info(f'Training network for client {net_id:2d}')
        # dataidxs = net_dataidx_map[net_id]
        # logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        
        # move the model to cuda device:
        net.to(device)
        c_nets[net_id].to(device)

        # noise_level = args.noise
        # if net_id == args.n_parties - 1:
        #     noise_level = 0

        # if args.noise_type == 'space':
        #     train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        # else:
        #     noise_level = args.noise / (args.n_parties - 1) * net_id
        #     train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        # train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        # n_epoch = args.epochs


        _, loc_test_acc, c_delta_para = train_net_scaffold(net_id, net, global_model, c_nets[net_id], c_global, train_dl_local[net_id], test_dl, args.n_epoch, args.lr, args.optimizer, device=device)

        c_nets[net_id].to('cpu')
        for key in total_delta:
            total_delta[key] += c_delta_para[key]

        # logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += loc_test_acc

    for key in total_delta:
        total_delta[key] /= args.n_parties
    c_global_para = c_global.state_dict()

    for key in c_global_para:
        if c_global_para[key].type() == 'torch.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.LongTensor)
        elif c_global_para[key].type() == 'torch.cuda.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
        else:
            #print(c_global_para[key].type())
            c_global_para[key] += total_delta[key]
    c_global.load_state_dict(c_global_para)
    avg_acc /= len(selected)
    # if args.alg == 'local_training':
    #     logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list


def local_train_net_moon(nets, selected, args, train_dl_local, test_dl=None, global_model = None, prev_model_pool = None, round=None, device="cpu"):
    avg_acc = 0.0
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        # dataidxs = net_dataidx_map[net_id]
        # logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        
        logger.info(f'Training network for client {net_id:2d}')
        net.to(device)

        # noise_level = args.noise
        # if net_id == args.n_parties - 1:
        #     noise_level = 0

        # if args.noise_type == 'space':
        #     train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        # else:
        #     noise_level = args.noise / (args.n_parties - 1) * net_id
        #     train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        # train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        # n_epoch = args.epochs

        prev_models=[]
        for i in range(len(prev_model_pool)):
            prev_models.append(prev_model_pool[i][net_id])
        # trainacc, testacc = train_net_moon(net_id, net, global_model, prev_models, train_dl_local, test_dl, n_epoch, args.lr,
        #                                       args.optimizer, args.mu, args.temperature, args, round, device=device)
        # logger.info("net %d final test acc %f" % (net_id, testacc))
        _, loc_test_acc = train_net_moon(net_id, net, global_model, prev_models, train_dl_local[i], test_dl, args.n_epoch, args.lr, args.optimizer, args.mu, args.temperature, args, round, device=device)
        avg_acc += loc_test_acc

    avg_acc /= len(selected)
    # if args.alg == 'local_training':
    #     logger.info("avg test acc %f" % avg_acc)
    global_model.to('cpu')
    nets_list = list(nets.values())
    return nets_list


if __name__ == '__main__':
    time_start = time.time()

    # torch.set_printoptions(profile="full")
    args = get_args()
    start_timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    tag = f"{args.dataset}-{args.model}-{args.alg}-N{args.n_parties}-beta{args.beta}-ep{args.epochs}-lr{args.lr}-round{args.comm_round}-{start_timestamp}"
    if args.alg == "moon":
        tag = tag + f"-mu{args.mu}"
    exp_dir = os.path.join(args.rootdir, tag) 
    args.logdir = os.path.join(exp_dir, f"seed{args.init_seed}")
    os.makedirs(args.logdir)

    if args.log_file_name is None:
        argument_path= f'experiment_log-{start_timestamp}'
    else:
        argument_path=args.log_file_name+'.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)

    if args.log_file_name is None:
        args.log_file_name = f'experiment_log-{start_timestamp}'
    log_path=args.log_file_name+'.log'
    logger = get_logger(logger_path=os.path.join(args.logdir, log_path))

    seed = args.init_seed
    logger.info("#" * 100)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)           

    data_set, data_info = get_dataset(dataset=args.dataset)
    args.n_class = data_info['num_classes']
    

    # split the train data
    logger.info("Partitioning data")
    _, net_dataidx_map = partition_labeldir(
        targets=data_set['train_labels'], 
        num_classes=data_info['num_classes'], 
        n_parties=args.n_parties,
        beta=args.beta
    )
    traindata_cls_counts = record_net_data_stats(data_set['train_labels'], net_dataidx_map, logger)
    train_dl_global, test_dl_global, _, _ = get_dataloader2(data_set['train_data'], data_set['test_data'], args.batch_size, 2*args.batch_size)
    
    # make the local trainloder for clients
    train_dl_loc=[]
    for party_id in range(args.n_parties):
        dataidxs = net_dataidx_map[party_id]
        # logger.info(f"Client {party_id:2d} is associated with {len(dataidxs):5f} data")
        party_train_dl, party_test_dl, _, _ = get_dataloader2(data_set['train_data'], data_set['test_data'], args.batch_size, 2*args.batch_size, dataidxs)
        train_dl_loc.append(party_train_dl)

    glob_loss, glob_acc = [], [], [], [] 

    if args.alg == 'scaffold':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        c_nets, _, _ = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        c_globals, _, _ = init_nets(args.net_config, 0, 1, args)
        c_global = c_globals[0]
        c_global_para = c_global.state_dict()
        for net_id, net in c_nets.items():
            net.load_state_dict(c_global_para)

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)


        for round in range(args.comm_round):
            logger.info("In comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net_scaffold(nets, selected, global_model, c_nets, c_global, args, train_dl_local=train_dl_loc, test_dl = test_dl_global, device=device)
            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)


            # logger.info('global n_training: %d' % len(train_dl_global))
            # logger.info('global n_test: %d' % len(test_dl_global))

            # global_model.to(device)
            # train_acc = compute_accuracy(global_model, train_dl_global, device=device)
            # test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device)

            # logger.info('>> Global Model Train accuracy: %f' % train_acc)
            # logger.info('>> Global Model Test accuracy: %f' % test_acc)

            # evaluate and log the performance of the global model
            acc, loss = evaluation(args, global_model, test_dl_global)
            glob_loss.append(loss)
            glob_acc.append(acc)

            if acc>best_glob_acc:
                best_glob_acc = acc

            logger.info(f'>> Global model test loss: {loss:.4f}'.format(loss))
            logger.info(f'>> Global model test accuracy: {acc:.4f}, historical best acc: {best_glob_acc:.4f}')


    elif args.alg == 'moon':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        old_nets_pool = []
        old_nets = copy.deepcopy(nets)
        for _, net in old_nets.items():
            net.eval()
            for param in net.parameters():
                param.requires_grad = False

        for round in range(args.comm_round):
            logger.info("In comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net_moon(nets, selected, args, net_dataidx_map, test_dl = test_dl_global, global_model=global_model,
                                 prev_model_pool=old_nets_pool, round=round, device=device)
            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            # logger.info('global n_training: %d' % len(train_dl_global))
            # logger.info('global n_test: %d' % len(test_dl_global))


            # train_acc = compute_accuracy(global_model, train_dl_global)
            # test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True)

            # logger.info('>> Global Model Train accuracy: %f' % train_acc)
            # logger.info('>> Global Model Test accuracy: %f' % test_acc)

            old_nets = copy.deepcopy(nets)
            for _, net in old_nets.items():
                net.eval()
                for param in net.parameters():
                    param.requires_grad = False
            if len(old_nets_pool) < 1:
                old_nets_pool.append(old_nets)
            else:
                old_nets_pool[0] = old_nets

            # evaluate and log the performance of the global model
            acc, loss = evaluation(args, global_model, test_dl_global)
            glob_loss.append(loss)
            glob_acc.append(acc)

            if acc>best_glob_acc:
                best_glob_acc = acc

            logger.info(f'>> Global model test loss: {loss:.4f}'.format(loss))
            logger.info(f'>> Global model test accuracy: {acc:.4f}, historical best acc: {best_glob_acc:.4f}')

   
    glob_loss, glob_acc, glob_balacc, glob_auc = np.array(glob_loss), np.array(glob_acc)
    # plot and save test loss curves
    plot_series(
        series=glob_loss,
        y_min=0,
        y_max=np.ceil(glob_loss.max()),
        series_name='Test loss',
        title='Test loss of global model',
        save_path=os.path.join(args.logdir, 'loss_fed_isic_{}_{}.png'.format(args.alg, args.model))
    )

    # plot and save test acc curves
    plot_series(
        series=glob_acc,
        y_min=0,
        y_max=1.0,
        series_name='Test acc',
        title='Test acc of global model',
        save_path=os.path.join(args.logdir, 'acc_fed_isic_{}_{}.png'.format(args.alg, args.model))
    )

    torch.save(
        {
            'global_model_test_loss': glob_loss,
            'global_model_test_acc': glob_acc,
            'global_model_test_auc': glob_auc,
            'global_model_test_bal_acc': glob_balacc,
        },
        os.path.join(args.logdir, 'learning_curve_{}_{}.pt'.format(args.alg, args.model))
    )
    
    # record the experiment trial result
    record_accuracy(dir_path=exp_dir, experiment_num=args.init_seed, acc=best_glob_acc)

    
    time_end = time.time()
    time_end_stamp = time.strftime('%Y-%m-%d %H:%M:%S') # time_end_stamp = time.strftime('%y-%m-%d-%H-%M-%S')
    sesseion_time = int((time_end-time_start)/60)   
    logger.info('\nSession completed at {}, time elapsed: {} mins. That\'s all folks.'.format(time_end_stamp, sesseion_time))
