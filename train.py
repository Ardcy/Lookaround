from tqdm import tqdm
from utils import progress_bar
from utils import set_random_seed, setup_logger, make_optimizer, make_scheduler, make_loss, WarmUpLR
import numpy as np
import argparse
import torch
from torch.multiprocessing import Pool
from data import build_dataloader, build_transforms
from models import init_model
import os
from timm.utils import accuracy
import time
import copy
from config import cfg
import time
from sam import SAM
from lookahead import Lookahead


def parse_option():

    parser = argparse.ArgumentParser(description='train lookaround')
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--yaml_path', type=str, default='.yaml')
    parser.add_argument('--out', type=str, default='')
    parser.add_argument('--train_mode', type=str, default='TRAIN3')
    parser.add_argument('--DATA_DIR', type=str, default="")
    parser.add_argument('--tnum', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--frequency', type=int, default=1)
    parser.add_argument('--optimizer', type=str, default="")
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument("--gpu_ids", type=str, default="")
    parser.add_argument('--pretrained', type=bool, default=None)

    args = parser.parse_args()

    if args.train_mode != '':
        cfg.TRAIN_MODE = args.train_mode
    cfg.merge_from_file(
        f'configs/{args.yaml_path}')

    if args.gpu_ids != "":
        cfg.MODEL.GPU_IDS = eval(args.gpu_ids)
    else:
        cfg.MODEL.GPU_IDS = [x for x in range(8)]
    cfg.SOLVER.FREQUENCY = args.frequency
    cfg.MODEL.DEVICE_ID = args.cuda_id
    cfg.SOLVER.BATCH_SIZE = args.batch_size
    cfg.DATA_DIR = args.DATA_DIR
    if args.out != '':
        cfg.OUTPUT_DIR = args.out
    if cfg.TRAIN_MODE == 'TRAINNM':
        cfg.BUILD_TRANSFORM_NUM = args.tnum
    if args.optimizer != '':
        cfg.SOLVER.OPTIMIZER_NAME = args.optimizer
    if args.lr != None:
        cfg.SOLVER.LR = args.lr
    if args.pretrained != None:
        cfg.MODEL.PRETRAINED = args.pretrained
    print(cfg.MODEL.PRETRAINED)
    return cfg, args


def train_imagenet_swa(cfg, is_greedy=False, T_num=6, is_image_net=True):
    device = 'cuda:' + str(cfg.MODEL.DEVICE_ID)

    TRANSFORM_list = ['RandomHorizontalFlip',
                      'RandomVerticalFlip', 'RandAugment']
    trainloaders = [build_dataloader(cfg, build_transforms(
        cfg, transform), cfg.DATA_DIR, is_train=True) for transform in TRANSFORM_list]
    test_t = build_transforms(cfg, 'test')
    testloader = build_dataloader(cfg, test_t, cfg.DATA_DIR, is_train=False)
    model = init_model(cfg)
    device_ids = cfg.MODEL.GPU_IDS

    if is_image_net or cfg.DATASET.IMAGESIZE == 224:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        device = 'cuda:' + str(device_ids[0])

    model = model.to(device)
    swa_model = torch.optim.swa_utils.AveragedModel(model)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_scheduler(cfg, optimizer)
    loss_function = make_loss(cfg)

    if is_image_net:
        swa_start = 70
    else:
        swa_start = 170

    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=0.0008)
    iter_per_epoch = len(trainloaders[0])
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch)

    output_dir = cfg.OUTPUT_DIR + args.yaml_path[0:-5] + '/'+f'TRAIN_SWA'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger = setup_logger(f"output_dir {cfg.TRAIN_MODE}", output_dir, 0)
    logger.info("Running with config:\n{}".format(cfg))
    test_acc = []
    best_acc1 = 0

    swa_model.train()
    for epoch in range(cfg.SOLVER.MAX_EPOCHS):
        train_iter1 = iter(trainloaders[0])
        train_iter2 = iter(trainloaders[1])
        train_iter3 = iter(trainloaders[2])
        logger.info(f"train model:  epoch: {epoch}")
        model.train()
        train_loss = 0
        correct1 = 0
        correct5 = 0
        total = 0
        tbar = tqdm(range(len(trainloaders[0])))
        s = time.time()
        net_time = 0
        iter_time = 0
        for batch_idx in tbar:
            for x in [train_iter1, train_iter2, train_iter3]:
                s = time.time()
                inputs, targets = next(x)
                iter_time += time.time() - s
                s = time.time()
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()

            if epoch < cfg.SOLVER.WARM:
                warmup_scheduler.step()
            train_loss += loss.item()
            if epoch > swa_start:
                swa_model.update_parameters(model)
            total += targets.size(0)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            correct1 += acc1.item() * cfg.SOLVER.BATCH_SIZE / 100
            correct5 += acc5.item() * cfg.SOLVER.BATCH_SIZE / 100
            net_time += time.time() - s
            s = time.time()
            tbar.set_postfix({"lr": optimizer.state_dict()['param_groups'][0]['lr'], "loss": round(
                train_loss/(batch_idx+1), 3), "Acc1": round(100.*correct1/total, 3), "Acc5": round(100.*correct5/total, 3), "iter_time": iter_time, "net_time": net_time})

        logger.info(
            f"lr: {optimizer.state_dict()['param_groups'][0]['lr']},loss:{round(train_loss/(batch_idx+1),3)},acc1:{round(100.*correct1/total,3)},acc5:{round(100.*correct5/total,3)}")

        if epoch > swa_start:
            swa_scheduler.step()
        else:
            scheduler.step()

        model.eval()
        logger.info(f"eval model epoch: {epoch}")
        test_loss = 0
        correct1 = 0
        correct5 = 0
        total = 0
        with torch.no_grad():
            tbar = tqdm(testloader)
            for batch_idx, (inputs, targets) in enumerate(tbar):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                test_loss += loss.item()
                total += targets.size(0)
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                correct1 += acc1.item()
                correct5 += acc5.item()
                tbar.set_postfix(
                    {"Acc1": round(100.*correct1/total, 3), "Acc5": round(100.*correct5/total, 3)})

        acc = (100.*correct1/total, 100.*correct5/total)
        test_acc.append(acc)
        logger.info(
            f"lr: {optimizer.state_dict()['param_groups'][0]['lr']},loss:{round(test_loss/(batch_idx+1),3)},acc1:{round(100.*correct1/total,3)},acc5:{round(100.*correct5/total,3)}")

        # save model
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        if best_acc1 < test_acc[-1][0]:
            best_acc1 = test_acc[-1][0]
            torch.save(checkpoint, output_dir +
                       f'/bestacc_{cfg.MODEL.NAME}.ckpt')

        if epoch % cfg.SAVE_FREQ == 0:
            torch.save(checkpoint, output_dir +
                       f'/{cfg.MODEL.NAME}_{epoch}.ckpt')

        torch.save(checkpoint, output_dir + f'/last_{cfg.MODEL.NAME}.ckpt')
        np.save(output_dir + '/test_acc.npy', np.array(test_acc))

    swa_model = swa_model.to(device)
    torch.optim.swa_utils.update_bn(trainloaders[0], swa_model, device=device)

    swa_model.eval()
    logger.info(f"swa_model: {epoch}")
    test_loss = 0
    correct1 = 0
    correct5 = 0
    total = 0
    with torch.no_grad():
        tbar = tqdm(testloader)
        for batch_idx, (inputs, targets) in enumerate(tbar):

            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = swa_model(inputs)
            loss = loss_function(outputs, targets)

            test_loss += loss.item()
            total += targets.size(0)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            correct1 += acc1.item()
            correct5 += acc5.item()

            tbar.set_postfix(
                {"Acc1": round(100.*correct1/total, 3), "Acc5": round(100.*correct5/total, 3)})
    logger.info(
        f"lr: {optimizer.state_dict()['param_groups'][0]['lr']},loss:{round(test_loss/(batch_idx+1),3)},acc1:{round(100.*correct1/total,3)},acc5:{round(100.*correct5/total,3)}")
    checkpoint = {
        'epoch': epoch,
        'model': swa_model.state_dict(),
    }

    torch.save(checkpoint, output_dir + f'/swa_model_{cfg.MODEL.NAME}.ckpt')

    return


def train_imagenet_swad(cfg, is_greedy=False, T_num=6, is_image_net=True):
    device = 'cuda:' + str(cfg.MODEL.DEVICE_ID)

    TRANSFORM_list = ['RandomHorizontalFlip',
                      'RandomVerticalFlip', 'RandAugment']
    trainloaders = [build_dataloader(cfg, build_transforms(
        cfg, transform), cfg.DATA_DIR, is_train=True) for transform in TRANSFORM_list]
    test_t = build_transforms(cfg, 'test')
    testloader = build_dataloader(cfg, test_t, cfg.DATA_DIR, is_train=False)
    model = init_model(cfg)
    device_ids = cfg.MODEL.GPU_IDS

    if is_image_net or cfg.DATASET.IMAGESIZE == 224:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        device = 'cuda:' + str(device_ids[0])

    model = model.to(device)
    optimizer = make_optimizer(cfg, model)
    scheduler = make_scheduler(cfg, optimizer)
    loss_function = make_loss(cfg)

    if is_image_net:
        swa_start = 70
    else:
        swa_start = 170

    swa_model = None
    iter_per_epoch = len(trainloaders[0])
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch)

    output_dir = cfg.OUTPUT_DIR + args.yaml_path[0:-5] + '/'+f'TRAIN_SWAD'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger = setup_logger(f"output_dir {cfg.TRAIN_MODE}", output_dir, 0)
    logger.info("Running with config:\n{}".format(cfg))
    test_acc = []
    best_acc1 = 0

    val_test_input = None
    val_test_target = None
    Loss_list_s = []
    Loss_list_t = []
    ll = None
    test_iter = iter(testloader)
    val_test_input, val_test_target = next(test_iter)

    for epoch in range(cfg.SOLVER.MAX_EPOCHS):
        train_iter1 = iter(trainloaders[0])
        train_iter2 = iter(trainloaders[1])
        train_iter3 = iter(trainloaders[2])
        logger.info(f"train model:  epoch: {epoch}")
        model.train()
        train_loss = 0
        correct1 = 0
        correct5 = 0
        total = 0
        tbar = tqdm(range(len(trainloaders[0])))
        s = time.time()
        net_time = 0
        iter_time = 0
        print("len = ", len(trainloaders[0]))
        for batch_idx in tbar:
            for x in [train_iter1, train_iter2, train_iter3]:
                s = time.time()
                inputs, targets = next(x)
                iter_time += time.time() - s
                s = time.time()
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()

                if epoch > swa_start:
                    model.eval()
                    lss = loss_function(model(val_test_input.to(
                        device)), val_test_target.to(device)).item()
                    model.train()
                    Loss_list_s.append(lss)
                    Loss_list_t.append(lss)
                    if len(Loss_list_s) > 3:
                        Loss_list_s = Loss_list_s[-3:]
                    if len(Loss_list_t) > 6:
                        Loss_list_t = Loss_list_t[-6:]
                    if len(Loss_list_s) == 3:
                        if Loss_list_s[0] == min(Loss_list_s):
                            ll = 1.2/3*sum(Loss_list_s)
                            swa_model = torch.optim.swa_utils.AveragedModel(
                                model)

                    if ll:
                        if not min(Loss_list_t) > ll:
                            swa_model.update_parameters(model)
                        else:
                            break

            if epoch < cfg.SOLVER.WARM:
                warmup_scheduler.step()

            train_loss += loss.item()
            total += targets.size(0)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            correct1 += acc1.item() * cfg.SOLVER.BATCH_SIZE / 100
            correct5 += acc5.item() * cfg.SOLVER.BATCH_SIZE / 100
            net_time += time.time() - s
            s = time.time()
            tbar.set_postfix({"lr": optimizer.state_dict()['param_groups'][0]['lr'], "loss": round(
                train_loss/(batch_idx+1), 3), "Acc1": round(100.*correct1/total, 3), "Acc5": round(100.*correct5/total, 3), "iter_time": iter_time, "net_time": net_time})

        logger.info(
            f"lr: {optimizer.state_dict()['param_groups'][0]['lr']},loss:{round(train_loss/(batch_idx+1),3)},acc1:{round(100.*correct1/total,3)},acc5:{round(100.*correct5/total,3)}")
        if ll:
            if min(Loss_list_t) > ll:
                break
        scheduler.step()

        model.eval()
        logger.info(f"eval model epoch: {epoch}")
        test_loss = 0
        correct1 = 0
        correct5 = 0
        total = 0
        with torch.no_grad():
            tbar = tqdm(testloader)
            for batch_idx, (inputs, targets) in enumerate(tbar):

                if batch_idx == 0:
                    val_test_input = copy.deepcopy(inputs)
                    val_test_target = copy.deepcopy(targets)
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                test_loss += loss.item()
                total += targets.size(0)
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                correct1 += acc1.item()
                correct5 += acc5.item()
                tbar.set_postfix(
                    {"Acc1": round(100.*correct1/total, 3), "Acc5": round(100.*correct5/total, 3)})

        acc = (100.*correct1/total, 100.*correct5/total)
        test_acc.append(acc)
        logger.info(
            f"lr: {optimizer.state_dict()['param_groups'][0]['lr']},loss:{round(test_loss/(batch_idx+1),3)},acc1:{round(100.*correct1/total,3)},acc5:{round(100.*correct5/total,3)}")

        # save model
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        if best_acc1 < test_acc[-1][0]:
            best_acc1 = test_acc[-1][0]
            torch.save(checkpoint, output_dir +
                       f'/bestacc_{cfg.MODEL.NAME}.ckpt')

        if epoch % cfg.SAVE_FREQ == 0:
            torch.save(checkpoint, output_dir +
                       f'/{cfg.MODEL.NAME}_{epoch}.ckpt')

        torch.save(checkpoint, output_dir + f'/last_{cfg.MODEL.NAME}.ckpt')

        np.save(output_dir + '/test_acc.npy', np.array(test_acc))

    swa_model = swa_model.to(device)
    torch.optim.swa_utils.update_bn(trainloaders[0], swa_model, device=device)

    swa_model.eval()
    logger.info(f"swad_model: {epoch}")
    test_loss = 0
    correct1 = 0
    correct5 = 0
    total = 0
    with torch.no_grad():
        tbar = tqdm(testloader)
        for batch_idx, (inputs, targets) in enumerate(tbar):

            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = swa_model(inputs)
            loss = loss_function(outputs, targets)

            test_loss += loss.item()

            total += targets.size(0)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            correct1 += acc1.item()
            correct5 += acc5.item()

            tbar.set_postfix(
                {"Acc1": round(100.*correct1/total, 3), "Acc5": round(100.*correct5/total, 3)})
    logger.info(
        f"lr: {optimizer.state_dict()['param_groups'][0]['lr']},loss:{round(test_loss/(batch_idx+1),3)},acc1:{round(100.*correct1/total,3)},acc5:{round(100.*correct5/total,3)}")
    checkpoint = {
        'epoch': epoch,
        'model': swa_model.state_dict(),
    }

    torch.save(checkpoint, output_dir + f'/swad_model_{cfg.MODEL.NAME}.ckpt')
    return


def trainnm_imagenet_lookaround(cfg, is_greedy=False, T_num=6, is_image_net=True):
    device = 'cuda:' + str(cfg.MODEL.DEVICE_ID)
    TRANSFORM_list = ['RandomHorizontalFlip',
                      'RandomHorizontalFlip', 'RandomHorizontalFlip']
    trainloaders = [build_dataloader(cfg, build_transforms(
        cfg, transform), cfg.DATA_DIR, is_train=True) for transform in TRANSFORM_list]
    test_t = build_transforms(cfg, 'test')
    testloader = build_dataloader(cfg, test_t, cfg.DATA_DIR, is_train=False)
    train_iter1 = iter(trainloaders[0])
    model = init_model(cfg)
    device_ids = cfg.MODEL.GPU_IDS
    if is_image_net or cfg.DATASET.IMAGESIZE == 224:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        device = 'cuda:' + str(device_ids[0])

    model = model.to(device)
    optimizer = make_optimizer(cfg, model)
    scheduler = make_scheduler(cfg, optimizer)
    loss_function = make_loss(cfg)

    iter_per_epoch = len(trainloaders[0])
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch)

    output_dir = cfg.OUTPUT_DIR + \
        args.yaml_path[0:-5] + '/'+f'TRAINNM_{cfg.SOLVER.FREQUENCY}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger = setup_logger(f"output_dir {cfg.TRAIN_MODE}", output_dir, 0)
    logger.info("Running with config:\n{}".format(cfg))
    test_acc = []
    best_acc1 = 0

    for epoch in range(cfg.SOLVER.MAX_EPOCHS):
        train_iter1 = iter(trainloaders[0])
        train_iter2 = iter(trainloaders[1])
        train_iter3 = iter(trainloaders[2])
        logger.info(f"train model:  epoch: {epoch}")
        model.train()
        train_loss = 0
        correct1 = 0
        correct5 = 0
        total = 0
        tbar = tqdm(range(len(trainloaders[0])))
        s = time.time()
        net_time = 0
        iter_time = 0
        print("len = ", len(trainloaders[0]))
        for batch_idx in tbar:

            for x in [train_iter1, train_iter2, train_iter3]:
                s = time.time()
                inputs, targets = next(x)
                iter_time += time.time() - s
                s = time.time()
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()

            if batch_idx % 10 == 0 and 'root' in cfg.OUTPUT_DIR:
                print("now end batch_idx = ", batch_idx, net_time, iter_time)
            if epoch < cfg.SOLVER.WARM:
                warmup_scheduler.step()
            train_loss += loss.item()
            total += targets.size(0)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            correct1 += acc1.item() * cfg.SOLVER.BATCH_SIZE / 100
            correct5 += acc5.item() * cfg.SOLVER.BATCH_SIZE / 100
            net_time += time.time() - s
            s = time.time()
            tbar.set_postfix({"lr": optimizer.state_dict()['param_groups'][0]['lr'], "loss": round(
                train_loss/(batch_idx+1), 3), "Acc1": round(100.*correct1/total, 3), "Acc5": round(100.*correct5/total, 3), "iter_time": iter_time, "net_time": net_time})

        logger.info(
            f"lr: {optimizer.state_dict()['param_groups'][0]['lr']},loss:{round(train_loss/(batch_idx+1),3)},acc1:{round(100.*correct1/total,3)},acc5:{round(100.*correct5/total,3)}")

        scheduler.step()
        model.eval()
        logger.info(f"eval model epoch: {epoch}")
        test_loss = 0
        correct1 = 0
        correct5 = 0
        total = 0
        with torch.no_grad():
            tbar = tqdm(testloader)
            for batch_idx, (inputs, targets) in enumerate(tbar):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                test_loss += loss.item()
                total += targets.size(0)
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                correct1 += acc1.item()
                correct5 += acc5.item()
                tbar.set_postfix(
                    {"Acc1": round(100.*correct1/total, 3), "Acc5": round(100.*correct5/total, 3)})

        acc = (100.*correct1/total, 100.*correct5/total)
        test_acc.append(acc)
        logger.info(
            f"lr: {optimizer.state_dict()['param_groups'][0]['lr']},loss:{round(test_loss/(batch_idx+1),3)},acc1:{round(100.*correct1/total,3)},acc5:{round(100.*correct5/total,3)}")

        # save model
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        if best_acc1 < test_acc[-1][0]:
            best_acc1 = test_acc[-1][0]
            torch.save(checkpoint, output_dir +
                       f'/bestacc_{cfg.MODEL.NAME}.ckpt')
        if epoch % cfg.SAVE_FREQ == 0:
            torch.save(checkpoint, output_dir +
                       f'/{cfg.MODEL.NAME}_{epoch}.ckpt')
        torch.save(checkpoint, output_dir + f'/last_{cfg.MODEL.NAME}.ckpt')
        np.save(output_dir + '/test_acc.npy', np.array(test_acc))
    return


def train_imagenet_lookahead(cfg, is_greedy=False, T_num=6, is_image_net=True):
    device = 'cuda:' + str(cfg.MODEL.DEVICE_ID)

    TRANSFORM_list = ['RandomHorizontalFlip',
                      'RandomVerticalFlip', 'RandAugment']
    trainloaders = [build_dataloader(cfg, build_transforms(
        cfg, transform), cfg.DATA_DIR, is_train=True) for transform in TRANSFORM_list]
    test_t = build_transforms(cfg, 'test')
    testloader = build_dataloader(cfg, test_t, cfg.DATA_DIR, is_train=False)
    train_iter1 = iter(trainloaders[0])

    model = init_model(cfg)
    device_ids = cfg.MODEL.GPU_IDS
    if is_image_net or cfg.DATASET.IMAGESIZE == 224:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        device = 'cuda:' + str(device_ids[0])

    model = model.to(device)
    base_optimizer = make_optimizer(cfg, model)
    optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
    loss_function = make_loss(cfg)

    scheduler = make_scheduler(cfg, optimizer)
    loss_function = make_loss(cfg)

    iter_per_epoch = len(trainloaders[0])
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch)

    output_dir = cfg.OUTPUT_DIR + args.yaml_path[0:-5] + '/'+f'TRAIN_LookAhead'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger = setup_logger(f"output_dir {cfg.TRAIN_MODE}", output_dir, 0)
    logger.info("Running with config:\n{}".format(cfg))
    test_acc = []
    best_acc1 = 0

    for epoch in range(cfg.SOLVER.MAX_EPOCHS):
        train_iter1 = iter(trainloaders[0])
        train_iter2 = iter(trainloaders[1])
        train_iter3 = iter(trainloaders[2])
        logger.info(f"train model:  epoch: {epoch}")
        model.train()
        train_loss = 0
        correct1 = 0
        correct5 = 0
        total = 0
        tbar = tqdm(range(len(trainloaders[0])))
        s = time.time()
        net_time = 0
        iter_time = 0
        for batch_idx in tbar:
            for x in [train_iter1, train_iter2, train_iter3]:
                s = time.time()
                inputs, targets = next(x)
                iter_time += time.time() - s
                s = time.time()
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()

            if batch_idx % 10 == 0 and 'root' in cfg.OUTPUT_DIR:
                print("now end batch_idx = ", batch_idx, net_time, iter_time)
            if epoch < cfg.SOLVER.WARM:
                warmup_scheduler.step()
            train_loss += loss.item()
            total += targets.size(0)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            correct1 += acc1.item() * cfg.SOLVER.BATCH_SIZE / 100
            correct5 += acc5.item() * cfg.SOLVER.BATCH_SIZE / 100
            net_time += time.time() - s
            s = time.time()
            tbar.set_postfix({"lr": optimizer.state_dict()['param_groups'][0]['lr'], "loss": round(
                train_loss/(batch_idx+1), 3), "Acc1": round(100.*correct1/total, 3), "Acc5": round(100.*correct5/total, 3), "iter_time": iter_time, "net_time": net_time})

        logger.info(
            f"lr: {optimizer.state_dict()['param_groups'][0]['lr']},loss:{round(train_loss/(batch_idx+1),3)},acc1:{round(100.*correct1/total,3)},acc5:{round(100.*correct5/total,3)}")

        scheduler.step()
        model.eval()
        logger.info(f"eval model epoch: {epoch}")
        test_loss = 0
        correct1 = 0
        correct5 = 0
        total = 0
        with torch.no_grad():
            tbar = tqdm(testloader)
            for batch_idx, (inputs, targets) in enumerate(tbar):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                test_loss += loss.item()
                total += targets.size(0)
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                correct1 += acc1.item()
                correct5 += acc5.item()
                tbar.set_postfix(
                    {"Acc1": round(100.*correct1/total, 3), "Acc5": round(100.*correct5/total, 3)})

        acc = (100.*correct1/total, 100.*correct5/total)
        test_acc.append(acc)
        logger.info(
            f"lr: {optimizer.state_dict()['param_groups'][0]['lr']},loss:{round(test_loss/(batch_idx+1),3)},acc1:{round(100.*correct1/total,3)},acc5:{round(100.*correct5/total,3)}")

        # save model
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        if best_acc1 < test_acc[-1][0]:
            best_acc1 = test_acc[-1][0]
            torch.save(checkpoint, output_dir +
                       f'/bestacc_{cfg.MODEL.NAME}.ckpt')

        if epoch % cfg.SAVE_FREQ == 0:
            torch.save(checkpoint, output_dir +
                       f'/{cfg.MODEL.NAME}_{epoch}.ckpt')

        torch.save(checkpoint, output_dir + f'/last_{cfg.MODEL.NAME}.ckpt')

        np.save(output_dir + '/test_acc.npy', np.array(test_acc))
    return


def train_imagenet_adamw(cfg, is_greedy=False, T_num=6, is_image_net=True):

    device = 'cuda:' + str(cfg.MODEL.DEVICE_ID)
    print(device)

    TRANSFORM_list = ['RandomHorizontalFlip',
                      'RandomVerticalFlip', 'RandAugment']
    trainloaders = [build_dataloader(cfg, build_transforms(
        cfg, transform), cfg.DATA_DIR, is_train=True) for transform in TRANSFORM_list]

    test_t = build_transforms(cfg, 'test')
    testloader = build_dataloader(cfg, test_t, cfg.DATA_DIR, is_train=False)
    train_iter1 = iter(trainloaders[0])

    model = init_model(cfg)
    device_ids = cfg.MODEL.GPU_IDS
    if is_image_net or cfg.DATASET.IMAGESIZE == 224:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        device = 'cuda:' + str(device_ids[0])
        model = model.to(device)
    else:
        model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.SOLVER.Adam_LR, betas=(
        cfg.SOLVER.Adam_Beta1, cfg.SOLVER.Adam_Beta2), weight_decay=cfg.SOLVER.Adam_weight_decay)
    loss_function = make_loss(cfg)
    scheduler = make_scheduler(cfg, optimizer)
    loss_function = make_loss(cfg)

    iter_per_epoch = len(trainloaders[0])
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch)

    output_dir = cfg.OUTPUT_DIR + args.yaml_path[0:-5] + '/'+f'TRAIN_ADAMW'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger = setup_logger(f"output_dir {cfg.TRAIN_MODE}", output_dir, 0)
    logger.info("Running with config:\n{}".format(cfg))
    test_acc = []
    best_acc1 = 0

    for epoch in range(cfg.SOLVER.MAX_EPOCHS):
        train_iter1 = iter(trainloaders[0])
        train_iter2 = iter(trainloaders[1])
        train_iter3 = iter(trainloaders[2])
        logger.info(f"train model:  epoch: {epoch}")
        model.train()
        train_loss = 0
        correct1 = 0
        correct5 = 0
        total = 0
        tbar = tqdm(range(len(trainloaders[0])))
        s = time.time()
        net_time = 0
        iter_time = 0
        print("len = ", len(trainloaders[0]))
        for batch_idx in tbar:
            for x in [train_iter1, train_iter2, train_iter3]:
                s = time.time()
                inputs, targets = next(x)
                iter_time += time.time() - s
                s = time.time()
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()
            if epoch < cfg.SOLVER.WARM:
                warmup_scheduler.step()
            train_loss += loss.item()

            total += targets.size(0)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            correct1 += acc1.item() * cfg.SOLVER.BATCH_SIZE / 100
            correct5 += acc5.item() * cfg.SOLVER.BATCH_SIZE / 100
            net_time += time.time() - s
            s = time.time()
            tbar.set_postfix({"lr": optimizer.state_dict()['param_groups'][0]['lr'], "loss": round(
                train_loss/(batch_idx+1), 3), "Acc1": round(100.*correct1/total, 3), "Acc5": round(100.*correct5/total, 3), "iter_time": iter_time, "net_time": net_time})

        logger.info(
            f"lr: {optimizer.state_dict()['param_groups'][0]['lr']},loss:{round(train_loss/(batch_idx+1),3)},acc1:{round(100.*correct1/total,3)},acc5:{round(100.*correct5/total,3)}")

        scheduler.step()
        model.eval()
        logger.info(f"eval model epoch: {epoch}")
        test_loss = 0
        correct1 = 0
        correct5 = 0
        total = 0
        with torch.no_grad():
            tbar = tqdm(testloader)
            for batch_idx, (inputs, targets) in enumerate(tbar):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                test_loss += loss.item()
                total += targets.size(0)
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                correct1 += acc1.item()
                correct5 += acc5.item()
                tbar.set_postfix(
                    {"Acc1": round(100.*correct1/total, 3), "Acc5": round(100.*correct5/total, 3)})

        acc = (100.*correct1/total, 100.*correct5/total)
        test_acc.append(acc)
        logger.info(
            f"lr: {optimizer.state_dict()['param_groups'][0]['lr']},loss:{round(test_loss/(batch_idx+1),3)},acc1:{round(100.*correct1/total,3)},acc5:{round(100.*correct5/total,3)}")

        # save model
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        if best_acc1 < test_acc[-1][0]:
            best_acc1 = test_acc[-1][0]
            torch.save(checkpoint, output_dir +
                       f'/bestacc_{cfg.MODEL.NAME}.ckpt')

        if epoch % cfg.SAVE_FREQ == 0:
            torch.save(checkpoint, output_dir +
                       f'/{cfg.MODEL.NAME}_{epoch}.ckpt')

        torch.save(checkpoint, output_dir + f'/last_{cfg.MODEL.NAME}.ckpt')

        np.save(output_dir + '/test_acc.npy', np.array(test_acc))
    return


def train_imagenet_sam(cfg, is_greedy=False, T_num=6, is_image_net=True):
    device = 'cuda:' + str(cfg.MODEL.DEVICE_ID)
    print(device)
    TRANSFORM_list = ['RandomHorizontalFlip',
                      'RandomVerticalFlip', 'RandAugment']
    trainloaders = [build_dataloader(cfg, build_transforms(
        cfg, transform), cfg.DATA_DIR, is_train=True) for transform in TRANSFORM_list]
    test_t = build_transforms(cfg, 'test')
    testloader = build_dataloader(cfg, test_t, cfg.DATA_DIR, is_train=False)
    train_iter1 = iter(trainloaders[0])
    model = init_model(cfg)
    device_ids = cfg.MODEL.GPU_IDS
    if is_image_net or cfg.DATASET.IMAGESIZE == 224:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        device = 'cuda:' + str(device_ids[0])

    model = model.to(device)
    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, lr=0.1, momentum=0.9)

    scheduler = make_scheduler(cfg, optimizer)
    loss_function = make_loss(cfg)

    iter_per_epoch = len(trainloaders[0])
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch)

    output_dir = cfg.OUTPUT_DIR + args.yaml_path[0:-5] + '/'+f'TRAIN_SAM'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger = setup_logger(f"output_dir {cfg.TRAIN_MODE}", output_dir, 0)
    logger.info("Running with config:\n{}".format(cfg))
    test_acc = []
    best_acc1 = 0

    for epoch in range(cfg.SOLVER.MAX_EPOCHS):
        train_iter1 = iter(trainloaders[0])
        train_iter2 = iter(trainloaders[1])
        train_iter3 = iter(trainloaders[2])
        logger.info(f"train model:  epoch: {epoch}")
        model.train()
        train_loss = 0
        correct1 = 0
        correct5 = 0
        total = 0
        tbar = tqdm(range(len(trainloaders[0])))
        s = time.time()
        net_time = 0
        iter_time = 0
        print("len = ", len(trainloaders[0]))
        for batch_idx in tbar:
            for x in [train_iter1, train_iter2, train_iter3]:
                s = time.time()
                inputs, targets = next(x)
                iter_time += time.time() - s
                s = time.time()
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                loss.backward()
                # sam
                optimizer.first_step(zero_grad=True)
                # make sure to do a full forward pass
                loss_function(model(inputs), targets).backward()
                optimizer.second_step(zero_grad=True)

            if epoch < cfg.SOLVER.WARM:
                warmup_scheduler.step()
            train_loss += loss.item()
            total += targets.size(0)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            correct1 += acc1.item() * cfg.SOLVER.BATCH_SIZE / 100
            correct5 += acc5.item() * cfg.SOLVER.BATCH_SIZE / 100
            net_time += time.time() - s
            s = time.time()
            tbar.set_postfix({"lr": optimizer.state_dict()['param_groups'][0]['lr'], "loss": round(
                train_loss/(batch_idx+1), 3), "Acc1": round(100.*correct1/total, 3), "Acc5": round(100.*correct5/total, 3), "iter_time": iter_time, "net_time": net_time})

        logger.info(
            f"lr: {optimizer.state_dict()['param_groups'][0]['lr']},loss:{round(train_loss/(batch_idx+1),3)},acc1:{round(100.*correct1/total,3)},acc5:{round(100.*correct5/total,3)}")

        scheduler.step()

        model.eval()
        logger.info(f"eval model epoch: {epoch}")
        test_loss = 0
        correct1 = 0
        correct5 = 0
        total = 0
        with torch.no_grad():
            tbar = tqdm(testloader)
            for batch_idx, (inputs, targets) in enumerate(tbar):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                test_loss += loss.item()
                total += targets.size(0)
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                correct1 += acc1.item()
                correct5 += acc5.item()
                tbar.set_postfix(
                    {"Acc1": round(100.*correct1/total, 3), "Acc5": round(100.*correct5/total, 3)})

        acc = (100.*correct1/total, 100.*correct5/total)
        test_acc.append(acc)
        logger.info(
            f"lr: {optimizer.state_dict()['param_groups'][0]['lr']},loss:{round(test_loss/(batch_idx+1),3)},acc1:{round(100.*correct1/total,3)},acc5:{round(100.*correct5/total,3)}")

        # save model
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        if best_acc1 < test_acc[-1][0]:
            best_acc1 = test_acc[-1][0]
            torch.save(checkpoint, output_dir +
                       f'/bestacc_{cfg.MODEL.NAME}.ckpt')
        if epoch % cfg.SAVE_FREQ == 0:
            torch.save(checkpoint, output_dir +
                       f'/{cfg.MODEL.NAME}_{epoch}.ckpt')
        torch.save(checkpoint, output_dir + f'/last_{cfg.MODEL.NAME}.ckpt')
        np.save(output_dir + '/test_acc.npy', np.array(test_acc))

    return


if __name__ == '__main__':

    cfg, args = parse_option()
    set_random_seed(cfg.SEED)

    if cfg.DATASET.NAME == 'imagenet':
        is_imagenet = True
    else:
        is_imagenet = False

    cfg.SOLVER.HEAD_NUM = 3
    cfg.SOLVER.FREQUENCY = 5
    cfg.BUILD_TRANSFORM_NUM = 3

    if args.train_mode == 'TRAIN_LOOKAROUND':
        trainnm_imagenet_lookaround(cfg, T_num=3, is_image_net=is_imagenet)
    elif args.train_mode == 'TRAIN_SGD':
        train_imagenet_swa(cfg, T_num=3, is_image_net=is_imagenet)
    elif args.train_mode == 'TRAIN_SWA':
        train_imagenet_swa(cfg, T_num=3, is_image_net=is_imagenet)
    elif args.train_mode == 'TRAIN_LOOKAHEAD':
        train_imagenet_lookahead(cfg, T_num=3, is_image_net=is_imagenet)
    elif args.train_mode == 'TRAIN_ADAMW':
        train_imagenet_adamw(cfg, T_num=3, is_image_net=is_imagenet)
    elif args.train_mode == 'TRAIN_SAM':
        train_imagenet_sam(cfg, T_num=3, is_image_net=is_imagenet)
    elif args.train_mode == 'TRAIN_SWAD':
        train_imagenet_swad(cfg, T_num=3, is_image_net=is_imagenet)
