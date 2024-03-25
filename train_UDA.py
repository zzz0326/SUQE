import os
import math
import time
import yaml
import argparse
import torch
import torch.optim as optim
import os.path as op
import numpy as np
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict
import utils  # my tool box
import dataset
from net_stdf import MFVQE, Position_Selector
import torch.nn as nn
import torch.nn.functional as F


def receive_arg():
    """Process all hyper-parameters and experiment settings.

    Record in opts_dict."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--opt_path', type=str, default='option_R3_mfqev2_4G.yml',
        help='Path to option YAML file.'
    )
    parser.add_argument(
        '--local_rank', type=int, default=0,
        help='Distributed launcher requires.'
    )
    args = parser.parse_args()

    with open(args.opt_path, 'r') as fp:
        opts_dict = yaml.load(fp, Loader=yaml.FullLoader)

    opts_dict['opt_path'] = args.opt_path
    opts_dict['train']['rank'] = args.local_rank

    if opts_dict['train']['exp_name'] == None:
        opts_dict['train']['exp_name'] = utils.get_timestr()

    opts_dict['train']['log_path'] = op.join(
        "exp", opts_dict['train']['exp_name'], "log.log"
    )
    opts_dict['train']['checkpoint_save_path_pre'] = op.join(
        "exp", opts_dict['train']['exp_name'], "ckp_"
    )

    opts_dict['train']['num_gpu'] = torch.cuda.device_count()
    if opts_dict['train']['num_gpu'] > 1:
        opts_dict['train']['is_dist'] = True
    else:
        opts_dict['train']['is_dist'] = False

    opts_dict['test']['restore_iter'] = int(
        opts_dict['test']['restore_iter']
    )

    return opts_dict

l2_regularization_coef = 0.001

def ema(alpha, model_t, model_s):
    for param_s, param_t in zip(model_s.parameters(), model_t.parameters()):
        # pdb.set_trace()
        param_t.data = alpha * param_t.data.detach() + (1 - alpha) * \
                       param_s.data.detach()

def main():
    # ==========
    # parameters
    # ==========
    loss_fun = nn.MSELoss()
    opts_dict = receive_arg()
    rank = opts_dict['train']['rank']
    unit = opts_dict['train']['criterion']['unit']
    num_iter = int(opts_dict['train']['num_iter'])
    interval_print = int(opts_dict['train']['interval_print'])
    interval_val = int(opts_dict['train']['interval_val'])

    # ==========
    # init distributed training
    # ==========

    if opts_dict['train']['is_dist']:
        utils.init_dist(
            local_rank=rank,
            backend='nccl'
        )

    # TO-DO: load resume states if exists
    pass

    # ==========
    # create logger
    # ==========

    if rank == 0:
        log_dir = op.join("exp", opts_dict['train']['exp_name'])
        utils.mkdir(log_dir)
        log_fp = open(opts_dict['train']['log_path'], 'w')

        # log all parameters
        msg = (
            f"{'<' * 10} Hello {'>' * 10}\n"
            f"Timestamp: [{utils.get_timestr()}]\n"
            f"\n{'<' * 10} Options {'>' * 10}\n"
            f"{utils.dict2str(opts_dict)}"
        )
        print(msg)
        log_fp.write(msg + '\n')
        log_fp.flush()

    # ==========
    # TO-DO: init tensorboard
    # ==========

    pass

    # ==========
    # fix random seed
    # ==========

    seed = opts_dict['train']['random_seed']
    # >I don't know why should rs + rank
    utils.set_random_seed(seed + rank)

    # ==========
    # Ensure reproducibility or Speed up
    # ==========

    # torch.backends.cudnn.benchmark = False  # if reproduce
    # torch.backends.cudnn.deterministic = True  # if reproduce
    torch.backends.cudnn.benchmark = True  # speed up

    # ==========
    # create train and val data prefetchers
    # ==========

    # create datasets
    train_ds_type = opts_dict['dataset']['train']['type']
    val_ds_type = opts_dict['dataset']['val']['type']
    radius = opts_dict['network']['radius']
    assert train_ds_type in dataset.__all__, \
        "Not implemented!"
    assert val_ds_type in dataset.__all__, \
        "Not implemented!"
    train_ds_cls = getattr(dataset, train_ds_type)
    val_ds_cls = getattr(dataset, val_ds_type)
    train_ds = train_ds_cls(
        opts_dict=opts_dict['dataset']['train'],
        radius=radius
    )
    val_ds = val_ds_cls(
        opts_dict=opts_dict['dataset']['val'],
        radius=radius
    )

    # create datasamplers
    train_sampler = utils.DistSampler(
        dataset=train_ds,
        num_replicas=opts_dict['train']['num_gpu'],
        rank=rank,
        ratio=opts_dict['dataset']['train']['enlarge_ratio']
    )
    val_sampler = None  # no need to sample val data

    # create dataloaders
    train_loader = utils.create_dataloader(
        dataset=train_ds,
        opts_dict=opts_dict,
        sampler=train_sampler,
        phase='train',
        seed=opts_dict['train']['random_seed']
    )
    val_loader = utils.create_dataloader(
        dataset=val_ds,
        opts_dict=opts_dict,
        sampler=val_sampler,
        phase='val'
    )
    assert train_loader is not None

    batch_size = opts_dict['dataset']['train']['batch_size_per_gpu'] * \
                 opts_dict['train']['num_gpu']  # divided by all GPUs
    num_iter_per_epoch = math.ceil(len(train_ds) * \
                                   opts_dict['dataset']['train']['enlarge_ratio'] / batch_size)
    num_epoch = math.ceil(num_iter / num_iter_per_epoch)
    val_num = len(val_ds)

    # create dataloader prefetchers
    tra_prefetcher = utils.CPUPrefetcher(train_loader)
    val_prefetcher = utils.CPUPrefetcher(val_loader)

    # ==========
    # create model
    # ==========

    model_T = MFVQE(opts_dict=opts_dict['network'])
    model_S = MFVQE(opts_dict=opts_dict['network'])
    model = Position_Selector(opts_dict=opts_dict['network'])
    model = model.to(rank)
    model_T = model_T.to(rank)
    model_S = model_S.to(rank)
    if opts_dict['train']['is_dist']:
        model = DDP(model, device_ids=[rank])
    if opts_dict['train']['is_dist']:
        model_T = DDP(model_T, device_ids=[rank])
    if opts_dict['train']['is_dist']:
        model_S = DDP(model_S, device_ids=[rank])

    # load pre-trained generator
    ckp_path = opts_dict['network']['stdf']['load_path']
    checkpoint = torch.load(ckp_path)
    state_dict = checkpoint['state_dict']
    if ('module.' in list(state_dict.keys())[0]) and (
            not opts_dict['train']['is_dist']):  # multi-gpu pre-trained -> single-gpu training
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove module
            new_state_dict[name] = v
        # model.load_state_dict(new_state_dict)
        model_T.load_state_dict(new_state_dict, False)
        model_S.load_state_dict(new_state_dict, False)
        print(f'loaded from {ckp_path}')
    elif ('module.' not in list(state_dict.keys())[0]) and (
            opts_dict['train']['is_dist']):  # single-gpu pre-trained -> multi-gpu training
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k  # add module
            new_state_dict[name] = v
        # model.load_state_dict(new_state_dict)
        model_T.load_state_dict(new_state_dict, False)
        model_S.load_state_dict(new_state_dict, False)
        print(f'loaded from {ckp_path}')
    else:  # the same way of training
        # model.load_state_dict(state_dict)
        model_T.load_state_dict(state_dict, False)
        model_S.load_state_dict(state_dict, False)
        # model_train.load_state_dict(state_dict, False)
        print(f'loaded from {ckp_path}')

    ckp_path = opts_dict['network']['stdf']['load_path_linear']
    checkpoint = torch.load(ckp_path)
    state_dict = checkpoint['state_dict']
    if ('module.' in list(state_dict.keys())[0]) and (
            not opts_dict['train']['is_dist']):  # multi-gpu pre-trained -> single-gpu training
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove module
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        # model_train.load_state_dict(new_state_dict, False)
        print(f'loaded from {ckp_path}')
    elif ('module.' not in list(state_dict.keys())[0]) and (
            opts_dict['train']['is_dist']):  # single-gpu pre-trained -> multi-gpu training
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k  # add module
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        # model_train.load_state_dict(new_state_dict, False)
        print(f'loaded from {ckp_path}')
    else:  # the same way of training
        model.load_state_dict(state_dict)
        # model_train.load_state_dict(state_dict, False)
        print(f'loaded from {ckp_path}')
    model.eval()
    model.requires_grad_(False)
    model_T.eval()
    model_T.requires_grad_(False)

    # ==========
    # define loss func & optimizer & scheduler & scheduler & criterion
    # ==========

    # define loss func
    assert opts_dict['train']['loss'].pop('type') == 'CharbonnierLoss', \
        "Not implemented."
    loss_func = utils.CharbonnierLoss(**opts_dict['train']['loss'])
    # print(model)
    # define optimizer
    assert opts_dict['train']['optim'].pop('type') == 'Adam', \
        "Not implemented."
    optimizer = optim.Adam(
        model_S.parameters(),
        **opts_dict['train']['optim']
    )
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

    # define scheduler
    # if opts_dict['train']['scheduler']['is_on']:
    #     assert opts_dict['train']['scheduler'].pop('type') == \
    #            'CosineAnnealingRestartLR', "Not implemented."
    #     del opts_dict['train']['scheduler']['is_on']
    #     scheduler = utils.CosineAnnealingRestartLR(
    #         optimizer,
    #         **opts_dict['train']['scheduler']
    #     )
    #     opts_dict['train']['scheduler']['is_on'] = True

    # define criterion
    assert opts_dict['train']['criterion'].pop('type') == \
           'PSNR', "Not implemented."
    criterion = utils.PSNR()

    #

    start_iter = 0  # should be restored
    start_epoch = start_iter // num_iter_per_epoch

    # display and log
    if rank == 0:
        msg = (
            f"\n{'<' * 10} Dataloader {'>' * 10}\n"
            f"total iters: [{num_iter}]\n"
            f"total epochs: [{num_epoch}]\n"
            f"iter per epoch: [{num_iter_per_epoch}]\n"
            f"val sequence: [{val_num}]\n"
            f"start from iter: [{start_iter}]\n"
            f"start from epoch: [{start_epoch}]"
        )
        print(msg)
        log_fp.write(msg + '\n')
        log_fp.flush()

    # ==========
    # evaluate original performance, e.g., PSNR before enhancement
    # ==========

    vid_num = val_ds.get_vid_num()
    if opts_dict['train']['pre-val'] and rank == 0:
        msg = f"\n{'<' * 10} Pre-evaluation {'>' * 10}"
        print(msg)
        log_fp.write(msg + '\n')

        per_aver_dict = {}
        for i in range(vid_num):
            per_aver_dict[i] = utils.Counter()
        pbar = tqdm(
            total=val_num,
            ncols=opts_dict['train']['pbar_len']
        )

        # fetch the first batch
        val_prefetcher.reset()
        val_data = val_prefetcher.next()

        while val_data is not None:
            # get data
            gt_data = val_data['gt'].to(rank)  # (B [RGB] H W)
            lq_data = val_data['lq'].to(rank)  # (B T [RGB] H W)
            index_vid = val_data['index_vid'].item()
            name_vid = val_data['name_vid'][0]  # bs must be 1!
            b, _, _, _, _ = lq_data.shape

            # eval
            batch_perf = np.mean(
                [criterion(lq_data[i, radius, ...], gt_data[i]) for i in range(b)]
            )  # bs must be 1!

            # log
            per_aver_dict[index_vid].accum(volume=batch_perf)

            # display
            pbar.set_description(
                "{:s}: [{:.3f}] {:s}".format(name_vid, batch_perf, unit)
            )
            pbar.update()

            # fetch next batch
            val_data = val_prefetcher.next()

        pbar.close()

        # log
        ave_performance = np.mean([
            per_aver_dict[index_vid].get_ave() for index_vid in range(vid_num)
        ])
        msg = "> ori performance: [{:.3f}] {:s}".format(ave_performance, unit)
        print(msg)
        log_fp.write(msg + '\n')
        log_fp.flush()

    if opts_dict['train']['is_dist']:
        torch.distributed.barrier()  # all processes wait for ending

    if rank == 0:
        msg = f"\n{'<' * 10} Training {'>' * 10}"
        print(msg)
        log_fp.write(msg + '\n')

        # create timer
        total_timer = utils.Timer()  # total tra + val time of each epoch

    # ==========
    # start training + validation (test)
    # ==========

    model_S.train()
    num_iter_accum = start_iter
    for current_epoch in range(start_epoch, num_epoch + 1):
        # shuffle distributed subsamplers before each epoch
        if opts_dict['train']['is_dist']:
            train_sampler.set_epoch(current_epoch)

        # fetch the first batch
        tra_prefetcher.reset()
        train_data = tra_prefetcher.next()
        loss_list = 0
        # train this epoch
        while train_data is not None:

            # over sign
            num_iter_accum += 1
            if num_iter_accum > num_iter:
                break
            x = np.random.randint(8, 216)
            y = np.random.randint(8, 2808)
            # get data
            gt_data = train_data['gt'][:, :, x:x+64, y:y+64].to(rank)  # (B [RGB] H W)
            lq_data_1 = train_data['lq'][:, :, :, x:x+64, y:y+64].to(rank)  # (B T [RGB] H W)
            # lq_h, lq_w = [80, 80]

            b, _, c, _, _ = lq_data_1.shape
            input_data_1 = torch.cat(
                [lq_data_1[:, :, i, ...] for i in range(c)],
                dim=1
            )  # B [R1 ... R7 G1 ... G7 B1 ... B7] H W
            # input_data_2 = torch.cat(
            #     [lq_data_2[:, :, i, ...] for i in range(c)],
            #     dim=1
            # )  # B [R1 ... R7 G1 ... G7 B1 ... B7] H W
            # search_range = 8
            prob_1, prob_2 = model(input_data_1)

            off_temp_1 = []
            off_temp_2 = []
            for i in range(b):
                off_temp_1.append(torch.argmax(prob_1[i, :]) - 8)
                off_temp_2.append(torch.argmax(prob_2[i, :]) - 8)

            Move= []


            for i in range(b):
                if prob_1[i, int(off_temp_1[i]) + 8] > prob_2[i, int(off_temp_2[i]) + 8] and int(off_temp_1[i]) != 0:
                    # 1. input data 2. 如果是0怎么处理 3. 获取输入数据
                    # lq_data_2.append(train_data['lq'][i, :, :, x + off_temp_1[i]:x + off_temp_1[i],
                    #      y:y+64].unsqueeze(0).to(rank))
                    Move.append([off_temp_1[i], 0])
                elif prob_2[i, int(off_temp_2[i]) + 8] > prob_1[0, int(off_temp_1[i]) + 8] and int(off_temp_2[i]) != 0:
                    # lq_data_2.append(train_data['lq'][i, :, :, x:x+64,
                    #      y + int(off_temp_2[i]):y + int(off_temp_2[i])].unsqueeze(0).to(rank))
                    Move.append([0, off_temp_2[i]])
                else:
                    # lq_data_2.append(train_data['lq'][i, :, :, x:x+64,
                    #                  y:y+64].unsqueeze(0).to(rank))
                    Move.append([0, 0])
            # lq_data_2 = []

            lq_data_2 = torch.cat(
                [train_data['lq'][i, :, :, x + Move[i][0]:x+64 + Move[i][0],
                 y+Move[i][1]:y+64+Move[i][1]].unsqueeze(0).to(rank) for i in range(b)],
                dim=0
            )
                    # lq_data_2 = train_data['lq_2'][:, :, :, 8+val_1:72+val_1, 8+val_2:72+val_2].to(rank)

            # lq_data_2 = torch.cat([lq_data_2[i] for i in range(b)])
            input_data_2 = torch.cat(
                [lq_data_2[:, :, i, ...] for i in range(c)],
                dim=1
            )

            Pseudo = model_S(input_data_1)
            output = model_T(input_data_2)
            if num_iter_accum % 100 == 0:
                ema(0.999, model_T, model_S)
            # PSNR_org = criterion(output, gt_data)

            loss = torch.mean(torch.stack(
                    [loss_func(Pseudo[i, :, 8:56, 8:56], output[i, :, 8-Move[i][0]:56-Move[i][0], 8-Move[i][1]:56-Move[i][1]]) for i in range(b)]
                ))  # cal loss
            # loss = loss_fun(probs_1, label_1) + loss_fun(probs_2, label_2) + l2_regularization_coef * l2_reg
            # print(loss)
            loss_list += loss
            # print(loss)
            loss.backward()  # cal grad
            optimizer.step()  # update parameters
            PSNR_enh = np.mean([criterion(model_S(input_data_1)[i], gt_data[i]) for i in range(b)])
            PSNR_org = np.mean([criterion(model_T(input_data_1)[i], gt_data[i]) for i in range(b)])
            print(PSNR_enh - PSNR_org)
            # scheduler.step()
            # print(psnr_0 / b, psnr_1 / b, psnr_2 / b)

            # loss = psnr_2 - psnr_1

            # lable_train = torch.zeros((b,), dtype=torch.long).to(rank)
            # for z in range(b):
            #
            #     val_1 = criterion(enhanced_data_1, gt_data)
            #     val_2 = criterion(enhanced_data_2, gt_data)
            #
            #     if val_1 > val_2:
            #         lable_train[z] = 0
            #     else:
            #         lable_train[z] = 1
            # lable_train.type(torch.long)
            # get loss
            # zero grad
            # loss = torch.mean(torch.stack(
            #     [loss_func(enhanced_data[i], gt_data[i]) for i in range(b)]
            # ))  # cal loss
            # loss = loss_fun(model_train(input_data_1, input_data_2), lable_train)
            # loss = -F.log_softmax(psnr_2 - psnr_1, dim=1)

            # 最小化输出 b，使用均方误差损失函数（mean squared error loss）
            # loss_b = F.mse_loss(psnr_1, torch.zeros_like(psnr_1))

            # 将两个损失函数组合起来，可以使用任何你需要的权重
            # loss = loss_a + loss_b

            # return loss

            # loss = psnr_2 - psnr_1

            # update learning rate


            if (num_iter_accum % interval_print == 0) and (rank == 0):
                # display & log
                lr = optimizer.param_groups[0]['lr']
                loss_item = loss.item()
                msg = (
                    f"iter: [{num_iter_accum}]/{num_iter}, "
                    f"epoch: [{current_epoch}]/{num_epoch - 1}, "
                    "lr: [{:.3f}]x1e-4, loss: [{:.7f}], total_loss: [{:.7}]".format(
                        lr * 1e4, loss_item, loss_list
                    )
                )
                loss_list = 0
                print(msg)
                log_fp.write(msg + '\n')

            if ((num_iter_accum % interval_val == 0) or \
                (num_iter_accum == num_iter)) and (rank == 0):
                # save model
                checkpoint_save_path = (
                    f"{opts_dict['train']['checkpoint_save_path_pre']}"
                    f"{num_iter_accum}_S"
                    ".pt"
                )
                state = {
                    'num_iter_accum': num_iter_accum,
                    'state_dict': model_S.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                # if opts_dict['train']['scheduler']['is_on']:
                #     state['scheduler'] = scheduler.state_dict()
                torch.save(state, checkpoint_save_path)

                checkpoint_save_path = (
                    f"{opts_dict['train']['checkpoint_save_path_pre']}"
                    f"{num_iter_accum}_T"
                    ".pt"
                )
                state = {
                    'num_iter_accum': num_iter_accum,
                    'state_dict': model_S.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                # if opts_dict['train']['scheduler']['is_on']:
                #     state['scheduler'] = scheduler.state_dict()
                torch.save(state, checkpoint_save_path)

                # validation
                # with torch.no_grad():
                #     per_aver_dict = {}
                #     for index_vid in range(vid_num):
                #         per_aver_dict[index_vid] = utils.Counter()
                #     pbar = tqdm(
                #         total=val_num,
                #         ncols=opts_dict['train']['pbar_len']
                #     )
                #
                #     # train -> eval
                #
                # # log
                # ave_per = np.mean([
                #     per_aver_dict[index_vid].get_ave() for index_vid in range(vid_num)
                # ])
                # msg = (
                #     "> model saved at {:s}\n"
                #     "> ave val per: [{:.3f}] {:s}"
                # ).format(
                #     checkpoint_save_path, ave_per, unit
                # )
                # print(msg)
                # log_fp.write(msg + '\n')
                log_fp.flush()

            if opts_dict['train']['is_dist']:
                torch.distributed.barrier()  # all processes wait for ending

            # fetch next batch
            train_data = tra_prefetcher.next()

        # end of this epoch (training dataloader exhausted)

    # end of all epochs

    # ==========
    # final log & close logger
    # ==========

    if rank == 0:
        total_time = total_timer.get_interval() / 3600
        msg = "TOTAL TIME: [{:.1f}] h".format(total_time)
        print(msg)
        log_fp.write(msg + '\n')

        msg = (
            f"\n{'<' * 10} Goodbye {'>' * 10}\n"
            f"Timestamp: [{utils.get_timestr()}]"
        )
        print(msg)
        log_fp.write(msg + '\n')

        log_fp.close()


if __name__ == '__main__':
    main()
