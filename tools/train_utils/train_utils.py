import glob
import os
import copy
import torch
import tqdm
from torch.nn.utils import clip_grad_norm_
import numpy as np


def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, 
                    model_teacher=None, model_copy=None, use_sub_data=False, cross_sample_prob=0.0,
                    tb_log=None, leave_pbar=False): # elodie teacher model/ use_sub_data, cross_sample_prob=0.0,
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)


    if cross_sample_prob>0:
        assert use_sub_data==False, 'use_sub_data is true, when cross_sample_prob>0 '
    #     cross_sample_array = np.random.choice([False, True], total_it_each_epoch, p=[cross_sample_prob, 1-cross_sample_prob])
    
    for cur_it in range(total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        optimizer.zero_grad()
        
        # print("batch:",batch)
        if model_teacher is not None: #elodie
            # print("\nbatch origin:\n",batch)
            batch_teacher = copy.deepcopy(batch)
            batch_teacher.pop('16lines')

            if cross_sample_prob>0:
                enable = np.random.choice([False, True], replace=False, p=[cross_sample_prob,1-cross_sample_prob])
                if enable:
                    batch['points'] = batch['16lines']['points_16lines']
                    batch['voxels'] = batch['16lines']['voxels']
                    batch['voxel_coords'] = batch['16lines']['voxel_coords']
                    batch['voxel_num_points'] = batch['16lines']['voxel_num_points']
            else:
                batch['points'] = batch['16lines']['points_16lines']
                batch['voxels'] = batch['16lines']['voxels']
                batch['voxel_coords'] = batch['16lines']['voxel_coords']
                batch['voxel_num_points'] = batch['16lines']['voxel_num_points']

            if 'voxel_coords_inbox' in batch['16lines']:
                batch['voxel_coords_inbox'] = batch['16lines']['voxel_coords_inbox']

            batch.pop('16lines')

            if use_sub_data:
                batch_dict_sub = {
                    'voxels': copy.deepcopy(batch['16lines']['voxels']),
                    'voxel_coords': copy.deepcopy(batch['16lines']['voxel_coords']),
                    'voxel_num_points': copy.deepcopy(batch['16lines']['voxel_num_points']),
                    'batch_size': batch['batch_size'],
                    'gt_boxes': batch['gt_boxes'],
                    'sub_data':True,
                }
                loss, tb_dict, disp_dict = model_func(model, batch, batch_dict_teacher=batch_teacher, model_teacher=model_teacher, batch_dict_sub=batch_dict_sub)
            else:
                loss, tb_dict, disp_dict = model_func(model, batch, batch_dict_teacher=batch_teacher, model_teacher=model_teacher)
        else:
            batch_dict_sub = None
            if "16lines" in batch: # dangerous
                if use_sub_data:
                    batch_dict_sub = {
                        'voxels': copy.deepcopy(batch['16lines']['voxels']),
                        'voxel_coords': copy.deepcopy(batch['16lines']['voxel_coords']),
                        'voxel_num_points': copy.deepcopy(batch['16lines']['voxel_num_points']),
                        'batch_size': batch['batch_size'],
                        'sub_data':True,
                    }
                    batch.pop('16lines')  
                elif cross_sample_prob>0:
                    enable = np.random.choice([False, True], replace=False, p=[cross_sample_prob,1-cross_sample_prob])
                    # enable = cross_sample_array[cur_it]
                    if enable:
                        batch['points'] = batch['16lines']['points_16lines']
                        batch['voxels'] = batch['16lines']['voxels']
                        batch['voxel_coords'] = batch['16lines']['voxel_coords']
                        batch['voxel_num_points'] = batch['16lines']['voxel_num_points']
                        batch.pop('16lines')  
                else:
                    batch.pop('16lines')            
            loss, tb_dict, disp_dict = model_func(model, batch, batch_dict_sub=batch_dict_sub, model_copy=model_copy)
        loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()

        accumulated_iter += 1
        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})

        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    if key.find('/')!=-1: # elodie
                        name = key
                    else:
                        name = 'train/' + key
                    tb_log.add_scalar(name, val, accumulated_iter)
    if rank == 0:
        pbar.close()
    return accumulated_iter


def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, 
                model_teacher=None, model_copy=None, use_sub_data=False, cross_sample_prob=0.0, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False): # elodie teacher model
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader, model_func,
                model_teacher=model_teacher,
                model_copy=model_copy,
                use_sub_data=use_sub_data,
                cross_sample_prob=cross_sample_prob,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter
            ) # elodie model teacher

            # save trained model
            trained_epoch = cur_epoch + 1
            if (trained_epoch % ckpt_save_interval == 0 or trained_epoch==total_epochs) and rank == 0: #elodie total_epochs
                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)
