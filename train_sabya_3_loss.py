import os
import sys
import time
import argparse
import logging

import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from data import dataset
import modules
import util

logger = logging.getLogger(__name__)


def train_frequency_representation(args, fr_module, fr_optimizer, fr_criterion, fr_scheduler, train_loader, val_loader,
                                   epoch, tb_writer):
    """
    Train the frequency-representation module for one epoch
    """
    epoch_start_time = time.time()
    fr_module.train()
    loss_train_fr = 0
    loss_train_ae = 0
    loss_train = 0
    loss_train_stft = 0
    for batch_idx, (clean_signal, noise_signal, target_fr) in enumerate(train_loader):
        if args.use_cuda:
            noise_signal, clean_signal, target_fr = noise_signal.cuda(), clean_signal.cuda(), target_fr.cuda()
        fr_optimizer.zero_grad()
        output_ae, output_reshape, output_fr = fr_module(noise_signal)
        loss_ae = ae_criterion(output_ae, clean_signal)
        loss_stft = 0
        
        for i in range(len(output_reshape[1])):
            output_temp = torch.unsqueeze(output_reshape[:,i,:,:],dim=1)
            loss_stft = loss_stft + reshape_criterion(output_temp, target_fr)
        loss_stft = loss_stft/len(output_reshape[1])
        
        loss_fr = fr_criterion(output_fr, target_fr)
        
        loss = 3*loss_ae + loss_fr + 0*loss_stft
        loss.backward()
        fr_optimizer.step()
        fr_optimizer.zero_grad()
        loss_train_fr += loss_fr.data.item()
        loss_train_ae += loss_ae.data.item()
        loss_train_stft += loss_stft.data.item()
        loss_train += loss.data.item()

    fr_module.eval()
    loss_val_ae, loss_val_fr, loss_val_stft, loss_val = 0, 0, 0, 0
    for batch_idx, (clean_signal, noise_signal, target_fr) in enumerate(val_loader):
        if args.use_cuda:
            noise_signal, clean_signal, target_fr = noise_signal.cuda(), clean_signal.cuda(), target_fr.cuda()
        fr_optimizer.zero_grad()
        output_ae, output_reshape, output_fr = fr_module(noise_signal)
        loss_ae = ae_criterion(output_ae, clean_signal)
        loss_fr = fr_criterion(output_fr, target_fr)
        
        loss_stft = 0
        
        for i in range(len(output_reshape[1])):
            output_temp = torch.unsqueeze(output_reshape[:,i,:,:],dim=1)
            loss_stft = loss_stft + reshape_criterion(output_temp, target_fr)
        loss_stft = loss_stft/len(output_reshape[1])
        
        loss = 3*loss_ae + loss_fr + 0*loss_stft
        loss_val_fr += loss_fr.data.item()
        loss_val_ae += loss_ae.data.item()
        loss_val_stft += loss_stft.data.item()
        loss_val += loss.data.item()


    loss_train_fr /= args.n_training
    loss_val_fr /= args.n_validation
    loss_train_ae /= args.n_training
    loss_val_ae /= args.n_validation
    loss_train_stft /= args.n_training
    loss_val_stft /= args.n_validation
    loss_train /= args.n_training
    loss_val /= args.n_validation

    # tb_writer.add_scalar('ae_l2_training', loss_train_ae, epoch)
    # tb_writer.add_scalar('fr_l2_training', loss_train_fr, epoch)
    # tb_writer.add_scalar('total_l2_training', loss_train, epoch)
    
    # tb_writer.add_scalar('ae_l2_validation', loss_val_ae, epoch)
    # tb_writer.add_scalar('fr_l2_validation', loss_val_fr, epoch)
    # tb_writer.add_scalar('total_l2_validation', loss_val, epoch)

    # fr_scheduler.step(loss_val)
    # logger.info("Epochs: %d / %d, Time: %.1f, AE_train_loss %.2f, FR training L2 loss %.2f, AE_val_loss %.2f, FR validation L2 loss %.2f",
    #             epoch, args.n_epochs_fr, time.time() - epoch_start_time, loss_train_fr, loss_val_fr)
    
    print(f"Epoch {epoch}/{args.n_epochs_fr} | train_ae: {loss_train_ae:.4f} | train_fr: {loss_train_fr:.4f} | train_stft: {loss_train_stft:.4f} | total_train_loss: {loss_train:.4f}")
    print(f"Epoch {epoch}/{args.n_epochs_fr} | val_ae: {loss_val_ae:.4f} | val_fr: {loss_val_fr:.4f} | val_stft: {loss_val_stft:.4f} | total_val_loss: {loss_val:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # basic parameters
    parser.add_argument('--output_dir', type=str, default='./checkpoint/experiment_name', help='output directory')
    parser.add_argument('--no_cuda', action='store_true', help="avoid using CUDA when available")
    
    # dataset parameters
    parser.add_argument('--batch_size', type=int, default=64, help='batch size used during training')
    parser.add_argument('--signal_dim', type=int, default=1532, help='dimensionof the input signal')
    parser.add_argument('--fr_size', type=int, default=1000, help='size of the frequency representation')
    parser.add_argument('--max_n_freq', type=int, default=30,
                        help='for each signal the number of frequencies is uniformly drawn between 1 and max_n_freq')
    # frequency-representation (fr) module parameters
    parser.add_argument('--fr_module_type', type=str, default='fr', help='type of the fr module: [fr | psnet]')
    parser.add_argument('--fr_n_layers', type=int, default=10, help='number of convolutional layers in the fr module')
    parser.add_argument('--fr_n_filters', type=int, default=8, help='number of filters per layer in the fr module')
    parser.add_argument('--fr_kernel_size', type=int, default=3,
                        help='filter size in the convolutional blocks of the fr module')
    parser.add_argument('--fr_kernel_out', type=int, default=25, help='size of the conv transpose kernel')
    parser.add_argument('--fr_inner_dim', type=int, default=200, help='dimension after first linear transformation')
    parser.add_argument('--fr_upsampling', type=int, default=5,
                        help='stride of the transposed convolution, upsampling * inner_dim = fr_size')
    
    # training parameters
    parser.add_argument('--n_training', type=int, default=200000, help='# of training data')
    parser.add_argument('--n_validation', type=int, default=2000, help='# of validation data')
    parser.add_argument('--lr_fr', type=float, default=0.0001,
                        help='initial learning rate for adam optimizer used for the frequency-representation module')
    parser.add_argument('--n_epochs_fr', type=int, default=200, help='number of epochs used to train the fr module')
    parser.add_argument('--n_epochs_fc', type=int, default=0, help='number of epochs used to train the fc module')
    parser.add_argument('--save_epoch_freq', type=int, default=10,
                        help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--numpy_seed', type=int, default=100)
    parser.add_argument('--torch_seed', type=int, default=76)

    args = parser.parse_args()

    if torch.cuda.is_available() and not args.no_cuda:
        args.use_cuda = True
    else:
        args.use_cuda = False

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    file_handler = logging.FileHandler(filename=os.path.join(args.output_dir, 'run.log'))
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=handlers
    )

    tb_writer = SummaryWriter(args.output_dir)
    util.print_args(logger, args)

    np.random.seed(args.numpy_seed)
    torch.manual_seed(args.torch_seed)

    train_loader = dataset.make_train_data(args.n_training, args.batch_size)
    val_loader = dataset.make_eval_data(args.n_validation, args.batch_size)

    fr_module = modules.set_fr_module(args)

    fr_optimizer, fr_scheduler = util.set_optim(args, fr_module, 'fr')

    start_epoch = 1

    logger.info('[Network] Number of parameters in the frequency-representation module : %.3f M' % (
                util.model_parameters(fr_module) / 1e6))

    fr_criterion = torch.nn.MSELoss(reduction='sum')
    reshape_criterion = torch.nn.MSELoss(reduction='sum')
    ae_criterion = torch.nn.MSELoss(reduction='sum')


    for epoch in range(start_epoch, args.n_epochs_fc + args.n_epochs_fr + 1):
        train_frequency_representation(args=args, fr_module=fr_module, fr_optimizer=fr_optimizer, fr_criterion=fr_criterion,
                                       fr_scheduler=fr_scheduler, train_loader=train_loader, val_loader=val_loader, epoch=epoch, tb_writer=tb_writer)

        if epoch % args.save_epoch_freq == 0 or epoch == args.n_epochs_fr:
            util.save(fr_module, fr_optimizer, fr_scheduler, args, epoch, args.fr_module_type)
