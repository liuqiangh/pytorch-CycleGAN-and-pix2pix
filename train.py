"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import numpy as np

def validation(opt,dataset,model):
    if opt.eval:
        model.eval()
    val_loss=[]
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        val_loss.append(model.val_criterion())

    mean_loss=np.mean(val_loss)

    if least_loss:
        if mean_loss < min(least_loss):
            print("Saving model with new least loss: ", mean_loss)
            model.save_networks('best_' + str(epoch))
            least_loss.append(mean_loss)
    else:
        least_loss.append(mean_loss)
        model.save_networks('best_' + str(epoch))
        

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    # create a dataset given opt.dataset_mode and other options
    dataset = create_dataset(opt)
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training data = %d' % dataset_size)

    #validation
    opt_val = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt_val.num_threads = 0   # test code only supports num_threads = 1
    opt_val.batch_size = 1    # test code only supports batch_size = 1
    opt_val.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt_val.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt_val.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset_val = create_dataset(opt_val)  # create a dataset given opt.dataset_mode and other options
    least_loss = []

    # create a model given opt.model and other options
    model = create_model(opt)
    # regular setup: load and print networks; create schedulers
    model.setup(opt)
    # create a visualizer that display/save images and plots
    visualizer = Visualizer(opt)
    total_iters = 0                # the total number of training iterations

    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        # the number of training iterations in current epoch, reset to 0 every epoch
        epoch_iter = 0

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            # unpack data from dataset and apply preprocessing
            model.set_input(data)
            # calculate loss functions, get gradients, update network weights
            model.optimize_parameters()

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                # visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(
                    epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(
                        epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' %
                      (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        # update learning rates at the end of every epoch.

        validation(opt_val,dataset_val,model)

        model.update_learning_rate()
