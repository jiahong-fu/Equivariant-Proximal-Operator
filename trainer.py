import os
import math
from decimal import Decimal

import utility

import torch
import torch.nn.utils as utils
from tqdm import tqdm

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            loss = self.loss(sr, hr)
#            print(torch.mean(hr))
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
#        print(self.ckp.dir)
        torch.set_grad_enabled(False)
#        with torch.no_grad():

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.ckp.add_ssim(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
#        print(len(self.loader_test))
        self.model.eval()

        timer_test = utility.timer()
#        if self.args.save_results: self.ckp.begin_background()
        with torch.no_grad():
            for idx_data, d in enumerate(self.loader_test):
                for idx_scale, scale in enumerate(self.scale):
                    d.dataset.set_scale(idx_scale)
                    psnr_all = []
                    ssim_all = []
                    for lr, hr, filename in tqdm(d, ncols=80):
                        lr, hr = self.prepare(lr, hr)
    #                    print(lr.shape)
                        sr = self.model(lr, idx_scale)
                        sr = utility.quantize(sr, self.args.rgb_range)
    #                    print(sr.shape)
    #                    print(torch.mean(hr))
                        save_list = [sr]
                        temp_psnr = utility.calc_psnr(sr, hr, scale, self.args.rgb_range, dataset=d)
                        self.ckp.log[-1, idx_data, idx_scale] += temp_psnr
                        psnr_all.append(temp_psnr)
                        if self.args.ssim:
                            temp_ssim = utility.calc_ssim(sr, hr, scale, self.args.rgb_range, dataset=d)
                            self.ckp.ssim[-1, idx_data, idx_scale] += temp_ssim
                            ssim_all.append(temp_ssim)
                        if self.args.save_gt:
                            save_list.extend([lr, hr])

                        if self.args.save_results:
                            self.ckp.save_results(d, filename[0], save_list, scale)
                    
    #                self.ckp.write_log('\n')
                    self.ckp.save_psnr(d, scale, psnr_all, ssim_all)
                    self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                    self.ckp.ssim[-1, idx_data, idx_scale] /= len(d)
                    best = self.ckp.log.max(0)
                    self.ckp.write_log(
                        '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                            d.dataset.name,
                            scale,
                            self.ckp.log[-1, idx_data, idx_scale],
                            best[0][idx_data, idx_scale],
                            best[1][idx_data, idx_scale] + 1
                        )
                    )
                    if self.args.ssim: 
                        self.ckp.write_log(
                            '[{} x{}]\tSSIM: {:.4f} (Best: {:.4f} @epoch {})'.format(
                                d.dataset.name,
                                scale,
                                self.ckp.ssim[-1, idx_data, idx_scale],
                                self.ckp.ssim[best[1][idx_data, idx_scale] , idx_data, idx_scale],
                                best[1][idx_data, idx_scale] + 1
                            )
                        )                    
#                self.ckp.write_log('\n')

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

#        if self.args.save_results:
#            self.ckp.end_background()

        if not self.args.test_only:
            self.model.train() #19号加的这一行
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda:'+ self.args.device)
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs

