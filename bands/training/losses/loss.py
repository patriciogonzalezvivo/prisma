import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
# from icecream import ic
from .high_receptive_pl import HRFPL
import os

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G_encoder, G_mapping, G_synthesis, D, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2):
        super().__init__()
        self.device = device
        self.G_encoder = G_encoder
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        self.run_hrfpl = HRFPL(weight=5, weights_path=os.getcwd())

    def run_G(self, r_img, mask, c, sync):
        with misc.ddp_sync(self.G_encoder, sync):
            x_global, z, feats = self.G_encoder(r_img, c)
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(x_global, mask, feats, ws)
        return img, ws

    def run_D(self, img, c, sync):
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
        return logits
    

    def accumulate_gradients(self, phase, erased_img, real_img, mask, real_c, gen_c, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)
        
        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                g_inputs = torch.cat([0.5 - mask, erased_img], dim=1)
                gen_img, _ = self.run_G(g_inputs, mask, gen_c, sync=sync) # May get synced by Gpl.
                gen_img = gen_img * mask + real_img * (1 - mask)
                loss_rec = 10 * torch.nn.functional.l1_loss(gen_img, real_img)
                loss_pl = self.run_hrfpl(gen_img, real_img)
                
                if self.augment_pipe is not None:
                    gen_img = self.augment_pipe(gen_img)
                d_inputs = torch.cat([0.5 - mask, gen_img], dim=1)
                gen_logits = self.run_D(d_inputs, gen_c, sync=False)
                
                loss_G =  torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                loss_Gmain = loss_G.mean() + loss_rec + loss_pl
                training_stats.report('Loss/G/loss', loss_G)
                training_stats.report('Loss/G/rec_loss', loss_rec)
                training_stats.report('Loss/G/main_loss', loss_Gmain)
                training_stats.report('Loss/G/pl_loss', loss_pl)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                g_inputs = torch.cat([0.5 - mask, erased_img], dim=1)
                gen_img, _ = self.run_G(g_inputs, mask, gen_c, sync=sync) # May get synced by Gpl.
                gen_img = gen_img * mask + real_img * (1 - mask)
                if self.augment_pipe is not None:
                    gen_img = self.augment_pipe(gen_img)
                d_inputs = torch.cat([0.5 - mask, gen_img], dim=1)

                gen_logits = self.run_D(d_inputs, gen_c, sync=False) # Gets synced by loss_Dreal.
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))

            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                if self.augment_pipe is not None:
                    real_img_tmp = self.augment_pipe(real_img_tmp)
                d_inputs = torch.cat([0.5 - mask, real_img_tmp], dim=1)
                real_logits = self.run_D(d_inputs, real_c, sync=sync)

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)
                
                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
