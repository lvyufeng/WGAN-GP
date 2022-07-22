import os
import sys
import argparse
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore import ms_function, context
from tqdm import tqdm
from src.grad import value_and_grad, grad
from src.layers import Dense, Conv2dTranspose, Conv2d
from src.img_utils import to_image
from src.dataset import create_dataset

# context.set_context(mode=context.PYNATIVE_MODE)


os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=64, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)
latent_dim = opt.latent_dim
n_critic = opt.n_critic

class Generator(nn.Cell):
    def __init__(self):
        super(Generator, self).__init__()

        preprocess = nn.SequentialCell(
            Dense(128, 4 * 4 * 4 * latent_dim),
            nn.ReLU(),
        )
        block1 = nn.SequentialCell(
            Conv2dTranspose(4 * latent_dim, 2 * latent_dim, 5, pad_mode='valid'),
            nn.ReLU(),
        )
        block2 = nn.SequentialCell(
            Conv2dTranspose(2 * latent_dim, latent_dim, 5, pad_mode='valid'),
            nn.ReLU(),
        )
        deconv_out = Conv2dTranspose(latent_dim, 1, 8, stride=2, pad_mode='valid')

        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()

    def construct(self, input):
        # print(input.shape)
        output = self.preprocess(input)
        # print(output.shape)
        output = output.view(-1, 4 * latent_dim, 4, 4)
        #print output.size()
        output = self.block1(output)
        #print output.size()
        output = output[:, :, :7, :7]
        #print output.size()
        output = self.block2(output)
        #print output.size()
        output = self.deconv_out(output)
        output = self.sigmoid(output)
        #print output.size()
        # print(output.shape)
        return output.view(-1, *img_shape)

class Discriminator(nn.Cell):
    def __init__(self):
        super(Discriminator, self).__init__()

        main = nn.SequentialCell(
            Conv2d(1, latent_dim, 5, stride=2, padding=2, pad_mode='pad'),
            # nn.Linear(OUTPUT_DIM, 4*4*4*DIM),
            nn.ReLU(),
            Conv2d(latent_dim, 2 * latent_dim, 5, stride=2, padding=2, pad_mode='pad'),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            nn.ReLU(),
            Conv2d(2 * latent_dim, 4 * latent_dim, 5, stride=2, padding=2, pad_mode='pad'),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            nn.ReLU(),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            # nn.LeakyReLU(True),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            # nn.LeakyReLU(True),
        )
        self.main = main
        self.output = Dense(4 * 4 * 4 * latent_dim, 1)

    def construct(self, input):
        input = input.view(-1, 1, 28, 28)
        out = self.main(input)
        out = out.view(-1, 4 * 4 * 4 * latent_dim)
        out = self.output(out)
        return out.view(-1)


# Loss weight for gradient penalty
lambda_gp = 10

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()
generator.update_parameters_name('generator')
discriminator.update_parameters_name('discriminator')
generator.set_train()
discriminator.set_train()

# Optimizers
optimizer_G = nn.Adam(generator.trainable_params(), learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2)
optimizer_D = nn.Adam(discriminator.trainable_params(), learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2)
optimizer_G.update_parameters_name('optim_g')
optimizer_D.update_parameters_name('optim_d')

def compute_gradient_penalty(real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = ops.StandardNormal()((real_samples.shape[0], 1, 1, 1))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples))
    
    grad_fn = grad(discriminator)
    # Get gradient w.r.t. interpolates
    (gradients,) = grad_fn(interpolates)

    gradients = gradients.view(gradients.shape[0], -1)
    gradient_penalty = ((mnp.norm(gradients, 2, axis=1) - 1) ** 2).mean()
    return gradient_penalty

def discriminator_forward(real_imgs):
    # Sample noise as generator input
    z = ops.StandardNormal()((real_imgs.shape[0], 128))

    # Generate a batch of images
    fake_imgs = generator(z)

    # Real images
    real_validity = discriminator(real_imgs)
    # Fake images
    fake_validity = discriminator(fake_imgs)
    # Gradient penalty
    gradient_penalty = compute_gradient_penalty(real_imgs, fake_imgs)
    # Adversarial loss
    d_loss = -ops.reduce_mean(real_validity) + ops.reduce_mean(fake_validity) + lambda_gp * gradient_penalty
    
    return d_loss, z

def generator_forward(z):
    # Generate a batch of images
    fake_imgs = generator(z)
    # Loss measures generator's ability to fool the discriminator
    # Train on fake images
    fake_validity = discriminator(fake_imgs)
    g_loss = -ops.reduce_mean(fake_validity)

    return g_loss, fake_imgs

grad_generator_fn = value_and_grad(generator_forward,
                                   optimizer_G.parameters,
                                   has_aux=True)
grad_discriminator_fn = value_and_grad(discriminator_forward,
                                       optimizer_D.parameters,
                                       has_aux=True)

@ms_function
def train_step_d(imgs):
    (d_loss, (z,)), d_grads = grad_discriminator_fn(imgs)
    optimizer_D(d_grads)
    return d_loss, z

@ms_function
def train_step_g(z):
    (g_loss, (fake_imgs,)), g_grads = grad_generator_fn(z)
    optimizer_G(g_grads)

    return g_loss, fake_imgs

dataset = create_dataset('./dataset', 'train', opt.img_size, opt.batch_size, num_parallel_workers=opt.n_cpu)
dataset_size = dataset.get_dataset_size()

batches_done = 0

for epoch in range(opt.n_epochs):
    t = tqdm(total=dataset_size)
    t.set_description('Epoch %i' % epoch)
    for i, (imgs, _) in enumerate(dataset.create_tuple_iterator()):
        d_loss, z = train_step_d(imgs)
        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:
            g_loss, fake_imgs = train_step_g(z)
            if batches_done % opt.sample_interval == 0:
                to_image(fake_imgs[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
            batches_done += opt.n_critic
        t.set_postfix(g_loss=g_loss, d_loss=d_loss)
        t.update(1)
