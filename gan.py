import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Transformasi data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download dataset
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Buat DataLoader
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

# Tampilkan sample batch
data_iter = iter(train_loader)
images, labels = next(data_iter)

print("Images shape:", images.shape)  # [32, 1, 28, 28]
print("Labels:", labels[:10])  # 10 pertama labels



from __future__ import print_function
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm_notebook as tqdm
import os





# Alternative way to access Dataset and DataLoader
Dataset = torch.utils.data.Dataset
DataLoader = torch.utils.data.DataLoader

# Configuration
batch_size = 16
image_size = 32
random_transforms = [transforms.ColorJitter(), transforms.RandomRotation(degrees=20)]

# Transform pipeline for 64x64 images
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply(random_transforms, p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


imgs, label = next(iter(train_loader))
imgs = imgs.numpy().transpose(0, 2, 3, 1)

for i in range(5):
    plt.imshow(imgs[i])
    plt.show()







def weights_init(m):
    """
    Takes as input a neural network m that will initialize all its weights.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)






# Data loading with resize
transform = transforms.Compose([
    transforms.Resize(32),  # Resize to 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=64, shuffle=True
)

# Discriminator for 32x32 images
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is 1 x 32 x 32
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),  # Output: 64 x 16 x 16
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # Output: 128 x 8 x 8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # Output: 256 x 4 x 4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 1, 4, 1, 0, bias=False),  # Output: 1 x 1 x 1
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

# Generator for 32x32 output
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Initialize networks
netG = Generator()
netD = Discriminator()





EPOCH = 20 # play with me
LR = 0.0002
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(0.5, 0.999))






dataset = train_dataset
dataloader = dataloader
os.makedirs("./results", exist_ok=True)
for epoch in range(EPOCH):
    for i, data in enumerate(dataloader, 0):
        # 1st Step: Updating the weights of the neural network of the discriminator
        netD.zero_grad()

        # Training the discriminator with a real image of the dataset
        real,_ = data
        input = Variable(real)
        target = Variable(torch.ones(input.size()[0]))
        output = netD(input)
        errD_real = criterion(output, target)

        # Training the discriminator with a fake image generated by the generator
        noise = Variable(torch.randn(input.size()[0], 100, 1, 1))
        fake = netG(noise)
        target = Variable(torch.zeros(input.size()[0]))
        output = netD(fake.detach())
        errD_fake = criterion(output, target)

        # Backpropagating the total error
        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()

        # 2nd Step: Updating the weights of the neural network of the generator
        netG.zero_grad()
        target = Variable(torch.ones(input.size()[0]))
        output = netD(fake)
        errG = criterion(output, target)
        errG.backward()
        optimizerG.step()

        # 3rd Step: Printing the losses and saving the real images and the generated images of the minibatch every 100 steps
        print('[%d/%d][%d/%d] Loss_D: %.4f; Loss_G: %.4f' % (epoch, EPOCH, i, len(dataloader), errD.item(), errG.item()))
        if i % 100 == 0:
            vutils.save_image(real, '%s/real_samples.png' % "./results", normalize=True)
            fake = netG(noise)
            vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results", epoch), normalize=True)







nz = 100 # Change from 100 to 128

def show_generated_img(n_images=5):
    """Generate and display images using the trained GAN generator"""
    sample = []

    # Generate images in eval mode for better results
    netG.eval()

    with torch.no_grad():  # Disable gradients for inference
        for _ in range(n_images):
            noise = torch.randn(1, nz, 1, 1, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # Now uses nz=128
            gen_image = netG(noise).to("cpu").clone().detach().squeeze(0)
            gen_image = gen_image.numpy().transpose(1, 2, 0)

            # Denormalize from [-1, 1] to [0, 1] for proper display
            gen_image = (gen_image + 1) / 2
            gen_image = np.clip(gen_image, 0, 1)

            sample.append(gen_image)

    # Set generator back to training mode
    netG.train()

    # Create the plot with reasonable figure size
    figure, axes = plt.subplots(1, len(sample), figsize=(n_images * 3, 3))

    # Handle case where n_images = 1 (axes won't be an array)
    if n_images == 1:
        axes = [axes]

    for index, axis in enumerate(axes):
        axis.axis('off')
        image_array = sample[index]
        axis.imshow(image_array)
        axis.set_title(f'Generated {index + 1}')

    plt.tight_layout()
    plt.show()
    plt.close()

# Alternative Solution 2: Check what your generator actually expects
def check_generator_input():
    """Check what input size your generator expects"""
    netG.eval()

    # Try different noise sizes to find the right one
    test_sizes = 512

    for size in test_sizes:
        try:
            test_noise = torch.randn(1, size, 1, 1, device=device)
            with torch.no_grad():
                output = netG(test_noise)
            print(f"✓ Generator accepts noise size: {size}")
            print(f"  Output shape: {output.shape}")
            return size
        except Exception as e:
            print(f"✗ Generator does NOT accept noise size: {size}")
            print(f"  Error: {e}")

    netG.train()
    return None

# Alternative Solution 3: Modify noise generation to match exactly what generator expects
def show_generated_img_auto_detect(n_images=5):
    """Automatically detect the right noise size and generate images"""

    # Common noise sizes used in GANs
    possible_sizes = 512
    working_nz = None

    netG.eval()

    # Find the correct noise size
    for size in possible_sizes:
        try:
            test_noise = torch.randn(1, size, 1, 1, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            with torch.no_grad():
                _ = netG(test_noise)
            working_nz = size
            print(f"Found working noise size: {working_nz}")
            break
        except:
            continue

    if working_nz is None:
        print("Could not find a working noise size. Please check your generator architecture.")
        netG.train()
        return

    # Generate images with the correct noise size
    sample = []
    with torch.no_grad():
        for _ in range(n_images):
            noise = torch.randn(1, working_nz, 1, 1, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            gen_image = netG(noise).to("cpu").clone().detach().squeeze(0)
            gen_image = gen_image.numpy().transpose(1, 2, 0)

            # Denormalize from [-1, 1] to [0, 1] for proper display
            gen_image = (gen_image + 1) / 2
            gen_image = np.clip(gen_image, 0, 1)

            sample.append(gen_image)

    netG.train()

    # Display the images
    figure, axes = plt.subplots(1, len(sample), figsize=(n_images * 3, 3))

    if n_images == 1:
        axes = [axes]

    for index, axis in enumerate(axes):
        axis.axis('off')
        image_array = sample[index]
        axis.imshow(image_array)
        axis.set_title(f'Generated {index + 1}')

    plt.tight_layout()
    plt.show()
    plt.close()






show_generated_img(7)
