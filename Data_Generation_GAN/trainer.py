import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from datasets.dataset_Hackathon import Hackathon_dataset

def trainer_synapse(args, gan_model, disc_model, snapshot_path):
    
    with open('lists/lists_Hackathon/training_image.txt', 'r') as f:
        image_paths = [line.strip() for line in f]


    with open('lists/lists_Hackathon/training_label.txt', 'r') as f:
        labels = [int(line.strip()) for line in f]
    dataset = Hackathon_dataset(image_paths, labels)
    batch_size = args.batch-size
    num_workers = 4
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    epoch = args.epochs
    criterion = nn.BCELoss()  
    optimizer_gan = optim.Adam(gan_model.parameters(), lr=args.learning_rate_gan)
    optimizer_disc = optim.Adam(disc_model.parameters(), lr=args.learning_rate_disc)
    for epoch in range(args.epochs):
        # Training GAN model and Discriminator model
        for batch_images, batch_labels in tqdm(dataloader, total=len(dataloader)):
            batch_images, batch_labels = batch_images.cuda(), batch_labels.cuda()

            # Train the discriminator
            real_labels = torch.ones(batch_size, 1).cuda()
            fake_labels = torch.zeros(batch_size, 1).cuda()

            disc_model.zero_grad()

            real_output = disc_model(batch_images).view(-1)

            loss_disc_real = criterion(real_output, real_labels)

            noise = torch.randn(batch_size, args.noise_dim).cuda()
            generated_images = gan_model(noise)

            fake_output = disc_model(generated_images.detach()).view(-1)

            loss_disc_fake = criterion(fake_output, fake_labels)
            loss_discriminator = (loss_disc_real + loss_disc_fake) / 2.0

            loss_discriminator.backward()
            optimizer_disc.step()

            # Train the generator (GAN model)
            gan_model.zero_grad()
            fake_output_for_gan = disc_model(generated_images).view(-1)
            loss_gan = criterion(fake_output_for_gan, real_labels)

            loss_gan.backward()
            optimizer_gan.step()
            if batch_idx % args.print_interval == 0:
                print(f"Epoch [{epoch}/{args.epochs}] Batch [{batch_idx}/{len(dataloader)}] "
                      f"Discriminator Loss: {loss_discriminator.item():.4f} "
                      f"Generator Loss: {loss_gan.item():.4f}")

    # Save the models at the end of training if needed
    torch.save(gan_model.state_dict(), os.path.join(snapshot_path, 'gan_model.pth'))
    torch.save(disc_model.state_dict(), os.path.join(snapshot_path, 'discriminator_model.pth'))

    return "Training Finished!"
        