# train.py

import torch
import torch.nn.functional as F
from tqdm import tqdm
from config import device, train_dataset, batch_size

def train(dataloader, model, optimizer, epochs):

    losses = []
    
    for epoch in tqdm(range(epochs), desc='Epochs'):
        running_loss = 0.0
        batch_progress = tqdm(dataloader, desc='Batches', leave=False)
        
        for iter, (images, labels) in enumerate(batch_progress):
            # Move images to the appropriate device (GPU/CPU)
            images = images.to(device)
            
            tgt = images.clone()
            
            pred = model(images)
            

            loss = F.binary_cross_entropy(pred, tgt)
            
            optimizer.zero_grad()  
            loss.backward()       
            optimizer.step()       
            
            running_loss += loss.item()
            
          
            avg_loss = running_loss * batch_size / len(train_dataset)
            
           
            losses.append(loss.item())
        
       
        tqdm.write(f'----\nEpoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}\n')
    
    return losses

def plot_training_curve(losses):
    
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Iteration')
    plt.ylabel('Binary Cross-Entropy Loss')
    plt.grid(True)
    plt.show()