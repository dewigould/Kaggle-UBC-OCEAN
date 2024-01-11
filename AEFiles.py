class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    
def train_autoencoder(train_loader,valid_loader, model, criterion, optimizer, reg_strength=1e-5,num_epochs=10):
    model.train()
    valid_losses_list = []
    training_loss_list = []
    for epoch in range(num_epochs):
        bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for steps,data in bar:
            inputs, _ = data['image'].to(CONFIG['device'],dtype=torch.float32), data['label'].to(CONFIG['device'])
            optimizer.zero_grad()
            outputs = model(inputs)
            #loss = criterion(outputs,inputs)
            reconstruction_loss = criterion(outputs, inputs)
            
            # L2 regularization
            reg_loss = 0.0
            for param in model.parameters():
                reg_loss += torch.norm(param, p=2)

            loss = reconstruction_loss + reg_strength * reg_loss            
            

            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
        print(f'Training - Epoch [{epoch + 1}/{num_epochs}], Reconstruction Loss: {reconstruction_loss.item()}, Regularization Loss: {reg_strength * reg_loss.item()}')
        training_loss_list.append(loss.item())
    
        # Validation
        model.eval()
        total_valid_loss = 0.0
        losses_within_epoch = []
        with torch.no_grad():
            for data in valid_loader:
                inputs, _ = data['image'].to(CONFIG['device'], dtype=torch.float32), data['label'].to(CONFIG['device'])
                outputs = model(inputs)
                valid_loss = criterion(outputs, inputs)
                total_valid_loss += valid_loss.item()
                losses_within_epoch.append(valid_loss.item())
        
        average_valid_loss = total_valid_loss / len(valid_loader)
        print(f'Validation - Epoch [{epoch + 1}/{num_epochs}], Loss: {average_valid_loss}')
        valid_losses_list.append(average_valid_loss)
    print(f'Validation losses for each epoch: {valid_losses_list}')
    print(f'Training losses for each epoch: {training_loss_list}')

    torch.save(model.state_dict(), 'autoencoder.pth')
    
    
    
    
    
def evaluate_autoencoder(test_loader, model, origin_df, batchsize,threshold=0.075):
    model.eval()
    anoms = []
    image_ids = []
    probs = []
    with torch.no_grad():
        bar = tqdm(enumerate(test_loader), total = len(test_loader))
        for steps,data in bar:
            inputs, _ = data['image'].to(CONFIG['device'],dtype=torch.float32), data['label'].to(CONFIG['device'])
            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            batch_start_index = steps * batchsize
            batch_end_index = (steps + 1) * batchsize
            current_batch_ids = origin_df.iloc[batch_start_index:batch_end_index]['image_id'].tolist()
        
            # Check if the loss is above the threshold, indicating an anomaly
            probs.append(loss.item())
            if loss.item() > threshold:
                print("Anomaly detected!")
                #anoms.append(1)
                anoms.extend([1] * len(inputs))
            else:
                #anoms.append(0)
                anoms.extend([0] * len(inputs))
        
            image_ids.extend(current_batch_ids)

    return (anoms, image_ids,probs)


      