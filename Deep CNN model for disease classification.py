from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from skimage import io
import torch
from skimage import color

#preprocess Chest X Ray datasets
class ChestXrayDataset(Dataset):
    """Chest X-ray dataset from https://nihcc.app.box.com/v/ChestXray-NIHCC."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file filename information.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.data_frame.iloc[idx, 0])

        image = io.imread(img_name,as_gray=True)
        
        image = (image - image.mean()) / image.std()
            
        image_class = self.data_frame.iloc[idx, -1]

        sample = {'x': image[None,:], 'y': image_class}

        if self.transform:
            sample = self.transform(sample)

        return sample
        
        
        
class Conv_model(nn.Module):
    def __init__(self):
        super(Conv_model,self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride = 2, padding = 1)
        self.activation1_1 = nn.ReLU()
        self.pooling1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.res1_1 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, stride = 1, padding = 1)
        self.activation1_2 = nn.ReLU()
        self.res1_2 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, stride = 1, padding = 1)
        self.activation1_3 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)
        self.activation2_1 = nn.ReLU()
        self.pooling2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.res2_1 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.activation2_2 = nn.ReLU()
        self.res2_2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.activation2_3 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 2, padding = 1)
        self.activation3_1 = nn.ReLU()
        self.pooling3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.res3_1 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.activation3_2 = nn.ReLU()
        self.res3_2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.activation3_3 = nn.ReLU()
        
        self.fc1 = nn.Linear(16384,128)
        self.fc2 = nn.Linear(128,3)
        
    def forward(self, x):
        out = self.activation1_1(self.conv1(x))
        out = self.pooling1(out)
        residual = out
        out = self.activation1_2(self.res1_1(out))
        out = self.res1_2(out)
        out = out + residual
        out = self.activation1_3(out)
        
        out = self.activation2_1(self.conv2(out))
        out = self.pooling2(out)
        residual = out
        out = self.activation2_2(self.res2_1(out))
        out = self.res2_2(out)
        out = out + residual
        out = self.activation2_3(out)
        
        out = self.activation3_1(self.conv3(out))
        out = self.pooling3(out)
        residual = out
        out = self.activation3_2(self.res3_1(out))
        out = self.res3_2(out)
        out = out + residual
        out = self.activation3_3(out)
        
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
        
        
from torch import optim
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()



import time
device = torch.device('cuda')

train_df_path = 'HW2_randomTrainSet.csv'
val_df_path = 'HW2_randomValidationSet.csv'
transformed_dataset = {'train': ChestXrayDataset(train_df_path,'./HW2/images/'), \
                       'validate':ChestXrayDataset(val_df_path,'./HW2/images/')}
bs = 10
dataloader = {x: DataLoader(transformed_dataset[x], batch_size=bs, shuffle=True, num_workers=0) \
              for x in ['train', 'validate']}
data_sizes ={x: len(transformed_dataset[x]) for x in ['train', 'validate']}

def train_model(model, dataloader, optimizer, loss_fn, num_epochs = 10, verbose = False, scheduler=None):
    acc_dict = {'train':[],'validate':[]}
    loss_dict = {'train':[],'validate':[]}
    best_acc = 0
    phases = ['train','validate']
    since = time.time()
    for i in range(num_epochs):
        print('Epoch: {}/{}'.format(i, num_epochs-1))
        print('-'*10)
        for p in phases:
            running_correct = 0
            running_loss = 0
            running_total = 0
            if p == 'train':
                model.train()
            else:
                model.eval()
                
            for data in dataloader[p]:
                optimizer.zero_grad()
                image = data['x'].to(device,dtype=torch.float)
                label = data['y'].to(device,dtype=torch.long)
                output = model(image)
                loss = loss_fn(output, label)
                _, preds = torch.max(output, dim = 1)
                num_imgs = image.size()[0]
                running_correct += torch.sum(preds ==label).item()
                running_loss += loss.item()*num_imgs
                running_total += num_imgs
                if p== 'train':
                    loss.backward()
                    optimizer.step()
            epoch_acc = float(running_correct/running_total)
            epoch_loss = float(running_loss/running_total)
            if verbose or (i%10 == 0):
                print('Phase:{}, epoch loss: {:.4f} Acc: {:.4f}'.format(p, epoch_loss, epoch_acc))
            
            acc_dict[p].append(epoch_acc)
            loss_dict[p].append(epoch_loss)
            if p == 'validate':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()
            else:
                if scheduler:
                    scheduler.step()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val acc: {:4f}'.format(best_acc))
    
    model.load_state_dict(best_model_wts)



model = Conv_model().to(device)
# weight initialization
for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        torch.nn.init.xavier_uniform_(m.weight,gain=1.0)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
model, acc_dict, loss_dict = train_model(model=model, dataloader=dataloader, optimizer=optimizer,\
                                         scheduler=None, loss_fn=criterion, num_epochs = 10, verbose = True)
                                         


import matplotlib.pyplot as plt
train_loss = loss_dict['train']
validate_loss = loss_dict['validate']
train_acc = acc_dict['train']
validate_acc = acc_dict['validate']
fig, ax = plt.subplots()
line1, = ax.plot(train_loss,label='train loss')
line2, = ax.plot(validate_loss,label='validate loss')
ax.legend()
plt.show()
