
import torch
from torchvision import datasets, transforms
import torchvision.models as models
import pandas as pd
import torch.nn.functional as F
from torch import optim
from fastai import *
from fastai.vision import *
import matplotlib as plt
from sklearn.model_selection import train_test_split
from random import shuffle
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from skimage import io
from skimage import color


#########################
#read data
df = pd.read_csv('train.csv')


##########################
#split data
import numpy as np

label = [i for i in df['diagnosis']]
alist = []
blist = []
clist = []
dlist = []
elist = []

#find the index of rows containing the three labels
for i in range(len(df)):
    if label[i] == 0:
        alist.append(i)
    elif label[i] == 1:
        blist.append(i)
    elif label[i] == 2:
        clist.append(i)
    elif label[i] == 3:
        blist.append(i)
    elif label[i] == 4:
        clist.append(i)
            
        
df_a = df.loc[alist]
df_b = df.loc[blist]
df_c = df.loc[clist]
df_d = df.loc[dlist]
df_e = df.loc[elist]

#splitting the dataset 
train_a, validate_a, test_a = np.split(df_a.sample(frac=1), [int(.7*len(df_a)), int(.8*len(df_a))])
train_b, validate_b, test_b = np.split(df_b.sample(frac=1), [int(.7*len(df_b)), int(.8*len(df_b))])
train_c, validate_c, test_c = np.split(df_c.sample(frac=1), [int(.7*len(df_c)), int(.8*len(df_c))])
train_d, validate_d, test_d = np.split(df_d.sample(frac=1), [int(.7*len(df_d)), int(.8*len(df_d))])
train_e, validate_e, test_e = np.split(df_e.sample(frac=1), [int(.7*len(df_e)), int(.8*len(df_e))])


#combine the data from three datasets
train=pd.concat([train_a,train_b,train_c,train_d,train_e])
validate=pd.concat([validate_a,validate_b,validate_c,validate_d,validate_e])
test=pd.concat([test_a,test_b,test_c,test_d,test_e])

#shuffle the data after combination
train = train.sample(frac=1)
validate = validate.sample(frac=1)
test = test.sample(frac=1)


#####################################
#convert the single label to multilabel

y_train = pd.get_dummies(train['diagnosis']).values
y_train_multi = np.empty(y_train.shape, dtype=y_train.dtype)
y_train_multi[:, 4] = y_train[:, 4]

for i in range(3, -1, -1):
    y_train_multi[:, i] = np.logical_or(y_train[:, i], y_train_multi[:, i+1])

train['diagnosis'] = y_train_multi


y_val = pd.get_dummies(validate['diagnosis']).values
y_val_multi = np.empty(y_val.shape, dtype=y_val.dtype)
y_val_multi[:, 4] = y_val[:, 4]

for i in range(3, -1, -1):
    y_val_multi[:, i] = np.logical_or(y_val[:, i], y_val_multi[:, i+1])

validate['diagnosis'] = y_val_multi


y_test = pd.get_dummies(test['diagnosis']).values
y_test_multi = np.empty(y_test.shape, dtype=y_test.dtype)
y_test_multi[:, 4] = y_test[:, 4]

for i in range(3, -1, -1):
    y_test_multi[:, i] = np.logical_or(y_test[:, i], y_test_multi[:, i+1])

test['diagnosis'] = y_test_multi




#####################################
#Preprocessing and DataLoader

train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(896),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

validation_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(896),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def crop_image1(img,tol=7):
    # img is image data
    # tol  is tolerance
        
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img


class DenseLoader(Dataset):
   
    def __init__(self, df, root_dir, transform=None):
        self.target = target
        self.data_frame = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):


        
        img_name = os.path.join(self.root_dir,
                                self.data_frame.iloc[idx, 0])
        
        image = io.imread(img_name)

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = crop_image_from_gray(img)
        img = cv2.resize(img, (224,224))
        image = cv2.addWeighted(img,4,cv2.GaussianBlur(img, (0,0), desired_size/30) ,-4 ,128)

        if len(image.shape) > 2 and image.shape[2] == 4:
            image = image[:,:,0]
            
        image=np.repeat(image[None,...],3,axis=0)
            
        image_class = self.data_frame.iloc[idx, -1]

        y = self.target[idx]
        # check for minority class so that it can do data augmentation for minority class
        if (y == 0) and self.transform: 
            image = self.transform(image)

            
        sample = {'x': image, 'y': image_class}

        return sample

root_dir = 'blindness/project/images/'

transformed_dataset = {'train': DenseLoader(train,root_dir), \
                       'validate': DenseLoader(validate,root_dir), \
                       'test': DenseLoader(test,root_dir)}


dataloader = {x: DataLoader(transformed_dataset[x], batch_size=batch_size, shuffle=True, num_workers=0) \
              for x in ['train', 'validate','test']}




#####################################
#Building model
#Customize Densenet Model

densenet = models.densenet161()

class Net(nn.Module):
    def __init__(self , model):
        super(Net, self).__init__()
        self.densenet_layer = nn.Sequential(*list(model)
        self.pool_layer = nn.AdaptiveAvgPool2d((1,1))
        self.regular = nn.Dropout(p=0.5)
        self.Linear_layer = nn.Linear(1000, 5)
        
    def forward(self, x):
        x = self.densenet_layer(x)
        x = self.pool_layer(x)
        x = self.regular(x)
        x = x.view(x.size(0), -1) 
        x = F.sigmoid(self.Linear_layer(x))
        
        return x

dense = Net(densenet)
model = dense.cuda()

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        torch.nn.init.xavier_uniform_(m.weight.data,gain=1.0)
model.apply(weights_init)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)


#############################################
#training model
def train_model(model, dataloader, optimizer, scheduler, loss_fn, num_epochs = 10, verbose = False):
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
    
    return model, acc_dict, loss_dict, best_model_wts

model, acc_dict, loss_dict, best_model_wts = train_model(model=model, dataloader=dataloader, optimizer=optimizer,\
                                         scheduler=True, loss_fn=criterion, num_epochs = 20, verbose = True)







#############################################
#plots
train_loss = loss_dict['train']
validate_loss = loss_dict['validate']
train_acc = acc_dict['train']
validate_acc = acc_dict['validate']
fig, ax = plt.subplots()
line1, = ax.plot(train_loss,label='train loss')
line2, = ax.plot(validate_loss,label='validate loss')
ax.legend()
plt.show()

fig, ax = plt.subplots()
line1, = ax.plot(train_acc,label='train accuracy')
line2, = ax.plot(validate_acc,label='validate accuracy')
ax.legend()
plt.show()

#########
def evaluate_model(model, dataloader):   
    from sklearn.preprocessing import label_binarize
    
    model.eval()
    y_test_label = []
    y_test = np.array([[0,0,0]])
    y_score = np.array([[0,0,0]])
    prediction = []
    for data in dataloader['test']:
        image = data['x'].to(device,dtype=torch.float)
        label = data['y'].to('cpu',dtype=torch.long)
        y_test_label = y_test_label+label.tolist()
        #convert the label to confusion matrix
        label = label_binarize(label, classes=[0, 1, 2])
        y_test = np.concatenate((y_test,label),axis = 0)
        output = model(image)
        output = output.to('cpu')
        #get the prediction matrix
        y_score = np.concatenate((y_score,output.detach().numpy()),axis = 0)
    #convert the matrix to a specific classification
    y_test = y_test[1:]
    y_score = y_score[1:]
    for i in y_score:
        prediction.append(list(i).index(max(i)))
    
    return y_test_label,y_test,prediction,y_score

#ROC curve
def ROC(y_test,y_score):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = y_test.shape[1]
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    lw = 2
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
    return None

#confusion matrix
def confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        pass
        #print('Confusion matrix, without normalization')

    #print(cm)
    #print(cm.shape)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(-0.5, cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return None


y_test_label,y_test,pre,y_score = evaluate_model(model.load_state_dict(best_model_wts), dataloader)
ROC(y_test,y_score)
confusion_matrix_plot(y_test_label, pre, target_names)


