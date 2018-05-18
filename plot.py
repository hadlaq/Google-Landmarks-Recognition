import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from logging_utils import *
sns.set()
sns.set_context("paper", font_scale=1.5)


model_path = 'model=vgg16_optimizer=adam_lr=0.002_reg=5e-05_batch_size=5_epochs=2_dropout=0.0/1'

train_loss = reader('logs/'+model_path+'/train_loss_hist')
train_acc = reader('logs/'+model_path+'/train_acc_hist')
dev_loss = reader('logs/'+model_path+'/dev_loss_hist')
dev_acc = reader('logs/'+model_path+'/dev_acc_hist')
epochs = range(1, len(train_loss)+1)
# plot loss
plt.figure()
plt.plot(epochs, train_loss, label='train')
plt.plot(epochs, dev_loss, label='dev')
plt.title('Loss history')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('logs/'+model_path+'/loss.pdf')

# plot acc
plt.figure()
plt.plot(epochs, train_acc, label='train')
plt.plot(epochs, dev_acc, label='dev')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Clasification accuracy')
plt.legend()
plt.savefig('logs/'+model_path+'/acc.pdf')
