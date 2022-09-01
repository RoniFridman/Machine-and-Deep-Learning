import sys
import torch
import torchvision.datasets as dsets
import torchvision.transforms as T
from utils import create_train_loader, train_cont, train_disc
from VariationalAutoencoder import VariationalAutoencoder
from DiscreteVAE import DiscreteVAE
import pandas as pd

''' Command line:
$ python main.py home/student/HW3
'''
PATH_TO_DATA_ROOT = sys.argv[1]  # 'home/student/HW3'

# Constants:
BATCH_SIZE = 32
HEIGHT, WIDTH = 64, 64
LATENT_DIMS = 10
CATAGORICAL_DIM = 40
NUM_EPOCHS = 30
LEARNING_RATE_C = 0.0001
LEARNING_RATE_D = 0.0001
annel_rate = 0.003
temp_min = 0.5

# Transform:
tr = T.Compose([T.Resize(size=(HEIGHT, WIDTH)),
                T.ToTensor(),
                ])

train_dataset = dsets.CelebA(root=PATH_TO_DATA_ROOT,
                             split='valid',
                             transform=tr,
                             download=False)

train_loader = create_train_loader(sys.argv, train_dataset, BATCH_SIZE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# train model

model_c = VariationalAutoencoder(latent_dims=LATENT_DIMS, size=HEIGHT).to(device)
optimizer_c = torch.optim.Adam(model_c.parameters(), lr=LEARNING_RATE_C)
loss_c = train_cont(model_c, train_loader, optimizer_c, epochs=NUM_EPOCHS)
loss_c_df = pd.DataFrame(loss_c, columns=['accumulative BCE', 'accumulative KLD', 'Epoch'])
loss_c_df.to_csv('loss_continues.csv')

torch.save(model_c, 'model.pkl')

model_d = DiscreteVAE(LATENT_DIMS, CATAGORICAL_DIM, HEIGHT).to(device)
optimizer_d = torch.optim.Adam(model_d.parameters(), lr=LEARNING_RATE_D)
loss_d = train_disc(model_d, train_loader, optimizer_d, annel_rate, temp_min, epochs=NUM_EPOCHS, temp=1.0, hard=False)

model_c.discrete_vae = model_d
loss_d_df = pd.DataFrame(loss_d, columns=['accumulative BCE', 'accumulative KLD', 'Epoch'])
loss_d_df.to_csv('loss_discrete.csv')


torch.save(model_c, 'model.pkl')

if __name__ == "__main__":
    pass
