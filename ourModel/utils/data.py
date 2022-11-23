import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader 
class Mydataset(Dataset):
    def __init__(self,dataset, batch_size, x, y):
        self.train = dataset[0]
        self.val = dataset[1]
        self.test  = dataset[2]
        self.batch_size = batch_size
        self.x = x
        self.y = y

    def __getitem__(self,index):
        i = index // self.batch_size
        j = index  % self.batch_size
        k = int(self.batch_size / (self.x + self.y))
        if j >= k * self.x:
            return self.test[(i * k * self.y + (j - k * self.x)) % len(self.test)]
        else:
            return self.train[i * k * self.x + j]
    def __len__(self):
        return len(self.train)* (self.x + self.y)

# if __name__=="__main__":
#     a = torch.tensor([[9,10,11,12,13,14], [9,10,11,12,13,14], [1,2,3,4]])
#     c = Mydataset(a, 4, 1, 3)
#     ids  = DataLoader(dataset=c,batch_size=4, shuffle=False)
#     for d in ids:
#         print(d)
        

    


