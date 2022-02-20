from torch.autograd import Function
import torch


class Sou1Loss(torch.nn.Module):
    def __init__(self):
        super(Sou1Loss, self).__init__()
        self.lossSou1 = torch.nn.MSELoss(reduce=True, reduction='mean')
        self.lossSou2 = torch.nn.CrossEntropyLoss()

    def forward(self, tar_represent, sou1_represent, RS, R):
        lossSou1 = self.lossSou1(tar_represent, sou1_represent)
        lossSou2 = self.lossSou2(RS, R)
        return lossSou1+lossSou2


if __name__ == "__main__":
    newLoss = Sou1Loss()
    print(newLoss(1, 2))
