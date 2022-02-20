from torch.autograd import Function
import torch


class TarLoss(torch.nn.Module):
    def __init__(self):
        super(TarLoss, self).__init__()
        self.lossTarget = torch.nn.CrossEntropyLoss()

    def forward(self, RT, R):
        # R = R.squeeze()
        # print(RT.size(), R.size())
        lossTarget = self.lossTarget(RT, R.long())
        return lossTarget


if __name__ == "__main__":
    newLoss = TarLoss()
    print(newLoss(1, 2))
