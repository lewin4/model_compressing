from src import train_resnet, train_unet, train_hrnet, evaluate_resnet

# import torch
# test.main()

# import torch as t
# from torch import nn
# from torch.autograd import Variable as V
#
#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         # 等价与self.register_parameter('param1' ,nn.Parameter(t.randn(3, 3)))
#         self.param1 = nn.Parameter(t.rand(3, 3))
#         self.submodel1 = nn.Linear(3, 4)
#
#     def forward(self, input):
#         x = self.param1.mm(input)
#         x = self.submodel11(x)
#         return x
#
#
# net = Net()

if __name__ == "__main__":
    evaluate_resnet.main()