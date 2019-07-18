import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from torch.optim import Adam
# from torchvision import datasets, transforms

# from sklearn.metrics import confusion_matrix


# class CocoDet:
#     def __init__(self, batch_size):
#         ds_transform = transforms.Compose([
#                        transforms.Resize((img_size,img_size)),
#                        transforms.Grayscale(),
#                        transforms.ToTensor()
#                        #transforms.Normalize((0.1307,), (0.3081,))
#                    ])
#
#         train_dataset = datasets.CocoDetection(train_path, train_anno_path,
#                                                transform=ds_transform, target_transform = target_t)
#         test_dataset = datasets.CocoDetection(val_path, val_anno_path,
#                                                transform=ds_transform, target_transform = target_t)
#
#
#         self.train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#         self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


#
reduced = 6 ########### beurk change
# reduced = 7
USE_CUDA = True
img_size = 28


class ConvLayer(nn.Module):
    def __init__(self, in_channels=3, out_channels=256, kernel_size=9):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=1
                             )
    def forward(self, x):
        return F.relu(self.conv(x))


class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9):
        super(PrimaryCaps, self).__init__()

        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=0) for _ in range(num_capsules)])
        # self.capsules = nn.ModuleList([
            # nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1) for _ in range(num_capsules)])


    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), 32 * reduced * reduced, -1)
        return self.squash(u)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm *  input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor

#out_channels = 16
class DigitCaps(nn.Module):
    def __init__(self, num_capsules=3, num_routes=32 * reduced * reduced, in_channels=8, out_channels=32):
        super(DigitCaps, self).__init__()

        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules

        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)

        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)

        b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))
        if USE_CUDA:
            b_ij = b_ij.cuda()

        num_iterations = 3
        for iteration in range(num_iterations):
            # print('size b_j: ', b_ij.size())
            c_ij = F.softmax(b_ij)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)

            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

        return v_j.squeeze(1)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True) + 1e-8
        output_tensor = squared_norm *  input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # '''
        # 32 * 3 = nb_classes * nb_channels
        self.reconstruction_layers = nn.Sequential(
            nn.Linear(32 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, img_size*img_size*3),
            nn.Sigmoid()
        )

        # self.out = torch.nn.Tanh()

    def forward(self, x, data):
        # print("size of x before recon:", x.size())
        classes = torch.sqrt((x ** 2).sum(2))
        classes = F.softmax(classes)
        # print("classes size: ", classes.size())

        _, max_length_indices = classes.max(dim=1)
        # masked = Variable(torch.sparse.torch.eye(2))
        masked = Variable(torch.sparse.torch.eye(classes.size(1)))
        if USE_CUDA:
            masked = masked.cuda()
        masked = masked.index_select(dim=0, index=max_length_indices.squeeze(1).data)
        # print("masked size: ", masked.size())
        reconstructions = self.reconstruction_layers((x * masked[:, :, None, None]).view(x.size(0), -1))
        reconstructions = reconstructions.view(-1, 3, img_size, img_size)
        #multiplication by mask = way of conditioning reconstruction ?

        return reconstructions, masked



class CapsNet(nn.Module):
    def __init__(self, img_size = 28, num_filters = 512, num_classes = 3):
        super(CapsNet, self).__init__()
        self.conv_layer = ConvLayer(in_channels=num_filters)
        self.primary_capsules = PrimaryCaps()
        self.digit_capsules = DigitCaps(num_capsules=num_classes)
        self.decoder = Decoder()
        self.mse_loss = nn.MSELoss()
        self._img_size = img_size

    def forward(self, data):
        output = self.digit_capsules(self.primary_capsules(self.conv_layer(data)))
        # output = self.digit_capsules(self.primary_capsules(data))

        reconstructions, masked = self.decoder(output, data)
        return output, reconstructions, masked

    def loss(self, data, x, target, reconstructions):
        # print("margin loss: ", self.margin_loss(x, target))
        # print("recon loss: ", self.reconstruction_loss(data, reconstructions))

        # _, non_zero = target.max(dim=1)
        # non_zero = non_zero.view(-1).cuda()
        # y = torch.ones(non_zero.size()).cuda()
        # y1 = torch.zeros(non_zero.size()).cuda()
        # mask = torch.where(non_zero > 0, y, y1).cuda()
        # pos_nb = mask.sum(dim = 0)
        # # print('pos_nb ', pos_nb)
        # coeff = pos_nb/target.size(0)

        # print(coeff)
        # print('target: ',target)
        # print('non_zero: ', non_zero)
        # print('mask: ', mask)
        # rec_loss2 = self.reconstruction_loss(data, reconstructions)
        # print('rec_loss2 size: ', rec_loss2.size())
        # We don't penalize the network if it can't reconstruct background class

        rec_loss = self.reconstruction_loss(data, reconstructions)# * coeff#mask
        # print('rec_loss: ', rec_loss)

        # print('rec_loss.size ', rec_loss.size())
        # margin_loss = self.margin_loss(x, target)
        # print('margin_loss size: ', margin_loss.size())

        return self.margin_loss(x, target) + rec_loss#.mean() #self.reconstruction_loss(data, reconstructions)

    def margin_loss(self, x, labels, size_average=True):
        batch_size = x.size(0)

        # weights = torch.sum(labels, 0).repeat(batch_size,1) + 1.0#/batch_size
        # weights = weights/batch_size

        # print("size labels: ", labels.size())
        # print('weights: ', weights)
        #norm of x = probability of being each class
        v_c = torch.sqrt((x**2).sum(dim=2, keepdim=True))
        # print("V_C: ", v_c)
        # print(v_c.size())

        #if prediction is correct at proba 0.9 or above
        left = F.relu(0.9 - v_c).view(batch_size, -1)
        # print("left size: ", left.size())
        #if prediction is incorrect w/ proba 0.1 or above
        # right = F.relu(v_c - 0.1).view(batch_size, -1)
        right = F.relu(v_c - 0.1).view(batch_size, -1)

        # loss = left side if capsule corresponds to pred, right ow
        loss = labels * left + 0.5 * (1.0 - labels) * right
        # loss = labels * left * weights + 0.5 * (1.0 - labels) * weights * right # attempt at adding weights
        # print('loss_size: ',loss.size())
        # print('loss: ', loss)
        # print('rec_loss', loss*weights)

        # mean of loss on the batch
        loss = loss.sum(dim=1).mean()
        #print('ml: ', loss)

        return loss

    def reconstruction_loss(self, data, reconstructions):
        loss = self.mse_loss(reconstructions.view(reconstructions.size(0), -1),
                             data.view(reconstructions.size(0), -1))
        #print 'rec: ', loss * 0.0005
        # return loss * 0.0005
        return loss #* 0.5
        # return loss * 0.000005



# capsule_net = CapsNet()

#
#
#
# for epoch in range(n_epochs):
#     capsule_net.train()
#     train_loss = 0
#     print(epoch)
#     for batch_id, (data, target) in enumerate(cocod.train_loader):
#         #print(data.type())
#         target = torch.sparse.torch.eye(2).index_select(dim=0, index=target)
#         data, target = Variable(data.float()), Variable(target.float())
#         #print(data.type())
#         if USE_CUDA:
#             data, target = data.cuda(), target.cuda()
#
#         optimizer.zero_grad()
#         output, reconstructions, masked = capsule_net(data)
#         loss = capsule_net.loss(data, output, target, reconstructions)
#         loss.backward()
#         optimizer.step()
#
#         train_loss += loss.item()#loss.data[0]
#
#         if batch_id % 10 == 0:
#             y_pred = np.argmax(masked.data.cpu().numpy(), 1)
#             y_true = np.argmax(target.data.cpu().numpy(), 1)
#             cm = confusion_matrix(y_true,y_pred)
#             tn, fp, fn, tp = cm.ravel()
#             print(cm)
#
#             print("train accuracy:", sum(np.argmax(masked.data.cpu().numpy(), 1) ==
#                                    np.argmax(target.data.cpu().numpy(), 1)) / float(batch_size))
#             print("train precision:", tp/(tp+fp))
#             print("train recall:", tp/(tp+fn))
#             #print(output.sum())
#
#     print(train_loss / len(cocod.train_loader))
#     #print( np.argmax(masked.data.cpu().numpy()))
#
#     capsule_net.eval()
#     test_loss = 0
#     for batch_id, (data, target) in enumerate(cocod.test_loader):
#
#         target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
#         data, target = Variable(data), Variable(target)
#
#         if USE_CUDA:
#             data, target = data.cuda(), target.cuda()
#
#         output, reconstructions, masked = capsule_net(data)
#         loss = capsule_net.loss(data, output, target, reconstructions)
#
#         test_loss += loss.item()#loss.data[0]
#
#         if batch_id % 10 == 0:
#             print("test accuracy:", sum(np.argmax(masked.data.cpu().numpy(), 1) ==
#                                    np.argmax(target.data.cpu().numpy(), 1)) / float(batch_size))
#
#     print(test_loss / len(mnist.test_loader))
