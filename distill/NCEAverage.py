import torch
from torch.autograd import Function
from torch import nn
from .alias_multinomial import AliasMethod
import math


class NCESoftmax(nn.Module):

    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5):
        super(NCESoftmax, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K

        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_l', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_ab', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, l, ab, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_l = self.params[2].item()
        Z_ab = self.params[3].item()

        momentum = self.params[4].item()
        batchSize = l.size(0)
        outputSize = self.memory_l.size(0)
        inputSize = self.memory_l.size(1)

        # original score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)
        # sample
        weight_l = torch.index_select(self.memory_l, 0, idx.view(-1)).detach()
        weight_l = weight_l.view(batchSize, K + 1, inputSize)
        out_ab = torch.bmm(weight_l, ab.view(batchSize, inputSize, 1))
        # out_ab = torch.exp(torch.div(out_ab, T))
        out_ab = torch.div(out_ab, T)
        # sample
        weight_ab = torch.index_select(self.memory_ab, 0, idx.view(-1)).detach()
        weight_ab = weight_ab.view(batchSize, K + 1, inputSize)
        out_l = torch.bmm(weight_ab, l.view(batchSize, inputSize, 1))
        # out_l = torch.exp(torch.div(out_l, T))
        out_l = torch.div(out_l, T)

        # set Z if haven't been set yet
        if Z_l < 0:
            # self.params[2] = out_l.mean() * outputSize
            self.params[2] = 1
            Z_l = self.params[2].clone().detach().item()
            print("normalization constant Z_l is set to {:.1f}".format(Z_l))
        if Z_ab < 0:
            # self.params[3] = out_ab.mean() * outputSize
            self.params[3] = 1
            Z_ab = self.params[3].clone().detach().item()
            print("normalization constant Z_ab is set to {:.1f}".format(Z_ab))

        # compute out_l, out_ab
        # out_l = torch.div(out_l, Z_l).contiguous()
        # out_ab = torch.div(out_ab, Z_ab).contiguous()
        out_l = out_l.contiguous()
        out_ab = out_ab.contiguous()

        # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_l, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(l, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_l = l_pos.div(l_norm)
            self.memory_l.index_copy_(0, y, updated_l)

            ab_pos = torch.index_select(self.memory_ab, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(ab, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_ab = ab_pos.div(ab_norm)
            self.memory_ab.index_copy_(0, y, updated_ab)

        return out_l, out_ab


class NCEAverage(nn.Module):

    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5):
        super(NCEAverage, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K

        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_l', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_ab', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, l, ab, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_l = self.params[2].item()
        Z_ab = self.params[3].item()

        momentum = self.params[4].item()
        batchSize = l.size(0)
        outputSize = self.memory_l.size(0)
        inputSize = self.memory_l.size(1)

        # original score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)
        # sample
        weight_l = torch.index_select(self.memory_l, 0, idx.view(-1)).detach()
        weight_l = weight_l.view(batchSize, K + 1, inputSize)
        out_ab = torch.bmm(weight_l, ab.view(batchSize, inputSize, 1))
        out_ab = torch.exp(torch.div(out_ab, T))
        # sample
        weight_ab = torch.index_select(self.memory_ab, 0, idx.view(-1)).detach()
        weight_ab = weight_ab.view(batchSize, K + 1, inputSize)
        out_l = torch.bmm(weight_ab, l.view(batchSize, inputSize, 1))
        out_l = torch.exp(torch.div(out_l, T))

        # set Z if haven't been set yet
        if Z_l < 0:
            self.params[2] = out_l.mean() * outputSize
            Z_l = self.params[2].clone().detach().item()
            print("normalization constant Z_l is set to {:.1f}".format(Z_l))
        if Z_ab < 0:
            self.params[3] = out_ab.mean() * outputSize
            Z_ab = self.params[3].clone().detach().item()
            print("normalization constant Z_ab is set to {:.1f}".format(Z_ab))

        # compute out_l, out_ab
        out_l = torch.div(out_l, Z_l).contiguous()
        out_ab = torch.div(out_ab, Z_ab).contiguous()

        # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_l, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(l, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_l = l_pos.div(l_norm)
            self.memory_l.index_copy_(0, y, updated_l)

            ab_pos = torch.index_select(self.memory_ab, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(ab, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_ab = ab_pos.div(ab_norm)
            self.memory_ab.index_copy_(0, y, updated_ab)

        return out_l, out_ab


class NCEAverageWithZ(nn.Module):

    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, z=None):
        super(NCEAverageWithZ, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K

        if z is None or z <= 0:
            z = -1
        else:
            pass
        self.register_buffer('params', torch.tensor([K, T, z, z, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_l', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_ab', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, l, ab, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_l = self.params[2].item()
        Z_ab = self.params[3].item()

        momentum = self.params[4].item()
        batchSize = l.size(0)
        outputSize = self.memory_l.size(0)
        inputSize = self.memory_l.size(1)

        # original score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)
        # sample
        weight_l = torch.index_select(self.memory_l, 0, idx.view(-1)).detach()
        weight_l = weight_l.view(batchSize, K + 1, inputSize)
        out_ab = torch.bmm(weight_l, ab.view(batchSize, inputSize, 1))
        out_ab = torch.exp(torch.div(out_ab, T))
        # sample
        weight_ab = torch.index_select(self.memory_ab, 0, idx.view(-1)).detach()
        weight_ab = weight_ab.view(batchSize, K + 1, inputSize)
        out_l = torch.bmm(weight_ab, l.view(batchSize, inputSize, 1))
        out_l = torch.exp(torch.div(out_l, T))

        # set Z if haven't been set yet
        if Z_l < 0:
            self.params[2] = out_l.mean() * outputSize
            Z_l = self.params[2].clone().detach().item()
            print("normalization constant Z_l is set to {:.1f}".format(Z_l))
        if Z_ab < 0:
            self.params[3] = out_ab.mean() * outputSize
            Z_ab = self.params[3].clone().detach().item()
            print("normalization constant Z_ab is set to {:.1f}".format(Z_ab))

        # compute out_l, out_ab
        out_l = torch.div(out_l, Z_l).contiguous()
        out_ab = torch.div(out_ab, Z_ab).contiguous()

        # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_l, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(l, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_l = l_pos.div(l_norm)
            self.memory_l.index_copy_(0, y, updated_l)

            ab_pos = torch.index_select(self.memory_ab, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(ab, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_ab = ab_pos.div(ab_norm)
            self.memory_ab.index_copy_(0, y, updated_ab)

        return out_l, out_ab


class NCEAverageFull(nn.Module):

    def __init__(self, inputSize, outputSize, T=0.07, momentum=0.5):
        super(NCEAverageFull, self).__init__()
        self.nLem = outputSize

        self.register_buffer('params', torch.tensor([T, -1, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_l', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_ab', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, l, ab, y):
        T = self.params[0].item()
        Z_l = self.params[1].item()
        Z_ab = self.params[2].item()

        momentum = self.params[3].item()
        batchSize = l.size(0)
        outputSize = self.memory_l.size(0)
        inputSize = self.memory_l.size(1)

        # score computation
        idx1 = y.unsqueeze(1).expand(-1, inputSize).unsqueeze(1).expand(-1, 1, -1)
        idx2 = torch.zeros(batchSize).long().cuda()
        idx2 = idx2.unsqueeze(1).expand(-1, inputSize).unsqueeze(1).expand(-1, 1, -1)
        # sample
        weight_l = self.memory_l.clone().detach().unsqueeze(0).expand(batchSize, outputSize, inputSize)
        weight_l_1 = weight_l.gather(dim=1, index=idx1)
        weight_l_2 = weight_l.gather(dim=1, index=idx2)
        weight_l.scatter_(1, idx1, weight_l_2)
        weight_l.scatter_(1, idx2, weight_l_1)
        out_ab = torch.bmm(weight_l, ab.view(batchSize, inputSize, 1))
        out_ab = torch.exp(torch.div(out_ab, T))
        # sample
        weight_ab = self.memory_ab.clone().detach().unsqueeze(0).expand(batchSize, outputSize, inputSize)
        weight_ab_1 = weight_ab.gather(dim=1, index=idx1)
        weight_ab_2 = weight_ab.gather(dim=1, index=idx2)
        weight_ab.scatter_(1, idx1, weight_ab_2)
        weight_ab.scatter_(1, idx2, weight_ab_1)
        out_l = torch.bmm(weight_ab, l.view(batchSize, inputSize, 1))
        out_l = torch.exp(torch.div(out_l, T))

        # set Z if haven't been set yet
        if Z_l < 0:
            self.params[1] = out_l.mean() * outputSize
            Z_l = self.params[1].clone().detach().item()
            print("normalization constant Z_l is set to {:.1f}".format(Z_l))
        if Z_ab < 0:
            self.params[2] = out_ab.mean() * outputSize
            Z_ab = self.params[2].clone().detach().item()
            print("normalization constant Z_ab is set to {:.1f}".format(Z_ab))

        # compute out_l, out_ab
        out_l = torch.div(out_l, Z_l).contiguous()
        out_ab = torch.div(out_ab, Z_ab).contiguous()

        # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_l, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(l, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_l = l_pos.div(l_norm)
            self.memory_l.index_copy_(0, y, updated_l)

            ab_pos = torch.index_select(self.memory_ab, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(ab, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_ab = ab_pos.div(ab_norm)
            self.memory_ab.index_copy_(0, y, updated_ab)

        return out_l, out_ab


class NCEAverageFullSoftmax(nn.Module):

    def __init__(self, inputSize, outputSize, T=1, momentum=0.5):
        super(NCEAverageFullSoftmax, self).__init__()
        self.nLem = outputSize

        self.register_buffer('params', torch.tensor([T, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_l', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_ab', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, l, ab, y):
        T = self.params[0].item()
        momentum = self.params[1].item()
        batchSize = l.size(0)
        outputSize = self.memory_l.size(0)
        inputSize = self.memory_l.size(1)

        # score computation
        # weight_l = self.memory_l.unsqueeze(0).expand(batchSize, outputSize, inputSize).detach()
        weight_l = self.memory_l.clone().unsqueeze(0).expand(batchSize, outputSize, inputSize).detach()
        out_ab = torch.bmm(weight_l, ab.view(batchSize, inputSize, 1))
        out_ab = out_ab.div(T)
        out_ab = out_ab.squeeze().contiguous()

        # weight_ab = self.memory_ab.unsqueeze(0).expand(batchSize, outputSize, inputSize).detach()
        weight_ab = self.memory_ab.clone().unsqueeze(0).expand(batchSize, outputSize, inputSize).detach()
        out_l = torch.bmm(weight_ab, l.view(batchSize, inputSize, 1))
        out_l = out_l.div(T)
        out_l = out_l.squeeze().contiguous()

        # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_l, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(l, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_l = l_pos.div(l_norm)
            self.memory_l.index_copy_(0, y, updated_l)

            ab_pos = torch.index_select(self.memory_ab, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(ab, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_ab = ab_pos.div(ab_norm)
            self.memory_ab.index_copy_(0, y, updated_ab)

        return out_l, out_ab

    def update_memory(self, l, ab, y):
        momentum = self.params[1].item()
        # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_l, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(l, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_l = l_pos.div(l_norm)
            self.memory_l.index_copy_(0, y, updated_l)

            ab_pos = torch.index_select(self.memory_ab, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(ab, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_ab = ab_pos.div(ab_norm)
            self.memory_ab.index_copy_(0, y, updated_ab)
