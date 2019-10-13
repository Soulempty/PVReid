from torch.nn.modules import loss
from utils.TripletLoss import TripletLoss
from utils.CrossEntropy import CrossEntropyLoss


class Loss(loss._Loss):
    def __init__(self,lsm=False, margin=1.2):
        super(Loss, self).__init__()
        self.lsm = lsm
        self.margin = margin

    def forward(self, outputs, labels):
        num_cls = outputs[1][0].size(-1)
        cross_entropy_loss = CrossEntropyLoss(num_cls, label_smooth=self.lsm)
        loss_sum = 0
        triplet_loss = TripletLoss(margin=self.margin)
        if isinstance(outputs[0],(tuple,list)):
            Triplet_Loss = [triplet_loss(output, labels) for output in outputs[0]]
            Triplet_Loss = sum(Triplet_Loss) / len(Triplet_Loss)
        else:
            Triplet_Loss = triplet_loss(outputs[0], labels) 
            
        if isinstance(outputs[1], (tuple,list)):
            crossen = [cross_entropy_loss(output, labels) for output in outputs[1]]
            crossen = sum(crossen) / len(crossen)
        else:
            crossen = cross_entropy_loss(outputs[1], labels)
        loss_sum = crossen *2 + Triplet_Loss
        print('\rtotal loss:%.2f  Triplet_Loss:%.2f  CrossEntropy_Loss:%.2f' % (
            loss_sum.data.cpu().numpy(),
            Triplet_Loss.data.cpu().numpy(),
            crossen.data.cpu().numpy()),
              end=' ')
        return loss_sum
