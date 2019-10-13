import torch


def extract_feature(model, loader):
    features = torch.FloatTensor()

    for (inputs, labels, box1, box2, box3, box4) in loader:

        ff = torch.FloatTensor(inputs.size(0), 256).zero_()#2048  2304   2080  3584 4224 4608

        for i in range(2):
            if i == 1:
                inputs = inputs.index_select(3, torch.arange(inputs.size(3) - 1, -1, -1).long())
            input_img = inputs.to('cuda')
            outputs = model(input_img)
            f = outputs[-1].data.cpu()
            ff = ff + f

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, ff), 0)
    return features
