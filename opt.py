import argparse

parser = argparse.ArgumentParser(description='reid')

parser.add_argument('--data_path',
                    default="../vehicle/VeRi/",
                    help='path of Market-1501-v15.09.15')

parser.add_argument('--mode',
                    default='train', choices=['train', 'evaluate', 'vis'],
                    help='train or evaluate ')
parser.add_argument('--ngpu', default=1, type=int, help='gpus')
parser.add_argument('--device', default='0', type=str, help='gpus')
parser.add_argument('--query_image',
                    default='0001_c1s1_001051_00.jpg',
                    help='path to the image you want to query')
parser.add_argument('--save_path',default='./answer1.txt', help='The directory of query files path')

parser.add_argument('--weight',
                    default='weights/model1.pt',
                    help='load weights ')

parser.add_argument('--epoch',
                    default=600,
                    help='number of epoch to train')
parser.add_argument('--freeze', action='store_true',help="evaluation only")
parser.add_argument('--lr',
                    default=2e-4,
                    help='initial learning_rate')

parser.add_argument('--lr_scheduler',
                    default=[240,320],
                    help='MultiStepLR,decay the learning rate')
parser.add_argument('--resume',action='store_true',help="load weights")

parser.add_argument("--batchid",
                    default=5,
                    help='the batch for id')

parser.add_argument("--batchimage",
                    default=4,
                    help='the batch of per id')
parser.add_argument("--batchsize",
                    default=20,
                    help='the batch size for test')
parser.add_argument("--batchtest",
                    default=24,
                    help='the batch size for test')
parser.add_argument('--name', default='Spark1', type=str, help='gpus')
parser.add_argument('--dtype', default='vehivle', type=str, help='vehicle or person')
parser.add_argument('--data_name', default='veri', type=str, help='vehicleid, veri, market, msmt')
opt = parser.parse_args()
