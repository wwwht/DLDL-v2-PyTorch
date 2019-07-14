import argparse

parser = argparse.ArgumentParser(description='Age Estimator')
"""dataset options"""
parser.add_argument('--train_img', type=str,
                    default='path to images for training')
parser.add_argument('--train_label', type=str,
                    default='path to .csv file which contains labels of images for training')
parser.add_argument('--val_img', type=str,
                    default='path to images for test')
parser.add_argument('--val_label', type=str,
                    default='path to .csv file which contains labels of images for test')
"""optimizer options"""
parser.add_argument("--lr", type=float, default=1e-3, help='learning rate')
parser.add_argument('--optimizer', default='SGD', choices=('SGD', 'ADAM', 'NADAM', 'RMSprop'), help='optimizer to use (SGD | ADAM | NADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--dampening', type=float, default=0, help='SGD dampening')
parser.add_argument('--nesterov', action='store_true', help='SGD nesterov')
parser.add_argument('--beta1', type=float, default=0.9, help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999, help='ADAM beta2')
parser.add_argument('--amsgrad', action='store_true', help='ADAM amsgrad')
parser.add_argument('--epsilon', type=float, default=1e-8, help='ADAM epsilon for numerical stability')
parser.add_argument('--gamma', type=float, default=0.8, help='learning rate decay factor for step decay')
parser.add_argument('--reset', action='store_true', help='reset the training')
parser.add_argument("--epochs", type=int, default=60, help='number of epochs to train')
parser.add_argument("--train_batch_size", type=int, default=128)
parser.add_argument("--val_batch_size", type=int, default=1)
parser.add_argument("--height", type=int, default=224, help='height of input image')
parser.add_argument("--width", type=int, default=224, help='width of input image')

"""model"""
parser.add_argument("--model_name", type=str, default='TinyAge', help='which model to train')
parser.add_argument('--nThread', type=int, default=12, help='number of threads for data loading')
args = parser.parse_args()
