"""Unified command-line options for F2Mix training and testing."""

import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ('true', '1', 'yes')


def parse_patch_size(s: str):
    return tuple(map(int, s.split(',')))


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='F2Mix: fine-grained augmentation and '
                                                 'feature copy-paste consistency')
    # Mode / model
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='run training or testing')
    parser.add_argument('--backbone', type=str, default='densenet',
                        choices=['densenet', 'resnet', 'res2net', 'vgg', 'pvt', 'pvt_v2', 'swin'],
                        help='backbone network (F2Mix is plug-and-play)')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes')

    # Data
    parser.add_argument('--data_path', type=str, default='your/path/to/dataset.hdf5',
                        help='HDF5 file with train/val/test groups (path + label)')
    parser.add_argument('--input_size', type=parse_patch_size, default=(48, 256, 256),
                        help='resampled volume size (D, H, W)')
    parser.add_argument('--num_workers', type=int, default=4, help='dataloader workers')

    # Training
    parser.add_argument('--total_epoch', type=int, default=150, help='total training epochs')
    parser.add_argument('--pre_epoch', type=int, default=0, help='epoch to resume from')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--base_lr', type=float, default=1e-5, help='base learning rate')
    parser.add_argument('--opt', type=str, default='SGD', choices=['SGD', 'AdamW'],
                        help='optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--seed', type=int, default=2024, help='random seed')

    # F2Mix specific
    parser.add_argument('--cp_patch', type=parse_patch_size, default=(180, 180),
                        help='in-plane copy-paste patch size (ph, pw)')
    parser.add_argument('--consistency_loss', type=str, default='mi',
                        choices=['mi', 'mse', 'ncc'], help='feature consistency loss')
    parser.add_argument('--lambda_fcp', type=float, default=0.5,
                        help='weight of the feature copy-paste consistency loss')
    parser.add_argument('--weak_prob', type=float, default=0.1,
                        help='probability of each weak transformation')
    parser.add_argument('--no_fga', action='store_true',
                        help='disable the FGA augmented view (ablation)')

    # Device
    parser.add_argument('--gpu', type=str, default='cuda:0', help='device to use')
    parser.add_argument('--multi_gpu', type=str2bool, default=False, help='use DataParallel')
    parser.add_argument('--gpus', type=str, default='0,1', help='GPU ids for DataParallel')

    # Saving / resuming
    parser.add_argument('--save_dir', type=str, default='./results/your_dataset',
                        help='root directory for logs, checkpoints and models')
    parser.add_argument('--resume', type=str, default=None,
                        help='checkpoint path to resume training from')
    parser.add_argument('--save_from_epoch', type=int, default=50,
                        help='epoch after which periodic models/checkpoints are saved')
    parser.add_argument('--test_model', type=str,
                        default='./results/your_dataset/densenet/models/best_model.pth',
                        help='model weights used for testing')

    return parser.parse_args()
