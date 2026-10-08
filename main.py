"""F2Mix unified entry point for training and testing.

Usage:
    python main.py --mode train --backbone densenet --data_path <your/path/dataset.hdf5>
    python main.py --mode test  --backbone densenet --data_path <your/path/dataset.hdf5> \
                   --test_model <your/path/best_model.pth>

The training pipeline follows the paper:
  1. FGA generates the fine-grained augmented view B from the original A.
  2. ICP exchanges an in-plane patch between A and B -> A', B'.
  3. The backbone encodes A, B, A', B'; the shared expander decodes
     full-resolution feature maps.
  4. FCP repeats the copy-paste at feature level with the same position.
  5. The FR module refines the features for classification; the total loss is
     L = L_CE + lambda * L_F with L_F the (MI-based) consistency loss.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from Model import build_model
from Utils import (HDF5VolumeDataset, build_consistency_loss, image_copy_paste,
                   feature_copy_paste, compute_metrics, plot_roc, CsvLogger,
                   save_predictions, setup_seed, setup_logging, make_run_dir,
                   save_checkpoint, load_checkpoint, load_weights)
from Utils.Options import get_args


# --------------------------------------------------------------------------- #
# Device / model helpers
# --------------------------------------------------------------------------- #

def build_device_and_model(args, num_classes):
    """Instantiate the model and move it to the selected device(s)."""
    model = build_model(args.backbone, num_classes=num_classes,
                        input_size=tuple(args.input_size)).float()

    if args.multi_gpu and torch.cuda.is_available():
        gpu_ids = [int(g) for g in args.gpus.split(',')]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        if len(gpu_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    else:
        device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    return model.to(device), device


def build_optimizer(args, model):
    if args.opt == 'SGD':
        return optim.SGD(model.parameters(), lr=args.base_lr,
                         momentum=args.momentum, weight_decay=args.weight_decay)
    return optim.AdamW(model.parameters(), lr=args.base_lr,
                       weight_decay=args.weight_decay)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate(model, device, loader):
    """Evaluate on a dataset; at test time the same volume feeds all views."""
    model.eval()
    y_true, y_prob = [], []
    for img_ori, _img_aug, label in loader:
        img_ori, label = img_ori.float().to(device), label.long().to(device)
        # No ICP/FGA at inference: A is replicated across the four views.
        _, _, _, _, logits = model(img_ori, img_ori, img_ori, img_ori)
        y_prob.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        y_true.extend(label.cpu().numpy())
    return compute_metrics(y_true, y_prob), y_true, y_prob


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def train_one_epoch(args, model, device, loader, optimizer, ce_loss,
                    consistency_loss, epoch, total_epochs):
    model.train()
    running_loss, running_ce, running_fcp = 0.0, 0.0, 0.0
    correct, samples = 0, 0

    for it, (img_ori, img_aug, label) in enumerate(loader):
        img_ori = img_ori.float().to(device)
        img_aug = img_aug.float().to(device)
        label = label.long().to(device)

        # ICP: image-level copy-paste between the original and FGA views.
        cp_ori, cp_aug, position = image_copy_paste(img_ori, img_aug, args.cp_patch)

        # Encode the four views; the expander returns full-resolution maps.
        f_a, f_b, f_a_cp, f_b_cp, logits = model(img_ori, img_aug, cp_ori, cp_aug)

        # FCP: repeat the copy-paste at feature level with the same position.
        f_a_cp_exchanged, f_b_cp_exchanged = feature_copy_paste(f_a_cp, f_b_cp, position)

        # Feature copy-paste consistency loss.
        loss_fcp = consistency_loss(f_a, f_a_cp_exchanged) + \
            consistency_loss(f_b, f_b_cp_exchanged)
        # Prediction loss.
        loss_ce = ce_loss(logits, label)
        loss = loss_ce + args.lambda_fcp * loss_fcp

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_ce += loss_ce.item()
        running_fcp += loss_fcp.item()
        correct += (logits.argmax(1) == label).sum().item()
        samples += label.size(0)

        if (it + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{total_epochs}] '
                  f'Iter [{it + 1}/{len(loader)}] '
                  f'Loss: {loss.item():.4f} (CE: {loss_ce.item():.4f}, '
                  f'FCP: {loss_fcp.item():.4f})', flush=True)

    n = max(len(loader), 1)
    return (running_loss / n, running_ce / n, running_fcp / n, correct / max(samples, 1) * 100)


def run_train(args, logger):
    run_dir = make_run_dir(args.save_dir, args.backbone,
                           tag=f'{args.consistency_loss}_{args.cp_patch[0]}')
    logger.info(f'Run directory: {run_dir}')

    model, device = build_device_and_model(args, args.num_classes)
    optimizer = build_optimizer(args, model)

    ce_loss = nn.CrossEntropyLoss()
    consistency_loss = build_consistency_loss(args.consistency_loss)

    train_set = HDF5VolumeDataset(args.data_path, 'train',
                                  input_size=tuple(args.input_size),
                                  use_fga=not args.no_fga,
                                  weak_prob=args.weak_prob)
    val_set = HDF5VolumeDataset(args.data_path, 'val',
                                input_size=tuple(args.input_size),
                                use_fga=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    start_epoch = args.pre_epoch
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, device=str(device))
        logger.info(f'Resumed from {args.resume} at epoch {start_epoch}')

    csv_logger = CsvLogger(os.path.join(run_dir, 'train_log.csv'),
                           ['epoch', 'loss', 'loss_ce', 'loss_fcp', 'train_acc',
                            'val_acc', 'val_auc', 'val_sen', 'val_spe'])

    best_auc = -1.0
    for epoch in range(start_epoch, args.total_epoch):
        loss, loss_ce, loss_fcp, train_acc = train_one_epoch(
            args, model, device, train_loader, optimizer, ce_loss,
            consistency_loss, epoch, args.total_epoch)

        val_metrics, _, _ = evaluate(model, device, val_loader)
        logger.info(f'epoch:{epoch} # loss:{loss:.4f} # ce:{loss_ce:.4f} # '
                    f'fcp:{loss_fcp:.4f} # train_acc:{train_acc:.2f} # '
                    f'val_acc:{val_metrics["acc"]:.4f} # val_auc:{val_metrics["auc"]:.4f} # '
                    f'val_sen:{val_metrics["sen"]:.4f} # val_spe:{val_metrics["spe"]:.4f}')
        csv_logger.log(epoch, f'{loss:.4f}', f'{loss_ce:.4f}', f'{loss_fcp:.4f}',
                       f'{train_acc:.2f}', f'{val_metrics["acc"]:.4f}',
                       f'{val_metrics["auc"]:.4f}', f'{val_metrics["sen"]:.4f}',
                       f'{val_metrics["spe"]:.4f}')

        # Model selection by the highest validation AUC (paper protocol).
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_path = os.path.join(run_dir, 'models', 'best_model.pth')
            torch.save((model.module if hasattr(model, 'module') else model).state_dict(),
                       best_path)
            logger.info(f'New best val AUC {best_auc:.4f} -> {best_path}')

        # Periodic snapshots.
        if epoch >= args.save_from_epoch:
            torch.save((model.module if hasattr(model, 'module') else model).state_dict(),
                       os.path.join(run_dir, 'models',
                                    f'epoch{epoch}_auc{val_metrics["auc"]:.4f}_model.pth'))
            save_checkpoint(epoch + 1, model, optimizer,
                            os.path.join(run_dir, 'checkpoints',
                                         f'checkpoint_epoch_{epoch}.pth'))

    logger.info(f'Training finished. Best val AUC: {best_auc:.4f}')


# --------------------------------------------------------------------------- #
# Testing
# --------------------------------------------------------------------------- #

def run_test(args, logger):
    model, device = build_device_and_model(args, args.num_classes)
    load_weights(args.test_model, model, device=str(device))
    logger.info(f'Loaded weights from {args.test_model}')

    test_set = HDF5VolumeDataset(args.data_path, 'test',
                                 input_size=tuple(args.input_size), use_fga=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    metrics, y_true, y_prob = evaluate(model, device, test_loader)
    logger.info(f'Test  ACC: {metrics["acc"]:.4f} # AUC: {metrics["auc"]:.4f} # '
                f'SEN: {metrics["sen"]:.4f} # SPE: {metrics["spe"]:.4f}')
    print(f'Test  ACC: {metrics["acc"]:.4f} # AUC: {metrics["auc"]:.4f} # '
          f'SEN: {metrics["sen"]:.4f} # SPE: {metrics["spe"]:.4f}')

    test_dir = os.path.join(args.save_dir, args.backbone, 'test')
    os.makedirs(test_dir, exist_ok=True)
    plot_roc(y_true, y_prob, os.path.join(test_dir, 'roc.png'))
    save_predictions(os.path.join(test_dir, 'predictions.csv'), y_true, y_prob,
                     (np.asarray(y_prob) >= 0.5).astype(int))


# --------------------------------------------------------------------------- #

def main():
    args = get_args()
    setup_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logging(os.path.join(args.save_dir, f'{args.backbone}_{args.mode}.log'))
    logger.info(f'Backbone: {args.backbone} | Mode: {args.mode} | '
                f'Dataset: {os.path.basename(args.data_path)}')

    if args.mode == 'train':
        run_train(args, logger)
    else:
        run_test(args, logger)


if __name__ == '__main__':
    main()
