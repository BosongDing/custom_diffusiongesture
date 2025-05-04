from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime
import matplotlib
import pprint
import sys
import time

from data_loader.lmdb_data_loader_trinity import *
from model.pose_diffusion import PoseDiffusion
from parse_args_diffusion import parse_args
from train_eval.train_diffusion import train_iter_diffusion
from utils.average_meter import AverageMeter
from utils.vocab_utils import build_vocab
import utils.train_utils

matplotlib.use('Agg')  # we don't use interactive GUI
[sys.path.append(i) for i in ['.', '..']]
device = torch.device("cuda:0")


def init_model(args, _device):
    # init model
    if args.model == 'pose_diffusion':
        print("init diffusion model")
        diffusion = PoseDiffusion(args).to(_device)
    return diffusion


def train_epochs(args, train_data_loader, lang_model, pose_dim, speaker_model=None):
    start = time.time()
    loss_meters = [AverageMeter('loss'), AverageMeter('var_loss'), AverageMeter('gen'), AverageMeter('dis'),
                   AverageMeter('KLD'), AverageMeter('DIV_REG')]

    tb_path = args.name + '_' + str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    tb_writer = SummaryWriter(log_dir=str(Path(args.model_save_path).parent / 'tensorboard_runs' / tb_path))

    # interval params
    print_interval = int(len(train_data_loader) / 5)
    save_model_epoch_interval = 50

    diffusion_model = init_model(args, device)

    optimizer = optim.Adam(diffusion_model.parameters(),lr=args.learning_rate, betas=(0.5, 0.999))

    # training
    global_iter = 0
    for epoch in range(args.epochs):

        # save model
        if (epoch % save_model_epoch_interval == 0 and epoch > 0) or epoch == args.epochs - 1:

            state_dict = diffusion_model.state_dict()

            save_name = '{}/{}_checkpoint_{:03d}.bin'.format(args.model_save_path, args.name, epoch)

            utils.train_utils.save_checkpoint({
                'args': args, 'epoch': epoch, 'lang_model': lang_model, 'speaker_model': speaker_model,
                'pose_dim': pose_dim, 'state_dict': state_dict,
            }, save_name)

        # train iter
        iter_start_time = time.time()
        for iter_idx, data in enumerate(train_data_loader, 0):
            global_iter += 1
            _, _, _, _, target_vec, in_audio, _, _ = data

            batch_size = target_vec.size(0)
            in_audio = in_audio.to(device)
            target_vec = target_vec.to(device)

            # train
            loss = []

            if args.model == 'pose_diffusion':
                loss = train_iter_diffusion(args, in_audio, target_vec,
                                      diffusion_model, optimizer)

            # loss values
            for loss_meter in loss_meters:
                name = loss_meter.name
                if name in loss:
                    loss_meter.update(loss[name], batch_size)

            # write to tensorboard
            for key in loss.keys():
                tb_writer.add_scalar(key + '/train', loss[key], global_iter)

            # print training status
            if (iter_idx + 1) % print_interval == 0:
                print_summary = 'EP {} ({:3d}) | {:>8s}, {:.0f} samples/s | '.format(
                    epoch, iter_idx + 1, utils.train_utils.time_since(start),
                    batch_size / (time.time() - iter_start_time))
                for loss_meter in loss_meters:
                    if loss_meter.count > 0:
                        print_summary += '{}: {:.3f}, '.format(loss_meter.name, loss_meter.avg)
                        loss_meter.reset()
                logging.info(print_summary)

            iter_start_time = time.time()

    tb_writer.close()

def main(config):
    args = config['args']

    # random seed
    if args.random_seed >= 0:
        utils.train_utils.set_random_seed(args.random_seed)

    # set logger
    utils.train_utils.set_logger(args.model_save_path, os.path.basename(__file__).replace('.py', '.log'))

    logging.info("PyTorch version: {}".format(torch.__version__))
    logging.info("CUDA version: {}".format(torch.version.cuda))
    logging.info("{} GPUs, default {}".format(torch.cuda.device_count(), device))
    logging.info(pprint.pformat(vars(args)))

    collate_fn = default_collate_fn

    # dataset
    mean_dir_vec = np.load('/home/bsd/cospeech/DiffGesture/data/trinity/mean_dir_vec.npy')
    
    # Use LMDB-based dataset for faster data loading
    logging.info("Creating training dataset...")
    train_dataset = TrinityLMDBDataset(
        data_dir='/home/bsd/cospeech/DiffGesture/data/trinity/allRec',
        audio_dir='/home/bsd/cospeech/DiffGesture/data/trinity/allRecAudio',
        n_poses=34,
        subdivision_stride=10,
        original_fps=60,
        target_fps=15,
        mean_dir_vec=mean_dir_vec
    )
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=args.loader_workers, pin_memory=True,
                              collate_fn=collate_fn
                              )
    
    logging.info("Creating validation dataset...")
    val_dataset = TrinityLMDBDataset(
        data_dir='/home/bsd/cospeech/DiffGesture/data/trinity/allTestMotion',
        audio_dir='/home/bsd/cospeech/DiffGesture/data/trinity/allTestAudio',
        n_poses=34,
        subdivision_stride=10,
        original_fps=60,
        target_fps=15,
        mean_dir_vec=mean_dir_vec
    )
    
    logging.info("Creating test dataset...")
    test_dataset = TrinityLMDBDataset(
        data_dir='/home/bsd/cospeech/DiffGesture/data/trinity/allTestMotion',
        audio_dir='/home/bsd/cospeech/DiffGesture/data/trinity/allTestAudio',
        n_poses=34,
        subdivision_stride=10,
        original_fps=60,
        target_fps=15,
        mean_dir_vec=mean_dir_vec
    )

    # No need for vocab model with Trinity dataset (we use dummy word sequences)
    lang_model = None
    
    # Train the model
    logging.info("Starting training...")
    pose_dim = args.pose_dim
    train_epochs(args, train_loader, lang_model,
                 pose_dim=pose_dim, speaker_model=None)


if __name__ == '__main__':
    _args = parse_args()
    main({'args': _args})
