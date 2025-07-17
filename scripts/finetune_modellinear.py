import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import logging
import os
import numpy as np
from pathlib import Path

# Import your existing modules
from model.pose_diffusion import PoseDiffusion
from model.diffusion_util import TransformerModel, VarianceSchedule
from model.diffusion_net import DiffusionNet
from data_loader.lmdb_data_loader_expressive import SpeechMotionDataset, default_collate_fn
from parse_args_diffusion import parse_args
from train_eval.train_diffusion import train_iter_diffusion
from utils.train_utils import set_logger, set_random_seed, save_checkpoint
from utils.vocab_utils import build_vocab
from data_loader.lmdb_data_loader_new import LMDBDataset

class AdaptedPoseDiffusion(nn.Module):
    """
    A modified PoseDiffusion model that adapts to different pose dimensions
    while preserving the pretrained transformer blocks.
    """
    
    def __init__(self, args, new_pose_dim):
        super().__init__()
        
        self.args = args
        self.new_pose_dim = new_pose_dim
        self.original_pose_dim = 126  # TED Expressive's pose dimension
        
        # Same configuration as original model
        self.pre_length = args.n_pre_poses
        self.gen_length = args.n_poses - args.n_pre_poses
        self.input_context = args.input_context
        
        # Audio encoder (will be frozen by default)
        self.audio_encoder = self._create_audio_encoder()
        
        # Calculate new input dimensions
        # Original: 32 (audio) + 126 (pose) + 1 (bit) = 159
        # New: 32 (audio) + new_pose_dim + 1 (bit)
        self.new_in_size = 32 + new_pose_dim + 1
        
        # Classifier-free guidance components with new dimensions
        self.classifier_free = args.classifier_free
        if self.classifier_free:
            self.null_cond_prob = args.null_cond_prob
            self.null_cond_emb = nn.Parameter(torch.randn(1, self.new_in_size))
        
        # Create adapted diffusion network
        self.diffusion_net = self._create_adapted_diffusion_net(args, new_pose_dim)
    
    def _create_audio_encoder(self):
        """Create the audio encoder (same as original)"""
        from model.pose_diffusion import WavEncoder
        return WavEncoder()
    
    def _create_adapted_diffusion_net(self, args, new_pose_dim):
        """Create an adapted diffusion network with new input/output projections"""
        
        # Create a custom transformer that adapts input/output dimensions
        adapted_transformer = AdaptedTransformerModel(
            num_pose=args.n_poses,
            new_pose_dim=new_pose_dim,
            original_pose_dim=self.original_pose_dim,
            hidden_dim=args.diff_hidden_dim,
            depth=args.block_depth//2,
            decoder_depth=args.block_depth//2
        )
        
        # Create variance schedule (same as original)
        var_sched = VarianceSchedule(
            num_steps=500,
            beta_1=1e-4,
            beta_T=0.02,
            mode='linear'
        )
        
        return DiffusionNet(net=adapted_transformer, var_sched=var_sched)
    
    def load_pretrained_weights(self, checkpoint_path, freeze_audio=True, freeze_transformer=True):
        """Load pretrained weights and adapt to new architecture"""
        
        print(f"Loading pretrained weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        pretrained_state = checkpoint['state_dict']
        
        # Load audio encoder weights
        audio_state = {k.replace('audio_encoder.', ''): v 
                    for k, v in pretrained_state.items() 
                    if k.startswith('audio_encoder.')}
        self.audio_encoder.load_state_dict(audio_state)
        
        if freeze_audio:
            print("Freezing audio encoder")
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
        
        # Load transformer blocks (encoder and decoder)
        transformer_blocks_state = {}
        
        # Extract encoder blocks
        for k, v in pretrained_state.items():
            if 'blocks.' in k and 'decoder_blocks' not in k:
                new_key = k.replace('diffusion_net.net.', '')
                transformer_blocks_state[new_key] = v
        
        # Extract decoder blocks
        for k, v in pretrained_state.items():
            if 'decoder_blocks.' in k:
                new_key = k.replace('diffusion_net.net.', '')
                transformer_blocks_state[new_key] = v
        
        # Extract normalization layers
        for k, v in pretrained_state.items():
            if 'norm.' in k or 'decoder_norm.' in k:
                if 'diffusion_net.net.' in k:
                    new_key = k.replace('diffusion_net.net.', '')
                    transformer_blocks_state[new_key] = v
        
        # Load position embedding
        if 'diffusion_net.net.pos_embedding' in pretrained_state:
            transformer_blocks_state['pos_embedding'] = pretrained_state['diffusion_net.net.pos_embedding']
        
        # Load the weights into our adapted transformer
        self.diffusion_net.net.load_pretrained_transformer_weights(transformer_blocks_state)
        
        # NEW: Freeze transformer blocks if requested
        if freeze_transformer:
            print("Freezing transformer blocks")
            # Freeze encoder blocks
            for block in self.diffusion_net.net.blocks:
                for param in block.parameters():
                    param.requires_grad = False
            
            # Freeze decoder blocks
            for block in self.diffusion_net.net.decoder_blocks:
                for param in block.parameters():
                    param.requires_grad = False
            
            # Freeze normalization layers
            for param in self.diffusion_net.net.norm.parameters():
                param.requires_grad = False
            for param in self.diffusion_net.net.decoder_norm.parameters():
                param.requires_grad = False
            
            # Freeze position embedding
            self.diffusion_net.net.pos_embedding.requires_grad = False
        
        # Load variance schedule
        var_sched_state = {k.replace('diffusion_net.var_sched.', ''): v 
                        for k, v in pretrained_state.items() 
                        if k.startswith('diffusion_net.var_sched.')}
        self.diffusion_net.var_sched.load_state_dict(var_sched_state)
        
        print("Pretrained weights loaded successfully")
    # Forward methods (same as original PoseDiffusion)
    def get_loss(self, x, pre_seq, in_audio):
        if self.input_context == 'audio':
            audio_feat_seq = self.audio_encoder(in_audio)
            in_data = torch.cat((pre_seq, audio_feat_seq), dim=2)
        else:
            assert False
        
        if self.classifier_free:
            mask = torch.zeros((x.shape[0],), device=x.device).float().uniform_(0, 1) < self.null_cond_prob
            in_data = torch.where(mask.unsqueeze(1).unsqueeze(2), 
                                self.null_cond_emb.repeat(in_data.shape[1], 1).unsqueeze(0), 
                                in_data)
        
        neg_elbo = self.diffusion_net.get_loss(x, in_data)
        return neg_elbo
    
    def sample(self, pose_dim, pre_seq, in_audio):
        if self.input_context == 'audio':
            audio_feat_seq = self.audio_encoder(in_audio)
            in_data = torch.cat((pre_seq, audio_feat_seq), dim=2)
        
        if self.classifier_free:
            uncondition_embedding = self.null_cond_emb.repeat(in_data.shape[1], 1).unsqueeze(0)
            samples = self.diffusion_net.sample(self.args.n_poses, in_data, pose_dim, 
                                              uncondition_embedding=uncondition_embedding)
        else:
            samples = self.diffusion_net.sample(self.args.n_poses, in_data, pose_dim)
        
        return samples


class AdaptedTransformerModel(nn.Module):
    """
    A transformer model that adapts the input/output dimensions
    while preserving the pretrained transformer blocks.
    """
    
    def __init__(self, num_pose, new_pose_dim, original_pose_dim, 
                 hidden_dim, depth, decoder_depth):
        super().__init__()
        
        self.new_pose_dim = new_pose_dim
        self.original_pose_dim = original_pose_dim
        self.hidden_dim = hidden_dim
        
        # Calculate new input dimension
        # Original: 126 + 3 + (32 + 126 + 1) = 288
        # New: new_pose_dim + 3 + (32 + new_pose_dim + 1)
        self.new_embed_dim = new_pose_dim + 3 + (32 + new_pose_dim + 1)
        
        # New input projection layer
        self.linear = nn.Linear(self.new_embed_dim, hidden_dim)
        
        # Position embedding (reuse from pretrained)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_pose, hidden_dim))
        
        # These will be loaded from pretrained model
        self.blocks = nn.ModuleList([
            self._create_block(hidden_dim) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        
        self.decoder_blocks = nn.ModuleList([
            self._create_block(hidden_dim) for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        
        # New output projection layers
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LeakyReLU(True),
            nn.Linear(hidden_dim//2, new_pose_dim)
        )
        
        self._init_new_weights()
    
    def _create_block(self, hidden_dim):
        """Create a transformer block (will be replaced by pretrained weights)"""
        from timm.models.vision_transformer import Block
        return Block(hidden_dim, num_heads=8, mlp_ratio=4., 
                    qkv_bias=True, norm_layer=nn.LayerNorm)
    
    def _init_new_weights(self):
        """Initialize only the new layers"""
        # Initialize input projection
        torch.nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)
        
        # Initialize output layers
        for m in self.out.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def load_pretrained_transformer_weights(self, state_dict):
        """Load pretrained transformer blocks while keeping new input/output layers"""
        
        # Load transformer blocks
        for key, value in state_dict.items():
            if hasattr(self, key):
                if key in ['blocks', 'decoder_blocks', 'norm', 'decoder_norm', 'pos_embedding']:
                    if key == 'pos_embedding':
                        self.pos_embedding.data = value
                    else:
                        getattr(self, key).load_state_dict({
                            k.split('.', 1)[1]: v for k, v in state_dict.items() 
                            if k.startswith(key + '.')
                        })
        
        print("Loaded pretrained transformer blocks")
    
    def forward(self, x, beta, context):
        # Same forward logic as original TransformerModel
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)
        
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        time_emb = time_emb.repeat(1, x.shape[1], 1)
        ctx_emb = torch.cat([time_emb, context], dim=-1)
        
        x = torch.cat([x, ctx_emb], dim=2)
        
        # Project through new input layer
        x = self.linear(x)
        # Move position embedding to the same device as x
        x += self.pos_embedding.to(x.device)
        
        # Pass through pretrained transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # Project through new output layers
        return self.out(x)

def main():
    parser = argparse.ArgumentParser(description='Fine-tune diffusion model on new dataset')
    parser.add_argument('--config', help='Config file path')
    parser.add_argument('--checkpoint',help='Pretrained checkpoint path')
    parser.add_argument('--train_data_path', help='Path to new training data')
    parser.add_argument('--output_dir', default='./output/finetune_linear_beat_warmup', help='Output directory for checkpoints')
    parser.add_argument('--tune_audio_encoder', action='store_true', 
                       help='Whether to fine-tune audio encoder (default: False)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                       help='Learning rate for fine-tuning')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size (smaller for small dataset)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--save_interval', type=int, default=5, help='Save checkpoint interval')
    parser.add_argument('--log_interval', type=int, default=50, help='Logging interval for batches')
    
    args_finetune = parser.parse_args()
    args_finetune.checkpoint = "/home/bsd/cospeech/DiffGesture/output/train_diffusion_expressive/pose_diffusion_checkpoint_499.bin"
    # Load original config file
    original_args = parse_args()
    
    # Set up logging
    os.makedirs(args_finetune.output_dir, exist_ok=True)
    set_logger(args_finetune.output_dir, 'finetune.log')
    
    # Load the training dataset directly
    logging.info(f"Loading dataset from {args_finetune.train_data_path}")
    
    # The dataset already has calculated mean_dir_vec and mean_pose
    # It will load these from the existing lmdb path
    train_dataset = LMDBDataset(dataset_path="./data/beat_english_v0.2.1/beat_warmup_cache")
    
    # Get a single sample to determine the new pose dimension
    # The dataset returns: word_seq_tensor, extended_word_seq, pose_seq, vec_seq, audio, spectrogram, aux_info
    sample = train_dataset[0]
    pose_seq = sample[2]  # This is the pose sequence
    vec_seq = sample[3]   # This is the direction vector sequence
    
    # Extract the actual pose dimension from the data
    new_pose_dim = vec_seq.shape[-1]  # Get the dimension of the direction vectors
    logging.info(f"Detected new pose dimension: {new_pose_dim}")
    logging.info(f"Original pose dimension was: 126")
    logging.info(f"Dataset contains {len(train_dataset)} samples")
    
    # # Build vocabulary (reuse from the dataset's language model if it exists)
    # vocab_cache_path = os.path.join(os.path.dirname(args_finetune.train_data_path), 'vocab_cache.pkl')
    # if os.path.exists(vocab_cache_path):
    #     logging.info(f"Loading existing vocabulary from {vocab_cache_path}")
    #     import pickle
    #     with open(vocab_cache_path, 'rb') as f:
    #         lang_model = pickle.load(f)
    # else:
    #     logging.info("Building new vocabulary")
    #     lang_model = build_vocab('words', [train_dataset], vocab_cache_path,
    #                            original_args.wordembed_path, original_args.wordembed_dim)
    
    # train_dataset.set_lang_model(lang_model)
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args_finetune.batch_size,
        shuffle=True, 
        drop_last=True, 
        num_workers=4,
        pin_memory=True,
        collate_fn=default_collate_fn
    )
    
    # Create adapted model with the detected pose dimension
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    model = AdaptedPoseDiffusion(original_args, new_pose_dim).to(device)
    
    # Load pretrained weights
    model.load_pretrained_weights(args_finetune.checkpoint, 
                                freeze_audio=not args_finetune.tune_audio_encoder)
    
    # Setup optimizer - only optimize parameters that require gradients
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_optimize, lr=args_finetune.learning_rate)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    logging.info(f"Frozen parameters: {total_params - trainable_params:,}")
    
    # Simple learning rate scheduler for small datasets
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args_finetune.epochs)
    
    # Training loop
    logging.info("Starting fine-tuning on small dataset...")
    best_loss = float('inf')
    
    for epoch in range(args_finetune.epochs):
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        for batch_idx, data in enumerate(train_loader):
            _, _, _, _, target_vec, in_audio, _, _ = data
            
            target_vec = target_vec.to(device)
            in_audio = in_audio.to(device)
            
            # Prepare input sequence
            pre_seq = target_vec.new_zeros((target_vec.shape[0], target_vec.shape[1], 
                                          target_vec.shape[2] + 1))
            pre_seq[:, 0:original_args.n_pre_poses, :-1] = target_vec[:, 0:original_args.n_pre_poses]
            pre_seq[:, 0:original_args.n_pre_poses, -1] = 1  # Constraint bit
            
            # Forward pass
            optimizer.zero_grad()
            loss = model.get_loss(target_vec, pre_seq, in_audio)
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            # Log more frequently for small datasets
            if batch_idx % args_finetune.log_interval == 0:
                current_lr = scheduler.get_last_lr()[0]
                logging.info(f"Epoch [{epoch+1}/{args_finetune.epochs}] "
                           f"Batch [{batch_idx+1}/{len(train_loader)}] "
                           f"Loss: {loss.item():.6f} "
                           f"LR: {current_lr:.6f}")
        
        # Calculate average loss for the epoch
        avg_loss = epoch_loss / batch_count
        logging.info(f"Epoch [{epoch+1}/{args_finetune.epochs}] "
                    f"Average Loss: {avg_loss:.6f}")
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_checkpoint_path = os.path.join(args_finetune.output_dir, 'best_model.pt')
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'args': original_args,
                'new_pose_dim': new_pose_dim,
                'best_loss': best_loss,
                'train_data_path': args_finetune.train_data_path
            }, best_checkpoint_path)
            logging.info(f"Saved best model with loss {best_loss:.6f}")
        
        # Save periodic checkpoints
        if (epoch + 1) % args_finetune.save_interval == 0:
            checkpoint_path = os.path.join(args_finetune.output_dir, 
                                         f'checkpoint_epoch_{epoch+1}.pt')
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'args': original_args,
                'new_pose_dim': new_pose_dim,
                'train_data_path': args_finetune.train_data_path
            }, checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path}")
    
    logging.info(f"Fine-tuning completed! Best loss: {best_loss:.6f}")
    logging.info(f"Best model saved at: {best_checkpoint_path}")


if __name__ == '__main__':
    main()