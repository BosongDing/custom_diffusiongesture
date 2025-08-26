python scripts/model/AEs/train_vae.py --dataset beat --batch_per_dataset 512 --num_workers 8 --prefetch_factor 6 --ckpt_dir scripts/model/AEs/ckpt/vae_beat_beta_1e-4 --log_dir scripts/model/AEs/ckpt/vae_logs_beat_beta_1e-4
python scripts/model/AEs/train_vae.py --dataset trinity --batch_per_dataset 512 --num_workers 8 --prefetch_factor 6 --ckpt_dir scripts/model/AEs/ckpt/vae_trinity_beta1e-4 --log_dir scripts/model/AEs/ckpt/vae_logs_trinity_beta1e-4
python scripts/model/AEs/train_vae.py --dataset ted_expressive --batch_per_dataset 512 --num_workers 8 --prefetch_factor 6 --ckpt_dir scripts/model/AEs/ckpt/vae_ted_expressive_beta1e-4 --log_dir scripts/model/AEs/ckpt/vae_logs_ted_expressive_beta1e-4


python scripts/model/AEs/train_vae.py --dataset beat --batch_per_dataset 512 --num_workers 8 --beta 1e-3 --prefetch_factor 6 --ckpt_dir scripts/model/AEs/ckpt/vae_beat_beta_1e-3 --log_dir scripts/model/AEs/ckpt/vae_logs_beat_beta_1e-3
python scripts/model/AEs/train_vae.py --dataset trinity --batch_per_dataset 512 --num_workers 8 --beta 1e-3 --prefetch_factor 6 --ckpt_dir scripts/model/AEs/ckpt/vae_trinity_beta1e-3 --log_dir scripts/model/AEs/ckpt/vae_logs_trinity_beta1e-3
python scripts/model/AEs/train_vae.py --dataset ted_expressive --batch_per_dataset 512 --num_workers 8 --beta 1e-3 --prefetch_factor 6 --ckpt_dir scripts/model/AEs/ckpt/vae_ted_expressive_beta1e-3 --log_dir scripts/model/AEs/ckpt/vae_logs_ted_expressive_beta1e-3

python scripts/model/AEs/train_vae.py --dataset trinity --batch_per_dataset 512 --num_workers 8 --beta 1e-6 --prefetch_factor 6 --ckpt_dir scripts/model/AEs/ckpt/vae_trinity_beta1e-6 --log_dir scripts/model/AEs/ckpt/vae_logs_trinity_beta1e-6
python scripts/model/AEs/train_vae.py --dataset ted_expressive --batch_per_dataset 512 --num_workers 8 --beta 1e-6 --prefetch_factor 6 --ckpt_dir scripts/model/AEs/ckpt/vae_ted_expressive_beta1e-6 --log_dir scripts/model/AEs/ckpt/vae_logs_ted_expressive_beta1e-6

