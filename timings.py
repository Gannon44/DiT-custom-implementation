import time
import torch
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
import random
import numpy as np


# Much of this code is from the sample.py code from the original repo

def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000


    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    generation_times = []
    max_memory_usage = []

    for i in range(args.n_samples):
        random_class = random.randint(0, args.num_classes - 1)
        class_labels = [random_class]

        z = torch.randn(len(class_labels), 4, latent_size, latent_size, device=device)
        y = torch.tensor(class_labels, device=device)


        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * len(class_labels), device=device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)


        start_time = time.time()
        torch.cuda.reset_peak_memory_stats(device)


        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
        )
        samples, _ = samples.chunk(2, dim=0)
        samples = vae.decode(samples / 0.18215).sample


        generation_time = time.time() - start_time
        generation_times.append(generation_time)
        max_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        max_memory_usage.append(max_memory)

        save_image(samples, f"timings/{args.output_name}_{i + 1}.png", nrow=1, normalize=True, value_range=(-1, 1))

    # Print metrics summary
    print(f"Maximum generation time per image: {max(generation_times):.4f} seconds")
    print(f"Minimum generation time per image: {min(generation_times):.4f} seconds")
    print(f"Mean generation time per image: {np.mean(generation_times):.4f} seconds")
    print(f"Median generation time per image: {np.median(generation_times):.4f} seconds")

    print(f"Maximum memory usage per image: {max(max_memory_usage):.2f} MB")
    print(f"Minimum memory usage per image: {min(max_memory_usage):.2f} MB")
    print(f"Mean memory usage per image: {np.mean(max_memory_usage):.2f} MB")
    print(f"Median memory usage per image: {np.median(max_memory_usage):.2f} MB")

    print(f"Number of samples generated: {args.n_samples}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--output-name", type=str, default="sample")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--n-samples", type=int, default=8, help="Number of images to generate.")
    args = parser.parse_args()
    main(args)
