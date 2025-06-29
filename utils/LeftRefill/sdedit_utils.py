import configargparse
import os
import random
import subprocess
from glob import glob

import numpy as np
import torch
from einops import repeat
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from natsort import natsorted
from omegaconf import OmegaConf
from PIL import Image
from test_inpainting import load_state_dict, torch_init_model
from tqdm import tqdm


def initialize_model(path):
    config = OmegaConf.load(os.path.join(path, "model_config.yaml"))
    model = instantiate_from_config(config.model)
    # repeat_sp_token = config['model']['params']['data_config']['repeat_sp_token']
    # sp_token = config['model']['params']['data_config']['sp_token']

    ckpt_list = glob(os.path.join(path, 'ckpts/epoch=*.ckpt'))
    if len(ckpt_list) > 1:
        resume_path = sorted(ckpt_list, key=lambda x: int(x.split('/')[-1].split('.ckpt')[0].split('=')[-1]))[-1]
    else:
        resume_path = ckpt_list[0]
    print('Load ckpt', resume_path)

    reload_weights = load_state_dict(resume_path, location='cpu')
    torch_init_model(model, reload_weights, key='none')
    if getattr(model, 'save_prompt_only', False):
        pretrained_weights = load_state_dict(f'{os.path.dirname(os.path.abspath(__file__))}/pretrained_models/512-inpainting-ema.ckpt', location='cpu')
        torch_init_model(model, pretrained_weights, key='none')

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()
    sampler = DDIMSampler(model)

    return sampler


if not os.path.exists(f'{os.path.dirname(os.path.abspath(__file__))}/pretrained_models/512-inpainting-ema.ckpt'):
    os.makedirs(f'{os.path.dirname(os.path.abspath(__file__))}/pretrained_models', exist_ok=True)
    os.system(f'wget https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/resolve/main/512-inpainting-ema.ckpt -O {os.path.dirname(os.path.abspath(__file__))}/pretrained_models/512-inpainting-ema.ckpt')
target_image_size = 512
repeat_sp_token = 50
sp_token = "<special-token>"
root_path = f"{os.path.dirname(os.path.abspath(__file__))}/check_points/ref_guided_inpainting"    
sampler = initialize_model(path=root_path)


def make_batch_sd(
        image,
        mask,
        txt,
        device,
        num_samples=1):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    batch = {
        "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
        "txt": num_samples * [txt],
        "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
        "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
    }
    return batch


def inpaint(sampler, image, mask, prompt, seed, scale, ddim_steps, num_samples=1, w=512, h=512, strength=None, eta=0.0, use_ddim_inversion=True):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = sampler.model

    # print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")

    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, 4, h // 8, w // 8) # start code目前是純noise
    
    start_code = torch.from_numpy(start_code).to(
        device=device, dtype=torch.float32)
    
    
    
    
    with torch.no_grad(), torch.autocast("cuda"):
        batch = make_batch_sd(image, mask, txt=prompt, device=device, num_samples=num_samples)
        print(batch['image'].shape)
        c = model.cond_stage_model.encode(batch["txt"])

        c_cat = list()
        for ck in model.concat_keys:
            cc = batch[ck].float()
            if ck != model.masked_image_key:
                bchw = [num_samples, 4, h // 8, w // 8]
                cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
            else:
                cc = model.get_first_stage_encoding(
                    model.encode_first_stage(cc))
            c_cat.append(cc)
        c_cat = torch.cat(c_cat, dim=1)

        # cond
        cond = {"c_concat": [c_cat], "c_crossattn": [c]}

        # uncond cond
        uc_cross = model.get_unconditional_conditioning(num_samples)
        uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

        shape = [model.channels, h // 8, w // 8]
        
        # TODO: encode image to latent (w // 8, h // 8), replace start_code with below latent (need to add noise inside sample)
        if strength is not None:
            start_code = model.get_first_stage_encoding(model.encode_first_stage(batch["image"]))


        samples_cfg, intermediates = sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc_full,
            x_T=start_code, 
            strength=strength,
            use_ddim_inversion=use_ddim_inversion
        )
        x_samples_ddim = model.decode_first_stage(samples_cfg)
        pred = x_samples_ddim * batch['mask'] + batch['image'] * (1 - batch['mask'])

        result = torch.clamp((pred + 1.0) / 2.0, min=0.0, max=1.0)

        result = (result.cpu().numpy().transpose(0, 2, 3, 1) * 255)
        result = result[:, :, 512:]

    return [Image.fromarray(img.astype(np.uint8)) for img in result]

    

def pad_image(input_image):
    pad_w, pad_h = np.max(((2, 2), np.ceil(np.array(input_image.size) / 64).astype(int)), axis=0) * 64 - input_image.size
    im_padded = Image.fromarray(np.pad(np.array(input_image), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))
    return im_padded

def predict(source, reference, ddim_steps, num_samples, scale, seed, strength=None, eta=0.0, use_ddim_inversion=True):
    source_img = source["image"].convert("RGB")
    origin_w, origin_h = source_img.size
    ratio = origin_h / origin_w
    init_mask = source["mask"].convert("RGB")
    print('Source...', source_img.size)
    reference_img = reference.convert("RGB")
    print('Reference...', reference_img.size)

    source_img = source_img.resize((target_image_size, target_image_size), resample=Image.Resampling.BICUBIC)
    reference_img = reference_img.resize((target_image_size, target_image_size), resample=Image.Resampling.BICUBIC)
    init_mask = init_mask.resize((target_image_size, target_image_size), resample=Image.Resampling.BILINEAR)
    init_mask = np.array(init_mask)
    init_mask[init_mask > 0] = 255
    init_mask = Image.fromarray(init_mask)

    source_img = pad_image(source_img) # resize to integer multiple of 32
    reference_img = pad_image(reference_img)
    mask = pad_image(init_mask)  # resize to integer multiple of 32
    
    width, height = source_img.size
    width *= 2
    print("Inpainting...", width, height)
    # print("Prompt:", prompt)

    # get inputs
    image = np.concatenate([np.asarray(reference_img), np.asarray(source_img)], axis=1)
    image = Image.fromarray(image)
    mask = np.asarray(mask)
    mask = np.concatenate([np.zeros_like(mask), mask], axis=1)
    mask = Image.fromarray(mask)

    prompt = ""
    for i in range(repeat_sp_token):
        prompt = prompt + sp_token.replace('>', f'{i}> ')
    prompt = prompt.strip()
    print('Prompt:', prompt)

    result = inpaint(
        sampler=sampler,
        image=image,
        mask=mask,
        prompt=prompt,
        seed=seed,
        scale=scale,
        ddim_steps=ddim_steps,
        num_samples=num_samples,
        h=height, w=width,
        strength=strength,
        eta=eta,
        use_ddim_inversion=use_ddim_inversion
    )

    result = [r.resize((int(origin_w), origin_h), resample=Image.Resampling.BICUBIC) for r in result]
    for r in result:
        print(r.size)
    return result





def LeftRefill(ref_img_path, source_root, ref_root, mask_root, output_root, strength=None, scale=2.5, eta=1.0, use_ddim_inversion=True):
    # Model Setup
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    torch.set_grad_enabled(False)

    
    
    
    # Run
    ref_img = Image.open(ref_img_path)
    num_image = len(os.listdir(source_root))
    ref_list = natsorted(os.listdir(ref_root))
    source_list = natsorted(os.listdir(source_root))
    mask_list = natsorted(os.listdir(mask_root))
    assert len(ref_list) == len(source_list) == len(mask_list), "Number of images not match"
    
    # ref_img.save(os.path.join(output_root, ref_list[0]))
    # If use test view as ref, start from 1
    for i in tqdm(range(0, num_image)):
        source_img = Image.open(os.path.join(source_root, source_list[i]))
        mask_img = Image.open(os.path.join(mask_root, mask_list[i]))
        source = {"image": source_img, "mask": mask_img}
        result = predict(source, ref_img, ddim_steps=50, num_samples=1, scale=scale, seed=random.randint(0, 147483647), strength=strength, eta=eta, use_ddim_inversion=use_ddim_inversion) # strength=None(No SD Edit)
        
        mask_img_np = np.array(mask_img)
        result_img_np = np.array(result[0])
        source_img_np = np.array(source_img)
        result_img_np[mask_img_np == 0] = source_img_np[mask_img_np == 0]
        result_img = Image.fromarray(result_img_np)
        if ref_list[i].split('.')[0] == os.path.basename(ref_img_path).split('.')[0]:
            ref_img.save(os.path.join(output_root, ref_list[i]))
        else:
            result_img.save(os.path.join(output_root, ref_list[i]))
        

if __name__ == "__main__":    
    argparser = configargparse.ArgumentParser()
    argparser.add_argument('--config', is_config_file=True, help='config file path')
    argparser.add_argument('--dataset', '-d', type=str, default='360-USID', help='dataset name', choices=['360-USID', 'Other-360'])
    argparser.add_argument('--scene', '-s', type=str, help='scene name', required=True)
    argparser.add_argument('--strength', type=float, default=0.5, help='strength for sdedit')
    argparser.add_argument('--script', type=str, default='sdedit', help='script to run', choices=['sdedit'])
    argparser.add_argument("--eta", type=float, default=1.0, help="eta for sdedit")
    argparser.add_argument("--scale", type=float, default=2.5, help="scale for sdedit")
    argparser.add_argument("--use_ddim_inversion", '-u', action='store_true', help="use ddim inversion for sdedit")
    args = argparser.parse_args()
    
    dataset_name = args.dataset
    scene_name = args.scene
    reference_index = 99999
    if dataset_name == '360-USID':
        reference_index = -1
    elif dataset_name == 'Other-360':
        reference_index = 0
        
    use_ddim_inversion = args.use_ddim_inversion
    
    
    if args.script == 'sdedit':
        """
        This script is for SDEdit detail enhancement.
        
        Args:
            strength: strength for sdedit
            eta: eta for sdedit
            scale: scale for sdedit
            ref_img_path: path to the reference image
            source_root: path to the source images
            ref_root: path to the images for name matching
            mask_root: path to the mask images (unseen masks)
            output_root: path to the output images (detail enhanced images)
        """ 
        strength = args.strength
        ref_img_path = natsorted(glob(f"data/{dataset_name}/{scene_name}/reference/*"))[reference_index]
        source_root = f"output/{dataset_name}/{scene_name}/train/ours_object_inpaint_init/renders"
        ref_root = f"data/{dataset_name}/{scene_name}/images"
        mask_root = f"data/{dataset_name}/{scene_name}/unseen_masks_dilated"
        output_root = f"data/{dataset_name}/{scene_name}/inpaint"
        
        output_root = output_root
        os.makedirs(output_root, exist_ok=True)
        if strength == 1:
            strength = None # means no sdedit
        
        LeftRefill(ref_img_path, source_root, ref_root, mask_root, output_root, strength=strength, eta=args.eta, scale=args.scale, use_ddim_inversion=use_ddim_inversion)
    else:
        raise ValueError(f"Script {args.script} not found")