import os
import random
import argparse
import numpy as np
from PIL import Image, ImageOps
import random 
import json
import torch
from diffusers import CogVideoXDPMScheduler
import datetime
import subprocess

import insightface
from insightface.app import FaceAnalysis
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from diffusers.training_utils import free_memory
from diffusers.utils import export_to_video, load_image, load_video

from models.utils import process_face_embeddings_split, process_face_embeddings, process_face_embeddings_infer, get_af_matrix_infer
from models.transformer import BindyouravatarTransformer3DModel
from models.pipeline_bindyouravatar import BindyouravatarPipeline
from models.eva_clip import create_model_and_transforms
from models.eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from models.eva_clip.utils_qformer import resize_numpy_image_long
from peft import PeftMixedModel
from contextlib import redirect_stdout
from util.utils import merge_audio_video,get_routing_logits_from_tracking_mask_results,load_mixed_lora_weights

def get_random_seed():
    return random.randint(0, 2**32 - 1)

def generate_video(
    prompt: str,
    model_path: str,
    output_path: str = "./output",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    img_file_path: str = None,
    audio_model_path= None,
    face_model_path= None,
    audio_emb_path= None,
    audio_file= None,
    num_frames= 49,
    in_channels= 48,
    is_zero_audio_emb= False,
    lora_paths: list = None,
    router_path: str = None,
    lora_rank: int = 128,
    is_load_face: bool = True,
    speaker_pos: str = "left",
    resume_from_consisid: bool = False,
    draw_routing_logits: bool = False,
    draw_routing_logits_suffix: str = "default",
    draw_routing_logits_video_save_dir: str = None,
    is_only_load_transformer: bool = False,
    transformer_path: str = None,
    zero2cond_cfg_flag: bool = False,
    draw_routing_logits_use_softmax: bool = True,
    debug_routing_logits: bool = False,
    debug_routing_logits_zeros: bool = False,
    debug_routing_logits_ones: bool = False,
    is_teacher_forcing: bool = False,
    two_stage_generate: bool = False,
    img_bg_file_path: str = None,
    log_file_path: str = "logs/infer_load_model.log",
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - model_path (str): The path of the pre-trained model to be used.
    - lora_paths (list): The paths of the LoRA weights to be used.
    - lora_rank (int): The rank of the LoRA weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - seed (int): The seed for reproducibility.
    """
    # config
    if not os.path.exists(output_path): 
        os.makedirs(output_path, exist_ok=True)
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir): 
        os.makedirs(log_dir, exist_ok=True)
    with open(log_file_path, 'w') as f:
        f.write("")  
    file_count = len([f for f in os.listdir(output_path) if os.path.isfile(os.path.join(output_path, f))])
    filename = f"{output_path}/{seed}_{file_count:04d}.mp4" 
    device = "cuda"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    is_train_audio= True if audio_model_path and audio_emb_path else False

    # 0. load base models
    
    if os.path.exists(os.path.join(model_path, "transformer_ema")):
        subfolder = "transformer_ema"
    else:
        subfolder = "transformer"
        
    transformer_additional_kwargs={
        'torch_dtype': dtype,
        'is_train_audio': is_train_audio,
        'in_channels': in_channels,
        'draw_routing_logits': draw_routing_logits,
        'draw_routing_logits_suffix': draw_routing_logits_suffix+f"_{file_count:04d}",
        'draw_routing_logits_video_save_dir': filename+"_mask" if draw_routing_logits_video_save_dir is None else draw_routing_logits_video_save_dir,
        'draw_routing_logits_use_softmax': draw_routing_logits_use_softmax,
        'debug_routing_logits': debug_routing_logits,
        'debug_routing_logits_zeros': debug_routing_logits_zeros,
        'debug_routing_logits_ones': debug_routing_logits_ones,
        'is_teacher_forcing': is_teacher_forcing,
    }
    
    with open(log_file_path, 'a') as f:
        with redirect_stdout(f):
            print(f"--------------------------------start loading transformer--------------------------------")
            # transformer_path = model_path if transformer_path is None else transformer_path
            print(f"from pretrained transformer_path: {model_path}")
            transformer = BindyouravatarTransformer3DModel.from_pretrained_cus(model_path, subfolder=subfolder,transformer_additional_kwargs=transformer_additional_kwargs,)
    
    # 0.2 load audio model
    if is_train_audio:
        
        if not is_only_load_transformer:
            with open(log_file_path, 'a') as f:
                with redirect_stdout(f):
                    print(f"--------------------------------start loading audio_modules--------------------------------")
                    if resume_from_consisid:
                        transformer.load_audio_modules(audio_model_path, strict=False)
                    else:
                        transformer.load_audio_modules(audio_model_path,strict=False)
                    print(f"--------------------------------complete loading audio_modules--------------------------------")
        if len(audio_emb_path) == 1:
            audio_embs=torch.load(audio_emb_path)[[i for i in range(num_frames+4)]].unsqueeze(0).to(device, dtype=dtype) #(bs=1,f,12,768)
            if is_zero_audio_emb:
                audio_embs=torch.zeros_like(audio_embs).to(device, dtype=dtype)
            af_matrix = get_af_matrix_infer(speaker_pos).to(device, dtype=dtype).unsqueeze(0)
        elif len(audio_emb_path) == 2:
            left_path = audio_emb_path[0]
            right_path = audio_emb_path[1]
            try:
                left_audio_embs=torch.load(left_path)[[i for i in range(num_frames+4)]].unsqueeze(0).to(device, dtype=dtype) #(num_id=1,f,12,768)
                right_audio_embs=torch.load(right_path)[[i for i in range(num_frames+4)]].unsqueeze(0).to(device, dtype=dtype) #(num_id=1,f,12,768)
            except:
                left_audio_embs=torch.load(left_path)[:].unsqueeze(0).to(device, dtype=dtype) #(num_id=1,f,12,768)
                right_audio_embs=torch.load(right_path)[:].unsqueeze(0).to(device, dtype=dtype) #(num_id=1,f,12,768)
                assert left_audio_embs.shape[0] <= num_frames+4 and right_audio_embs.shape[0] <= num_frames+4
                num_frames_to_pad = num_frames+4-left_audio_embs.shape[1]
                num_frames_to_pad_right = num_frames+4-right_audio_embs.shape[1]
                left_audio_embs = torch.cat([left_audio_embs,torch.zeros(1,num_frames_to_pad,12,768).to(device, dtype=dtype)],dim=1)
                right_audio_embs = torch.cat([right_audio_embs,torch.zeros(1,num_frames_to_pad_right,12,768).to(device, dtype=dtype)],dim=1)
            audio_embs = torch.cat([left_audio_embs, right_audio_embs], dim=0).unsqueeze(0) #(bs=1,2,f,12,768)
            af_matrix = get_af_matrix_infer(speaker_pos).to(device, dtype=dtype).unsqueeze(0)
        else:
            raise ValueError("audio_emb_path must be a list of length 1 or 2")
    else:
        audio_embs=None
        af_matrix = None


    # 0.3 load face model
    if is_load_face and not is_only_load_transformer:
        if face_model_path is None:
            face_model_path=os.path.join(os.path.dirname(audio_model_path), "face_modules.pt")
        with open(log_file_path, 'a') as f:
            with redirect_stdout(f):
                print(f"--------------------------------start loading face_modules--------------------------------")
                print(f"face_model_path: {face_model_path}")
                if face_model_path:
                    if resume_from_consisid:
                        transformer.load_face_modules(face_model_path, strict=False)
                    else:
                        transformer.load_face_modules(face_model_path,strict=False)
                else:
                    raise ValueError("face_model_path is None")
                print(f"--------------------------------complete loading face_modules--------------------------------")

    # 0.4 load router weights
    if router_path:
        with open(log_file_path, 'a') as f:
            with redirect_stdout(f):
                transformer.load_router_modules(router_path)
                

    # 0.5 load LoRA weights
    flag_lora_ready = False
    if lora_paths and not is_only_load_transformer:
        transformer = load_mixed_lora_weights(transformer, lora_paths, lora_rank, log_file_path)
        flag_lora_ready = True
    transformer.eval()
    scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

    # 0.5 load transformer weights again(for lora and router only)
    # if transformer_path:
    #     transformer = load_lora_from_transformer_safetensors(transformer, transformer_path, lora_rank,flag_lora_ready,log_file_path)
    #     transformer = load_router_from_transformer_safetensors(transformer, transformer_path,log_file_path)

    # 0.6 load transformer weights again(for inpaint base model)
    if transformer_path:
        with open(log_file_path, 'a') as f:
            with redirect_stdout(f):
                print(f"--------------------------------start loading transformer--------------------------------")
                index_file = os.path.join(transformer_path,"transformer", "diffusion_pytorch_model.safetensors.index.json")

                with open(index_file, "r") as f:
                    index = json.load(f)

                weight_map = index["weight_map"]
                all_shards = set(weight_map.values())

                tensors_checkpoint = {}  
                from safetensors.torch import load_file
                for shard in all_shards:
                    shard_path = os.path.join(transformer_path,"transformer", shard)
                    shard_tensors = load_file(shard_path)
                    tensors_checkpoint.update(shard_tensors)
                missing_keys, unexpected_keys = transformer.load_state_dict(tensors_checkpoint,strict=False)
                print(f"len of missing_keys: {len(missing_keys)}, len of unexpected_keys: {len(unexpected_keys)}")
                print(f"missing_keys: {missing_keys}")
                print(f"unexpected_keys: {unexpected_keys}")
                print(f"--------------------------------complete loading transformer--------------------------------")

    try:
        is_kps = transformer.config.is_kps
    except:
        is_kps = False

    # 1. load face helper models
    face_helper = FaceRestoreHelper(
        upscale_factor=1,
        face_size=512,
        crop_ratio=(1, 1),
        det_model='retinaface_resnet50',
        save_ext='png',
        device=device,
        model_rootpath=os.path.join(model_path, "face_encoder")
    )
    face_helper.face_parse = None
    face_helper.face_parse = init_parsing_model(model_name='bisenet', device=device, model_rootpath=os.path.join(model_path, "face_encoder"))
    face_helper.face_det.eval()
    face_helper.face_parse.eval()

    model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', os.path.join(model_path, "face_encoder", "EVA02_CLIP_L_336_psz14_s6B.pt"), force_custom_clip=True)
    face_clip_model = model.visual
    face_clip_model.eval()

    eva_transform_mean = getattr(face_clip_model, 'image_mean', OPENAI_DATASET_MEAN)
    eva_transform_std = getattr(face_clip_model, 'image_std', OPENAI_DATASET_STD)
    if not isinstance(eva_transform_mean, (list, tuple)):
        eva_transform_mean = (eva_transform_mean,) * 3
    if not isinstance(eva_transform_std, (list, tuple)):
        eva_transform_std = (eva_transform_std,) * 3
    eva_transform_mean = eva_transform_mean
    eva_transform_std = eva_transform_std

    face_main_model = FaceAnalysis(name='antelopev2', root=os.path.join(model_path, "face_encoder"), providers=['CUDAExecutionProvider'])
    handler_ante = insightface.model_zoo.get_model(f'{model_path}/face_encoder/models/antelopev2/glintr100.onnx', providers=['CUDAExecutionProvider'])
    face_main_model.prepare(ctx_id=0, det_size=(640, 640))
    handler_ante.prepare(ctx_id=0)
        
    face_clip_model.to(device, dtype=dtype)
    face_helper.face_det.to(device)
    face_helper.face_parse.to(device)
    transformer.to(device, dtype=dtype)
    free_memory()
    
    pipe = BindyouravatarPipeline.from_pretrained(model_path, transformer=transformer, scheduler=scheduler, torch_dtype=dtype)
    pipe.fuse_lora(lora_scale=1 / lora_rank)

    # 2. Set Scheduler.
    scheduler_args = {}
    if "variance_type" in pipe.scheduler.config:
        variance_type = pipe.scheduler.config.variance_type
        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"
        scheduler_args["variance_type"] = variance_type

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, **scheduler_args)

    # 3. Enable CPU offload for the model.
    pipe.to(device)

    # turn on if you don't have multiple GPUs or enough GPU memory(such as H100) and it will cost more time in inference, it may also reduce the quality
    # pipe.enable_model_cpu_offload()
    # pipe.enable_sequential_cpu_offload()
    # pipe.vae.enable_slicing()
    # pipe.vae.enable_tiling()
    
    # process face data 
    img_file_path_list = img_file_path
        
    print(img_file_path_list)
    print(prompt)

    id_cond_list = []
    id_vit_hidden_list = [] 

    id_image_list = []
    for img_file_path in img_file_path_list:
        id_image = np.array(load_image(image=img_file_path).convert("RGB"))
        id_image = resize_numpy_image_long(id_image, 1024)
        id_image_list.append(id_image)
    id_cond_list, id_vit_hidden_list, align_crop_face_image, face_kps, _ = process_face_embeddings_split(face_helper, face_clip_model, handler_ante, 
                                                                            eva_transform_mean, eva_transform_std, 
                                                                            face_main_model, device, dtype, id_image_list, 
                                                                            original_id_images=id_image_list, is_align_face=True, 
                                                                            cal_uncond=False)
    if img_bg_file_path:
        _, _, image_bg, _ = process_face_embeddings_infer(face_helper, face_clip_model, handler_ante,
                                                                            eva_transform_mean, eva_transform_std,
                                                                            face_main_model, device, dtype,
                                                                            img_bg_file_path, is_align_face=False, skip_embedding=True)
        use_inpaint = True
    else:
        image_bg = align_crop_face_image
        use_inpaint = False
    if is_kps:
        kps_cond = face_kps
    else:
        kps_cond = None
    print("kps_cond: ", kps_cond, "align_face: ", align_crop_face_image.size(), )
    print("id_cond: ", len(id_cond_list), ) 

    tensor = align_crop_face_image.cpu().detach()
    tensor = tensor.squeeze()
    tensor = tensor.permute(1, 2, 0)
    tensor = tensor.numpy() * 255
    tensor = tensor.astype(np.uint8)
    image = ImageOps.exif_transpose(Image.fromarray(tensor))

    prompt = prompt.strip('"')
    
    generator = torch.Generator(device).manual_seed(seed) if seed else None
    
    
    with torch.no_grad(): 
        video_generate = pipe(
            prompt=prompt,
            image=image,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            use_dynamic_cfg=False,
            guidance_scale=guidance_scale,
            generator=generator,
            id_vit_hidden=id_vit_hidden_list,
            id_cond=id_cond_list,
            kps_cond=kps_cond,
            audio_embs=audio_embs,
            af_matrix=af_matrix,
            zero2cond_cfg_flag=zero2cond_cfg_flag,
            routing_logits_zeros_flag=True if two_stage_generate else False,
            image_bg=image_bg,
            use_inpaint=use_inpaint,
        ).frames[0]
    export_to_video(video_generate, filename, fps=25)
    if two_stage_generate:
        video_path = filename
        output_path = os.path.dirname(video_path)
        
        sam2_cmd = [
            "python", 
            "tools/sam2_tools.py",
            "--video_folder", video_path,
            "--output_path", output_path
        ]
        
        try:
            subprocess.run(sam2_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"SAM2 processing failed: {e}")
            raise e
        

        tracking_mask_results_dir = os.path.join(output_path,os.path.basename(video_path).split(".")[0] ,"tracking_mask_results")
        routing_logits = get_routing_logits_from_tracking_mask_results(tracking_mask_results_dir,is_draw_video=True,video_save_dir=os.path.splitext(filename)[0] + "_sam2.mp4")
        routing_logits = routing_logits.to(device, dtype=dtype)

        pipe.to(device)
        video_generate = pipe(
            prompt=prompt,
            image=image,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            use_dynamic_cfg=False,
            guidance_scale=guidance_scale,
            generator=generator,
            id_vit_hidden=id_vit_hidden_list,
            id_cond=id_cond_list,
            kps_cond=kps_cond,
            audio_embs=audio_embs,
            af_matrix=af_matrix,
            zero2cond_cfg_flag=zero2cond_cfg_flag,
            routing_logits_zeros_flag=False,
            routing_logits_forcing=routing_logits,
            image_bg=image_bg,
            use_inpaint=use_inpaint,
        ).frames[0]
        os.rename(filename, os.path.splitext(filename)[0] + "_ori.mp4")

        export_to_video(video_generate, filename, fps=25)
    

    # 5. Export the generated frames to a video file. 

    log_filename = os.path.splitext(filename)[0] + '.log'
    with open("logs/infer_result.log", 'a') as f1, open(log_filename, 'a') as f2:
        with redirect_stdout(f1):
            print(f"================================================")
            print(f"current time: {datetime.datetime.now()}")
            print(f"dir: {filename}")
            print(f"audio_model_path: {audio_model_path}")
            print(f"lora_paths: {lora_paths}")
            print(f"prompt: {prompt}")
            print(f"audio_emb_path: {audio_emb_path}")
            print(f"img_file_path: {img_file_path_list}")
            print(f"speaker_pos: {speaker_pos}")
            print(f"seed: {seed}")
            print(f"guidance_scale: {guidance_scale}")
            print(f"================================================")
        with redirect_stdout(f2):
            print(f"================================================")
            print(f"current time: {datetime.datetime.now()}")
            print(f"dir: {filename}")
            print(f"audio_model_path: {audio_model_path}")
            print(f"lora_paths: {lora_paths}")
            print(f"prompt: {prompt}")
            print(f"audio_emb_path: {audio_emb_path}")
            print(f"img_file_path: {img_file_path_list}")
            print(f"speaker_pos: {speaker_pos}")
            print(f"seed: {seed}")
            print(f"guidance_scale: {guidance_scale}")
            print(f"================================================")

    # append audio to video
    if audio_file:
        time_to_skip_audio = 0.08
        time_to_skip_video = 0
        output_file = os.path.splitext(filename)[0] + "_output.mp4"
        merge_audio_video(audio_file, filename, output_file, time_to_skip_audio=time_to_skip_audio,time_to_skip_video=time_to_skip_video, skip_first_frame=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a multimodal input")
    
    # ==================== Model Path Arguments ====================
    parser.add_argument("--model_path", type=str, default="pretrained", help="Path to the pre-trained model")
    parser.add_argument("--transformer_path", type=str, default="", help="Path to the transformer model")
    
    # ==================== Input Arguments ====================
    parser.add_argument("--img_file_path", nargs='+', default=['asserts/0.jpg', 'asserts/1.jpg'], help="List of input image file paths")
    parser.add_argument("--img_bg_file_path", type=str, default="", help="Path to the background image file")
    parser.add_argument("--prompt", type=str, default="Two men in half bodies, are seated in a dimly lit room, possibly an office or meeting room, with a formal atmosphere , they are engaged in a conversation.", help="Text description for video generation")
    
    # ==================== Output Arguments ====================
    parser.add_argument("--output_path", type=str, default="./results", help="Path where the generated video will be saved")
    parser.add_argument("--log_file_path", type=str, default="logs/infer_load_model.log", help="Path for the model loading log file")
    
    # ==================== Generation Control Arguments ====================
    parser.add_argument("--guidance_scale", type=float, default=7.0, help="Scale for classifier-free guidance")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of steps for the inference process")
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument("--num_frames", type=int, default=49, help="Number of frames in the generated video")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type for computation (float16 or bfloat16)")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for reproducibility")
    
    # ==================== Audio Arguments ====================
    parser.add_argument("--audio_model_path", type=str, default="", help="Path to the audio model")
    parser.add_argument("--audio_emb_path", nargs='+', default=[], help="List of audio embedding file paths")
    parser.add_argument("--audio_file", type=str, default="", help="Path to the audio file")
    parser.add_argument("--is_zero_audio_emb", action='store_true', help="Whether to set audio embeddings to zero")
    parser.add_argument("--speaker_pos", type=str, default="left", help="Speaker position (left/right)")
    
    # ==================== Face Arguments ====================
    parser.add_argument("--face_model_path", type=str, default="", help="Path to the face model")
    parser.add_argument("--no_load_face", action='store_true', help="Whether to skip loading face model")
    
    # ==================== LoRA and Router Arguments ====================
    parser.add_argument("--lora_paths", type=str, nargs='+', default=[], help="List of LoRA weight file paths")
    parser.add_argument("--router_path", type=str, default="", help="Path to the router model")
    parser.add_argument("--is_only_load_transformer", action='store_true', help="Whether to load only transformer, not separate pt modules")
    
    # ==================== Advanced Generation Arguments ====================
    parser.add_argument("--zero2cond_cfg_flag", action='store_true', help="Whether to set the first element of cfg to 0")
    parser.add_argument("--two_stage_generate", action='store_true', help="Whether to perform two-stage generation")
    args = parser.parse_args()
    assert len(args.img_file_path) == 2 

    generate_video(
        prompt=args.prompt,
        model_path=args.model_path,
        output_path=args.output_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=torch.float16 if args.dtype == "float16" else torch.bfloat16,
        seed=args.seed,
        img_file_path=args.img_file_path,
        img_bg_file_path=args.img_bg_file_path if args.img_bg_file_path != "" else None,
        lora_paths=args.lora_paths if args.lora_paths != [] else None,
        router_path=args.router_path if args.router_path != "" else None,
        audio_model_path=args.audio_model_path if args.audio_model_path != "" else None,
        face_model_path=args.face_model_path if args.face_model_path != "" else None,
        audio_emb_path=args.audio_emb_path if args.audio_emb_path != [] else None,
        audio_file=args.audio_file if args.audio_file != "" else None,
        num_frames=args.num_frames,
        is_zero_audio_emb=args.is_zero_audio_emb,
        is_load_face=not args.no_load_face,
        speaker_pos=args.speaker_pos,
        is_only_load_transformer=args.is_only_load_transformer,
        transformer_path=args.transformer_path if args.transformer_path != "" else None,
        zero2cond_cfg_flag=args.zero2cond_cfg_flag,
        two_stage_generate=args.two_stage_generate,
        log_file_path=args.log_file_path
    )
