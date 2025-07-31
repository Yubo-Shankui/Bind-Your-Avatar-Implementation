import cv2
import math
import numpy as np
from PIL import Image

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize
from transformers import T5EncoderModel, T5Tokenizer
from typing import List, Optional, Tuple, Union
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from diffusers.utils import load_image
from PIL import Image, ImageOps

def tensor_to_pil(src_img_tensor):
    img = src_img_tensor.clone().detach()
    if img.dtype == torch.bfloat16:
        img = img.to(torch.float32)
    img = img.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img.astype(np.uint8)
    pil_image = Image.fromarray(img)
    return pil_image
    

def _get_t5_prompt_embeds(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("`text_input_ids` must be provided when the tokenizer is not specified.")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds


def encode_prompt(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_embeds = _get_t5_prompt_embeds(
        tokenizer,
        text_encoder,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
        text_input_ids=text_input_ids,
    )
    return prompt_embeds


def compute_prompt_embeddings(
    tokenizer, text_encoder, prompt, max_sequence_length, device, dtype, requires_grad: bool = False
):
    if requires_grad:
        prompt_embeds = encode_prompt(
            tokenizer,
            text_encoder,
            prompt,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
    else:
        with torch.no_grad():
            prompt_embeds = encode_prompt(
                tokenizer,
                text_encoder,
                prompt,
                num_videos_per_prompt=1,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
    return prompt_embeds



def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    patch_size_t: int = None,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
    base_height: int = 480,
    base_width: int = 720,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

    if patch_size_t is None:
        # Version 1.0
        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
        )
    else:
        # Version 1.5
        base_num_frames = (num_frames + patch_size_t - 1) // patch_size_t

        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(base_size_height, base_size_width),
        )

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin



def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    return _totensor(imgs, bgr2rgb, float32)


def to_gray(img):
    x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    x = x.repeat(1, 3, 1, 1)
    return x


def draw_kps(image_pil, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil



def process_face_embeddings_split(face_helper, clip_vision_model, handler_ante, eva_transform_mean, eva_transform_std, app, device, weight_dtype, images, original_id_images=None, is_align_face=True, cal_uncond=False,skip_bg_removal=False,skip_embedding=False):
    """ 
    only for inference with image set.

    Args:
        images: list of numpy rgb image, range [0, 255]
    """ 
    id_cond_list = []
    id_vit_hidden_list = []  
    new_img = Image.new("RGB", (720, 480), color=(255, 255, 255))  
    processed_faces = []  # Store processed face images

    i = 0
    for image in images: 
        face_helper.clean_all()
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # (724, 502, 3)
        
        height, width, _ = image.shape 
        face_helper.read_image(image_bgr)
        face_helper.get_face_landmarks_5(only_center_face=False)
        if len(face_helper.all_landmarks_5) < 1:
            return None, None, None, None, None

        id_ante_embedding = None 
        face_kps = None  

        if face_kps is None:
            face_kps = face_helper.all_landmarks_5[0]
        face_helper.align_warp_face()
        
        if len(face_helper.cropped_faces) < 1:
            print("facexlib align face fail")
            return None, None, None, None, None
            # raise RuntimeError('facexlib align face fail')
        align_face = face_helper.cropped_faces[0]  # (512, 512, 3)  # RGB

        # incase insightface didn't detect face
        if id_ante_embedding is None:
            # print('fail to detect face using insightface, extract embedding on align face')
            id_ante_embedding = handler_ante.get_feat(align_face)

        id_ante_embedding = torch.from_numpy(id_ante_embedding).to(device, weight_dtype)  # torch.Size([512])
        
        if id_ante_embedding.ndim == 1:
            id_ante_embedding = id_ante_embedding.unsqueeze(0)  # torch.Size([1, 512])

        # parsing
        if is_align_face:
            input = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0  # torch.Size([1, 3, 512, 512])
            input = input.to(device)
            # return_face_features_image = return_face_features_image_2 = input 
            parsing_out = face_helper.face_parse(normalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
            parsing_out = parsing_out.argmax(dim=1, keepdim=True)  # torch.Size([1, 1, 512, 512])
            bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
            bg = sum(parsing_out == i for i in bg_label).bool()
            white_image = torch.ones_like(input)  # torch.Size([1, 3, 512, 512])
            # only keep the face features
            return_face_features_image = torch.where(bg, white_image, to_gray(input))  # torch.Size([1, 3, 512, 512])
            return_face_features_image_2 = torch.where(bg, white_image, input)  # torch.Size([1, 3, 512, 512])

            return_face_features_image_2 = return_face_features_image_2 * 255.0

            
            # Convert the processed image to a PIL image for visualization
            processed_face = tensor_to_pil(return_face_features_image_2[0]) 
            processed_faces.append(processed_face)
        else:
            original_image_bgr = cv2.cvtColor(original_id_images[i], cv2.COLOR_RGB2BGR)
            input = img2tensor(original_image_bgr, bgr2rgb=True).unsqueeze(0) / 255.0  # torch.Size([1, 3, 512, 512])
            input = input.to(device)
            return_face_features_image = return_face_features_image_2 = input
        
        if not skip_embedding:
            # transform img before sending to eva-clip-vit 
            face_features_image = resize(return_face_features_image, clip_vision_model.image_size,
                                        InterpolationMode.BICUBIC)  # torch.Size([1, 3, 336, 336])
            face_features_image = normalize(face_features_image, eva_transform_mean, eva_transform_std)
            id_cond_vit, id_vit_hidden = clip_vision_model(face_features_image.to(weight_dtype), return_all_features=False, return_hidden=True, shuffle=False)  # torch.Size([1, 768]),  list(torch.Size([1, 577, 1024]))
            id_cond_vit_norm = torch.norm(id_cond_vit, 2, 1, True)
            id_cond_vit = torch.div(id_cond_vit, id_cond_vit_norm)

            id_cond = torch.cat([id_ante_embedding, id_cond_vit], dim=-1)  # torch.Size([1, 512]), torch.Size([1, 768])  ->  torch.Size([1, 1280]) 
        else:
            id_cond = id_vit_hidden = None
        id_cond_list.append(id_cond)
        id_vit_hidden_list.append(id_vit_hidden)

        # Use the processed images for visualization
        img = processed_faces[-1] if processed_faces else Image.fromarray(align_face, 'RGB')
        img = img.resize((360, 360)) 
        new_img.paste(img, (360*i, 60)) 
        i += 1 
    
    # If the ./tests/paste_img directory exists, save new_img
    import os
    if os.path.exists('./tests/paste_img'):
        new_img.save('./tests/paste_img/new_img.png')

    new_img = np.array(new_img)
    new_img = img2tensor(new_img, bgr2rgb=False).unsqueeze(0) / 255.0 

    
    return id_cond_list, id_vit_hidden_list, new_img, face_kps, None  #return_face_features_image_2, face_kps    # torch.Size([1, 1280]), list(torch.Size([1, 577, 1024])) 



def process_face_embeddings(
    face_helper_1,
    clip_vision_model,
    face_helper_2,
    eva_transform_mean,
    eva_transform_std,
    app,
    device,
    weight_dtype,
    image,
    original_id_image=None,
    is_align_face=True,
    skip_bg_removal=False,
    skip_embedding=False,
):
    """
    Process face embeddings from an image, extracting relevant features such as face embeddings, landmarks, and parsed
    face features using a series of face detection and alignment tools.

    Args:
        face_helper_1: Face helper object (first helper) for alignment and landmark detection.
        clip_vision_model: Pre-trained CLIP vision model used for feature extraction.
        face_helper_2: Face helper object (second helper) for embedding extraction.
        eva_transform_mean: Mean values for image normalization before passing to EVA model.
        eva_transform_std: Standard deviation values for image normalization before passing to EVA model.
        app: Application instance used for face detection.
        device: Device (CPU or GPU) where the computations will be performed.
        weight_dtype: Data type of the weights for precision (e.g., `torch.float32`).
        image: Input image in RGB format with pixel values in the range [0, 255].
        original_id_image: (Optional) Original image for feature extraction if `is_align_face` is False.
        is_align_face: Boolean flag indicating whether face alignment should be performed.
        skip_bg_removal: Boolean flag indicating whether to skip background removal for the returned face features image.

    Returns:
        Tuple:
            - id_cond: Concatenated tensor of Ante face embedding and CLIP vision embedding
            - id_vit_hidden: Hidden state of the CLIP vision model, a list of tensors.
            - return_face_features_image_2: Processed face features image after normalization and parsing.
            - face_kps: Keypoints of the face detected in the image.
    """

    face_helper_1.clean_all()
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # get antelopev2 embedding
    face_info = app.get(image_bgr)
    if len(face_info) > 0:
        face_info = sorted(face_info, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))[
            -1
        ]  # only use the maximum face
        id_ante_embedding = face_info["embedding"]  # (512,)
        face_kps = face_info["kps"]
    else:
        id_ante_embedding = None
        face_kps = None

    # using facexlib to detect and align face
    face_helper_1.read_image(image_bgr)
    face_helper_1.get_face_landmarks_5(only_center_face=True)
    if face_kps is None:
        face_kps = face_helper_1.all_landmarks_5[0]
    face_helper_1.align_warp_face()
    if len(face_helper_1.cropped_faces) == 0:
        raise RuntimeError("facexlib align face fail")
    align_face = face_helper_1.cropped_faces[0]  # (512, 512, 3)  # RGB

    # incase insightface didn't detect face
    if id_ante_embedding is None:
        # print("fail to detect face using insightface, extract embedding on align face")
        id_ante_embedding = face_helper_2.get_feat(align_face)

    id_ante_embedding = torch.from_numpy(id_ante_embedding).to(device, weight_dtype)  # torch.Size([512])
    if id_ante_embedding.ndim == 1:
        id_ante_embedding = id_ante_embedding.unsqueeze(0)  # torch.Size([1, 512])

    # parsing
    if is_align_face:
        input = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0  # torch.Size([1, 3, 512, 512])
        input = input.to(device)
        
        # Always generate a version with background removal for feature extraction
        parsing_out = face_helper_1.face_parse(normalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
        parsing_out = parsing_out.argmax(dim=1, keepdim=True)  # torch.Size([1, 1, 512, 512])
        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        bg = sum(parsing_out == i for i in bg_label).bool()
        white_image = torch.ones_like(input)  # torch.Size([1, 3, 512, 512])
        # only keep the face features
        return_face_features_image = torch.where(bg, white_image, to_gray(input))  # torch.Size([1, 3, 512, 512])
        
        # Decide whether to remove the background of the returned image based on skip_bg_removal
        if skip_bg_removal:
            return_face_features_image_2 = input  # Do not remove background, use the input image directly
        else:
            return_face_features_image_2 = torch.where(bg, white_image, input)  # Remove background
    else:
        original_image_bgr = cv2.cvtColor(original_id_image, cv2.COLOR_RGB2BGR)
        input = img2tensor(original_image_bgr, bgr2rgb=True).unsqueeze(0) / 255.0  # torch.Size([1, 3, 512, 512])
        input = input.to(device)
        return_face_features_image = return_face_features_image_2 = input

    if not skip_embedding:
            # transform img before sending to eva-clip-vit
        face_features_image = resize(
            return_face_features_image, clip_vision_model.image_size, InterpolationMode.BICUBIC
        )  # torch.Size([1, 3, 336, 336])
        face_features_image = normalize(face_features_image, eva_transform_mean, eva_transform_std)
        id_cond_vit, id_vit_hidden = clip_vision_model(
            face_features_image.to(weight_dtype), return_all_features=False, return_hidden=True, shuffle=False
        )  # torch.Size([1, 768]),  list(torch.Size([1, 577, 1024]))
        id_cond_vit_norm = torch.norm(id_cond_vit, 2, 1, True)
        id_cond_vit = torch.div(id_cond_vit, id_cond_vit_norm)

        id_cond = torch.cat(
            [id_ante_embedding, id_cond_vit], dim=-1
        )  # torch.Size([1, 512]), torch.Size([1, 768])  ->  torch.Size([1, 1280])
    else:
        id_cond = id_vit_hidden = None


    # draw image_2
    if skip_bg_removal:
        processed_face = tensor_to_pil(return_face_features_image_2.squeeze(0) * 255.0) 
        import os
        if os.path.exists('./tests/paste_img'):
            processed_face.save('./tests/paste_img/processed_face.png')

    return (
        id_cond,
        id_vit_hidden,
        return_face_features_image_2,
        face_kps,
    )  # torch.Size([1, 1280]), list(torch.Size([1, 577, 1024]))


def process_face_embeddings_infer(
    face_helper_1,
    clip_vision_model,
    face_helper_2,
    eva_transform_mean,
    eva_transform_std,
    app,
    device,
    weight_dtype,
    img_file_path,
    is_align_face=True,
    skip_bg_removal=False,
    skip_embedding=False,
):
    """
    Process face embeddings from an input image for inference, including alignment, feature extraction, and embedding
    concatenation.

    Args:
        face_helper_1: Face helper object (first helper) for alignment and landmark detection.
        clip_vision_model: Pre-trained CLIP vision model used for feature extraction.
        face_helper_2: Face helper object (second helper) for embedding extraction.
        eva_transform_mean: Mean values for image normalization before passing to EVA model.
        eva_transform_std: Standard deviation values for image normalization before passing to EVA model.
        app: Application instance used for face detection.
        device: Device (CPU or GPU) where the computations will be performed.
        weight_dtype: Data type of the weights for precision (e.g., `torch.float32`).
        img_file_path: Path to the input image file (string) or a numpy array representing an image.
        is_align_face: Boolean flag indicating whether face alignment should be performed (default: True).
        skip_bg_removal: Boolean flag indicating whether to skip background removal for the returned face features image.

    Returns:
        Tuple:
            - id_cond: Concatenated tensor of Ante face embedding and CLIP vision embedding.
            - id_vit_hidden: Hidden state of the CLIP vision model, a list of tensors.
            - image: Processed face image after feature extraction and alignment.
            - face_kps: Keypoints of the face detected in the image.
    """

    # Load and preprocess the input image
    if isinstance(img_file_path, str):
        image = np.array(load_image(image=img_file_path).convert("RGB"))
    else:
        image = np.array(ImageOps.exif_transpose(Image.fromarray(img_file_path)).convert("RGB"))

    # Resize image to ensure the longer side is 1024 pixels
    image = resize_numpy_image_long(image, 1024)
    original_id_image = image

    # Process the image to extract face embeddings and related features
    id_cond, id_vit_hidden, align_crop_face_image, face_kps = process_face_embeddings(
        face_helper_1,
        clip_vision_model,
        face_helper_2,
        eva_transform_mean,
        eva_transform_std,
        app,
        device,
        weight_dtype,
        image,
        original_id_image,
        is_align_face,
        skip_bg_removal,
        skip_embedding,
    )

    # Convert the aligned cropped face image (torch tensor) to a numpy array
    tensor = align_crop_face_image.cpu().detach()
    tensor = tensor.squeeze()
    tensor = tensor.permute(1, 2, 0)
    tensor = tensor.numpy() * 255
    tensor = tensor.astype(np.uint8)
    image = ImageOps.exif_transpose(Image.fromarray(tensor))

    return id_cond, id_vit_hidden, image, face_kps







def resize_numpy_image_long(image, resize_long_edge=768):
    """
    Resize the input image to a specified long edge while maintaining aspect ratio.

    Args:
        image (numpy.ndarray): Input image (H x W x C or H x W).
        resize_long_edge (int): The target size for the long edge of the image. Default is 768.

    Returns:
        numpy.ndarray: Resized image with the long edge matching `resize_long_edge`, while maintaining the aspect
        ratio.
    """

    h, w = image.shape[:2]
    if max(h, w) <= resize_long_edge:
        return image
    k = resize_long_edge / max(h, w)
    h = int(h * k)
    w = int(w * k)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return image

def cfg_id_vit_hidden(id_vit_hidden, zero2cond_cfg_flag=False):
    if id_vit_hidden is not None:
        # Process two-level list
        id_vit_hidden_cfg = []
        for outer_list in id_vit_hidden:  # The first level of the list
            outer_list_cfg = []
            for tensor in outer_list:  # The second level of the list
                # Duplicate tensor to match batch size after cfg
                tensor_cfg = torch.cat([tensor if not zero2cond_cfg_flag else torch.zeros_like(tensor), tensor], dim=0)
                outer_list_cfg.append(tensor_cfg)
            id_vit_hidden_cfg.append(outer_list_cfg)
        id_vit_hidden = id_vit_hidden_cfg
    else:
        raise ValueError("id_vit_hidden is None")
    return id_vit_hidden

def cfg_id_cond(id_cond, zero2cond_cfg_flag=False):
    if id_cond is not None:
        # Process the list of tensors
        id_cond_cfg = []
        for tensor in id_cond:  # Each tensor in the list
            # Duplicate tensor to match batch size after cfg
            tensor_cfg = torch.cat([tensor if not zero2cond_cfg_flag else torch.zeros_like(tensor), tensor], dim=0)
            id_cond_cfg.append(tensor_cfg)
        id_cond = id_cond_cfg
    else:
        raise ValueError("id_cond is None")
    return id_cond


def get_af_matrix_infer(speaker_pos):
    # Create a 2x2 audio-face fusion matrix. 
    # It's an identity matrix if speaker_pos is "left", and 1 - identity if "right".
    if speaker_pos  == "left":
        af_matrix = torch.eye(2)
    elif speaker_pos == "right":
        af_matrix = 1 - torch.eye(2)
    else:
        raise ValueError("speaker is not left or right")
        
    return af_matrix

def focal_loss(pred, target, alpha=0.5, gamma=2.0, eps=1e-6):
    """
    Calculate Focal Loss
    pred: prediction, range [0,1]
    target: target, range [0,1]
    alpha: class weight factor
    gamma: focusing parameter
    eps: numerical stability parameter
    """
    pred = torch.clamp(pred, eps, 1-eps)
    target = torch.clamp(target, eps, 1-eps)
    
    ce_loss = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)

    pt = pred * target + (1 - pred) * (1 - target)
    focal_weight = (1 - pt) ** gamma
    alpha_weight = alpha * target + (1 - alpha) * (1 - target)
    
    loss = alpha_weight * focal_weight * ce_loss
    
    return loss

def bce_loss(pred, target, alpha=0.5, gamma=2.0, eps=1e-6):
    pred = torch.clamp(pred, eps, 1-eps)
    ce_loss = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
    return ce_loss
