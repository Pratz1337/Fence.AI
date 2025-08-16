import torch
import os
import numpy as np
import yaml
from PIL import Image
from torchvision.transforms import CenterCrop, Resize, Compose, InterpolationMode
from scripts.clip.processing import make_normalize
from scripts.clip.fusion import apply_fusion
from scripts.clip.networks import create_architecture, load_weights


def get_config(model_name, weights_dir='/teamspace/studios/this_studio/Guard.ai/backend/scripts/models/clip/weights'):
    with open(os.path.join(weights_dir, model_name, 'config.yaml')) as fid:
        data = yaml.load(fid, Loader=yaml.FullLoader)
    model_path = os.path.join(weights_dir, model_name, data['weights_file'])
    return data['model_name'], model_path, data['arch'], data['norm_type'], data['patch_size']


def analyze_image(image, weights_dir='/teamspace/studios/this_studio/Guard.ai/backend/scripts/models/clip/weights', models_list=None, fusion='soft_or_prob', device='cpu'):
    if models_list is None:
        models_list = os.listdir(weights_dir)
    
    # Initialize dictionaries to store models and transformations
    models_dict = dict()
    transform_dict = dict()
    
    # Load and prepare models
    print("Loading models:")
    for model_name in models_list:
        print(model_name, flush=True)
        _, model_path, arch, norm_type, patch_size = get_config(model_name, weights_dir=weights_dir)

        model = load_weights(create_architecture(arch), model_path)
        model = model.to(device).eval()

        transform = list()
        if patch_size is None:
            print('input none', flush=True)
            transform_key = f'none_{norm_type}'
        elif patch_size == 'Clip224':
            print('input resize: Clip224', flush=True)
            transform.append(Resize(224, interpolation=InterpolationMode.BICUBIC))
            transform.append(CenterCrop((224, 224)))
            transform_key = f'Clip224_{norm_type}'
        elif isinstance(patch_size, tuple) or isinstance(patch_size, list):
            print(f'input resize: {patch_size}', flush=True)
            transform.append(Resize(*patch_size))
            transform.append(CenterCrop(patch_size[0]))
            transform_key = f'res{patch_size[0]}_{norm_type}'
        elif patch_size > 0:
            print(f'input crop: {patch_size}', flush=True)
            transform.append(CenterCrop(patch_size))
            transform_key = f'crop{patch_size}_{norm_type}'
        
        transform.append(make_normalize(norm_type))
        transform = Compose(transform)
        transform_dict[transform_key] = transform
        models_dict[model_name] = (transform_key, model)
        print(flush=True)

    # Process the image
    results = {}
    with torch.no_grad():
        do_models = list(models_dict.keys())
        do_transforms = set([models_dict[_][0] for _ in do_models])
        
        # Prepare transformed images
        transformed_images = {}
        for k in transform_dict:
            transformed_images[k] = transform_dict[k](image.convert('RGB')).unsqueeze(0)

        # Run inference with each model
        for model_name in do_models:
            transform_key, model = models_dict[model_name]
            out_tens = model(transformed_images[transform_key].to(device)).cpu().numpy()

            if out_tens.shape[1] == 1:
                out_tens = out_tens[:, 0]
            elif out_tens.shape[1] == 2:
                out_tens = out_tens[:, 1] - out_tens[:, 0]
            else:
                assert False
            
            if len(out_tens.shape) > 1:
                logit = np.mean(out_tens, (1, 2))[0]
            else:
                logit = out_tens[0]

            results[model_name] = float(logit)
    
    # Apply fusion if specified
    if fusion is not None and len(results) > 1:
        model_results = np.array([results[model] for model in do_models])
        results['fusion'] = float(apply_fusion(model_results.reshape(1, -1), fusion, axis=-1)[0])
    
    return results