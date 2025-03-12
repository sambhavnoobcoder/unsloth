# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from transformers import (
    BitsAndBytesConfig,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
)
try:
    from transformers import AutoModelForImageTextToText
    AutoModelForVision2Seq = AutoModelForImageTextToText
except:
    from transformers import AutoModelForVision2Seq
pass
from .llama import *
from ..kernels import (
    post_patch_loss_function,
)
from ._utils import __version__
from peft import LoraConfig, TaskType, get_peft_model as _get_peft_model
from transformers import set_seed as transformers_set_seed
from unsloth_zoo.peft_utils import (
    get_peft_regex,
    SKIP_QUANTIZATION_MODULES,
    requires_grad_for_gradient_checkpointing,
)
from triton import __version__ as triton_version
from unsloth_zoo.utils import _get_dtype
from unsloth_zoo.patching_utils import patch_model_and_tokenizer
from unsloth_zoo.training_utils import prepare_model_for_training
import types
import functools
import os
import gc
import platform
import requests
from PIL import Image
import torchvision.transforms.functional as F

__all__ = [
    "FastBaseModel",
]


def process_image_sizes(model_config, max_image_size=None):
    """
    Determine the target image size from config or provided parameter.
    
    Args:
        model_config: The model configuration object
        max_image_size: Optional user-specified max image size (int or tuple)
        
    Returns:
        Tuple of (width, height) for target image size
    """
    # First check if user explicitly provided a size
    if max_image_size is not None:
        if isinstance(max_image_size, int):
            return (max_image_size, max_image_size)
        elif isinstance(max_image_size, (tuple, list)) and len(max_image_size) == 2:
            return tuple(max_image_size)
        else:
            raise ValueError("max_image_size must be an integer or tuple of two integers (width, height)")
    
    # Otherwise try to get from config
    if hasattr(model_config, "vision_config"):
        vision_config = model_config.vision_config
        if hasattr(vision_config, "image_size"):
            if isinstance(vision_config.image_size, int):
                return (vision_config.image_size, vision_config.image_size)
            elif isinstance(vision_config.image_size, (list, tuple)):
                return tuple(vision_config.image_size)
    
    # If we reach here, no size was specified
    return None


def resize_images(images, target_size, keep_aspect_ratio=True):
    """
    Resize images to target size.
    
    Args:
        images: PyTorch tensor of shape (B, C, H, W) or PIL Image
        target_size: Tuple of (width, height)
        keep_aspect_ratio: Whether to preserve aspect ratio
        
    Returns:
        Resized images
    """
    if target_size is None:
        return images
    
    # Handle single PIL image
    if isinstance(images, Image.Image):
        if keep_aspect_ratio:
            # Create a copy to avoid modifying the original
            img_copy = images.copy()
            img_copy.thumbnail(target_size, Image.LANCZOS)
            return img_copy
        else:
            return images.resize(target_size, Image.LANCZOS)
    
    # Handle tensor of images
    if torch.is_tensor(images):
        # Check if this is a flattened tensor (Qwen2-VL specific format)
        if len(images.shape) == 2:
            # For flattened tensors, we need to preserve the second dimension
            # and only resize the first dimension proportionally
            orig_size = images.shape[0]
            target_size_1d = min(orig_size, target_size[0] * target_size[1])
            
            # Create a new tensor with the target size
            resized = torch.zeros((target_size_1d, images.shape[1]), 
                                  dtype=images.dtype, 
                                  device=images.device)
            
            # Copy data from original tensor, truncating if necessary
            copy_size = min(orig_size, target_size_1d)
            resized[:copy_size] = images[:copy_size]
            
            return resized
        
        # Standard image tensor with shape (B, C, H, W)
        elif len(images.shape) == 4:
            b, c, h, w = images.shape
            
            if h <= target_size[1] and w <= target_size[0]:
                # No need to resize if already smaller
                return images
                
            if keep_aspect_ratio:
                # Calculate new dimensions preserving aspect ratio
                aspect_ratio = w / h
                if w > h:
                    new_w = min(w, target_size[0])
                    new_h = int(new_w / aspect_ratio)
                    if new_h > target_size[1]:
                        new_h = target_size[1]
                        new_w = int(new_h * aspect_ratio)
                else:
                    new_h = min(h, target_size[1])
                    new_w = int(new_h * aspect_ratio)
                    if new_w > target_size[0]:
                        new_w = target_size[0]
                        new_h = int(new_w / aspect_ratio)
                
                # Ensure minimum size of 1
                new_w = max(1, new_w)
                new_h = max(1, new_h)
            else:
                new_w, new_h = target_size
            
            return F.resize(images, [new_h, new_w], antialias=True)
    
    return images


def unsloth_base_fast_generate(
    self,
    *args,
    **kwargs,
):
    FastBaseModel.for_inference(self)
    dtype = _get_dtype(self.config.torch_dtype)

    # Check if VLM
    is_vlm = (
        x.endswith(("ForConditionalGeneration", "ForVisionText2Text"))
        for x in self.config.architectures
    )
    is_vlm = is_vlm or hasattr(self.config, "vision_config")

    # Remove token_type_ids
    kwargs.pop("token_type_ids", None)

    # VLMs do not allow logits_to_keep
    if not is_vlm:
        kwargs["logits_to_keep"] = 1
    else:
        kwargs.pop("logits_to_keep", None)
        kwargs.pop("num_logits_to_keep", None)

    # Check pad_token
    model_eos_token_id = getattr(self.config, "eos_token_id", None)
    if model_eos_token_id is not None and hasattr(model_eos_token_id, "__iter__"):
        model_eos_token_id = model_eos_token_id[0]

    kwargs["pad_token_id"] = kwargs.pop("pad_token_id", model_eos_token_id)

    # Resize pixel values if present and we have a target size
    if "pixel_values" in kwargs and hasattr(self, "unsloth_target_image_size") and self.unsloth_target_image_size is not None:
        pixel_values = kwargs["pixel_values"]
        kwargs["pixel_values"] = resize_images(pixel_values, self.unsloth_target_image_size)

    # Look for other potential image tensors (different models use different keys)
    for key in kwargs:
        if any(img_key in key for img_key in ["image", "pixel", "vision"]) and torch.is_tensor(kwargs[key]):
            if hasattr(self, "unsloth_target_image_size") and self.unsloth_target_image_size is not None:
                kwargs[key] = resize_images(kwargs[key], self.unsloth_target_image_size)

    # Convert pixel values to the right dtype
    try: 
        if "pixel_values" in kwargs:
            kwargs["pixel_values"] = kwargs["pixel_values"].to(dtype)
    except: pass

    # Mixed precision autocast
    with torch.inference_mode(), torch.autocast(device_type = "cuda", dtype = dtype):
        output = self._old_generate(*args, **kwargs)
    pass

    FastBaseModel.for_training(self)
    return output


class FastBaseModel:

    @staticmethod
    def from_pretrained(
        model_name        = "unsloth/Llama-3.2-1B-Instruct",
        max_seq_length    = None,
        dtype             = None,
        load_in_4bit      = True,
        load_in_8bit      = False,
        full_finetuning   = False,
        token             = None,
        device_map        = "sequential",
        trust_remote_code = False,
        model_types       = None,
        tokenizer_name    = None,
        auto_model        = AutoModelForVision2Seq,
        max_image_size    = None,
        use_gradient_checkpointing = "unsloth",
        **kwargs,
    ):
        if trust_remote_code:
            print(
                "Unsloth: WARNING `trust_remote_code` is True.\n"\
                "Are you certain you want to do remote code execution?"
            )
        pass
        if token is None: token = get_token()
        SUPPORTS_BFLOAT16 = is_bfloat16_supported()
        gpu_stats = torch.cuda.get_device_properties(0)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

        from importlib.metadata import version as importlib_version
        try:    vllm_version = f" vLLM: {importlib_version('vllm')}."
        except: vllm_version = ""

        statistics = \
           f"==((====))==  Unsloth {__version__}: Fast {model_types[0].title()} patching. Transformers: {transformers_version}.{vllm_version}\n"\
           f"   {chr(92)}{chr(92)}   /|    {gpu_stats.name}. Num GPUs = {torch.cuda.device_count()}. Max memory: {max_memory} GB. Platform: {platform_system}.\n"\
           f"O^O/ {chr(92)}_/ {chr(92)}    Torch: {torch.__version__}. CUDA: {gpu_stats.major}.{gpu_stats.minor}. CUDA Toolkit: {torch.version.cuda}. Triton: {triton_version}\n"\
           f"{chr(92)}        /    Bfloat16 = {str(SUPPORTS_BFLOAT16).upper()}. FA [Xformers = {xformers_version}. FA2 = {HAS_FLASH_ATTENTION}]\n"\
           f' "-____-"     Free license: http://github.com/unslothai/unsloth'
        print(statistics)

        # Warn about fast transfers
        old_hf_transfer = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0")
        if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0") == "1":
            print("Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!")
        pass
        # Return old flag
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = old_hf_transfer
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        get_statistics() # For debugging - we use a download counter to see if environments are not breaking 

        if dtype is None:
            dtype = torch.float16 if not SUPPORTS_BFLOAT16 else torch.bfloat16
        elif dtype == torch.bfloat16 and not SUPPORTS_BFLOAT16:
            logger.warning_once("Device does not support bfloat16. Will change to float16.")
            dtype = torch.float16

        assert(dtype == torch.float16 or dtype == torch.bfloat16 or dtype == torch.float32)

        bnb_config = None
        if full_finetuning and (load_in_4bit or load_in_8bit):
            print("Unsloth: You selected full finetuning support, but 4bit / 8bit is enabled - disabling LoRA / QLoRA.")
            load_in_4bit = False
            load_in_8bit = False
        pass

        if load_in_4bit and load_in_8bit:
            raise RuntimeError("Unsloth: Can only load in 4bit or 8bit, not both!")
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit              = True,
                bnb_4bit_use_double_quant = True,
                bnb_4bit_quant_type       = "nf4",
                bnb_4bit_compute_dtype    = dtype,
                llm_int8_skip_modules     = SKIP_QUANTIZATION_MODULES,
            )
        elif load_in_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit              = True,
                llm_int8_skip_modules     = SKIP_QUANTIZATION_MODULES,
            )
        elif not load_in_4bit and not load_in_8bit and not full_finetuning:
            print("Unsloth: LoRA, QLoRA and full finetuning all not selected. Switching to QLoRA.")
            load_in_4bit = True
            bnb_config = BitsAndBytesConfig(
                load_in_4bit              = True,
                bnb_4bit_use_double_quant = True,
                bnb_4bit_quant_type       = "nf4",
                bnb_4bit_compute_dtype    = dtype,
                llm_int8_skip_modules     = SKIP_QUANTIZATION_MODULES,
            )
        pass

        if full_finetuning:
            os.environ["UNSLOTH_ENABLE_FULL_FINETUNING"] = "1"
            if dtype == torch.bfloat16:
                print("Unsloth: Using bfloat16 full finetuning which cuts memory usage by 50%.")
            else:
                print("Unsloth: Float16 full finetuning uses more memory since we upcast weights to float32.")
        else:
            os.environ["UNSLOTH_ENABLE_FULL_FINETUNING"] = "0"
        pass

        kwargs.pop("attn_implementation", None); # No need since we auto call it

        # Cannot be None, since HF now checks for the config
        if load_in_4bit: kwargs["quantization_config"] = bnb_config

        model = auto_model.from_pretrained(
            model_name,
            device_map              = device_map,
            torch_dtype             = dtype,
            # quantization_config   = bnb_config,
            token                   = token,
            trust_remote_code       = trust_remote_code,
            # attn_implementation   = "sdpa", [TODO] Pixtral for eg fails
            **kwargs,
        )
        # Return old flag
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = old_hf_transfer

        # Counteract saved tokenizers
        tokenizer_name = model_name if tokenizer_name is None else tokenizer_name
        auto_processor = AutoProcessor if auto_model is AutoModelForVision2Seq else AutoTokenizer
        tokenizer = auto_processor.from_pretrained(
            tokenizer_name,
            padding_side = "right",
            token        = token,
        )
        # Add padding side as well
        if hasattr(tokenizer, "tokenizer"):
            tokenizer.tokenizer.padding_side = "right"

        model, tokenizer = patch_tokenizer(model, tokenizer)
        model = post_patch_loss_function(model)
        # Fix other stuff like BnB compute data types
        model, tokenizer = patch_model_and_tokenizer(
            model,
            tokenizer,
            downcast_rope = False,
            fix_embeddings = False,
        )

        # Log Unsloth version for future fastpaths for inference
        if hasattr(model, "config"):
            model.config.update({"unsloth_version" : __version__})
        pass
        patch_saving_functions(model, vision = True)
        patch_saving_functions(tokenizer, vision = True)

        # Fix gradient accumulation
        from transformers.trainer import Trainer
        patch_gradient_accumulation_fix(Trainer)

        # Save tokenizer for inference purposes
        tokenizer.padding_side = "left" # Force inference
        if hasattr(tokenizer, "tokenizer"):
            tokenizer.tokenizer.padding_side = "left" # Force inference
        m = model
        while hasattr(m, "model"):
            m.max_seq_length = max_seq_length
            m._saved_temp_tokenizer = tokenizer
            # Also set is_loaded_in_8bit to disable incorrect DDP
            m.is_loaded_in_8bit = True if not full_finetuning else False
            m = m.model
        pass
        m.max_seq_length = max_seq_length
        m._saved_temp_tokenizer = tokenizer
        # Also set is_loaded_in_8bit to disable incorrect DDP
        m.is_loaded_in_8bit = True if not full_finetuning else False

        # Patch generate
        if model.generate.__name__ != "unsloth_base_fast_generate":
            model._old_generate = model.generate
            unsloth_base_fast_generate.__doc__ = model._old_generate.__doc__
            model.generate = types.MethodType(unsloth_base_fast_generate, model)

        # Get model config to extract image size information if needed
        config = AutoConfig.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=trust_remote_code,
        )
        
        # Process image size parameters
        target_image_size = process_image_sizes(config, max_image_size)
        
        # Store the target image size in model for later use
        model.unsloth_target_image_size = target_image_size
        
        # Patch image processing based on processor type
        if hasattr(tokenizer, "image_processor"):
            original_preprocess = tokenizer.image_processor.preprocess
            
            def patched_preprocess(images, **kwargs):
                # First resize the images if needed
                if target_image_size is not None:
                    if isinstance(images, list):
                        images = [resize_images(img, target_image_size) for img in images]
                    else:
                        images = resize_images(images, target_image_size)
                
                # Then call the original preprocess
                return original_preprocess(images, **kwargs)
            
            tokenizer.image_processor.preprocess = patched_preprocess
        
        # For MllamaProcessor-like processors that don't expose image_processor
        elif hasattr(tokenizer, "__call__") and "MllamaProcessor" in tokenizer.__class__.__name__:
            original_call = tokenizer.__call__
            
            def patched_call(*args, **kwargs):
                # Extract images if present
                images = kwargs.pop("images", None)
                image = kwargs.pop("image", None)
                
                # Use whichever is provided
                img = images if images is not None else image
                
                # Resize images if needed
                if img is not None and target_image_size is not None:
                    if isinstance(img, list):
                        img = [resize_images(i, target_image_size) for i in img]
                    else:
                        img = resize_images(img, target_image_size)
                
                # Call original __call__ with the right parameter name
                if img is not None:
                    kwargs["image"] = img  # MllamaProcessor uses "image" not "images"
                
                return original_call(*args, **kwargs)
            
            tokenizer.__call__ = patched_call
        
        # For any other processor, try to patch common methods
        else:
            # Look for common image processing methods
            for method_name in ["process_images", "preprocess", "preprocess_images"]:
                if hasattr(tokenizer, method_name) and callable(getattr(tokenizer, method_name)):
                    original_method = getattr(tokenizer, method_name)
                    
                    def patched_method(images, **kwargs):
                        # Resize images if needed
                        if target_image_size is not None:
                            if isinstance(images, list):
                                images = [resize_images(img, target_image_size) for img in images]
                            else:
                                images = resize_images(images, target_image_size)
                        
                        # Call original method
                        return original_method(images, **kwargs)
                    
                    setattr(tokenizer, method_name, patched_method)
                    break
        
        # Post patches
        model = FastBaseModel.post_patch_model(
            model,
            use_gradient_checkpointing = use_gradient_checkpointing,
        )
        # Clear deleted GPU items
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
        pass
        return model, tokenizer
    pass


    @staticmethod
    def get_peft_model(
        model,
        r                          = 16,
        target_modules             = None,
        lora_alpha                 = 16,
        lora_dropout               = 0,
        bias                       = "none",
        finetune_vision_layers     = True,
        finetune_language_layers   = True,
        finetune_attention_modules = True,
        finetune_mlp_modules       = True,
        layers_to_transform        = None,
        layers_pattern             = None,
        use_gradient_checkpointing = True,
        random_state               = 3407,
        max_seq_length             = 2048, # not used anymore
        use_rslora                 = False,
        modules_to_save            = None,
        init_lora_weights          = True,
        loftq_config               = {},
        temporary_location         = "_unsloth_temporary_saved_buffers",
        **kwargs,
    ):
        if os.environ.get("UNSLOTH_ENABLE_FULL_FINETUNING", "0") == "1":
            print("Unsloth: Full finetuning is enabled, so .get_peft_model has no effect")
            return model
        pass
        transformers_set_seed(random_state)

        if type(r) is not int:
            raise TypeError(f"Unsloth: Rank of {str(r)} must be an integer.")
        if r <= 0:
            raise TypeError(f"Unsloth: Rank of {str(r)} must be larger than 0.")

        if isinstance(model, PeftModelForCausalLM):
            raise RuntimeError("Unsloth: You already added LoRA adapters to your model!")

        if target_modules == "all-linear":
            finetune_vision_layers     = True
            finetune_language_layers   = True
            finetune_attention_modules = True
            finetune_mlp_modules       = True
        pass
        if target_modules is None:
            target_modules = get_peft_regex(
                model,
                finetune_vision_layers     = finetune_vision_layers,
                finetune_language_layers   = finetune_language_layers,
                finetune_attention_modules = finetune_attention_modules,
                finetune_mlp_modules       = finetune_mlp_modules,
            )
        else:
            assert(type(target_modules) in (list, tuple,))
        pass

        # Clear deleted GPU items
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
        pass
        max_seq_length = model.max_seq_length
        lora_config = LoraConfig(
            r               = r,
            lora_alpha      = lora_alpha,
            target_modules  = target_modules,
            lora_dropout    = lora_dropout,
            bias            = bias,
            task_type       = TaskType.CAUSAL_LM,
        )
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing = use_gradient_checkpointing,
        )
        model = _get_peft_model(model, lora_config)
        # Enable gradients on modules which are trainable
        requires_grad_for_gradient_checkpointing(model)

        model = FastBaseModel.post_patch_model(model, use_gradient_checkpointing)
        model.max_seq_length = max_seq_length

        # Clear deleted GPU items
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
        pass
        patch_saving_functions(model, vision = True)

        # Add for_inference and for_training
        model.for_training  = functools.partial(FastBaseModel.for_training,  model)
        model.for_inference = functools.partial(FastBaseModel.for_inference, model)
        return model
    pass


    @staticmethod
    def post_patch_model(
        model,
        use_gradient_checkpointing = True,
    ):
        full_finetuning = os.environ.get("UNSLOTH_ENABLE_FULL_FINETUNING", "0") == "1"

        float32_mixed_precision = True
        if _get_dtype(model.config.torch_dtype) == torch.bfloat16:
            # Use bfloat16 precision for full finetuning
            float32_mixed_precision = False

        model = prepare_model_for_training(
            model,
            use_gradient_checkpointing = use_gradient_checkpointing,
            use_reentrant              = True,
            full_finetuning            = full_finetuning,
            train_layernorms           = full_finetuning,
            train_embedding            = full_finetuning,
            train_lm_head              = full_finetuning,
            float32_mixed_precision    = float32_mixed_precision,
        )

        from transformers.trainer import Trainer 
        if Trainer._inner_training_loop.__name__ != "_fast_inner_training_loop":
            raise RuntimeError(
                'Unsloth currently does not work on multi GPU setups - sadly we are a 2 brother team so '\
                'enabling it will require much more work, so we have to prioritize. Please understand!\n'\
                'We do have a separate beta version, which you can contact us about!\n'\
                'Thank you for your understanding and we appreciate it immensely!'
            )
        pass
        patch_saving_functions(model, vision = True)

        # Patch tokenizer to pad to the right
        m = model
        while hasattr(m, "model"):
            if hasattr(m, "_saved_temp_tokenizer"):
                if hasattr(m._saved_temp_tokenizer, "tokenizer"):
                    m._saved_temp_tokenizer.tokenizer.padding_side = "right"
            pass
            # Also set is_loaded_in_8bit to disable incorrect DDP
            m.is_loaded_in_8bit = True if not full_finetuning else False
            m = m.model
        pass
        if hasattr(m, "_saved_temp_tokenizer"):
            if hasattr(m._saved_temp_tokenizer, "tokenizer"):
                m._saved_temp_tokenizer.tokenizer.padding_side = "right"
        pass
        # Also set is_loaded_in_8bit to disable incorrect DDP
        m.is_loaded_in_8bit = True if not full_finetuning else False

        # Clear deleted GPU items
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
        pass
        # Add for_inference and for_training
        model.for_training  = functools.partial(FastBaseModel.for_training,  model)
        model.for_inference = functools.partial(FastBaseModel.for_inference, model)

        # Patch generate
        if model.generate.__name__ != "unsloth_base_fast_generate":
            model._old_generate = model.generate
            unsloth_base_fast_generate.__doc__ = model._old_generate.__doc__
            model.generate = types.MethodType(unsloth_base_fast_generate, model)
        return model
    pass


    @staticmethod
    def for_inference(model):
        if not hasattr(model, "parameters"):
            raise TypeError("Unsloth: I think you're passing a tokenizer, not the model to for_inference!")

        def _for_inference(m):
            if hasattr(m, "gradient_checkpointing"): m.gradient_checkpointing = False
            if hasattr(m, "training"): m.training = False
            # Pad tokenizer to the left
            if hasattr(m, "_saved_temp_tokenizer"): m._saved_temp_tokenizer.padding_side = "left"
            # Set a flag for generation!
            m._flag_for_generation = True
        pass
        m = model
        while hasattr(m, "model"):
            _for_inference(m)
            m = m.model
        _for_inference(m)

        # Also disable training for embeddings for NEFTune
        if hasattr(model, "get_input_embeddings"):
            embeddings = model.get_input_embeddings()
            if hasattr(embeddings, "training"): embeddings.training = False
        pass
        if hasattr(model, "get_output_embeddings"):
            embeddings = model.get_output_embeddings()
            if hasattr(embeddings, "training"): embeddings.training = False
        pass
        return model
    pass


    @staticmethod
    def for_training(model, use_gradient_checkpointing = True):
        if not hasattr(model, "parameters"):
            raise TypeError("Unsloth: I think you're passing a tokenizer, not the model to for_training!")

        # Delete all fast inference loras
        for param in model.parameters():
            if hasattr(param, "_fast_lora"):
                del param._fast_lora
        pass

        def _for_training(m):
            if hasattr(m, "gradient_checkpointing"): m.gradient_checkpointing = use_gradient_checkpointing
            if hasattr(m, "training"): m.training = True
            # Pad tokenizer to the left
            if hasattr(m, "_saved_temp_tokenizer"): m._saved_temp_tokenizer.padding_side = "right"
            # Set a flag for generation!
            if hasattr(m, "_flag_for_generation"): del m._flag_for_generation
        pass
        m = model
        while hasattr(m, "model"):
            _for_training(m)
            m = m.model
        _for_training(m)

        # Also re-enable training for embeddings for NEFTune
        if hasattr(model, "get_input_embeddings"):
            embeddings = model.get_input_embeddings()
            if hasattr(embeddings, "training"): embeddings.training = True
        pass
        if hasattr(model, "get_output_embeddings"):
            embeddings = model.get_output_embeddings()
            if hasattr(embeddings, "training"): embeddings.training = True
        pass
        return model
    pass
pass
