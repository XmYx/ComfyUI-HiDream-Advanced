# -*- coding: utf-8 -*-
# HiDream ComfyUI Extended Nodes (Improved)
# Adds model download/loading, conditioning, latents, sampler nodes, VAE encode/decode,
# plus an "unsampler" node that inverts an image into noise latents via partial reverse diffusion.

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import gc
import os
import packaging.version
import transformers
import comfy.model_management
import comfy.utils
import huggingface_hub
from safetensors.torch import load_file

try:
    import flash_attn
    flash_attn_available = True
except ImportError:
    flash_attn_available = False

sdpa_available = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

try:
    import accelerate
    accelerate_available = True
except ImportError:
    accelerate_available = False

gptqmodel_available = False
autogptq_available = False
gptq_support_available = False

try:
    from transformers import GPTQConfig
    gptqmodel_available = True
    gptq_support_available = True
except ImportError:
    pass

try:
    import auto_gptq
    autogptq_available = True
    gptq_support_available = True
except ImportError:
    pass

try:
    import optimum
    optimum_available = True
except ImportError:
    optimum_available = False

try:
    from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
    from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
    bnb_available = True
except ImportError:
    bnb_available = False
    TransformersBitsAndBytesConfig = None
    DiffusersBitsAndBytesConfig = None

from transformers import LlamaForCausalLM, AutoTokenizer

# Attempt hi_diffusers imports
try:
    from .hi_diffusers.models.transformers.transformer_hidream_image import HiDreamImageTransformer2DModel
    from .hi_diffusers.pipelines.hidream_image.pipeline_hidream_image import HiDreamImagePipeline
    from .hi_diffusers.pipelines.hidream_image.pipeline_hidream_image_to_image import HiDreamImageToImagePipeline
    from .hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from .hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler
    hidream_classes_loaded = True
except ImportError as e:
    HiDreamImageTransformer2DModel = None
    HiDreamImagePipeline = None
    HiDreamImageToImagePipeline = None
    FlowUniPCMultistepScheduler = None
    FlashFlowMatchEulerDiscreteScheduler = None
    hidream_classes_loaded = False

print(f"Flash Attention available: {flash_attn_available}")
print(f"SDPA available: {sdpa_available}")
print(f"Accelerate available: {accelerate_available}")
print(f"GPTQ support available: {gptq_support_available}")
print(f"Optimum available: {optimum_available}")
print(f"BitsAndBytes available: {bnb_available}")
print(f"hi_diffusers classes loaded: {hidream_classes_loaded}")

ORIGINAL_MODEL_PREFIX = "HiDream-ai"
NF4_MODEL_PREFIX = "azaneko"
ORIGINAL_LLAMA_MODEL_NAME = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
NF4_LLAMA_MODEL_NAME = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
ALTERNATE_LLAMA_MODEL_NAME = "akhbar/Meta-Llama-3.1-8B-Instruct-abliterated-GPTQ"
ALTERNATE_NF4_LLAMA_MODEL_NAME = "akhbar/Meta-Llama-3.1-8B-Instruct-abliterated-GPTQ"

MODEL_CONFIGS = {
    "full-nf4": {
        "path": f"{NF4_MODEL_PREFIX}/HiDream-I1-Full-nf4",
        "guidance_scale": 5.0, "num_inference_steps": 50, "shift": 3.0,
        "scheduler_class": "FlowUniPCMultistepScheduler",
        "is_nf4": True, "is_fp8": False, "requires_bnb": False, "requires_gptq_deps": True
    },
    "dev-nf4": {
        "path": f"{NF4_MODEL_PREFIX}/HiDream-I1-Dev-nf4",
        "guidance_scale": 0.0, "num_inference_steps": 28, "shift": 6.0,
        "scheduler_class": "FlashFlowMatchEulerDiscreteScheduler",
        "is_nf4": True, "is_fp8": False, "requires_bnb": False, "requires_gptq_deps": True
    },
    "fast-nf4": {
        "path": f"{NF4_MODEL_PREFIX}/HiDream-I1-Fast-nf4",
        "guidance_scale": 0.0, "num_inference_steps": 16, "shift": 3.0,
        "scheduler_class": "FlashFlowMatchEulerDiscreteScheduler",
        "is_nf4": True, "is_fp8": False, "requires_bnb": False, "requires_gptq_deps": True
    },
    "full": {
        "path": f"{ORIGINAL_MODEL_PREFIX}/HiDream-I1-Full",
        "guidance_scale": 5.0, "num_inference_steps": 50, "shift": 3.0,
        "scheduler_class": "FlowUniPCMultistepScheduler",
        "is_nf4": False, "is_fp8": False, "requires_bnb": True, "requires_gptq_deps": False
    },
    "dev": {
        "path": f"{ORIGINAL_MODEL_PREFIX}/HiDream-I1-Dev",
        "guidance_scale": 0.0, "num_inference_steps": 28, "shift": 6.0,
        "scheduler_class": "FlashFlowMatchEulerDiscreteScheduler",
        "is_nf4": False, "is_fp8": False, "requires_bnb": True, "requires_gptq_deps": False
    },
    "fast": {
        "path": f"{ORIGINAL_MODEL_PREFIX}/HiDream-I1-Fast",
        "guidance_scale": 0.0, "num_inference_steps": 16, "shift": 3.0,
        "scheduler_class": "FlashFlowMatchEulerDiscreteScheduler",
        "is_nf4": False, "is_fp8": False, "requires_bnb": True, "requires_gptq_deps": False
    }
}

orig_count = len(MODEL_CONFIGS)
if not hidream_classes_loaded:
    MODEL_CONFIGS = {}
if not bnb_available:
    MODEL_CONFIGS = {k: v for k,v in MODEL_CONFIGS.items() if not v.get("requires_bnb",False)}
if not optimum_available or not gptq_support_available:
    MODEL_CONFIGS = {k:v for k,v in MODEL_CONFIGS.items() if not v.get("requires_gptq_deps",False)}
filtered_count = len(MODEL_CONFIGS)

if filtered_count == 0:
    print("CRITICAL ERROR: No HiDream models available.")
elif filtered_count < orig_count:
    print("Warning: Some HiDream models disabled due to missing dependencies.")

MODEL_CACHE = {}
DEBUG_CACHE = True

bnb_llm_config = None
bnb_transformer_4bit_config = None
model_dtype = torch.bfloat16
if bnb_available:
    from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
    from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
    bnb_llm_config = TransformersBitsAndBytesConfig(load_in_4bit=True)
    bnb_transformer_4bit_config = DiffusersBitsAndBytesConfig(load_in_4bit=True)

def get_scheduler_instance(scheduler_name, shift_value):
    if not hidream_classes_loaded:
        raise RuntimeError("HiDream schedulers not available.")
    if scheduler_name == "FlowUniPCMultistepScheduler":
        return FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=shift_value, use_dynamic_shifting=False)
    elif scheduler_name == "FlashFlowMatchEulerDiscreteScheduler":
        return FlashFlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift_value, use_dynamic_shifting=False)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

def pil2tensor(image: Image.Image):
    """Convert a PIL image to ComfyUI-compatible float32 torch tensor [1,H,W,C]."""
    if image is None:
        return None
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        np_array = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np_array).unsqueeze(0)
        return tensor
    except:
        try:
            tensor = comfy.utils.pil2tensor(image)
            return tensor
        except:
            return None

def load_models(model_type, use_alternate_llm=False):
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown or incompatible model_type: {model_type}")
    config = MODEL_CONFIGS[model_type]
    model_path = config["path"]
    is_nf4 = config["is_nf4"]
    scheduler_name = config["scheduler_class"]
    shift = config["shift"]
    requires_bnb = config.get("requires_bnb", False)
    requires_gptq_deps = config.get("requires_gptq_deps", False)

    if requires_bnb and not bnb_available:
        raise ImportError(f"Model '{model_type}' requires BitsAndBytes (bnb).")
    if requires_gptq_deps and (not optimum_available or not gptq_support_available):
        raise ImportError(f"Model '{model_type}' requires GPTQ/Optimum libraries.")

    cache_key = f"{model_type}_{'alternate' if use_alternate_llm else 'standard'}"
    if DEBUG_CACHE:
        print(f"Cache check for key: {cache_key}")
        print(f"Cache contains: {list(MODEL_CACHE.keys())}")

    if cache_key in MODEL_CACHE:
        pipe, stored_config = MODEL_CACHE[cache_key]
        if pipe is not None and hasattr(pipe, 'transformer') and pipe.transformer is not None:
            print(f"Using cached model for {cache_key}")
            return pipe, config
        else:
            print(f"Cache entry invalid for {cache_key}, reloading")
            MODEL_CACHE.pop(cache_key, None)

    print(f"--- Loading Model Type: {model_type} ---")
    text_encoder_load_kwargs = {
        "low_cpu_mem_usage": True,
        "torch_dtype": model_dtype
    }
    # LLM loading
    if is_nf4:
        llama_model_name = (
            ALTERNATE_NF4_LLAMA_MODEL_NAME
            if use_alternate_llm else
            NF4_LLAMA_MODEL_NAME
        )
        if accelerate_available:
            if hasattr(torch.cuda, 'get_device_properties') and torch.cuda.is_available():
                total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                max_mem = int(total_mem * 0.4)
                text_encoder_load_kwargs["max_memory"] = {0: f"{max_mem}GiB"}
            text_encoder_load_kwargs["device_map"] = "auto"
        else:
            pass
    else:
        llama_model_name = (
            ALTERNATE_LLAMA_MODEL_NAME
            if use_alternate_llm else
            ORIGINAL_LLAMA_MODEL_NAME
        )
        if bnb_llm_config:
            text_encoder_load_kwargs["quantization_config"] = bnb_llm_config
        if flash_attn_available:
            text_encoder_load_kwargs["attn_implementation"] = "flash_attention_2"
        elif sdpa_available:
            text_encoder_load_kwargs["attn_implementation"] = "sdpa"
        else:
            text_encoder_load_kwargs["attn_implementation"] = "eager"

    tokenizer = AutoTokenizer.from_pretrained(llama_model_name, use_fast=False)

    # text encoder
    if is_nf4:
        from transformers import AutoConfig, AutoModelForCausalLM, GPTQConfig
        config_obj = AutoConfig.from_pretrained(llama_model_name)
        config_obj.rope_scaling = {"type": "linear", "factor": 1.0}
        text_encoder_load_kwargs["config"] = config_obj
        text_encoder_load_kwargs["low_cpu_mem_usage"] = True
        if gptqmodel_available:
            gptq_kwargs = dict(bits=4)
            text_encoder_load_kwargs["quantization_config"] = GPTQConfig(**gptq_kwargs)
            text_encoder = AutoModelForCausalLM.from_pretrained(llama_model_name, **text_encoder_load_kwargs)
        elif autogptq_available:
            from auto_gptq import AutoGPTQForCausalLM
            text_encoder = AutoGPTQForCausalLM.from_pretrained(llama_model_name, **text_encoder_load_kwargs)
        else:
            raise ImportError("No supported GPTQModel backend.")
    else:
        text_encoder = LlamaForCausalLM.from_pretrained(llama_model_name, **text_encoder_load_kwargs)

    if "device_map" not in text_encoder_load_kwargs:
        text_encoder.to("cuda")

    # image transformer
    transformer_load_kwargs = {
        "subfolder": "transformer",
        "torch_dtype": model_dtype,
        "low_cpu_mem_usage": True
    }
    if not is_nf4 and bnb_transformer_4bit_config:
        transformer_load_kwargs["quantization_config"] = bnb_transformer_4bit_config

    transformer = HiDreamImageTransformer2DModel.from_pretrained(model_path, **transformer_load_kwargs)
    transformer.to("cuda")

    # scheduler
    scheduler = get_scheduler_instance(scheduler_name, shift)

    # pipeline
    pipe = HiDreamImagePipeline.from_pretrained(
        model_path,
        scheduler=scheduler,
        tokenizer_4=tokenizer,
        text_encoder_4=text_encoder,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True
    )
    pipe.transformer = transformer
    try:
        pipe.to("cuda")
    except:
        pass

    # NF4 offload
    if is_nf4 and hasattr(pipe, "enable_sequential_cpu_offload"):
        try:
            pipe.enable_sequential_cpu_offload()
        except:
            pass

    MODEL_CACHE[cache_key] = (pipe, config)
    return pipe, config

def global_cleanup():
    """Optional global cleanup function for external usage."""
    print("HiDream: Performing global cleanup...")
    torch.cuda.synchronize()
    if torch.cuda.is_available():
        before_mem = torch.cuda.memory_allocated() / 1024**2
        print(f"  Memory before cleanup: {before_mem:.2f} MB")
    HiDreamModelLoader.cleanup_models()
    gc.collect(); gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        after_mem = torch.cuda.memory_allocated() / 1024**2
        print(f"  Memory after cleanup: {after_mem:.2f} MB")
    return True


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NODE DEFINITIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class HiDreamModelLoader:
    RETURN_TYPES = ("HIDREAM_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "HiDream"

    @classmethod
    def cleanup_models(cls):
        print("HiDream: Cleaning up all cached models...")
        keys_to_del = list(MODEL_CACHE.keys())
        for key in keys_to_del:
            print(f"  Removing '{key}'...")
            try:
                pipe_to_del, _ = MODEL_CACHE.pop(key)
                if hasattr(pipe_to_del, 'transformer'):
                    pipe_to_del.transformer = None
                if hasattr(pipe_to_del, 'text_encoder_4'):
                    pipe_to_del.text_encoder_4 = None
                if hasattr(pipe_to_del, 'tokenizer_4'):
                    pipe_to_del.tokenizer_4 = None
                if hasattr(pipe_to_del, 'scheduler'):
                    pipe_to_del.scheduler = None
                del pipe_to_del
            except Exception as e:
                print(f"  Error cleaning up {key}: {e}")
        gc.collect()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("HiDream: Cache cleared")

    @classmethod
    def INPUT_TYPES(cls):
        available_model_types = list(MODEL_CONFIGS.keys())
        if not available_model_types:
            return {"required": {
                "error": ("STRING", {"default": "No models available...", "multiline": True})
            }}
        default_model = (
            "fast-nf4" if "fast-nf4" in available_model_types
            else "fast" if "fast" in available_model_types
            else available_model_types[0]
        )
        return {
            "required": {
                "model_type": (available_model_types, {"default": default_model}),
                "use_alternate_llm": ("BOOLEAN", {"default": False}),
            }
        }

    def load(self, model_type, use_alternate_llm=False):
        if not MODEL_CONFIGS or model_type not in MODEL_CONFIGS:
            return ({"pipe": None, "config": None},)
        try:
            pipe, config = load_models(model_type, use_alternate_llm)
            return ({"pipe": pipe, "config": config},)
        except Exception as e:
            print(f"Error loading model {model_type}: {e}")
            return ({"pipe": None, "config": None},)


class HiDreamConditioning:
    RETURN_TYPES = ("HIDREAM_CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "setup"
    CATEGORY = "HiDream"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("HIDREAM_MODEL",),
                "primary_prompt": ("STRING", {"multiline": True, "default": "A fantasy portrait."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "clip_l_prompt": ("STRING", {"multiline": True, "default": ""}),
                "openclip_prompt": ("STRING", {"multiline": True, "default": ""}),
                "t5_prompt": ("STRING", {"multiline": True, "default": ""}),
                "llama_prompt": ("STRING", {"multiline": True, "default": ""}),
                "llm_system_prompt": ("STRING", {"multiline": True,
                    "default": "You are a creative AI assistant that helps create images."
                }),
                "clip_l_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "openclip_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "t5_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "llama_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "max_length_clip_l": ("INT", {"default": 77, "min": 64, "max": 218}),
                "max_length_openclip": ("INT", {"default": 77, "min": 64, "max": 218}),
                "max_length_t5": ("INT", {"default": 128, "min": 64, "max": 512}),
                "max_length_llama": ("INT", {"default": 128, "min": 64, "max": 2048}),
            }
        }

    def setup(self, model, primary_prompt, negative_prompt,
              clip_l_prompt, openclip_prompt, t5_prompt, llama_prompt,
              llm_system_prompt,
              clip_l_weight, openclip_weight, t5_weight, llama_weight,
              max_length_clip_l, max_length_openclip, max_length_t5, max_length_llama):
        cond = {
            "primary_prompt": primary_prompt,
            "negative_prompt": negative_prompt,
            "clip_l_prompt": clip_l_prompt,
            "openclip_prompt": openclip_prompt,
            "t5_prompt": t5_prompt,
            "llama_prompt": llama_prompt,
            "llm_system_prompt": llm_system_prompt,
            "clip_l_weight": clip_l_weight,
            "openclip_weight": openclip_weight,
            "t5_weight": t5_weight,
            "llama_weight": llama_weight,
            "max_length_clip_l": max_length_clip_l,
            "max_length_openclip": max_length_openclip,
            "max_length_t5": max_length_t5,
            "max_length_llama": max_length_llama,
        }
        return (cond,)


class HiDreamCombineConditioning:
    RETURN_TYPES = ("HIDREAM_CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "combine"
    CATEGORY = "HiDream"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning_1": ("HIDREAM_CONDITIONING",),
                "conditioning_2": ("HIDREAM_CONDITIONING",),
                "operation": (["add", "subtract", "blend"], {"default": "add"}),
                "blend_weight_1": ("FLOAT", {"default": 0.5, "min": 0.0, "max":1.0, "step":0.05}),
            }
        }

    def combine(self, conditioning_1, conditioning_2, operation="add", blend_weight_1=0.5):
        c1 = conditioning_1.copy()
        c2 = conditioning_2.copy()

        if operation == "add":
            for k in ["primary_prompt","clip_l_prompt","openclip_prompt","t5_prompt","llama_prompt"]:
                c1[k] = (c1[k].strip() + " " + c2[k].strip()).strip()
            c1["negative_prompt"] = (c1["negative_prompt"].strip() + " " + c2["negative_prompt"].strip()).strip()
            for wkey in ["clip_l_weight","openclip_weight","t5_weight","llama_weight"]:
                c1[wkey] = (c1[wkey] + c2[wkey]) / 2.0

        elif operation == "subtract":
            for k in ["primary_prompt","clip_l_prompt","openclip_prompt","t5_prompt","llama_prompt","negative_prompt"]:
                c1[k] = c1[k].replace(c2[k], "")
            for wkey in ["clip_l_weight","openclip_weight","t5_weight","llama_weight"]:
                c1[wkey] = max(0.0, c1[wkey] - c2[wkey])

        elif operation == "blend":
            w2 = 1.0 - blend_weight_1
            for k in ["primary_prompt","clip_l_prompt","openclip_prompt","t5_prompt","llama_prompt"]:
                p1 = c1[k].strip()
                p2 = c2[k].strip()
                if p1 and p2:
                    c1[k] = f"({p1} *{blend_weight_1:.2f}) AND ({p2} *{w2:.2f})"
                elif p1:
                    c1[k] = f"{p1} *{blend_weight_1:.2f}"
                else:
                    c1[k] = f"{p2} *{w2:.2f}"
            neg1 = c1["negative_prompt"].strip()
            neg2 = c2["negative_prompt"].strip()
            if neg1 and neg2:
                c1["negative_prompt"] = f"({neg1} *{blend_weight_1:.2f}) AND ({neg2} *{w2:.2f})"
            elif neg1:
                c1["negative_prompt"] = f"{neg1} *{blend_weight_1:.2f}"
            else:
                c1["negative_prompt"] = f"{neg2} *{w2:.2f}"

            for wkey in ["clip_l_weight","openclip_weight","t5_weight","llama_weight"]:
                v1 = c1[wkey]
                v2 = c2[wkey]
                c1[wkey] = blend_weight_1*v1 + w2*v2

        return (c1,)


class HiDreamLATENT:
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "create"
    CATEGORY = "HiDream"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 64}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 8})
            }
        }

    def create(self, seed, width, height, batch_size=1):
        width = (width // 64) * 64
        height = (height // 64) * 64
        try:
            device = comfy.model_management.get_torch_device()
        except:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        generator = torch.Generator(device=device).manual_seed(seed)
        data = {
            "seed": seed,
            "width": width,
            "height": height,
            "batch_size": batch_size,
            "generator": generator
        }
        return (data,)


class HiDreamSampler:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sample"
    CATEGORY = "HiDream"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("HIDREAM_MODEL",),
                "conditioning": ("HIDREAM_CONDITIONING",),
                "latent": ("LATENT",),
                "override_steps": ("INT", {"default": -1, "min": -1, "max": 200}),
                "override_cfg": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 20.0, "step": 0.1}),
                "scheduler_override": (["Default","UniPC","Euler","Karras Euler","Karras Exponential"], {"default":"Default"})
            }
        }

    def sample(self, model, conditioning, latent,
               override_steps, override_cfg, scheduler_override):
        pipe = model["pipe"]
        config = model["config"]
        if pipe is None or config is None:
            return (torch.zeros((1, 512, 512, 3)),)

        is_nf4 = config.get("is_nf4", False)
        base_steps = config["num_inference_steps"]
        base_cfg = config["guidance_scale"]
        shift = config["shift"]
        base_sched = config["scheduler_class"]

        steps = override_steps if override_steps >= 0 else base_steps
        cfg = override_cfg if override_cfg >= 0.0 else base_cfg

        if scheduler_override != "Default":
            if scheduler_override == "UniPC":
                pipe.scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=shift, use_dynamic_shifting=False)
            elif scheduler_override == "Euler":
                pipe.scheduler = FlashFlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift, use_dynamic_shifting=False)
            elif scheduler_override == "Karras Euler":
                pipe.scheduler = FlashFlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift, use_dynamic_shifting=False, use_karras_sigmas=True)
            elif scheduler_override == "Karras Exponential":
                pipe.scheduler = FlashFlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift, use_dynamic_shifting=False, use_exponential_sigmas=True)
        else:
            pipe.scheduler = get_scheduler_instance(base_sched, shift)

        seed = latent["seed"]
        width = latent["width"]
        height = latent["height"]
        generator = latent["generator"]

        prompt_clip_l = conditioning["clip_l_prompt"].strip() or conditioning["primary_prompt"]
        prompt_openclip = conditioning["openclip_prompt"].strip() or conditioning["primary_prompt"]
        prompt_t5 = conditioning["t5_prompt"].strip() or conditioning["primary_prompt"]
        prompt_llama = conditioning["llama_prompt"].strip() or conditioning["primary_prompt"]
        llm_system_prompt = conditioning["llm_system_prompt"]
        negative_prompt = conditioning["negative_prompt"].strip() or None

        clip_l_weight = conditioning["clip_l_weight"]
        openclip_weight = conditioning["openclip_weight"]
        t5_weight = conditioning["t5_weight"]
        llama_weight = conditioning["llama_weight"]
        max_length_clip_l = conditioning["max_length_clip_l"]
        max_length_openclip = conditioning["max_length_openclip"]
        max_length_t5 = conditioning["max_length_t5"]
        max_length_llama = conditioning["max_length_llama"]

        pbar = comfy.utils.ProgressBar(steps)
        def progress_callback(pipe, i, t, callback_kwargs):
            pbar.update_absolute(i+1)
            return callback_kwargs

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not is_nf4:
            pipe.to(device)

        outputs = None
        try:
            with torch.inference_mode():
                outputs = pipe(
                    prompt=prompt_clip_l,
                    prompt_2=prompt_openclip,
                    prompt_3=prompt_t5,
                    prompt_4=prompt_llama,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    guidance_scale=cfg,
                    num_inference_steps=steps,
                    num_images_per_prompt=1,
                    generator=generator,
                    max_sequence_length_clip_l=max_length_clip_l,
                    max_sequence_length_openclip=max_length_openclip,
                    max_sequence_length_t5=max_length_t5,
                    max_sequence_length_llama=max_length_llama,
                    llm_system_prompt=llm_system_prompt,
                    clip_l_scale=clip_l_weight,
                    openclip_scale=openclip_weight,
                    t5_scale=t5_weight,
                    llama_scale=llama_weight,
                    callback_on_step_end=progress_callback,
                    callback_on_step_end_tensor_inputs=["latents"],
                ).images
        except Exception as e:
            print(f"Error during sampling: {e}")
            return (torch.zeros((1, height, width, 3)),)
        finally:
            pbar.update_absolute(steps)

        if not outputs:
            return (torch.zeros((1, height, width, 3)),)

        out_tensor = pil2tensor(outputs[0])
        if out_tensor is None:
            return (torch.zeros((1, height, width, 3)),)
        if out_tensor.dtype != torch.float32:
            out_tensor = out_tensor.to(torch.float32)

        try:
            comfy.model_management.soft_empty_cache()
        except:
            pass
        return (out_tensor,)


class HiDreamImg2Img:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "render"
    CATEGORY = "HiDream"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("HIDREAM_MODEL",),
                "conditioning": ("HIDREAM_CONDITIONING",),
                "image": ("IMAGE",),
                "denoising_strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max":1.0}),
                "override_steps": ("INT", {"default": -1, "min": -1, "max": 200}),
                "override_cfg": ("FLOAT", {"default": -1.0, "min": -1.0, "max":20.0, "step":0.1}),
                "scheduler_override": (["Default","UniPC","Euler","Karras Euler","Karras Exponential"], {"default":"Default"})
            }
        }

    def preprocess_image(self, image_tensor, target_height=None, target_width=None):
        b, h, w, c = image_tensor.shape
        if target_height is None:
            target_height = h
        if target_width is None:
            target_width = w
        target_width = (target_width // 16) * 16
        target_height = (target_height // 16) * 16

        x = image_tensor.permute(0,3,1,2)
        x_resized = F.interpolate(x, size=(target_height, target_width), mode='bicubic', align_corners=False)
        return x_resized.permute(0,2,3,1)

    def render(self, model, conditioning, image,
               denoising_strength, override_steps, override_cfg, scheduler_override):
        pipe = model["pipe"]
        config = model["config"]
        if pipe is None or config is None:
            return (torch.zeros((1, 512, 512, 3)),)

        is_nf4 = config.get("is_nf4", False)
        base_steps = config["num_inference_steps"]
        base_cfg = config["guidance_scale"]
        shift = config["shift"]
        base_sched = config["scheduler_class"]

        steps = override_steps if override_steps >= 0 else base_steps
        cfg = override_cfg if override_cfg >= 0.0 else base_cfg

        if scheduler_override != "Default":
            if scheduler_override == "UniPC":
                pipe.scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=shift, use_dynamic_shifting=False)
            elif scheduler_override == "Euler":
                pipe.scheduler = FlashFlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift, use_dynamic_shifting=False)
            elif scheduler_override == "Karras Euler":
                pipe.scheduler = FlashFlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift, use_dynamic_shifting=False, use_karras_sigmas=True)
            elif scheduler_override == "Karras Exponential":
                pipe.scheduler = FlashFlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift, use_dynamic_shifting=False, use_exponential_sigmas=True)
        else:
            pipe.scheduler = get_scheduler_instance(base_sched, shift)

        prompt_clip_l = conditioning["clip_l_prompt"].strip() or conditioning["primary_prompt"]
        prompt_openclip = conditioning["openclip_prompt"].strip() or conditioning["primary_prompt"]
        prompt_t5 = conditioning["t5_prompt"].strip() or conditioning["primary_prompt"]
        prompt_llama = conditioning["llama_prompt"].strip() or conditioning["primary_prompt"]
        llm_system_prompt = conditioning["llm_system_prompt"]
        negative_prompt = conditioning["negative_prompt"].strip() or None

        clip_l_weight = conditioning["clip_l_weight"]
        openclip_weight = conditioning["openclip_weight"]
        t5_weight = conditioning["t5_weight"]
        llama_weight = conditioning["llama_weight"]
        max_length_clip_l = conditioning["max_length_clip_l"]
        max_length_openclip = conditioning["max_length_openclip"]
        max_length_t5 = conditioning["max_length_t5"]
        max_length_llama = conditioning["max_length_llama"]

        processed_input = self.preprocess_image(image)
        b, hh, ww, cc = processed_input.shape

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not is_nf4:
            pipe.to(device)

        try:
            i2i_pipe = HiDreamImageToImagePipeline(
                scheduler=pipe.scheduler,
                vae=pipe.vae,
                text_encoder=pipe.text_encoder,
                tokenizer=pipe.tokenizer,
                text_encoder_2=pipe.text_encoder_2,
                tokenizer_2=pipe.tokenizer_2,
                text_encoder_3=pipe.text_encoder_3,
                tokenizer_3=pipe.tokenizer_3,
                text_encoder_4=pipe.text_encoder_4,
                tokenizer_4=pipe.tokenizer_4
            )
            i2i_pipe.transformer = pipe.transformer
        except Exception as e:
            print(f"Error preparing Img2Img pipeline: {e}")
            return (torch.zeros((1, hh, ww, 3)),)

        outputs = None
        try:
            with torch.inference_mode():
                outputs = i2i_pipe(
                    prompt=prompt_clip_l,
                    prompt_2=prompt_openclip,
                    prompt_3=prompt_t5,
                    prompt_4=prompt_llama,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    init_image=processed_input,
                    denoising_strength=denoising_strength,
                    num_images_per_prompt=1,
                    generator=torch.Generator(device=device),
                    max_sequence_length_clip_l=max_length_clip_l,
                    max_sequence_length_openclip=max_length_openclip,
                    max_sequence_length_t5=max_length_t5,
                    max_sequence_length_llama=max_length_llama,
                    llm_system_prompt=llm_system_prompt,
                    clip_l_scale=clip_l_weight,
                    openclip_scale=openclip_weight,
                    t5_scale=t5_weight,
                    llama_scale=llama_weight,
                ).images
        except Exception as e:
            print(f"Error during img2img: {e}")
            return (torch.zeros((1, hh, ww, 3)),)

        if not outputs:
            return (torch.zeros((1, hh, ww, 3)),)

        out_tensor = pil2tensor(outputs[0])
        if out_tensor is None:
            return (torch.zeros((1, hh, ww, 3)),)
        if out_tensor.dtype != torch.float32:
            out_tensor = out_tensor.to(torch.float32)
        try:
            comfy.model_management.soft_empty_cache()
        except:
            pass
        return (out_tensor,)


class HiDreamVAEEncode:
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "encode"
    CATEGORY = "HiDream"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("HIDREAM_MODEL",),
                "image": ("IMAGE",),
            }
        }

    def encode(self, model, image):
        pipe = model["pipe"]
        if pipe is None:
            return ({"vae_latents": None},)

        if not hasattr(pipe, "vae"):
            print("No 'vae' found in pipeline. Cannot encode.")
            return ({"vae_latents": None},)

        x = image
        b, h, w, c = x.shape
        x = x.permute(0,3,1,2).to(dtype=torch.float32)
        with torch.no_grad():
            x = x.to(device=pipe.vae.device)
            latents = pipe.vae.encode(x).latent_dist.sample().to(torch.float32)
        data = {
            "vae_latents": latents,
            "latent_width": w,
            "latent_height": h,
            "latent_b": b,
            "latent_c": latents.shape[1],
        }
        return (data,)


class HiDreamVAEDecode:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = "HiDream"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("HIDREAM_MODEL",),
                "latent": ("LATENT",),
            }
        }

    def decode(self, model, latent):
        pipe = model["pipe"]
        if pipe is None:
            return (torch.zeros((1, 64, 64, 3)),)
        if not hasattr(pipe, "vae"):
            print("No 'vae' found in pipeline. Cannot decode.")
            return (torch.zeros((1, 64, 64, 3)),)

        latents = latent.get("vae_latents", None)
        if latents is None:
            print("No latents found in input. Returning blank.")
            return (torch.zeros((1, 64, 64, 3)),)

        b = latent.get("latent_b", 1)
        w = latent.get("latent_width", 64)
        h = latent.get("latent_height", 64)

        with torch.no_grad():
            latents = latents.to(device=pipe.vae.device, dtype=torch.float32)
            out = pipe.vae.decode(latents).sample
        out = out.clamp(-1,1)
        out = (out + 1) * 0.5
        out = out.detach().to(torch.float32)
        out = out.permute(0,2,3,1)
        return (out,)


class HiDreamUnsampler:
    """
    "Unsampling" (reverse diffusion) node that attempts to invert an image into noise latents.
    Encodes the image to latents, then runs partial reverse steps to produce final "noise" latents.
    Returns a LATENT structure with "vae_latents" containing the reversed latents.
    """
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("unsampled_latent",)
    FUNCTION = "unsample"
    CATEGORY = "HiDream"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("HIDREAM_MODEL",),
                "conditioning": ("HIDREAM_CONDITIONING",),
                "image": ("IMAGE",),
                "reverse_steps": ("INT", {"default":20, "min":1, "max":200}),
                "override_cfg": ("FLOAT", {"default":7.0, "min":0.0, "max":20.0, "step":0.1}),
                "scheduler_override": (["Default","UniPC","Euler","Karras Euler","Karras Exponential"], {"default":"Default"})
            }
        }

    def unsample(self, model, conditioning, image, reverse_steps, override_cfg, scheduler_override):
        pipe = model["pipe"]
        config = model["config"]
        if pipe is None or config is None:
            return ({"vae_latents": None},)

        if not hasattr(pipe, "vae"):
            print("No 'vae' found in pipeline. Inversion not possible.")
            return ({"vae_latents": None},)

        is_nf4 = config.get("is_nf4", False)
        base_steps = config["num_inference_steps"]
        base_cfg = config["guidance_scale"]
        shift = config["shift"]
        base_sched = config["scheduler_class"]

        inv_steps = reverse_steps
        cfg = override_cfg if override_cfg >= 0.0 else base_cfg

        if scheduler_override != "Default":
            if scheduler_override == "UniPC":
                pipe.scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=shift, use_dynamic_shifting=False)
            elif scheduler_override == "Euler":
                pipe.scheduler = FlashFlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift, use_dynamic_shifting=False)
            elif scheduler_override == "Karras Euler":
                pipe.scheduler = FlashFlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift, use_dynamic_shifting=False, use_karras_sigmas=True)
            elif scheduler_override == "Karras Exponential":
                pipe.scheduler = FlashFlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift, use_dynamic_shifting=False, use_exponential_sigmas=True)
        else:
            pipe.scheduler = get_scheduler_instance(base_sched, shift)

        if not is_nf4:
            pipe.to("cuda")

        # Encode image -> latents
        b, h, w, c = image.shape
        with torch.no_grad():
            x = image.permute(0,3,1,2).to(dtype=torch.float32, device=pipe.vae.device)
            latents = pipe.vae.encode(x).latent_dist.sample().to(torch.float32)

        # Prepare partial "inversion" logic
        # We'll do a naive 'DDIM-like' stepping backwards. Not guaranteed to perfectly invert,
        # but a partial approximation for demonstration.

        # define timesteps
        timesteps = pipe.scheduler.timesteps
        # for safety if reverse_steps > len(timesteps), clamp:
        inv_steps = min(inv_steps, len(timesteps)-1)
        reverse_t = timesteps[:inv_steps].flip(dims=[0])  # pick early timesteps, reverse them

        # gather prompts
        prompt_clip_l = conditioning["clip_l_prompt"].strip() or conditioning["primary_prompt"]
        prompt_openclip = conditioning["openclip_prompt"].strip() or conditioning["primary_prompt"]
        prompt_t5 = conditioning["t5_prompt"].strip() or conditioning["primary_prompt"]
        prompt_llama = conditioning["llama_prompt"].strip() or conditioning["primary_prompt"]
        llm_system_prompt = conditioning["llm_system_prompt"]
        negative_prompt = conditioning["negative_prompt"].strip() or None

        clip_l_weight = conditioning["clip_l_weight"]
        openclip_weight = conditioning["openclip_weight"]
        t5_weight = conditioning["t5_weight"]
        llama_weight = conditioning["llama_weight"]
        max_length_clip_l = conditioning["max_length_clip_l"]
        max_length_openclip = conditioning["max_length_openclip"]
        max_length_t5 = conditioning["max_length_t5"]
        max_length_llama = conditioning["max_length_llama"]

        # get the unconditional / text conditions from pipeline
        # We'll do a direct call with single step sub-sample. We do manual unet calls for partial reverse
        def get_text_embeddings():
            return pipe._encode_prompts(
                prompt_clip_l,
                prompt_openclip,
                prompt_t5,
                prompt_llama,
                negative_prompt,
                max_length_clip_l,
                max_length_openclip,
                max_length_t5,
                max_length_llama,
                llm_system_prompt,
                clip_l_weight,
                openclip_weight,
                t5_weight,
                llama_weight,
            )

        with torch.inference_mode():
            text_embeddings, uncond_embeddings = get_text_embeddings()
            # latents shape: [B, C, H/8, W/8] typically
            latents = latents.to(pipe.unet.device)
            for i, t in enumerate(reverse_t):
                t_input = t.repeat(latents.shape[0])
                # classifier-free guidance
                alpha = cfg
                # approximate unet forward
                with torch.autocast(pipe.unet.device.type, pipe.unet.dtype):
                    noise_pred_uncond = pipe.unet(latents, t_input, encoder_hidden_states=uncond_embeddings).sample
                    noise_pred_text = pipe.unet(latents, t_input, encoder_hidden_states=text_embeddings).sample
                    noise_pred = noise_pred_uncond + alpha*(noise_pred_text - noise_pred_uncond)

                # step the scheduler backwards
                latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # store final latents in dict
        data = {
            "vae_latents": latents.detach().cpu(),
            "latent_width": w,
            "latent_height": h,
            "latent_b": b,
            "latent_c": latents.shape[1]
        }
        return (data,)


NODE_CLASS_MAPPINGS = {
    "HiDreamModelLoader": HiDreamModelLoader,
    "HiDreamConditioning": HiDreamConditioning,
    "HiDreamCombineConditioning": HiDreamCombineConditioning,
    "HiDreamLATENT": HiDreamLATENT,
    "HiDreamSampler": HiDreamSampler,
    "HiDreamImg2Img": HiDreamImg2Img,
    "HiDreamVAEEncode": HiDreamVAEEncode,
    "HiDreamVAEDecode": HiDreamVAEDecode,
    "HiDreamUnsampler": HiDreamUnsampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HiDreamModelLoader": "HiDream: Model Loader",
    "HiDreamConditioning": "HiDream: Conditioning Setup",
    "HiDreamCombineConditioning": "HiDream: Combine Conditioning",
    "HiDreamLATENT": "HiDream: LATENT Setup",
    "HiDreamSampler": "HiDream: Sampler (Text2Img)",
    "HiDreamImg2Img": "HiDream: Img2Img",
    "HiDreamVAEEncode": "HiDream: VAE Encode",
    "HiDreamVAEDecode": "HiDream: VAE Decode",
    "HiDreamUnsampler": "HiDream: Unsampler (Reverse Diffusion)"
}

try:
    import comfy.model_management as model_management
    if hasattr(model_management, 'unload_all_models'):
        original_unload = model_management.unload_all_models
        def wrapped_unload():
            print("HiDream: ComfyUI is unloading all models, cleaning HiDream cache...")
            HiDreamModelLoader.cleanup_models()
            return original_unload()
        model_management.unload_all_models = wrapped_unload
        print("HiDream: Registered with ComfyUI memory management.")
except Exception as e:
    print(f"HiDream: Could not register cleanup with model_management: {e}")

print("-"*50)
print("HiDream Extended Nodes Ready (VAE Encode/Decode, 'Unsampler' -> latents).")
print(f"Available Models: {list(MODEL_CONFIGS.keys())}")
print("-"*50)
