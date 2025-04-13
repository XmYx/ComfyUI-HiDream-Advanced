import torch
import math
import einops
from typing import Any, Callable, Dict, List, Optional, Union
from .pipeline_hidream_image import HiDreamImagePipeline, calculate_shift, retrieve_timesteps
from .pipeline_output import HiDreamImagePipelineOutput
from ...schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from ...schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler
from diffusers.utils import is_torch_xla_available

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

class HiDreamImageToImagePipeline(HiDreamImagePipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        prompt_4: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        negative_prompt_4: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 128,
        max_sequence_length_clip_l: Optional[int] = None,
        max_sequence_length_openclip: Optional[int] = None,
        max_sequence_length_t5: Optional[int] = None,
        max_sequence_length_llama: Optional[int] = None,
        llm_system_prompt: str = "",
        clip_l_scale: float = 1.0,
        openclip_scale: float = 1.0,
        t5_scale: float = 1.0,
        llama_scale: float = 1.0,
        # Add img2img specific parameters
        init_image: Optional[torch.FloatTensor] = None,
        denoising_strength: float = 0.75,
    ):
        # Handle dimensions
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        division = self.vae_scale_factor * 2
        
        # Force dimensions to be divisible by division without any area scaling
        width = int(width // division * division)
        height = int(height // division * division)
        
        # Ensure minimum dimensions
        width = max(width, division)
        height = max(height, division)
        
        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False
        
        # Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
            
        device = self._execution_device
        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        
        # Encode prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            prompt_4=prompt_4,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            negative_prompt_4=negative_prompt_4,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            max_sequence_length_clip_l=max_sequence_length_clip_l,
            max_sequence_length_openclip=max_sequence_length_openclip,
            max_sequence_length_t5=max_sequence_length_t5,
            max_sequence_length_llama=max_sequence_length_llama,
            llm_system_prompt=llm_system_prompt,
            clip_l_scale=clip_l_scale,
            openclip_scale=openclip_scale,
            t5_scale=t5_scale,
            llama_scale=llama_scale,
            lora_scale=lora_scale,
        )

        # --- UNLOAD TEXT ENCODER 4 (LLM) AFTER EMBEDS ---
        import gc
        try:
            if hasattr(self, "text_encoder_4") and self.text_encoder_4 is not None:
                print("[HiDreamImagePipeline] Unloading text_encoder_4 (LLM) from memory/GPU after embedding.")
                # Use to_empty if available for robust unloading (esp. GPTQ/BnB models)
                if hasattr(self.text_encoder_4, "to_empty"):
                    self.text_encoder_4.to_empty(device="meta")
                else:
                    self.text_encoder_4.to("meta")
                del self.text_encoder_4
                self.text_encoder_4 = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception as e:
            print(f"[HiDreamImagePipeline] Could not fully unload LLM: {e}")
        # --- CONTINUE TO LATENT PREP/DENOISING ---
        
        if self.do_classifier_free_guidance:
            prompt_embeds_arr = []
            for n, p in zip(negative_prompt_embeds, prompt_embeds):
                if len(n.shape) == 3:
                    prompt_embeds_arr.append(torch.cat([n, p], dim=0))
                else:
                    prompt_embeds_arr.append(torch.cat([n, p], dim=1))
            prompt_embeds = prompt_embeds_arr
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        
        # Prepare latent variables - this is where we handle img2img
        num_channels_latents = self.transformer.config.in_channels
        
        # If we have an init_image, we want to encode it to latents
        if init_image is not None:
            # Preprocess the input image to latent representation
            init_image = init_image.to(device=device, dtype=self.vae.dtype)
            
            # Ensure correct shape [B, C, H, W]
            # ComfyUI typically provides [B, H, W, C]
            if init_image.shape[3] == 3:  # [B, H, W, C]
                init_image = init_image.permute(0, 3, 1, 2)
                
            # Scale to [-1, 1]
            init_image = 2 * init_image - 1.0
            
            # Encode the image to latent space
            latents = self.vae.encode(init_image).latent_dist.sample(generator=generator)
            latents = latents * self.vae.config.scaling_factor
            
            # If we're working with a batch of 1, repeat for each image_per_prompt
            if latents.shape[0] == 1 and batch_size * num_images_per_prompt > 1:
                latents = latents.repeat(batch_size * num_images_per_prompt, 1, 1, 1)
        else:
            # For regular txt2img, prepare random latents
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                pooled_prompt_embeds.dtype,
                device,
                generator,
                latents,
            )
            
        # Prepare for different aspect ratios
        if latents.shape[-2] != latents.shape[-1]:
            B, C, H, W = latents.shape
            pH, pW = H // self.transformer.config.patch_size, W // self.transformer.config.patch_size
            img_sizes = torch.tensor([pH, pW], dtype=torch.int64).reshape(-1)
            img_ids = torch.zeros(pH, pW, 3)
            img_ids[..., 1] = img_ids[..., 1] + torch.arange(pH)[:, None]
            img_ids[..., 2] = img_ids[..., 2] + torch.arange(pW)[None, :]
            img_ids = img_ids.reshape(pH * pW, -1)
            img_ids_pad = torch.zeros(self.transformer.max_seq, 3)
            img_ids_pad[:pH*pW, :] = img_ids
            img_sizes = img_sizes.unsqueeze(0).to(latents.device)
            img_ids = img_ids_pad.unsqueeze(0).to(latents.device)
            if self.do_classifier_free_guidance:
                img_sizes = img_sizes.repeat(2 * B, 1)
                img_ids = img_ids.repeat(2 * B, 1, 1)
        else:
            img_sizes = img_ids = None
            
        # Prepare timesteps
        mu = calculate_shift(self.transformer.max_seq)
        scheduler_kwargs = {"mu": mu}
        
        if isinstance(self.scheduler, FlowUniPCMultistepScheduler):
            self.scheduler.set_timesteps(num_inference_steps, device=device, shift=math.exp(mu))
            timesteps = self.scheduler.timesteps
        else:
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler,
                num_inference_steps,
                device,
                sigmas=sigmas,
                **scheduler_kwargs,
            )
        
        # For img2img, we need to modify the timesteps based on denoising_strength
        if init_image is not None and denoising_strength > 0.0:
            # Calculate the starting timestep based on denoising strength
            start_step = int(num_inference_steps * (1.0 - denoising_strength))
            
            # Skip steps based on denoising strength
            if start_step > 0:
                timesteps = timesteps[start_step:]
                print(f"Starting denoising from step {start_step}/{num_inference_steps} (strength: {denoising_strength})")
                
                # Create noise
                noise = torch.randn(latents.shape, dtype=latents.dtype, device=device, generator=generator)
                
                # Get starting timestep
                t_start = timesteps[0].unsqueeze(0)
                
                # Set the scheduler's step index for proper noise scaling
                self.scheduler._step_index = start_step
                
                # Apply noise using the appropriate scheduler method
                if isinstance(self.scheduler, FlowUniPCMultistepScheduler):
                    print(f"Using UniPC add_noise with timestep {t_start}")
                    latents = self.scheduler.add_noise(
                        original_samples=latents,
                        noise=noise,
                        timesteps=t_start
                    )
                else:  # FlashFlowMatchEulerDiscreteScheduler or variants
                    print(f"Using FlashFlow scale_noise with timestep {t_start}")
                    latents = self.scheduler.scale_noise(
                        sample=latents,
                        timestep=t_start,
                        noise=noise
                    )
        
        # Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                    
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])
                
                if latent_model_input.shape[-2] != latent_model_input.shape[-1]:
                    B, C, H, W = latent_model_input.shape
                    patch_size = self.transformer.config.patch_size
                    pH, pW = H // patch_size, W // patch_size
                    out = torch.zeros(
                        (B, C, self.transformer.max_seq, patch_size * patch_size),
                        dtype=latent_model_input.dtype,
                        device=latent_model_input.device
                    )
                    latent_model_input = einops.rearrange(latent_model_input, 'B C (H p1) (W p2) -> B C (H W) (p1 p2)', p1=patch_size, p2=patch_size)
                    out[:, :, 0:pH*pW] = latent_model_input
                    latent_model_input = out
                
                noise_pred = self.transformer(
                    hidden_states = latent_model_input,
                    timesteps = timestep,
                    encoder_hidden_states = prompt_embeds,
                    pooled_embeds = pooled_prompt_embeds,
                    img_sizes = img_sizes,
                    img_ids = img_ids,
                    return_dict = False,
                )[0]
                
                noise_pred = -noise_pred
                
                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug
                        latents = latents.to(latents_dtype)
                
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                
                # call the callback, if provided
                progress_bar.update()
                
                if XLA_AVAILABLE:
                    xm.mark_step()
        
        # Post-processing
        if output_type == "latent":
            image = latents
        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
        
        # Offload all models
        self.maybe_free_model_hooks()
        
        if not return_dict:
            return (image,)
        
        return HiDreamImagePipelineOutput(images=image)
