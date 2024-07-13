from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.notebook import tqdm
import torch
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
import DDIM_inversion
from torch.optim.adam import Adam
from PIL import Image

T = torch.Tensor
TN = T | None

def load_1024(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((1024, 1024)))
    return image

class NullInversion:

    def _get_text_embeddings(prompt: str, tokenizer, text_encoder, device):
        # Tokenize text and get embeddings
        text_inputs = tokenizer(prompt, padding='max_length', max_length=tokenizer.model_max_length, truncation=True, return_tensors='pt')
        text_input_ids = text_inputs.input_ids

        with torch.no_grad():
            prompt_embeds = text_encoder(
                text_input_ids.to(device),
                output_hidden_states=True,
            )

        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        if prompt == '':
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
            return negative_prompt_embeds, negative_pooled_prompt_embeds
        return prompt_embeds, pooled_prompt_embeds


    def _encode_text_sdxl(self, prompt: str) -> tuple[dict[str, T], T]:
        device = self.model._execution_device
        prompt_embeds, pooled_prompt_embeds, = self._get_text_embeddings(prompt, self.model.tokenizer, self.model.text_encoder, device)
        prompt_embeds_2, pooled_prompt_embeds2, = self._get_text_embeddings( prompt, self.model.tokenizer_2, self.model.text_encoder_2, device)
        prompt_embeds = torch.cat((prompt_embeds, prompt_embeds_2), dim=-1)
        text_encoder_projection_dim = self.model.text_encoder_2.config.projection_dim
        add_time_ids = self.model._get_add_time_ids((1024, 1024), (0, 0), (1024, 1024), torch.float16,
                                            text_encoder_projection_dim).to(device)
        added_cond_kwargs = {"text_embeds": pooled_prompt_embeds2, "time_ids": add_time_ids}
        return added_cond_kwargs, prompt_embeds


    def _encode_text_sdxl_with_negative(self, prompt: str) -> tuple[dict[str, T], T]:
        added_cond_kwargs, prompt_embeds = self._encode_text_sdxl(self.model, prompt)
        added_cond_kwargs_uncond, prompt_embeds_uncond = self._encode_text_sdxl(self.model, "")
        prompt_embeds = torch.cat((prompt_embeds_uncond, prompt_embeds, ))
        added_cond_kwargs = {"text_embeds": torch.cat((added_cond_kwargs_uncond["text_embeds"], added_cond_kwargs["text_embeds"])),
                            "time_ids": torch.cat((added_cond_kwargs_uncond["time_ids"], added_cond_kwargs["time_ids"])),}
        return added_cond_kwargs, prompt_embeds
    
  
    def prev_step(self, model_output: T, timestep: int, sample: T):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[int(timestep)]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    def next_step(self, model_output: T, timestep: int, sample: T) -> T:
        timestep, next_timestep = min(timestep - self.model.scheduler.config.num_train_timesteps // self.model.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.model.scheduler.alphas_cumprod[int(timestep)] if timestep >= 0 else self.model.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.model.scheduler.alphas_cumprod[int(next_timestep)]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_only(self, latent: T, t: T, context: T, added_cond_kwargs: dict[str, T]):
        latents_input = torch.cat([latent] * 2)
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context, added_cond_kwargs=added_cond_kwargs)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)

        return noise_pred_uncond, noise_prediction_text

    def get_noise_pred(self, latent: T, t: T, context: T,  added_cond_kwargs: dict[str, T], is_forward=True):
        latents_input = torch.cat([latent] * 2)
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context, added_cond_kwargs=added_cond_kwargs)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @property
    def scheduler(self):
        return self.model.scheduler

    def null_optimization(self, latents, num_inner_steps, epsilon):
        added_cond_kwargs, context = self._encode_text_sdxl_with_negative(self.prompt)
        uncond_embeddings, cond_embeddings = context.chunk(2)

        uncond_embeddings_list = []
        latent_cur = latents[-1]
        bar = tqdm(total=num_inner_steps * self.ddim_steps)
        for i in range(self.ddim_steps):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]

            for j in range(num_inner_steps):
                noise_pred_uncond, noise_pred_cond = self.get_noise_pred_only(latent_cur, t, context, added_cond_kwargs)
                noise_pred_cond.requires_grad = False
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, context, added_cond_kwargs, False)
        bar.close()
        return uncond_embeddings_list


    def invert(self, image_path: str, offsets=(0,0,0,0), num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False):
        
        image_gt = load_1024(image_path, *offsets)
        if verbose:
            print("DDIM inversion...")
        ddim_latents = DDIM_inversion.ddim_inversion(image_gt)
        if verbose:
            print("Null-text optimization...")
        uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
        return image_gt, ddim_latents, uncond_embeddings
        
    
    def __init__(self, model, prompt, ddim_steps, guidance_scale):
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False)
        
        self.guidance_scale = guidance_scale
        self.ddim_steps = ddim_steps
        self.model = model
        self.model.scheduler.set_timesteps(self.ddim_steps)
        self.prompt = prompt


def latent2image(model: StableDiffusionXLPipeline, latents):
        latents = 1 / (model.vae.config.scaling_factor) * latents.detach()
        image = model.vae.decode(latents)['sample']
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).astype(np.uint8)
        return image


def diffusion_step(model, latent, context, t, guidance_scale, added_cond_kwargs):
    latents_input = torch.cat([latent] * 2)
    noise_pred = model.unet(latents_input, t, encoder_hidden_states=context, added_cond_kwargs=added_cond_kwargs)["sample"]
    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latent)["prev_sample"]
    
    return latents

@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt:  List[str],
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    start_time=50,
    return_type='image'
):
    added_cond_kwargs, prompt_embedds = DDIM_inversion._encode_text_sdxl_with_negative(model, prompt)

    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        uncond_emb, cond_emb = prompt_embedds.chunk(2)
        if uncond_embeddings is not None:
            context = torch.cat([uncond_embeddings[i], cond_emb])
        else:
            context = prompt_embedds
        latent = diffusion_step(model, latent, context, t, guidance_scale, added_cond_kwargs)
        
    if return_type == 'image':
        image = latent2image(model.vae, latent)
    else:
        image = latent
    
    return image, latent



