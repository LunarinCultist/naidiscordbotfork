import io
import json
import base64
import asyncio
import discord
import logging
import re
import requests
from hashlib import md5
from datetime import datetime, timedelta
from typing import Optional, Tuple, Coroutine, Dict, List
from PIL import Image, PngImagePlugin
from redbot.core import commands, app_commands, Config
from redbot.core.bot import Red
from novelai_api import NovelAIError
from novelai_api.ImagePreset import ImageModel, ImagePreset, ImageSampler, ImageGenerationType, UCPreset

import novelai.constants as const
from novelai.naiapi import NaiAPI
from novelai.imageview import ImageView, RetryView

log = logging.getLogger("red.crab-cogs.novelai")

def round_to_nearest(x, base):
    return int(base * round(x/base))

def scale_to_size(width, height, size):
    scale = (size / (width * height)) ** 0.5
    return int(width * scale), int(height * scale)

def fix_discord_emoji_corruption(text):
    import re
    pattern = r'(\d+(?:\.\d+)?):🧑‍🎨([^:]+)::'
    fixed_text = re.sub(pattern, r'\1::artist:\2::', text)
    return fixed_text

def check_nsfw_content(prompt, negative_prompt = "", character_prompts = None):
    nsfw_keywords = [
        "nsfw", "pussy", "gigantic breasts", "large breasts", "cleavage", 
        "penis", "nipple", "cameltoe", "sex", "anal", "fellatio", 
        "bondage", "loli", "fingering", "masturbation", "dildo"
    ]
    
    text_to_check = prompt.lower()
    if negative_prompt:
        text_to_check += " " + negative_prompt.lower()
    
    if character_prompts:
        for char in character_prompts:
            text_to_check += " " + char.get("prompt", "").lower()
            text_to_check += " " + char.get("negative", "").lower()
    
    for keyword in nsfw_keywords:
        if keyword in text_to_check:
            return True
    
def process_prompt_with_text_logic(prompt, base_prompt):
    prompt = prompt.replace(';', ':')
    
    prompt_lower = prompt.lower()
    wants_text = 'text' in prompt_lower or 'english text' in prompt_lower
    
    processed_base = base_prompt
    if wants_text and base_prompt:
        processed_base = re.sub(r',?\s*no text\s*,?', '', base_prompt, flags=re.IGNORECASE)
        processed_base = re.sub(r',\s*,', ',', processed_base)
        processed_base = processed_base.strip(' ,')
    
    if processed_base:
        combined = f"{prompt.strip(' ,')}, {processed_base}" if prompt else processed_base
    else:
        combined = prompt
    
    if combined:
        text_sections = []
        remaining_parts = []
        
        parts = [part.strip() for part in combined.split(',')]
        
        for part in parts:
            if part.strip().lower().startswith('text:'):
                text_sections.append(part.strip())
            elif part.strip():
                remaining_parts.append(part.strip())
        
        if text_sections:
            if remaining_parts:
                combined = ', '.join(remaining_parts) + ', ' + ', '.join(text_sections)
            else:
                combined = ', '.join(text_sections)
        else:
            combined = ', '.join(remaining_parts)
    
    return combined

def parse_vibe_bundle(file_content):
    try:
        bundle_data = json.loads(file_content.decode('utf-8'))
        
        if (bundle_data.get("identifier") != "novelai-vibe-transfer-bundle" or 
            "vibes" not in bundle_data or 
            not isinstance(bundle_data["vibes"], list)):
            log.warning("Invalid vibe bundle structure")
            return None
        
        reference_encodings = []
        reference_strengths = []
        
        for vibe in bundle_data["vibes"]:
            if (vibe.get("identifier") != "novelai-vibe-transfer" or
                "encodings" not in vibe or
                "importInfo" not in vibe):
                log.warning("Invalid vibe structure in bundle")
                continue
            
            vibe_type = vibe.get("type", "image")
            
            encodings = vibe["encodings"]
            v45_encodings = encodings.get("v4-5full", {})
            
            if not v45_encodings:
                log.warning("No v4-5full encodings found in vibe")
                continue
            
            encoding_key = None
            encoding_value = None
            
            if vibe_type == "encoding":
                if "unknown" in v45_encodings:
                    encoding_data = v45_encodings["unknown"]
                    encoding_value = encoding_data.get("encoding")
                else:
                    encoding_key = next(iter(v45_encodings.keys()), None)
                    if encoding_key:
                        encoding_data = v45_encodings[encoding_key]
                        encoding_value = encoding_data.get("encoding")
            else:
                encoding_key = next(iter(v45_encodings.keys()), None)
                if encoding_key:
                    encoding_data = v45_encodings[encoding_key]
                    encoding_value = encoding_key
            
            if not encoding_value:
                log.warning(f"No encoding data found in vibe (type: {vibe_type})")
                continue
            
            reference_encodings.append(encoding_value)
            
            strength = vibe["importInfo"].get("strength", 0.6)
            reference_strengths.append(strength)
        
        if not reference_encodings:
            log.warning("No valid vibes found in bundle")
            return None
        
        if len(reference_encodings) > 4:
            log.warning(f"Vibe bundle has {len(reference_encodings)} vibes (max 4 allowed)")
            return "too_many_vibes"
        
        log.info(f"Successfully parsed vibe bundle with {len(reference_encodings)} vibes")
        return reference_encodings, reference_strengths
        
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        log.warning(f"Failed to parse vibe bundle: {e}")
        return None
    except Exception as e:
        log.error(f"Unexpected error parsing vibe bundle: {e}")
        return None


class NovelAI(commands.Cog):

    def __init__(self, bot):
        super().__init__()
        self.bot = bot
        self.api = None
        self.queue = []
        self.queue_task = None
        self.generating = {}
        self.user_last_img = {}
        self.last_generation_datetime = datetime.min
        self.config = Config.get_conf(self, identifier=66766566169)
        
        defaults_user = {
            "base_prompt": const.DEFAULT_PROMPT,
            "base_negative_prompt": const.DEFAULT_NEGATIVE_PROMPT,
            "resolution": "832,1216",
            "guidance": 5.0,
            "model": "nai-diffusion-4-5-full",
            "auto_text_processing": True,
        }
        
        self.config.register_user(**defaults_user)

    async def cog_load(self):
        await self.try_create_api()

    async def red_delete_data_for_user(self, requester, user_id):
        await self.config.user_from_id(user_id).clear()

    async def try_create_api(self):
        api = await self.bot.get_shared_api_tokens("novelai")
        username, password = api.get("username"), api.get("password")
        if username and password:
            self.api = NaiAPI(username, password)
            return True
        else:
            return False

    async def get_user_model_safe(self, user):
        stored_model = await self.config.user(user).model()
        if stored_model in const.MODELS:
            return stored_model
        else:
            default_model = "nai-diffusion-4-5-full"
            await self.config.user(user).model.set(default_model)
            log.info(f"Reset invalid model '{stored_model}' to default for user {user.id}")
            return default_model

    async def get_user_resolution_safe(self, user):
        stored_resolution = await self.config.user(user).resolution()
        if stored_resolution in const.RESOLUTION_TITLES:
            return stored_resolution
        else:
            default_resolution = "832,1216"
            await self.config.user(user).resolution.set(default_resolution)
            log.info(f"Reset invalid resolution '{stored_resolution}' to default for user {user.id}")
            return default_resolution

    def claim_second_key(self, timeout=1):
        try:
            resp = requests.post('http://localhost:8080/claim', timeout=timeout)
            return resp.json().get('success', False)
        except Exception as e:
            log.warning(f"Failed to claim second key: {e}")
            return False

    def release_second_key(self, timeout=1):
        try:
            requests.post('http://localhost:8080/release', timeout=timeout)
        except Exception as e:
            log.warning(f"Failed to release second key: {e}")

    def check_key_status(self, timeout=1):
        try:
            resp = requests.get('http://localhost:8080/status', timeout=timeout)
            data = resp.json()
            return not data.get('busy', True)
        except Exception as e:
            log.warning(f"Failed to check key status: {e}")
            return False

    async def consume_queue(self):
        new = True
        while self.queue:
            task, ctx = self.queue.pop(0)
            alive = True
            if not new:
                try:
                    await ctx.edit_original_response(content="`Generating image...`")
                except (discord.errors.NotFound, discord.errors.HTTPException):
                    self.generating[ctx.user.id] = False
                    alive = False
                    log.warning(f"Interaction expired for user {ctx.user.id}, skipping generation")
                except Exception:
                    log.exception("Editing message in queue")
            if self.queue:
                _ = asyncio.create_task(self.edit_queue_messages())
            if alive:
                await task
            await asyncio.sleep(2)
            new = False

    async def edit_queue_messages(self):
        tasks = []
        for i, (task, ctx) in enumerate(self.queue):
            try:
                tasks.append(ctx.edit_original_response(content=f"`Position in queue: {i + 1}`"))
            except (discord.errors.NotFound, discord.errors.HTTPException):
                self.generating[ctx.user.id] = False
                continue
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def queue_add(self, ctx, prompt, preset, model, should_spoiler = False):
        self.generating[ctx.user.id] = True
        self.queue.append((self.fulfill_novelai_request(ctx, prompt, preset, model, should_spoiler), ctx))
        if not self.queue_task or self.queue_task.done():
            self.queue_task = asyncio.create_task(self.consume_queue())

    def get_loading_message(self):
        return f"`Position in queue: {len(self.queue) + 1}`" if self.queue_task and not self.queue_task.done() else "`Generating image...`"

    @app_commands.command(name="novelai", description="Generate anime images with NovelAI v4.5.")
    @app_commands.describe(
        prompt="Gets added to your base prompt (/novelaidefaults)",
        negative_prompt="Gets added to your base negative prompt (/novelaidefaults)",
        seed="Random number that determines image generation.",
        vibe_bundle="Upload a (.naiv4vibebundle) file for style transfer. Preferably an encoding only.",
        char1_prompt="First character description (optional)",
        char1_negative="First character negative prompt (optional)", 
        char2_prompt="Second character description (optional)",
        char2_negative="Second character negative prompt (optional)",
        char3_prompt="Third character description (optional)",
        char3_negative="Third character negative prompt (optional)",
        char4_prompt="Fourth character description (optional)",
        char4_negative="Fourth character negative prompt (optional)",
        **const.PARAMETER_DESCRIPTIONS
    )
    @app_commands.choices(**const.PARAMETER_CHOICES)
    async def novelai(self,
                      ctx: discord.Interaction,
                      prompt: str,
                      negative_prompt: Optional[str] = None,
                      seed: Optional[int] = None,
                      vibe_bundle: Optional[discord.Attachment] = None,
                      char1_prompt: Optional[str] = None,
                      char1_negative: Optional[str] = None,
                      char2_prompt: Optional[str] = None, 
                      char2_negative: Optional[str] = None,
                      char3_prompt: Optional[str] = None,
                      char3_negative: Optional[str] = None,
                      char4_prompt: Optional[str] = None,
                      char4_negative: Optional[str] = None,
                      resolution: Optional[str] = None,
                      guidance: Optional[app_commands.Range[float, 0.0, 10.0]] = None,
                      model: Optional[ImageModel] = None):
        
        if not ctx.guild:
            return await ctx.response.send_message("This command can only be used in servers.", ephemeral=True)
                
        if not self.api and not await self.try_create_api():
            return await ctx.response.send_message(
                "NovelAI username and password not set. The bot owner needs to set them like this:\n"
                "[p]set api novelai username,USERNAME\n[p]set api novelai password,PASSWORD")

        if self.generating.get(ctx.user.id, False):
            return await ctx.response.send_message("Your current image must finish generating before you can request another one.", ephemeral=True)
        
        if ctx.user.id in self.user_last_img and (datetime.now() - self.user_last_img[ctx.user.id]).total_seconds() < 3:
            eta = self.user_last_img[ctx.user.id] + timedelta(seconds=3)
            return await ctx.response.send_message(f"You may use this command again {discord.utils.format_dt(eta, 'R')}.", ephemeral=True)

        vibe_data = None
        if vibe_bundle:
            valid_extensions = ['.naiv4vibebundle', '.json']
            if not any(vibe_bundle.filename.lower().endswith(ext) for ext in valid_extensions):
                return await ctx.response.send_message("Vibe bundle must be a .naiv4vibebundle or .json file.", ephemeral=True)
            
            if vibe_bundle.size > 5 * 1024 * 1024:
                return await ctx.response.send_message("Vibe bundle file is too large (max 5MB).", ephemeral=True)
            
            try:
                vibe_file_content = await vibe_bundle.read()
                vibe_data = parse_vibe_bundle(vibe_file_content)
                
                if vibe_data is None:
                    return await ctx.response.send_message("Invalid vibe bundle file. Please ensure it's a valid NovelAI vibe bundle.", ephemeral=True)
                
                if vibe_data == "too_many_vibes":
                    return await ctx.response.send_message("This vibe bundle has more than 4 vibes, which would cost premium currency (Anlas). Please use a bundle with 4 or fewer vibes.", ephemeral=True)
                
                log.info(f"User {ctx.user.id} uploaded vibe bundle with {len(vibe_data[0])} vibes")
                
            except Exception as e:
                log.error(f"Error processing vibe bundle: {e}")
                return await ctx.response.send_message("Failed to process vibe bundle file.", ephemeral=True)

        character_prompts = []
        if char1_prompt:
            character_prompts.append({
                "prompt": char1_prompt,
                "negative": char1_negative or ""
            })
        if char2_prompt:
            character_prompts.append({
                "prompt": char2_prompt, 
                "negative": char2_negative or ""
            })
        if char3_prompt:
            character_prompts.append({
                "prompt": char3_prompt, 
                "negative": char3_negative or ""
            })
        if char4_prompt:
            character_prompts.append({
                "prompt": char4_prompt, 
                "negative": char4_negative or ""
            })

        result = await self.prepare_novelai_request(ctx, prompt, negative_prompt, seed, resolution, guidance, model, character_prompts, vibe_data)
        if not result:
            return
        prompt, preset, model = result
        
        should_spoiler = check_nsfw_content(prompt, negative_prompt, character_prompts)
        
        message = self.get_loading_message()
        self.queue_add(ctx, prompt, preset, model, should_spoiler)
        await ctx.response.send_message(content=message)

    @app_commands.command(name="novelai-img2img", description="Convert img2img with NovelAI v4.5.")
    @app_commands.describe(
        image="The image you want to use as a base for img2img.",
        strength="How much you want the image to change. 0.7 is default.",
        noise="Adds new detail to your image. 0 is default.",
        prompt="Gets added to your base prompt (/novelaidefaults)",
        negative_prompt="Gets added to your base negative prompt (/novelaidefaults)",
        seed="Random number that determines image generation.",
        guidance="The intensity of the prompt.",
        char1_prompt="First character description (optional)",
        char1_negative="First character negative prompt (optional)", 
        char2_prompt="Second character description (optional)",
        char2_negative="Second character negative prompt (optional)",
        char3_prompt="Third character description (optional)",
        char3_negative="Third character negative prompt (optional)",
        char4_prompt="Fourth character description (optional)",
        char4_negative="Fourth character negative prompt (optional)"
    )
    async def novelai_img(self,
                          ctx: discord.Interaction,
                          image: discord.Attachment,
                          strength: app_commands.Range[float, 0.0, 1.0],
                          noise: app_commands.Range[float, 0.0, 1.0],
                          prompt: str,
                          negative_prompt: Optional[str] = None,
                          seed: Optional[int] = None,
                          guidance: Optional[app_commands.Range[float, 0.0, 10.0]] = None,
                          model: Optional[ImageModel] = None,
                          char1_prompt: Optional[str] = None,
                          char1_negative: Optional[str] = None,
                          char2_prompt: Optional[str] = None, 
                          char2_negative: Optional[str] = None,
                          char3_prompt: Optional[str] = None,
                          char3_negative: Optional[str] = None,
                          char4_prompt: Optional[str] = None,
                          char4_negative: Optional[str] = None):
        
        if not ctx.guild:
            return await ctx.response.send_message("This command can only be used in servers.", ephemeral=True)

        if "image" not in image.content_type or not image.width or not image.height or not (image.size / 1024 / 1024) <= 2:
            return await ctx.response.send_message("Image must be valid and less than 2 MB.", ephemeral=True)

        width, height = scale_to_size(image.width, image.height, 1024*1024)
        resolution = f"{round_to_nearest(width, 64)},{round_to_nearest(height, 64)}"
        
        character_prompts = []
        if char1_prompt:
            character_prompts.append({
                "prompt": char1_prompt,
                "negative": char1_negative or ""
            })
        if char2_prompt:
            character_prompts.append({
                "prompt": char2_prompt, 
                "negative": char2_negative or ""
            })
        
        result = await self.prepare_novelai_request(ctx, prompt, negative_prompt, seed, resolution, guidance, model, character_prompts, None)
        if not result:
            return
        await ctx.response.defer()

        prompt, preset, model = result
        
        should_spoiler = check_nsfw_content(prompt, negative_prompt, character_prompts)
        
        preset.strength = strength
        preset.noise = noise
        fp = io.BytesIO()
        await image.save(fp)
        if image.width*image.height > const.MAX_UPLOADED_IMAGE_SIZE:
            try:
                width, height = scale_to_size(image.width, image.height, const.MAX_UPLOADED_IMAGE_SIZE)
                resized_image = Image.open(fp).resize((width, height), Image.Resampling.LANCZOS)
                fp = io.BytesIO()
                resized_image.save(fp, "PNG")
                fp.seek(0)
            except Image.UnidentifiedImageError:
                log.exception("Resizing image")
                return await ctx.followup.send(":warning: Failed to resize image. Please try sending a smaller image.")
        preset.image = base64.b64encode(fp.read()).decode()

        message = self.get_loading_message()
        self.queue_add(ctx, prompt, preset, model, should_spoiler)
        await ctx.edit_original_response(content=message)

    async def prepare_novelai_request(self,
                                    ctx,
                                    prompt,
                                    negative_prompt,
                                    seed,
                                    resolution,
                                    guidance,
                                    model,
                                    character_prompts = None,
                                    vibe_data = None):
        
        if not self.api and not await self.try_create_api():
            return await ctx.response.send_message(
                "NovelAI username and password not set. The bot owner needs to set them like this:\n"
                "[p]set api novelai username,USERNAME\n[p]set api novelai password,PASSWORD")

        if self.generating.get(ctx.user.id, False):
            content = "Your current image must finish generating before you can request another one."
            return await ctx.response.send_message(content, ephemeral=True)
        if ctx.user.id in self.user_last_img and (datetime.now() - self.user_last_img[ctx.user.id]).total_seconds() < 3:
            eta = self.user_last_img[ctx.user.id] + timedelta(seconds=3)
            content = f"You may use this command again {discord.utils.format_dt(eta, 'R')}."
            return await ctx.response.send_message(content, ephemeral=True)

        base_prompt = await self.config.user(ctx.user).base_prompt()
        base_neg = await self.config.user(ctx.user).base_negative_prompt()

        if negative_prompt:
            negative_prompt = negative_prompt.replace(';', ':')
            negative_prompt = fix_discord_emoji_corruption(negative_prompt)

        auto_text_processing = await self.config.user(ctx.user).auto_text_processing()
        if base_prompt and auto_text_processing:
            prompt = process_prompt_with_text_logic(prompt, base_prompt)
        elif base_prompt:
            prompt = f"{prompt.strip(' ,')}, {base_prompt}" if prompt else base_prompt
        elif prompt:
            prompt = prompt
        else:
            prompt = base_prompt or ""
        
        prompt = fix_discord_emoji_corruption(prompt)

        if base_neg:
            negative_prompt = f"{negative_prompt.strip(' ,')}, {base_neg}" if negative_prompt else base_neg
        
        resolution = resolution or await self.get_user_resolution_safe(ctx.user)

        preset = ImagePreset()
        preset.n_samples = 1
        try:
            preset.resolution = tuple(int(num) for num in resolution.split(","))
        except (ValueError, TypeError):
            preset.resolution = (1024, 1024)
            
        preset.uc = negative_prompt or const.DEFAULT_NEGATIVE_PROMPT
        preset.uc_preset = UCPreset.Preset_None
        preset.quality_toggle = False
        
        preset.sampler = ImageSampler.k_euler_ancestral
        preset.scale = guidance if guidance is not None else await self.config.user(ctx.user).guidance()
        preset.cfg_rescale = 0.0
        preset.decrisper = False
        preset.noise_schedule = "karras"
        
        preset._settings["params_version"] = 3
        preset._settings["legacy"] = False
        preset._settings["qualityToggle"] = True
        preset._settings["prefer_brownian"] = True
        preset._settings["deliberate_euler_ancestral_bug"] = False
        preset._settings["steps"] = const.V45_STEPS
        preset._settings["use_coords"] = False
        preset._settings["dynamic_thresholding"] = False
        preset._settings["use_order"] = True
        
        if vibe_data:
            reference_encodings, reference_strengths = vibe_data
            preset._settings["reference_image_multiple"] = reference_encodings
            preset._settings["reference_strength_multiple"] = reference_strengths
            log.info(f"Added {len(reference_encodings)} vibe references to request")
        
        if character_prompts:
            characters = []
            for char in character_prompts:
                char_prompt = char["prompt"].replace(';', ':')
                char_negative = char["negative"].replace(';', ':')
                
                char_prompt = fix_discord_emoji_corruption(char_prompt)
                char_negative = fix_discord_emoji_corruption(char_negative)
                
                characters.append({
                    "prompt": char_prompt,
                    "uc": char_negative,
                    "position": ""
                })
            preset.characters = characters
        
        model = model or ImageModel(await self.get_user_model_safe(ctx.user))
        preset.seed = seed if seed else 0
        
        return prompt, preset, model

    async def fulfill_novelai_request(self,
                                      ctx,
                                      prompt,
                                      preset,
                                      model,
                                      should_spoiler = False):
        try:
            if ctx.is_expired():
                log.warning(f"Interaction expired for user {ctx.user.id} before generation started")
                self.generating[ctx.user.id] = False
                return
        except:
            pass
            
        second_key_claimed = False
        max_wait_time = 60 
        wait_start = datetime.now()
        
        while not second_key_claimed and (datetime.now() - wait_start).total_seconds() < max_wait_time:
            second_key_claimed = self.claim_second_key()
            if not second_key_claimed:
                try:
                    remaining_time = max_wait_time - (datetime.now() - wait_start).total_seconds()
                    if remaining_time > 0:
                        await ctx.edit_original_response(content=f"`Waiting for second key... ({int(remaining_time)}s)`")
                        await asyncio.sleep(3)
                    else:
                        break
                except (discord.errors.NotFound, discord.errors.HTTPException):
                    log.warning(f"Interaction expired for user {ctx.user.id} during key wait")
                    self.generating[ctx.user.id] = False
                    if second_key_claimed:
                        self.release_second_key()
                    return
        
        if not second_key_claimed:
            log.info("Second key remained busy for too long, proceeding with single generation")
            try:
                await ctx.edit_original_response(content="`Second key busy - generating single image...`")
            except (discord.errors.NotFound, discord.errors.HTTPException):
                log.warning(f"Interaction expired for user {ctx.user.id} during status update")
                self.generating[ctx.user.id] = False
                return
        else:
            log.info("Successfully claimed second key for generation")
            vibe_status = ""
            if preset._settings.get("reference_image_multiple"):
                vibe_count = len(preset._settings["reference_image_multiple"])
                vibe_status = f" with {vibe_count} vibes"
            try:
                await ctx.edit_original_response(content=f"`Generating image{vibe_status}...`")
            except (discord.errors.NotFound, discord.errors.HTTPException):
                log.warning(f"Interaction expired for user {ctx.user.id} during status update")
                self.generating[ctx.user.id] = False
                if second_key_claimed:
                    self.release_second_key()
                return
        
        try:
            while (datetime.now() - self.last_generation_datetime).total_seconds() < 2:
                await asyncio.sleep(1)
            
            try:
                for retry in range(4):
                    try:
                        async with self.api as wrapper:
                            action = ImageGenerationType.IMG2IMG if preset._settings.get("image", None) else ImageGenerationType.NORMAL
                            self.last_generation_datetime = datetime.now()
                            
                            async for _, img in wrapper.api.high_level.generate_image(prompt, model, preset, action):
                                image_bytes = img
                            break
                    except NovelAIError as error:
                        if error.status not in (500, 520, 408, 522, 524) or retry == 3:
                            raise
                        log.warning("NovelAI encountered an error." if error.status in (500, 520) else "Timed out.")
                        if retry == 1:
                            try:
                                await ctx.edit_original_response(content="`Generating image...` :warning:")
                            except (discord.errors.NotFound, discord.errors.HTTPException):
                                log.warning(f"Interaction expired for user {ctx.user.id} during retry warning")
                                return
                        await asyncio.sleep(retry + 2)
            except Exception as error:
                view = RetryView(self, prompt, preset, model, ctx.user.id)
                if isinstance(error, (discord.errors.NotFound, discord.errors.HTTPException)):
                    log.warning(f"Interaction expired for user {ctx.user.id} during generation")
                    return
                if isinstance(error, NovelAIError):
                    if error.status == 401:
                        content = ":warning: Failed to authenticate NovelAI account."
                    elif error.status == 402:
                        content = ":warning: The subscription and/or credits have run out for this NovelAI account."
                    elif error.status in (500, 520, 408, 522, 524):
                        content = "NovelAI seems to be experiencing an outage, and multiple retries have failed. Please be patient and try again soon."
                        view = None
                    elif error.status == 429:
                        content = "Bot is not allowed to generate multiple images at the same time. Please wait a minute."
                        view = None
                    elif error.status == 400:
                        content = "Failed to generate image: " + (error.message or "A validation error occured.")
                    elif error.status == 409:
                        content = "Failed to generate image: " + (error.message or "A conflict error occured.")
                    else:
                        content = f"Failed to generate image: Error {error.status}."
                    log.warning(content)
                else:
                    content = "Failed to generate image! Contact the bot owner if the problem persists."
                    log.error(f"Generating image: {type(error).__name__} - {error}")
                
                try:
                    msg = await ctx.edit_original_response(content=content, view=view)
                    if view:
                        view.message = msg
                except (discord.errors.NotFound, discord.errors.HTTPException):
                    log.warning(f"Interaction expired for user {ctx.user.id} during error response")
                return
            finally:
                self.generating[ctx.user.id] = False
                self.user_last_img[ctx.user.id] = datetime.now()

                image = Image.open(io.BytesIO(image_bytes))
                comment = json.loads(image.info["Comment"])
                seed = comment["seed"]
                del comment["signed_hash"]
                image.info["Comment"] = json.dumps(comment)
                pnginfo = PngImagePlugin.PngInfo()
                for key, val in image.info.items():
                    pnginfo.add_text(str(key), str(val))
                fp = io.BytesIO()
                image.save(fp, "png", pnginfo=pnginfo)
                fp.seek(0)
                image_bytes = fp.read()

                resolution = preset._settings.get('resolution', (1024, 1024))
                if hasattr(resolution, 'value'):
                    resolution = resolution.value
                resolution_str = f"{resolution[0]}x{resolution[1]}"
                
                vibe_info = ""
                if preset._settings.get("reference_image_multiple"):
                    vibe_count = len(preset._settings["reference_image_multiple"])
                    vibe_info = f", Vibes: {vibe_count}"
                
                spoiler_info = ", Spoilered: Yes" if should_spoiler else ""
                
                log.info(f"Generated image - User: {ctx.user.name} ({ctx.user.id}), "
                        f"Model: {model}, Resolution: {resolution_str}, Seed: {seed}{vibe_info}{spoiler_info}")
                log.info(f"Base Prompt: {prompt}")
                log.info(f"Base Negative prompt: {preset._settings.get('uc', '')}")
                
                characters = preset._settings.get('characters', [])
                if characters:
                    for i, char in enumerate(characters, 1):
                        log.info(f"Character {i} Prompt: {char.get('prompt', '')}")
                        if char.get('uc'):
                            log.info(f"Character {i} Negative: {char.get('uc', '')}")

                name = md5(image_bytes).hexdigest() + ".png"
                
                if should_spoiler:
                    name = f"SPOILER_{name}"
                
                file = discord.File(io.BytesIO(image_bytes), name)
                view = ImageView(self, prompt, preset, seed, model, ctx.user.id)
                
                try:
                    msg = await ctx.edit_original_response(content=None, attachments=[file], view=view)
                    view.message = msg

                    imagescanner = self.bot.get_cog("ImageScanner")
                    if imagescanner:
                        if imagescanner.always_scan_generated_images or ctx.channel.id in imagescanner.scan_channels:
                            img_info = imagescanner.convert_novelai_info(image.info)
                            imagescanner.image_cache[msg.id] = ({1: img_info}, {1: image_bytes})
                            await msg.add_reaction("🔎")
                except (discord.errors.NotFound, discord.errors.HTTPException):
                    log.warning(f"Interaction expired for user {ctx.user.id} during final response")
                    return
        except (discord.errors.NotFound, discord.errors.HTTPException):
            log.warning(f"Interaction expired for user {ctx.user.id}")
        except Exception:
            log.exception("Fulfilling request")
        finally:
            if second_key_claimed:
                self.release_second_key()
            self.generating[ctx.user.id] = False

    @app_commands.command(name="novelaidefaults", description="Views or updates your personal default values for /novelai")
    @app_commands.describe(
        base_prompt="Gets added after each prompt. \"none\" to delete, \"default\" to reset.",
        base_negative_prompt="Gets added after each negative prompt. \"none\" to delete, \"default\" to reset.",
        auto_text_processing="Enable/disable automatic 'no text' removal when text tags are detected.",
        **const.PARAMETER_DESCRIPTIONS
    )
    @app_commands.choices(**const.PARAMETER_CHOICES)
    async def novelaidefaults(self,
                              ctx: discord.Interaction,
                              base_prompt: Optional[str] = None,
                              base_negative_prompt: Optional[str] = None,
                              auto_text_processing: Optional[bool] = None,
                              resolution: Optional[str] = None,
                              guidance: Optional[app_commands.Range[float, 0.0, 10.0]] = None,
                              model: Optional[ImageModel] = None):
        
        if base_prompt is not None:
            base_prompt = base_prompt.strip(" ,")
            if base_prompt.lower() == "none":
                base_prompt = None
            elif base_prompt.lower() == "default":
                base_prompt = const.DEFAULT_PROMPT
            await self.config.user(ctx.user).base_prompt.set(base_prompt)
        
        if base_negative_prompt is not None:
            base_negative_prompt = base_negative_prompt.strip(" ,")
            if base_negative_prompt.lower() == "none":
                base_negative_prompt = None
            elif base_negative_prompt.lower() == "default":
                base_negative_prompt = const.DEFAULT_NEGATIVE_PROMPT
            await self.config.user(ctx.user).base_negative_prompt.set(base_negative_prompt)
        
        if auto_text_processing is not None:
            await self.config.user(ctx.user).auto_text_processing.set(auto_text_processing)
        
        if resolution is not None:
            await self.config.user(ctx.user).resolution.set(resolution)
        
        if guidance is not None:
            await self.config.user(ctx.user).guidance.set(guidance)
        
        if model is not None:
            await self.config.user(ctx.user).model.set(model.value)

        embed = discord.Embed(title="NovelAI v4.5 default settings", color=0xffffff)
        prompt = str(await self.config.user(ctx.user).base_prompt())
        neg = str(await self.config.user(ctx.user).base_negative_prompt())
        embed.add_field(name="Base prompt", value=prompt[:1000] + "..." if len(prompt) > 1000 else prompt, inline=False)
        embed.add_field(name="Base negative prompt", value=neg[:1000] + "..." if len(neg) > 1000 else neg, inline=False)
        
        resolution_value = await self.get_user_resolution_safe(ctx.user)
        embed.add_field(name="Resolution", value=const.RESOLUTION_TITLES[resolution_value])
        embed.add_field(name="Guidance", value=f"{await self.config.user(ctx.user).guidance():.1f}")
        
        model_value = await self.get_user_model_safe(ctx.user)
        embed.add_field(name="Model", value=const.MODELS[model_value])
        embed.add_field(name="Auto Text Processing", value="Enabled" if await self.config.user(ctx.user).auto_text_processing() else "Disabled")
        embed.add_field(name="Sampler", value="Euler Ancestral (Fixed)")
        embed.add_field(name="Noise Schedule", value="Karras (Fixed)")
        await ctx.response.send_message(embed=embed, ephemeral=True)