import re
import discord
from datetime import datetime, timedelta
from discord.ui import View
from novelai_api.ImagePreset import ImagePreset, ImageModel
from typing import Optional

from novelai.constants import VIEW_TIMEOUT


class ImageView(View):
    def __init__(self, cog, prompt: str, preset: ImagePreset, seed: int, model: ImageModel, original_user_id: int):
        super().__init__(timeout=VIEW_TIMEOUT)
        self.cog = cog
        self.prompt = prompt
        self.preset = preset
        self.seed = seed
        self.model = model
        self.original_user_id = original_user_id  # Store the original user ID
        self.deleted = False
        self.message: Optional[discord.Message] = None

    async def message_edit_callback(self, ctx: discord.Interaction):
        if not self.is_finished() and not self.deleted:
            await ctx.message.edit(view=self)

    @discord.ui.button(emoji="🌱", style=discord.ButtonStyle.grey)
    async def seed(self, ctx: discord.Interaction, _: discord.Button):
        embed = discord.Embed(title="Generation seed", description=f"{self.seed}", color=0x77B255)
        await ctx.response.send_message(embed=embed, ephemeral=True)

    @discord.ui.button(emoji="♻", style=discord.ButtonStyle.grey)
    async def recycle(self, ctx: discord.Interaction, btn: discord.Button):
        # Block DMs entirely (matching main cog behavior)
        if not ctx.guild:
            return await ctx.response.send_message("This command can only be used in servers.", ephemeral=True)
    
        # Simple 3 second cooldown check (matching main cog behavior)
        if self.cog.generating.get(ctx.user.id, False):
            content = "Your current image must finish generating before you can request another one."
            return await ctx.response.send_message(content, ephemeral=True)
        
        if ctx.user.id in self.cog.user_last_img and (datetime.now() - self.cog.user_last_img[ctx.user.id]).total_seconds() < 3:
            eta = self.cog.user_last_img[ctx.user.id] + timedelta(seconds=3)
            content = f"You may use this command again {discord.utils.format_dt(eta, 'R')}."
            return await ctx.response.send_message(content, ephemeral=True)

        self.preset.seed = 0
        btn.disabled = True
        await ctx.message.edit(view=self)
        btn.disabled = False  # re-enables it after the task calls back

        content = self.cog.get_loading_message()
        self.cog.queue_add(ctx, self.prompt, self.preset, self.model)
        await ctx.response.send_message(content=content)

    @discord.ui.button(emoji="🗑️", style=discord.ButtonStyle.grey)
    async def delete(self, ctx: discord.Interaction, _: discord.Button):
        # Use the stored original user ID instead of trying to parse it
        if not ctx.guild or ctx.user.id == self.original_user_id or ctx.channel.permissions_for(ctx.user).manage_messages:
            self.deleted = True
            self.stop()
            imagelog = self.cog.bot.get_cog("ImageLog")
            if imagelog:
                imagelog.manual_deleted_by[ctx.message.id] = ctx.user.id
            await ctx.message.delete()
        else:
            await ctx.response.send_message("Only a moderator or the user who requested the image may delete it.", ephemeral=True)

    async def on_timeout(self) -> None:
        if self.message and not self.deleted:
            await self.message.edit(view=None)


class RetryView(View):
    def __init__(self, cog, prompt: str, preset: ImagePreset, model: ImageModel, original_user_id: int):
        super().__init__(timeout=VIEW_TIMEOUT)
        self.cog = cog
        self.prompt = prompt
        self.preset = preset
        self.model = model
        self.original_user_id = original_user_id  # Store the original user ID
        self.deleted = False
        self.message: Optional[discord.Message] = None

    @discord.ui.button(emoji="🔁", style=discord.ButtonStyle.grey)
    async def retry(self, ctx: discord.Interaction, _: discord.Button):
        # Block DMs entirely (matching main cog behavior)
        if not ctx.guild:
            return await ctx.response.send_message("This command can only be used in servers.", ephemeral=True)
    
        # Simple generation check (no special owner privileges needed)
        if self.cog.generating.get(ctx.user.id, False):
            content = "Your current image must finish generating before you can request another one."
            return await ctx.response.send_message(content, ephemeral=True)

        self.deleted = True
        self.stop()
        await ctx.message.edit(view=None)
        content = self.cog.get_loading_message()
        self.cog.queue_add(ctx, self.prompt, self.preset, self.model)
        await ctx.response.send_message(content=content)

    async def on_timeout(self) -> None:
        if self.message and not self.deleted:
            await self.message.edit(view=None)