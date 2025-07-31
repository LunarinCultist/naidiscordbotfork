import re
from redbot.core.app_commands import Choice
from collections import OrderedDict

VIEW_TIMEOUT = 5 * 60

MAX_IMAGE_SIZE = 2*1024*1024 
MAX_UPLOADED_IMAGE_SIZE = 1920*1080

DEFAULT_PROMPT = "masterpiece, location, no text"
DEFAULT_NEGATIVE_PROMPT = "blurry, lowres, upscaled, artistic error, film grain, scan artifacts, worst quality, bad quality, jpeg artifacts, very displeasing, chromatic aberration, halftone, multiple views, logo, too many watermarks, negative space, blank page, ai-generated"

RESOLUTION_TITLES = OrderedDict({
    "1024,1024": "Square (1024x1024)",
    
    "832,1216": "Portrait (832x1216)",
    "896,1152": "Portrait (896x1152)",
    "768,1280": "Portrait (768x1280)", 
    "704,1344": "Portrait (704x1344)",
    "640,1408": "Portrait (640x1408)",
    "576,1472": "Portrait (576x1472)",
    "512,1536": "Portrait (512x1536)",
    
    "1216,832": "Landscape (1216x832)",
    "1152,896": "Landscape (1152x896)",
    "1280,768": "Landscape (1280x768)",
    "1344,704": "Landscape (1344x704)",
    "1408,640": "Landscape (1408x640)",
    "1472,576": "Landscape (1472x576)",
    "1536,512": "Landscape (1536x512)",
})


MODELS = OrderedDict({
    "nai-diffusion-4-5-curated": "Anime v4.5 Curated",
    "nai-diffusion-4-5-full": "Anime v4.5 Full"
})

PARAMETER_DESCRIPTIONS = {
    "resolution": "The aspect ratio of your image.",
    "guidance": "The intensity of the prompt.",
    "model": "The model to use for generation.",
}

PARAMETER_CHOICES = {
    "resolution": [Choice(name=title, value=value) for value, title in RESOLUTION_TITLES.items()],
    "model": [Choice(name=title, value=value) for value, title in MODELS.items()],
}

V45_STEPS = 28
V45_DEFAULTS = {
    "params_version": 3,
    "legacy": False,
    "qualityToggle": True,
    "prefer_brownian": True,
    "dynamic_thresholding": False,
    "use_coords": False,
    "use_order": True
}