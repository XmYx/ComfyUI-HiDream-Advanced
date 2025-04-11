# pipeline_hidream_image_to_image.py
from .pipeline_hidream_image import HiDreamImagePipeline
from .pipeline_output import HiDreamImagePipelineOutput

class HiDreamImageToImagePipeline(HiDreamImagePipeline):
    @torch.no_grad()
    def __call__(
        self,
        init_image=None,  # Add init_image parameter
        denoising_strength=0.75,  # Add denoising strength
        # All other existing parameters
    ):
        # Override implementation for img2img
        # Most of the code will be similar to original with modifications for init_image
