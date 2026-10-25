"""Hunyuan3D-2.1 pipeline: PBR textures (albedo + metallic/roughness) for single-photo
and text jobs. Three-photo jobs are forwarded to the 2.0 app (api.py), because 2.1 has
no multi-view shape model.

Deployed as its own Modal app so the 2.0 endpoint keeps serving while this one is
compared and rolled out. It reuses the "hunyuan3d-weights" volume.

    modal run inference/api_21.py::download_weights   # once, fills the weights volume
    modal deploy inference/api_21.py

Licence: Tencent Hunyuan 3D 2.1 Community License (commercial use allowed; not in the
EU, UK or South Korea; separate licence above 1M monthly active users). Some source
headers say "NON-COMMERCIAL"; the 2.0 code has the same headers. The LICENSE file is
the Community licence.
"""

import io
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import modal

# Pinned: the repo is not a package and its layout has changed between commits.
HUNYUAN21_REPO = "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1"
HUNYUAN21_COMMIT = "82920d643c0dc2f7bfd7255f45f62d386edfe60c"
REPO_DIR = "/root/hy21"
HF_CACHE = "/root/.cache/huggingface"

hunyuan21_image = (
    # 2.1 is tested on Python 3.10 + PyTorch 2.5.1/CUDA 12.4; bpy 4.0 needs Python 3.10.
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install(
        "git",
        "wget",
        "build-essential",
        # OpenCV / Open3D / headless Blender (bpy) runtime libraries
        "libgl1",
        "libglib2.0-0",
        "libegl1",
        "libopengl0",
        "libsm6",
        "libxext6",
        "libxrender1",
        "libxi6",
        "libxkbcommon0",
        "libxxf86vm1",
        "libxfixes3",
    )
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        # L40S is sm_89; 8.6 keeps an A10G fallback possible.
        "TORCH_CUDA_ARCH_LIST": "8.6;8.9",
        # The shape model downloads to HY3DGEN_MODELS, not the Hugging Face cache.
        # Point it at the volume so it isn't fetched again on every cold start.
        "HY3DGEN_MODELS": f"{HF_CACHE}/hy3dgen",
    })
    .run_commands(
        "python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124",
    )
    # requirements.txt at the pinned commit, minus the demo UI (gradio, pythreejs),
    # training (deepspeed) and Tencent's PyPI mirrors; plus our own needs.
    .pip_install(
        "ninja==1.11.1.1",
        "pybind11==2.13.4",
        "transformers==4.46.0",
        "diffusers==0.30.0",
        "accelerate==1.1.1",
        "pytorch-lightning==1.9.5",
        "huggingface-hub==0.30.2",
        "safetensors==0.4.4",
        "numpy==1.24.4",
        "scipy==1.14.1",
        "einops==0.8.0",
        "pandas==2.2.2",
        "opencv-python==4.10.0.84",
        "imageio==2.36.0",
        "scikit-image==0.24.0",
        "rembg==2.0.65",
        "realesrgan==0.3.0",
        "tb_nightly==2.18.0a20240726",
        "basicsr==1.4.2",
        "trimesh==4.4.7",
        "pymeshlab==2022.2.post3",
        "pygltflib==1.16.3",
        "xatlas==0.0.9",
        "open3d==0.18.0",
        "omegaconf==2.3.0",
        "pyyaml==6.0.2",
        "configargparse==1.7",
        "fastapi[standard]==0.115.12",
        "tqdm==4.66.5",
        "psutil==6.0.0",
        "cupy-cuda12x==13.4.1",
        "bpy==4.0",
        "onnxruntime==1.16.3",
        "torchmetrics==1.6.0",
        "pydantic==2.10.6",
        "timm",
        "torchdiffeq",
        # ours: S3, BiRefNet background removal, trimesh decimation fallback
        "boto3",
        "kornia",
        "fast-simplification",
    )
    .run_commands(
        f"git clone {HUNYUAN21_REPO} {REPO_DIR} && cd {REPO_DIR} && git checkout {HUNYUAN21_COMMIT}",
        f"cd {REPO_DIR}/hy3dpaint/custom_rasterizer && python -m pip install --no-build-isolation -e .",
        f"cd {REPO_DIR}/hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh",
        f"mkdir -p {REPO_DIR}/hy3dpaint/ckpt && wget -q -O {REPO_DIR}/hy3dpaint/ckpt/RealESRGAN_x4plus.pth "
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        # Fail the build here, not on the first job, if Blender can't load headless.
        "python -c \"import bpy; print('bpy', bpy.app.version_string)\"",
    )
    # The paint config uses paths relative to the repo root (hy3dpaint/cfgs/...).
    .workdir(REPO_DIR)
)

weights_volume = modal.Volume.from_name("hunyuan3d-weights", create_if_missing=True)
app = modal.App("hunyuan3d-21-pipeline")

# The 2.0 app (api.py) still handles three-photo jobs.
LEGACY_APP = "hunyuan3d-2-multimodal-pipeline"
LEGACY_CLASS = "Hunyuan3DMultimodalModel"

# Tunables
MODEL_REPO = "tencent/Hunyuan3D-2.1"
MAX_FACES_FOR_TEXTURING = 80_000
OCTREE_RESOLUTION = 384
SHAPE_SEED = 2025
# Texture: 6 views at 512 px. The paint pipeline bakes at 4096 and saves at half size,
# so the GLB gets 2048 px albedo / metallic / roughness maps.
PAINT_VIEWS = 6
PAINT_RESOLUTION = 512

T2I_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
T2I_STEPS = 30
T2I_GUIDANCE = 7.0
T2I_PROMPT_SUFFIX = (
    ", single object, centered, full view, plain white background, "
    "soft even studio lighting, high detail, 3D asset"
)
T2I_NEGATIVE_PROMPT = (
    "multiple objects, cropped, cut off, text, watermark, busy background, "
    "harsh shadows, blurry, low quality"
)

BIREFNET_MODEL = "ZhengPeng7/BiRefNet"
BIREFNET_REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"


def use_repo_modules():
    """The 2.1 repo isn't installable; its packages are imported from the checkout."""
    import sys

    for path in (f"{REPO_DIR}/hy3dpaint", f"{REPO_DIR}/hy3dshape", REPO_DIR):
        if path not in sys.path:
            sys.path.insert(0, path)
    # basicsr (used by Real-ESRGAN) imports a torchvision module removed in 0.17.
    from torchvision_fix import apply_fix

    apply_fix()


class BiRefNetRemover:
    """Background remover: RGB image in, RGBA cutout out."""

    def __init__(self):
        import torch
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation

        self.model = AutoModelForImageSegmentation.from_pretrained(
            BIREFNET_MODEL, revision=BIREFNET_REVISION, trust_remote_code=True
        )
        self.model.to("cuda").eval().half()
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.torch = torch

    def __call__(self, image):
        from torchvision.transforms.functional import to_pil_image

        image = image.convert("RGB")
        batch = self.transform(image).unsqueeze(0).to("cuda").half()
        with self.torch.no_grad():
            mask = self.model(batch)[-1].sigmoid()[0].float().cpu()
        cutout = image.copy()
        cutout.putalpha(to_pil_image(mask).resize(image.size))
        return cutout


def cleanup_and_decimate(mesh, max_faces: int):
    """Remove floaters/degenerate faces and decimate. Falls back to trimesh if pymeshlab fails."""
    from hy3dshape.postprocessors import DegenerateFaceRemover, FaceReducer, FloaterRemover

    try:
        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        mesh = FaceReducer()(mesh, max_facenum=max_faces)
    except Exception as e:
        print(f"--> pymeshlab postprocessing failed ({e}); using trimesh fallback", flush=True)
        parts = mesh.split(only_watertight=False)
        mesh = max(parts, key=lambda m: len(m.faces))
        mesh.remove_unreferenced_vertices()
        if len(mesh.faces) > max_faces:
            mesh = mesh.simplify_quadric_decimation(face_count=max_faces)
    return mesh


def describe_pbr(glb_path: str) -> str:
    """Which PBR maps the exported GLB actually carries (the point of 2.1)."""
    from pygltflib import GLTF2

    gltf = GLTF2().load(glb_path)
    if not gltf.materials:
        return "no materials"
    pbr = gltf.materials[0].pbrMetallicRoughness
    maps = []
    if pbr and pbr.baseColorTexture is not None:
        maps.append("albedo")
    if pbr and pbr.metallicRoughnessTexture is not None:
        maps.append("metallic-roughness")
    if gltf.materials[0].normalTexture is not None:
        maps.append("normal")
    return ", ".join(maps) or "no texture maps"


@app.function(image=hunyuan21_image, volumes={HF_CACHE: weights_volume}, timeout=3600)
def download_weights():
    """Run once: puts every model this app loads on the weights volume."""
    from huggingface_hub import snapshot_download

    use_repo_modules()
    from hy3dshape.utils.utils import smart_load_model

    # Shape model goes to HY3DGEN_MODELS (on the volume), via the repo's own loader.
    smart_load_model(MODEL_REPO, subfolder="hunyuan3d-dit-v2-1", use_safetensors=False, variant="fp16")
    snapshot_download(MODEL_REPO, allow_patterns=["hunyuan3d-paintpbr-v2-1/*"])
    snapshot_download("facebook/dinov2-giant")
    snapshot_download(BIREFNET_MODEL, revision=BIREFNET_REVISION)
    snapshot_download(T2I_MODEL, allow_patterns=["*.json", "*.txt", "*fp16.safetensors"])
    weights_volume.commit()
    print("--> Weights are on the volume.")


@app.cls(
    image=hunyuan21_image,
    # Texturing alone peaks around 21 GB and shape + texture around 29 GB, so both
    # pipelines stay resident on a 48 GB L40S instead of being swapped per stage.
    gpu="L40S",
    volumes={HF_CACHE: weights_volume},
    cpu=8,  # UV unwrap / baking / inpainting are CPU-bound
    memory=32768,
    timeout=1800,
    retries=0,
    max_containers=1,
    scaledown_window=120,
)
class Hunyuan21Model:
    @modal.enter()
    def setup(self):
        use_repo_modules()
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        from hy3dshape.rembg import BackgroundRemover
        from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

        print("--> Loading Hunyuan3D-2.1 shape pipeline...", flush=True)
        self.shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(MODEL_REPO)

        print("--> Loading Hunyuan3D-2.1 PBR paint pipeline...", flush=True)
        config = Hunyuan3DPaintConfig(PAINT_VIEWS, PAINT_RESOLUTION)
        # The class defaults assume the hy3dpaint folder as working directory; the
        # repo's own demo overrides them like this when run from the repo root.
        config.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
        config.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
        config.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
        self.paint_pipeline = Hunyuan3DPaintPipeline(config)

        try:
            self.bg_remover = BiRefNetRemover()
            print("--> Background remover: BiRefNet")
        except Exception as e:
            print(f"--> BiRefNet failed to load ({e}); falling back to rembg", flush=True)
            self.bg_remover = BackgroundRemover()
        self.t2i_pipeline = None

    @staticmethod
    def access_aws():
        import boto3

        region = os.environ.get("AWS_REGION", "us-east-1")
        token = os.environ.get("MODAL_IDENTITY_TOKEN")
        if not token:
            return boto3.client("s3", region_name=region)

        sts = boto3.client("sts", region_name=region)
        assumed_role = sts.assume_role_with_web_identity(
            RoleArn="arn:aws:iam::449563540770:role/ModalWebIdentity",
            RoleSessionName="modal-session",
            WebIdentityToken=token,
        )
        creds = assumed_role["Credentials"]
        return boto3.client(
            "s3",
            aws_access_key_id=creds["AccessKeyId"],
            aws_secret_access_key=creds["SecretAccessKey"],
            aws_session_token=creds["SessionToken"],
        )

    def resolve_image(self, image_bytes, prompt: Optional[str]):
        from PIL import Image

        if image_bytes:
            print("--> Input type: Direct Image")
            input_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        else:
            if self.t2i_pipeline is None:
                import torch
                from diffusers import AutoPipelineForText2Image

                print("--> Loading text-to-image pipeline on demand...")
                self.t2i_pipeline = AutoPipelineForText2Image.from_pretrained(
                    T2I_MODEL,
                    torch_dtype=torch.float16,
                    variant="fp16",
                )
                self.t2i_pipeline.enable_model_cpu_offload()

            print(f"--> Input type: Text Prompt ('{prompt}')")
            print("--> Stage 0: Generating 2D reference image from text...")
            input_image = self.t2i_pipeline(
                prompt + T2I_PROMPT_SUFFIX,
                negative_prompt=T2I_NEGATIVE_PROMPT,
                num_inference_steps=T2I_STEPS,
                guidance_scale=T2I_GUIDANCE,
                width=1024,
                height=1024,
            ).images[0]

        # Real transparency (a cut-out) skips background removal.
        if input_image.mode == "RGB":
            print("--> Removing background...", flush=True)
            input_image = self.bg_remover(input_image)
        elif input_image.mode == "RGBA" and input_image.getchannel("A").getextrema()[0] == 255:
            print("--> Removing background...", flush=True)
            input_image = self.bg_remover(input_image.convert("RGB"))
        return input_image

    @modal.method()
    def generate_3d(
        self,
        image_bytes: Optional[bytes],
        prompt: Optional[str],
        job_id: Optional[str] = None,
        output_bucket: Optional[str] = None,
    ) -> bytes | str:
        import shutil

        import torch

        torch.cuda.reset_peak_memory_stats()
        job_started = time.perf_counter()

        def log_stage(stage: str, started: float) -> None:
            torch.cuda.synchronize()
            peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"--> {stage} completed in {time.perf_counter() - started:.1f}s; peak GPU {peak_gb:.1f} GiB", flush=True)

        started = time.perf_counter()
        input_image = self.resolve_image(image_bytes, prompt)
        log_stage("Image preparation", started)

        print("--> Stage 1: Generating 3D mesh geometry (2.1)...", flush=True)
        started = time.perf_counter()
        mesh = self.shape_pipeline(
            image=input_image,
            octree_resolution=OCTREE_RESOLUTION,
            generator=torch.manual_seed(SHAPE_SEED),
        )[0]
        log_stage("Stage 1 shape generation", started)
        print(f"--> Raw mesh faces: {len(mesh.faces)}", flush=True)

        started = time.perf_counter()
        mesh = cleanup_and_decimate(mesh, MAX_FACES_FOR_TEXTURING)
        print(f"--> Cleanup/decimation in {time.perf_counter() - started:.1f}s; faces: {len(mesh.faces)}", flush=True)

        # The 2.1 paint pipeline works on files: OBJ in, OBJ + GLB out.
        work_dir = tempfile.mkdtemp(prefix="hy21-")
        try:
            white_mesh_path = os.path.join(work_dir, "white_mesh.obj")
            mesh.export(white_mesh_path)

            print("--> Stage 2: Painting PBR textures (2.1)...", flush=True)
            started = time.perf_counter()
            textured_obj = self.paint_pipeline(
                mesh_path=white_mesh_path,
                image_path=input_image,
                output_mesh_path=os.path.join(work_dir, "textured_mesh.obj"),
                use_remesh=False,  # already cleaned and decimated above
                save_glb=True,
            )
            log_stage("Stage 2 PBR texture painting", started)

            # convert_obj_to_glb swallows Blender errors and returns False; check the file
            # so a failed export fails the job instead of uploading nothing.
            glb_path = Path(textured_obj).with_suffix(".glb")
            if not glb_path.exists() or glb_path.stat().st_size == 0:
                raise RuntimeError(f"GLB export failed: {glb_path} is missing or empty.")
            glb_data = glb_path.read_bytes()
            print(
                f"--> GLB {len(glb_data) / (1024 ** 2):.1f} MiB, maps: {describe_pbr(str(glb_path))}; "
                f"total generation time {time.perf_counter() - job_started:.1f}s",
                flush=True,
            )
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

        if output_bucket:
            if not job_id:
                raise ValueError("job_id is required when output_bucket is provided.")
            output_key = f"assets/{job_id}.glb"
            self.access_aws().put_object(
                Bucket=output_bucket,
                Key=output_key,
                Body=glb_data,
                ContentType="model/gltf-binary",
            )
            print(f"--> Saved generated model to s3://{output_bucket}/{output_key}")
            return output_key

        return glb_data


@app.function(image=hunyuan21_image)
# requires_proxy_auth: Modal rejects requests (401) without valid Modal-Key and
# Modal-Secret headers, before a container starts. Same contract as api.py, so the
# caller only needs this endpoint's URL.
@modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
def trigger_job(payload: dict):
    from fastapi.responses import JSONResponse

    bucket = payload.get("bucket")
    key = payload.get("key")
    prompt = payload.get("prompt")
    input_mode = payload.get("input_mode", "single")
    views = payload.get("views")
    job_id = payload.get("jobId")
    output_bucket = payload.get("output_bucket")

    if not job_id or not output_bucket:
        return JSONResponse(status_code=400, content={"message": "jobId and output_bucket are required."})

    model = Hunyuan21Model()
    s3 = model.access_aws()

    if input_mode == "multiview":
        # 2.1 has no multi-view shape model: hand the job to the 2.0 app unchanged.
        required_views = ("front", "left", "back")
        if not views or any(
            not views.get(view, {}).get("bucket") or not views.get(view, {}).get("key")
            for view in required_views
        ):
            return JSONResponse(
                status_code=400,
                content={"message": "Multiview jobs require front, left, and back S3 references."},
            )
        multiview_images = {
            view: s3.get_object(Bucket=views[view]["bucket"], Key=views[view]["key"])["Body"].read()
            for view in required_views
        }
        legacy = modal.Cls.from_name(LEGACY_APP, LEGACY_CLASS)()
        legacy.generate_3d.spawn(
            image_bytes=None,
            prompt=prompt,
            job_id=job_id,
            output_bucket=output_bucket,
            model_mode="multiview",
            multiview_images=multiview_images,
        )
        print("Forwarded multiview job to the Hunyuan3D 2.0 app")
    elif input_mode == "single" and bucket and key:
        image_bytes = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        print(f"Running 2.1 Image-to-3D for file: {bucket}/{key}")
        model.generate_3d.spawn(image_bytes=image_bytes, prompt=prompt, job_id=job_id, output_bucket=output_bucket)
    elif input_mode == "prompt" and prompt:
        print(f"Running 2.1 Text-to-3D with prompt: '{prompt}'")
        model.generate_3d.spawn(image_bytes=None, prompt=prompt, job_id=job_id, output_bucket=output_bucket)
    else:
        return JSONResponse(
            status_code=400,
            content={"message": "Provide a single image, front/left/back images, or a prompt."},
        )

    return JSONResponse(
        status_code=202,
        content={
            "message": "Job accepted for background processing.",
            "jobId": job_id,
            "outputKey": f"assets/{job_id}.glb",
        },
    )


@app.local_entrypoint()
def main(
    bucket: Optional[str] = None,
    key: Optional[str] = None,
    prompt: Optional[str] = None,
    output_path: str = "output_21.glb",
):
    """Try one job without S3 output, e.g. for the comparison set:
    modal run inference/api_21.py --bucket my-bucket --key uploads/x/input.jpg
    """
    model = Hunyuan21Model()
    image_bytes = None
    if bucket and key:
        image_bytes = model.access_aws().get_object(Bucket=bucket, Key=key)["Body"].read()
    elif not prompt:
        raise SystemExit("Pass --bucket and --key for an image, or --prompt for text.")

    glb_data = model.generate_3d.remote(image_bytes=image_bytes, prompt=prompt)
    Path(output_path).write_bytes(glb_data)
    print(f"Saved {output_path}")
