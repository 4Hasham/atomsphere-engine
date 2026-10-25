import io
import os
import time
from pathlib import Path
from typing import Optional

import modal

# 1. Image definition with hy3dgen and 2D T2I support
hunyuan_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "git",
        "ffmpeg",
        "libgl1",
        "libglib2.0-0",
        "libopengl0",
        "libglu1-mesa",
        "libxrender1",
        "libxext6",
        "libsm6",
        "libxkbcommon0",
        "build-essential",
        "gcc-11",
        "g++-11",
        "cmake",
    )
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .run_commands(
        "python3 -m pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "boto3",
        "trimesh",
        "transformers==4.48.3",
        "diffusers==0.32.2",
        "huggingface_hub",
        "accelerate",
        "einops",
        "ninja",
        "pybind11",
        "fastapi[standard]",
        "pydantic",
        "rembg",
        "onnxruntime",
        "fast-simplification",
        # BiRefNet background removal (remote code needs these)
        "timm",
        "kornia",
    )
    .run_commands(
        "git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2 /root/Hunyuan3D-2",
    )
    .workdir("/root/Hunyuan3D-2")
    .run_commands(
        "pip install -e .",
        # Fails the image build early if pymeshlab can't load PLY (missing plugin libs)
        "python3 -c \"import trimesh, pymeshlab; trimesh.creation.box().export('/tmp/t.ply'); ms = pymeshlab.MeshSet(); ms.load_new_mesh('/tmp/t.ply'); print('pymeshlab OK')\"",
        "export CC=gcc-11 CXX=g++-11 TORCH_CUDA_ARCH_LIST=8.6 CUDA_HOME=/usr/local/cuda && cd hy3dgen/texgen/custom_rasterizer && python3 setup.py install",
        "export CC=gcc-11 CXX=g++-11 TORCH_CUDA_ARCH_LIST=8.6 CUDA_HOME=/usr/local/cuda && cd hy3dgen/texgen/differentiable_renderer && python3 setup.py build_ext --inplace",
    )
)

weights_volume = modal.Volume.from_name("hunyuan3d-weights", create_if_missing=True)
app = modal.App("hunyuan3d-2-multimodal-pipeline")

# Tunables
MAX_FACES_FOR_TEXTURING = 80_000
OCTREE_RESOLUTION = 384  # Hunyuan's default; 256 gave soft, blobby edges. 512 = max detail, slower
SHAPE_SEED = 2025  # fixed so the same input always gives the same mesh

# Text-to-image reference. SDXL base is OpenRAIL++ (commercial use allowed);
# sdxl-turbo was a non-commercial license and only 512 px at 4 steps.
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

# BiRefNet (MIT) gives cleaner cutout edges than rembg; edge halos end up smeared
# into the texture. Pinned because loading it runs code from the model repo.
BIREFNET_MODEL = "ZhengPeng7/BiRefNet"
BIREFNET_REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"


class BiRefNetRemover:
    """Background remover with the same call shape as hy3dgen's: RGB image in, RGBA out."""

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
    from hy3dgen.shapegen import DegenerateFaceRemover, FaceReducer, FloaterRemover

    try:
        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        mesh = FaceReducer()(mesh, max_facenum=max_faces)
    except Exception as e:
        print(f"--> pymeshlab postprocessing failed ({e}); using trimesh fallback", flush=True)
        # Drop floaters: keep only the largest connected component
        parts = mesh.split(only_watertight=False)
        mesh = max(parts, key=lambda m: len(m.faces))
        mesh.remove_unreferenced_vertices()
        if len(mesh.faces) > max_faces:
            mesh = mesh.simplify_quadric_decimation(face_count=max_faces)
    return mesh


@app.cls(
    image=hunyuan_image,
    gpu="A10G",  # 24GB VRAM handles the 2D T2I + 3D pipelines smoothly
    volumes={"/root/.cache/huggingface": weights_volume},
    cpu=8,  # UV unwrap / baking / inpainting are CPU-bound
    memory=32768,  # 32 GiB RAM
    timeout=1800,  # safety margin while tuning; lower again once timings are known
    retries=0,
    max_containers=1,
    scaledown_window=120,
)
class Hunyuan3DMultimodalModel:
    @modal.enter()
    def setup(self):
        from hy3dgen.rembg import BackgroundRemover
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        from hy3dgen.texgen import Hunyuan3DPaintPipeline

        import transformers

        print(f"--> Transformers version: {transformers.__version__}")
        print("--> Loading shape-generation pipeline...")
        self.shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-2"
        )
        self.multiview_pipeline = None
        print("--> Shape-generation pipeline loaded.")

        print("--> Loading texture-generation pipeline...")
        self.tex_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
            "tencent/Hunyuan3D-2"
        )
        # CPU offload removed: on a 24GB A10G the paint models fit once the shape
        # pipeline is moved off the GPU, and offload hooks only add transfer overhead.
        # If you hit CUDA OOM in stage 2, re-enable it:
        # self.tex_pipeline.enable_model_cpu_offload()
        print("--> Texture-generation pipeline loaded.")

        try:
            self.bg_remover = BiRefNetRemover()
            print("--> Background remover: BiRefNet")
        except Exception as e:
            print(f"--> BiRefNet failed to load ({e}); falling back to rembg", flush=True)
            self.bg_remover = BackgroundRemover()
        self.t2i_pipeline = None

    def get_shape_pipeline(self, model_mode: str):
        if model_mode == "single":
            return self.shape_pipeline
        if model_mode != "multiview":
            raise ValueError(f"Unsupported shape model mode: {model_mode}")

        if self.multiview_pipeline is None:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

            print("--> Loading Hunyuan3D-2mv shape pipeline on CPU...", flush=True)
            self.multiview_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                "tencent/Hunyuan3D-2mv",
                subfolder="hunyuan3d-dit-v2-mv",
                variant="fp16",
                device="cpu",
            )
            print("--> Hunyuan3D-2mv shape pipeline loaded.", flush=True)
        return self.multiview_pipeline

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

        # Create a boto3 client using the fresh temporary credentials
        creds = assumed_role["Credentials"]
        s3 = boto3.client(
            "s3",
            aws_access_key_id=creds["AccessKeyId"],
            aws_secret_access_key=creds["SecretAccessKey"],
            aws_session_token=creds["SessionToken"],
        )
        return s3

    def resolve_image(self, image_bytes, prompt: Optional[str]):
        from PIL import Image

        input_image = None
        if image_bytes:
            print("--> Input type: Direct Image")
            input_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        else:
            if self.t2i_pipeline is None:
                from diffusers import AutoPipelineForText2Image
                import torch

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

        # Hunyuan3D works best with a background-free (RGBA) input.
        # BackgroundRemover only needs to run if there is no alpha channel
        # already carrying a real cutout.
        if input_image.mode == "RGB":
            print("--> Removing background...", flush=True)
            input_image = self.bg_remover(input_image)
        elif input_image.mode == "RGBA":
            alpha_min = input_image.getchannel("A").getextrema()[0]
            if alpha_min == 255:  # fully opaque alpha = no real cutout
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
        model_mode: str = "single",
        multiview_images: Optional[dict[str, bytes]] = None,
    ) -> bytes | str:
        import gc
        import torch

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        job_started = time.perf_counter()

        def log_stage_complete(stage: str, stage_started: float) -> None:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - stage_started
            allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
            peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(
                f"--> {stage} completed in {elapsed:.1f}s; "
                f"GPU memory allocated={allocated_gb:.2f} GiB, "
                f"reserved={reserved_gb:.2f} GiB, peak allocated={peak_gb:.2f} GiB",
                flush=True,
            )

        stage_started = time.perf_counter()
        if model_mode == "multiview":
            required_views = ("front", "left", "back")
            if not multiview_images or any(not multiview_images.get(view) for view in required_views):
                raise ValueError("Multiview generation requires front, left, and back image bytes.")
            prepared_views = {
                view: self.resolve_image(multiview_images[view], prompt=None)
                for view in required_views
            }
            shape_input = prepared_views
            texture_images = [prepared_views[view] for view in required_views]
            print(
                f"--> Prepared multiview input sizes: "
                f"{ {view: image.size for view, image in prepared_views.items()} }",
                flush=True,
            )
        else:
            input_image = self.resolve_image(image_bytes, prompt)
            shape_input = input_image
            texture_images = input_image
            print(f"--> Prepared input image size: {input_image.size}", flush=True)
        log_stage_complete("Image preparation", stage_started)

        # --- Step 1: Shape Generation (Geometry) ---
        # The shape pipeline is moved to CPU at the end of every call, so on a warm
        # container it must be moved back to the GPU or stage 1 silently runs on CPU.
        shape_pipeline = self.get_shape_pipeline(model_mode)
        shape_pipeline.to("cuda")

        print(f"--> Stage 1: Generating 3D Mesh Geometry ({model_mode})...", flush=True)
        stage_started = time.perf_counter()
        mesh = shape_pipeline(
            image=shape_input,
            octree_resolution=OCTREE_RESOLUTION,
            generator=torch.manual_seed(SHAPE_SEED),
        )[0]
        log_stage_complete("Stage 1 shape generation", stage_started)
        print(f"--> Raw mesh faces: {len(mesh.faces)}", flush=True)

        # Free the GPU for the texture models
        shape_pipeline.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(
            "--> Moved shape pipeline to CPU before texture painting; "
            f"GPU memory allocated={torch.cuda.memory_allocated() / (1024 ** 3):.2f} GiB, "
            f"reserved={torch.cuda.memory_reserved() / (1024 ** 3):.2f} GiB",
            flush=True,
        )

        # --- Mesh cleanup + decimation (keeps texturing fast) ---
        stage_started = time.perf_counter()
        mesh = cleanup_and_decimate(mesh, MAX_FACES_FOR_TEXTURING)
        print(
            f"--> Mesh cleanup/decimation completed in "
            f"{time.perf_counter() - stage_started:.1f}s; "
            f"faces going into texturing: {len(mesh.faces)}",
            flush=True,
        )

        # --- Step 2: Texture Synthesis ---
        print("--> Stage 2: Painting High-Res Texture...", flush=True)
        stage_started = time.perf_counter()
        textured_mesh = self.tex_pipeline(mesh=mesh, image=texture_images)
        log_stage_complete("Stage 2 texture painting", stage_started)

        # Export to GLB binary stream
        stage_started = time.perf_counter()
        output_buffer = io.BytesIO()
        textured_mesh.export(output_buffer, file_type="glb")
        glb_data = output_buffer.getvalue()
        print(
            f"--> GLB export completed in {time.perf_counter() - stage_started:.1f}s "
            f"({len(glb_data) / (1024 ** 2):.1f} MiB); total generation time "
            f"{time.perf_counter() - job_started:.1f}s",
            flush=True,
        )

        if output_bucket:
            if not job_id:
                raise ValueError("job_id is required when output_bucket is provided.")
            s3 = self.access_aws()
            output_key = f"assets/{job_id}.glb"
            s3.put_object(
                Bucket=output_bucket,
                Key=output_key,
                Body=glb_data,
                ContentType="model/gltf-binary",
            )
            print(f"--> Saved generated model to s3://{output_bucket}/{output_key}")
            return output_key

        return glb_data


@app.function(image=hunyuan_image)
# requires_proxy_auth: Modal rejects requests (401) unless they carry valid
# Modal-Key and Modal-Secret headers from a Proxy Auth Token. The check runs
# before your code, so unauthenticated callers never start a container.
@modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
def trigger_job(payload: dict):
    from fastapi.responses import JSONResponse

    # Trigger background worker without waiting for completion
    model = Hunyuan3DMultimodalModel()

    bucket = payload.get("bucket")
    key = payload.get("key")
    prompt = payload.get("prompt")
    input_mode = payload.get("input_mode", "single")
    views = payload.get("views")
    job_id = payload.get("jobId")
    output_bucket = payload.get("output_bucket")

    if not job_id or not output_bucket:
        return JSONResponse(
            status_code=400,
            content={"message": "jobId and output_bucket are required."},
        )

    s3 = model.access_aws()
    multiview_images = None
    if input_mode == "multiview":
        required_views = ("front", "left", "back")
        if not views or any(
            not views.get(view, {}).get("bucket") or not views.get(view, {}).get("key")
            for view in required_views
        ):
            return JSONResponse(
                status_code=400,
                content={"message": "Multiview jobs require front, left, and back S3 references."},
            )

        multiview_images = {}
        for view in required_views:
            view_object = views[view]
            response = s3.get_object(Bucket=view_object["bucket"], Key=view_object["key"])
            multiview_images[view] = response["Body"].read()
        image_bytes = None
        print("Running multiview Image-to-3D using front, left, and back views")
    elif input_mode == "single" and bucket and key:
        obj = s3.get_object(Bucket=bucket, Key=key)
        image_bytes = obj["Body"].read()
        print(f"Running Image-to-3D for file: {bucket}/{key}")
    elif input_mode == "prompt" and prompt:
        image_bytes = None
        print(f"Running Text-to-3D with prompt: '{prompt}'")
    else:
        return JSONResponse(
            status_code=400,
            content={"message": "Provide a single image, front/left/back images, or a prompt."},
        )

    model.generate_3d.spawn(
        image_bytes=image_bytes,
        prompt=prompt,
        job_id=job_id,
        output_bucket=output_bucket,
        model_mode="multiview" if input_mode == "multiview" else "single",
        multiview_images=multiview_images,
    )

    return JSONResponse(
        status_code=202,
        content={
            "message": "Job accepted for background processing.",
            "jobId": job_id,
            "outputKey": f"assets/{job_id}.glb",
        },
    )


# Local Entrypoint for Testing Both Modalities
@app.local_entrypoint()
def main(
    prompt: Optional[str] = "A wooden treasure chest with iron bands, 3d game asset",
    bucket: Optional[str] = None,
    key: Optional[str] = None,
    output_path: str = "output.glb",
):
    model = Hunyuan3DMultimodalModel()

    if bucket and key:
        s3 = model.access_aws()
        obj = s3.get_object(Bucket=bucket, Key=key)
        image_bytes = obj["Body"].read()
        print(f"Running Image-to-3D for file: {bucket}/{key}")
    elif prompt:
        image_bytes = None
        print(f"Running Text-to-3D with prompt: '{prompt}'")
    else:
        return {"message": "Invalid payload."}

    glb_data = model.generate_3d.remote(image_bytes=image_bytes, prompt=prompt)
    if not isinstance(glb_data, bytes):
        raise RuntimeError("Local generation did not return GLB bytes.")

    out_file = Path(output_path)
    out_file.write_bytes(glb_data)
    print(f"Done! Saved 3D asset to {out_file.resolve()}")