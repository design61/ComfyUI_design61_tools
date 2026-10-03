import os
import re
import json
import wave
import shutil
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime

import torch
import folder_paths
import comfy.utils


# ============================================================
# 任意类型
# ============================================================

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


ANY_TYPE = AnyType("*")


# ============================================================
# 路径日期模板
#
# 支持：
# %date:yyyyMMddHHmmss%
# %date:yyyy-MM-dd_HH-mm-ss%
#
# 例如：
# Minimax/%date:yyyyMMddHHmmss%
# ============================================================

def expand_date_template(value: str):
    if not value:
        return value

    now = datetime.now()

    token_map = {
        "yyyy": now.strftime("%Y"),
        "MM": now.strftime("%m"),
        "dd": now.strftime("%d"),
        "HH": now.strftime("%H"),
        "mm": now.strftime("%M"),
        "ss": now.strftime("%S"),
    }

    def replace_date(match):
        fmt = match.group(1)

        # 注意替换顺序
        # 必须先替换长 token
        for token in [
            "yyyy",
            "MM",
            "dd",
            "HH",
            "mm",
            "ss",
        ]:
            fmt = fmt.replace(
                token,
                token_map[token],
            )

        return fmt

    return re.sub(
        r"%date:([^%]+)%",
        replace_date,
        value,
    )


# ============================================================
# 相对路径默认以 ComfyUI/output 为基础
# ============================================================

def resolve_frames_path(value: str):
    """
    序列帧目录解析规则：

    1. 支持日期模板
       tmp/%date:yyyyMMddHHmmss%

    2. 绝对路径：
       E:\\xxx\\frames
       → 原样使用

    3. 普通相对路径：
       tmp
       → ComfyUI/output/tmp

       Minimax/test
       → ComfyUI/output/Minimax/test

    4. ../ 路径：
       ../tmp
       → ComfyUI/tmp

       ../../tmp
       → ComfyUI 上一级/tmp
    """

    if not value or not value.strip():
        raise ValueError("frames_path 不能为空")

    # 日期变量
    value = expand_date_template(
        value.strip()
    )

    path = Path(
        value
    ).expanduser()

    # =============================================
    # 绝对路径直接使用
    # =============================================

    if path.is_absolute():
        return path.resolve()

    # =============================================
    # 相对路径统一基于 ComfyUI/output
    #
    # tmp
    # -> output/tmp
    #
    # ../xxx
    # -> output/../xxx
    # -> ComfyUI/xxx
    # =============================================

    output_root = Path(
        folder_paths.get_output_directory()
    )

    return (
        output_root
        / path
    ).resolve()


# ============================================================
# 自然排序
# ============================================================

def natural_key(value: str):
    return [
        int(text) if text.isdigit()
        else text.lower()

        for text in re.split(
            r"(\d+)",
            value,
        )
    ]


# ============================================================
# FFmpeg 查找
# ============================================================

def resolve_ffmpeg_path(user_ffmpeg_path: str):

    # --------------------------------------------------------
    # 用户指定
    # --------------------------------------------------------

    if user_ffmpeg_path and user_ffmpeg_path.strip():

        p = Path(
            expand_date_template(
                user_ffmpeg_path.strip()
            )
        ).expanduser()

        if p.is_file():
            return str(p.resolve())

        if p.is_dir():

            if os.name == "nt":

                candidates = [
                    p / "ffmpeg.exe",
                    p / "bin" / "ffmpeg.exe",
                ]

            else:

                candidates = [
                    p / "ffmpeg",
                    p / "bin" / "ffmpeg",
                ]

            for c in candidates:

                if c.is_file():
                    return str(c.resolve())

    # --------------------------------------------------------
    # 插件自带 ffmpeg
    # --------------------------------------------------------

    plugin_dir = Path(__file__).resolve().parent
    ffmpeg_dir = plugin_dir / "ffmpeg"

    if os.name == "nt":

        candidates = [
            ffmpeg_dir / "ffmpeg.exe",
            ffmpeg_dir / "bin" / "ffmpeg.exe",
        ]

    else:

        candidates = [
            ffmpeg_dir / "ffmpeg",
            ffmpeg_dir / "bin" / "ffmpeg",
        ]

    for c in candidates:

        if c.is_file():
            return str(c.resolve())

    # --------------------------------------------------------
    # PATH
    # --------------------------------------------------------

    system_ffmpeg = shutil.which(
        "ffmpeg"
    )

    if system_ffmpeg:
        return system_ffmpeg

    return None


# ============================================================
# concat path 转义
# ============================================================

def escape_concat_path(path: Path):

    s = path.resolve().as_posix()

    s = s.replace(
        "'",
        r"'\''",
    )

    return s


# ============================================================
# 编码器
# ============================================================

def get_video_encoder(codec, backend):

    table = {

        ("h264", "cpu"):
            "libx264",

        ("h265", "cpu"):
            "libx265",

        ("h264", "nvenc"):
            "h264_nvenc",

        ("h265", "nvenc"):
            "hevc_nvenc",
    }

    encoder = table.get(
        (
            codec.lower(),
            backend.lower(),
        )
    )

    if encoder is None:

        raise ValueError(
            f"不支持：{codec} / {backend}"
        )

    return encoder


# ============================================================
# preset
# ============================================================

def get_encoder_preset(
    preset,
    encoder,
):

    preset = preset.lower()

    if encoder in {
        "libx264",
        "libx265",
    }:

        mapping = {

            "p1": "ultrafast",
            "p2": "superfast",
            "p3": "veryfast",
            "p4": "fast",
            "p5": "medium",
            "p6": "slow",
            "p7": "veryslow",
        }

        return mapping.get(
            preset,
            preset if preset in {
                "ultrafast",
                "superfast",
                "veryfast",
                "faster",
                "fast",
                "medium",
                "slow",
                "slower",
                "veryslow",
            }
            else "medium"
        )

    if encoder in {
        "h264_nvenc",
        "hevc_nvenc",
    }:

        mapping = {

            "ultrafast": "p1",
            "superfast": "p2",
            "veryfast": "p3",
            "faster": "p4",
            "fast": "p4",
            "medium": "p5",
            "slow": "p6",
            "slower": "p7",
            "veryslow": "p7",
        }

        return mapping.get(
            preset,
            preset if preset in {
                "p1",
                "p2",
                "p3",
                "p4",
                "p5",
                "p6",
                "p7",
            }
            else "p5"
        )

    return preset


# ============================================================
# 输出路径
# ============================================================

def build_output_path(
    source_folder,
    output_path,
    container,
):

    folder_name = (
        source_folder.name
        or "video"
    )

    default_filename = (
        f"{folder_name}.{container}"
    )

    # --------------------------------------------------------
    # output_path 空
    # --------------------------------------------------------

    if not output_path.strip():

        out_dir = (
            Path(folder_paths.get_output_directory())
            / "FolderFramesToVideo"
        )

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        return (
            out_dir
            / default_filename
        )

    # --------------------------------------------------------
    # 日期模板 + 相对路径
    # --------------------------------------------------------

    out = resolve_frames_path(
        output_path
    )

    # 认为最后没有扩展名时，
    # 如果字符串明确以 / 或 \ 结束，则为目录
    if (
        output_path.endswith("/")
        or output_path.endswith("\\")
        or out.is_dir()
    ):

        out.mkdir(
            parents=True,
            exist_ok=True,
        )

        return (
            out
            / default_filename
        )

    wanted_suffix = (
        f".{container}"
    )

    if (
        out.suffix.lower()
        != wanted_suffix.lower()
    ):

        out = out.with_suffix(
            wanted_suffix
        )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return out


# ============================================================
# ComfyUI AUDIO → WAV
# ============================================================

def save_comfy_audio_to_wav(
    audio,
    output_wav,
):

    if audio is None:
        return None

    if not isinstance(
        audio,
        dict,
    ):

        raise ValueError(
            "AUDIO 格式错误"
        )

    waveform = audio.get(
        "waveform"
    )

    sample_rate = audio.get(
        "sample_rate"
    )

    if waveform is None:
        raise ValueError(
            "AUDIO 缺少 waveform"
        )

    if sample_rate is None:
        raise ValueError(
            "AUDIO 缺少 sample_rate"
        )

    if not torch.is_tensor(
        waveform
    ):

        raise ValueError(
            "waveform 不是 Tensor"
        )

    waveform = waveform.detach()

    # [B,C,T]
    if waveform.ndim == 3:
        waveform = waveform[0]

    # [T]
    elif waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    elif waveform.ndim != 2:

        raise ValueError(
            f"无法识别 waveform："
            f"{tuple(waveform.shape)}"
        )

    waveform = (
        waveform
        .float()
        .cpu()
        .clamp(
            -1.0,
            1.0,
        )
    )

    channels = waveform.shape[0]
    samples = waveform.shape[1]

    with wave.open(
        str(output_wav),
        "wb",
    ) as wav:

        wav.setnchannels(
            channels
        )

        wav.setsampwidth(2)

        wav.setframerate(
            int(sample_rate)
        )

        # 10秒一块
        chunk_size = (
            int(sample_rate)
            * 10
        )

        for start in range(
            0,
            samples,
            chunk_size,
        ):

            end = min(
                start + chunk_size,
                samples,
            )

            chunk = waveform[
                :,
                start:end
            ]

            chunk = (
                chunk
                .transpose(0, 1)
                .mul(32767.0)
                .round()
                .to(torch.int16)
                .contiguous()
                .numpy()
                .tobytes()
            )

            wav.writeframes(
                chunk
            )

    return output_wav


# ============================================================
# IMAGE tensor → 临时 PNG
#
# 注意：
# IMAGE 本身既然已经从上游传进来了，
# 它本来就已经在 RAM。
#
# 这里不会额外构造第二套 IMAGE batch，
# 只是逐张落盘给 FFmpeg。
# ============================================================

def save_image_batch_to_temp(
    images,
    temp_dir,
):

    from PIL import Image
    import numpy as np

    if images is None:
        return []

    if not torch.is_tensor(
        images
    ):

        raise ValueError(
            "IMAGE 输入不是 Tensor"
        )

    # ComfyUI IMAGE:
    # [B,H,W,C]

    if images.ndim != 4:

        raise ValueError(
            f"IMAGE 应为 [B,H,W,C]，"
            f"实际 {tuple(images.shape)}"
        )

    frame_paths = []

    frame_count = int(
        images.shape[0]
    )

    save_progress = comfy.utils.ProgressBar(
        frame_count
    )

    for i in range(
        frame_count
    ):

        # 只取当前一帧
        frame = (
            images[i]
            .detach()
            .cpu()
            .clamp(
                0.0,
                1.0,
            )
        )

        array = (
            frame
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .numpy()
        )

        path = (
            temp_dir
            / f"frame_{i + 1:08d}.png"
        )

        Image.fromarray(
            array
        ).save(
            path,
            compress_level=1,
        )

        frame_paths.append(
            path
        )

        save_progress.update_absolute(
            i + 1,
            frame_count,
        )

    return frame_paths


# ============================================================
# concat
# ============================================================

def create_concat_file(
    frames,
    concat_file,
    fps,
):

    duration = (
        1.0
        / float(fps)
    )

    with open(
        concat_file,
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:

        for frame in frames:

            path = escape_concat_path(
                frame
            )

            f.write(
                f"file '{path}'\n"
            )

            f.write(
                f"duration {duration:.12f}\n"
            )

        if frames:

            last = escape_concat_path(
                frames[-1]
            )

            f.write(
                f"file '{last}'\n"
            )


# ============================================================
# ComfyUI Workflow Metadata
# ============================================================

def escape_ffmetadata_value(value):
    """转义 FFmpeg FFMETADATA1 字段值。"""
    value = str(value)
    value = value.replace("\\", "\\\\")
    value = value.replace(";", "\\;")
    value = value.replace("#", "\\#")
    value = value.replace("=", "\\=")
    value = value.replace("\n", "\\\n")
    return value


def create_workflow_metadata_file(prompt, extra_pnginfo, metadata_file: Path):
    """
    写入 ComfyUI 可识别的视频元数据。

    主要字段：
      prompt   = 当前 API Prompt
      workflow = 当前前端 Workflow JSON

    返回 True 表示至少写入了一项元数据。
    """
    metadata = {}

    if prompt is not None:
        metadata["prompt"] = prompt

    if isinstance(extra_pnginfo, dict):
        for key, value in extra_pnginfo.items():
            metadata[key] = value

    if not metadata:
        return False

    with open(metadata_file, "w", encoding="utf-8", newline="\n") as f:
        f.write(";FFMETADATA1\n")
        for key, value in metadata.items():
            json_value = json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            f.write(
                f"{key}={escape_ffmetadata_value(json_value)}\n"
            )

    return True


# ============================================================
# FFmpeg command
# ============================================================

def build_ffmpeg_command(
    ffmpeg_path,
    concat_file,
    output_file,
    frame_count,
    fps,
    codec,
    backend,
    quality_value,
    preset,
    pix_fmt,
    overwrite,
    audio_file=None,
    audio_bitrate="192k",
    metadata_file=None,
):

    encoder = get_video_encoder(
        codec,
        backend,
    )

    real_preset = get_encoder_preset(
        preset,
        encoder,
    )

    duration = (
        frame_count
        / float(fps)
    )

    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel", "error",
        "-progress", "pipe:1",
        "-nostats",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
    ]

    # --------------------------------------------------------
    # 输入编号
    # 0 = 图片序列
    # 1... = 可选音频 / workflow metadata
    # --------------------------------------------------------

    next_input_index = 1
    audio_input_index = None
    metadata_input_index = None

    if audio_file is not None:
        audio_input_index = next_input_index
        cmd += [
            "-i", str(audio_file),
        ]
        next_input_index += 1

    if metadata_file is not None:
        metadata_input_index = next_input_index
        cmd += [
            "-f", "ffmetadata",
            "-i", str(metadata_file),
        ]
        next_input_index += 1

    # --------------------------------------------------------
    # Stream mapping
    # --------------------------------------------------------

    cmd += [
        "-map", "0:v:0",
    ]

    if audio_input_index is not None:
        cmd += [
            "-map", f"{audio_input_index}:a:0",
        ]

    if metadata_input_index is not None:
        cmd += [
            "-map_metadata", str(metadata_input_index),
        ]

    # --------------------------------------------------------
    # Video
    # --------------------------------------------------------

    cmd += [
        "-c:v", encoder,
        "-preset", real_preset,
    ]

    if backend == "cpu":
        cmd += [
            "-crf", str(quality_value),
        ]
    else:
        cmd += [
            "-rc", "vbr",
            "-cq", str(quality_value),
            "-b:v", "0",
        ]

    cmd += [
        "-pix_fmt", pix_fmt,
        "-r", str(fps),
        "-frames:v", str(frame_count),
    ]

    # --------------------------------------------------------
    # Audio
    # --------------------------------------------------------

    if audio_input_index is not None:
        cmd += [
            "-c:a", "aac",
            "-b:a", audio_bitrate,
        ]
    else:
        cmd += [
            "-an",
        ]

    # 图片帧数决定视频长度
    cmd += [
        "-t", f"{duration:.9f}",
    ]

    # MP4 / MOV 需要 use_metadata_tags 才能可靠保留自定义 metadata
    if output_file.suffix.lower() in {
        ".mp4",
        ".mov",
    }:
        if metadata_file is not None:
            cmd += [
                "-movflags", "+faststart+use_metadata_tags",
            ]
        else:
            cmd += [
                "-movflags", "+faststart",
            ]

    cmd += [
        "-y" if overwrite else "-n",
        str(output_file),
    ]

    return (
        cmd,
        encoder,
        real_preset,
    )


# ============================================================
# FFmpeg progress
# ============================================================

def run_ffmpeg_with_progress(
    cmd,
    frame_count,
):

    progress = comfy.utils.ProgressBar(
        frame_count
    )

    process = subprocess.Popen(

        cmd,

        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,

        text=True,
        encoding="utf-8",
        errors="ignore",

        bufsize=1,
    )

    last_frame = 0

    if process.stdout:

        for raw in process.stdout:

            line = raw.strip()

            if line.startswith(
                "frame="
            ):

                try:

                    current = int(
                        line.split(
                            "=",
                            1,
                        )[1]
                    )

                except Exception:

                    continue

                current = min(
                    current,
                    frame_count,
                )

                if current > last_frame:

                    progress.update_absolute(
                        current,
                        frame_count,
                    )

                    last_frame = current

            elif line == "progress=end":

                progress.update_absolute(
                    frame_count,
                    frame_count,
                )

    return_code = process.wait()

    error = ""

    if process.stderr:

        error = (
            process.stderr.read()
            or ""
        ).strip()

    return (
        return_code,
        error,
    )


# ============================================================
# Preview
# ============================================================

def create_video_ui_info(
    output_file,
    status,
):

    ui = {

        "text": [
            status
        ]
    }

    try:

        output_dir = Path(
            folder_paths.get_output_directory()
        ).resolve()

        target = (
            output_file
            .resolve()
        )

        if (
            target.parent == output_dir
            or output_dir in target.parents
        ):

            subfolder = str(
                target.parent.relative_to(
                    output_dir
                )
            ).replace(
                "\\",
                "/",
            )

            ui["videos"] = [

                {
                    "filename":
                        target.name,

                    "subfolder":
                        subfolder,

                    "type":
                        "output",

                    "format":
                        target.suffix
                        .lower()
                        .lstrip("."),
                }
            ]

    except Exception:
        pass

    return ui


# ============================================================
# Main V4
# ============================================================

class FolderFramesToVideoFFmpeg_design61:

    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):

        return {

            "required": {

                "frames_path": (
                    "STRING",
                    {
                        "default":
                            "tmp",
                        "multiline":
                            False,
                        "dynamicPrompts":
                            False,
                    },
                ),

                "fps": (
                    "INT",
                    {
                        "default": 24,
                        "min": 1,
                        "max": 240,
                    },
                ),

                "extensions": (
                    "STRING",
                    {
                        "default":
                            "png,jpg,jpeg,webp,bmp",
                    },
                ),

                "start_frame": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 999999999,
                    },
                ),

                "end_frame": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 999999999,
                    },
                ),

                "container": (
                    [
                        "mp4",
                        "mkv",
                        "mov",
                    ],
                ),

                "codec": (
                    [
                        "h264",
                        "h265",
                    ],
                ),

                "backend": (
                    [
                        "nvenc",
                        "cpu",
                    ],
                ),

                "quality_value": (
                    "INT",
                    {
                        "default": 18,
                        "min": 0,
                        "max": 51,
                    },
                ),

                "preset": (
                    [
                        "fast",
                        "medium",
                        "slow",

                        "p1",
                        "p2",
                        "p3",
                        "p4",
                        "p5",
                        "p6",
                        "p7",

                        "ultrafast",
                        "superfast",
                        "veryfast",
                        "faster",
                        "slower",
                        "veryslow",
                    ],
                ),

                "pix_fmt": (
                    [
                        "yuv420p",
                        "yuv422p",
                        "yuv444p",
                    ],
                ),

                "audio_bitrate": (
                    [
                        "128k",
                        "192k",
                        "256k",
                        "320k",
                    ],
                ),

                "output_path": (
                    "STRING",
                    {
                        "default": "Minimax/%date:yyyyMMddHHmmss%",
                    },
                ),

                "save_workflow": (
                    "BOOLEAN",
                    {
                        "default": True
                    },
                ),

                "delete_source_frames": (
                    "BOOLEAN",
                    {
                        "default": False
                    },
                ),

                "overwrite": (
                    "BOOLEAN",
                    {
                        "default": True
                    },
                ),

                "keep_concat_file": (
                    "BOOLEAN",
                    {
                        "default": False
                    },
                ),

                "ffmpeg_path": (
                    "STRING",
                    {
                        "default": "",
                    },
                ),
            },

            "optional": {

                "trigger": (
                    ANY_TYPE,
                    {}
                ),

                "image": (
                    "IMAGE",
                ),

                "audio": (
                    "AUDIO",
                ),
            },

            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = (
        ANY_TYPE,
        "IMAGE",
        "STRING",
        "INT",
        "STRING",
    )

    RETURN_NAMES = (
        "passthrough",
        "image",
        "video_path",
        "used_frame_count",
        "status",
    )

    FUNCTION = "run"

    CATEGORY = "FFmpeg/Video"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    # ========================================================

    def run(
        self,

        frames_path,
        fps,
        extensions,

        start_frame,
        end_frame,

        container,
        codec,
        backend,

        quality_value,
        preset,
        pix_fmt,

        audio_bitrate,

        output_path,
        save_workflow,

        delete_source_frames,
        overwrite,
        keep_concat_file,

        ffmpeg_path,

        trigger=None,
        image=None,
        audio=None,
        prompt=None,
        extra_pnginfo=None,
    ):

        temp_audio_file = None
        concat_file = None
        temp_image_dir = None
        metadata_file = None

        # IMAGE 输出：
        # 有输入时直接 passthrough
        output_image = image

        try:

            # =================================================
            # FFmpeg
            # =================================================

            ffmpeg = resolve_ffmpeg_path(
                ffmpeg_path
            )

            if ffmpeg is None:

                raise RuntimeError(
                    "未找到 FFmpeg。"
                )

            # =================================================
            # 模式选择
            #
            # IMAGE 优先级 > folder
            # =================================================

            image_mode = (
                image is not None
                and torch.is_tensor(image)
                and image.ndim == 4
                and image.shape[0] > 0
                and image.numel() > 0
            )

            # =================================================
            # IMAGE MODE
            # =================================================

            if image_mode:

                temp_image_dir = Path(
                    tempfile.mkdtemp(
                        prefix="comfy_video_frames_"
                    )
                )

                all_frames = (
                    save_image_batch_to_temp(
                        image,
                        temp_image_dir,
                    )
                )

                source_folder = (
                    temp_image_dir
                )

                source_description = (
                    "IMAGE 输入"
                )

            # =================================================
            # FOLDER MODE
            # =================================================

            else:

                source_folder = resolve_frames_path(
                    frames_path
                )

                if not source_folder.exists():

                    raise FileNotFoundError(
                        f"文件夹不存在："
                        f"{source_folder}"
                    )

                if not source_folder.is_dir():

                    raise ValueError(
                        f"不是文件夹："
                        f"{source_folder}"
                    )

                allowed = {

                    x.strip()
                    .lower()
                    .lstrip(".")

                    for x in extensions.split(",")

                    if x.strip()
                }

                all_frames = [

                    file

                    for file in source_folder.iterdir()

                    if (
                        file.is_file()
                        and
                        file.suffix
                        .lower()
                        .lstrip(".")
                        in allowed
                    )
                ]

                all_frames.sort(
                    key=lambda p:
                        natural_key(
                            p.name
                        )
                )

                source_description = (
                    str(source_folder)
                )

            # =================================================
            # frame range
            # =================================================

            total_count = len(
                all_frames
            )

            if total_count == 0:

                raise RuntimeError(
                    "没有找到任何视频帧。"
                )

            start_index = max(
                1,
                int(start_frame),
            )

            if int(end_frame) <= 0:

                end_index = (
                    total_count
                )

            else:

                end_index = min(
                    int(end_frame),
                    total_count,
                )

            if start_index > end_index:

                raise ValueError(
                    "start_frame 大于 end_frame"
                )

            frames = all_frames[
                start_index - 1:
                end_index
            ]

            frame_count = len(
                frames
            )

            # =================================================
            # 输出
            # =================================================

            final_output = (
                build_output_path(
                    source_folder,
                    output_path,
                    container,
                )
            )

            # =================================================
            # concat
            # =================================================

            if keep_concat_file:

                concat_file = (
                    Path(
                        folder_paths.get_temp_directory()
                    )
                    / "_FolderFramesToVideo_concat.txt"
                )

            else:

                fd, temp_name = (
                    tempfile.mkstemp(
                        prefix="comfy_concat_",
                        suffix=".txt",
                    )
                )

                os.close(fd)

                concat_file = Path(
                    temp_name
                )

            create_concat_file(
                frames,
                concat_file,
                fps,
            )

            # =================================================
            # audio
            # =================================================

            if audio is not None:

                fd, temp_audio_name = (
                    tempfile.mkstemp(
                        prefix="comfy_audio_",
                        suffix=".wav",
                    )
                )

                os.close(fd)

                temp_audio_file = Path(
                    temp_audio_name
                )

                save_comfy_audio_to_wav(
                    audio,
                    temp_audio_file,
                )

            # =================================================
            # workflow metadata
            # =================================================

            if save_workflow:

                fd, temp_metadata_name = (
                    tempfile.mkstemp(
                        prefix="comfy_workflow_",
                        suffix=".txt",
                    )
                )

                os.close(fd)

                metadata_file = Path(
                    temp_metadata_name
                )

                metadata_ok = create_workflow_metadata_file(
                    prompt=prompt,
                    extra_pnginfo=extra_pnginfo,
                    metadata_file=metadata_file,
                )

                if not metadata_ok:
                    metadata_file.unlink(
                        missing_ok=True
                    )
                    metadata_file = None

            # =================================================
            # command
            # =================================================

            cmd, encoder, real_preset = (
                build_ffmpeg_command(

                    ffmpeg_path=ffmpeg,

                    concat_file=concat_file,

                    output_file=final_output,

                    frame_count=frame_count,

                    fps=fps,

                    codec=codec,

                    backend=backend,

                    quality_value=quality_value,

                    preset=preset,

                    pix_fmt=pix_fmt,

                    overwrite=overwrite,

                    audio_file=temp_audio_file,

                    audio_bitrate=audio_bitrate,

                    metadata_file=metadata_file,
                )
            )

            # =================================================
            # encode progress
            # =================================================

            return_code, error = (
                run_ffmpeg_with_progress(
                    cmd,
                    frame_count,
                )
            )

            if return_code != 0:

                raise RuntimeError(
                    "FFmpeg 执行失败：\n"
                    + error
                )

            if not final_output.exists():

                raise RuntimeError(
                    "FFmpeg 完成但视频文件不存在。"
                )

            # =================================================
            # Delete source
            #
            # IMAGE 模式：
            # 只删除临时图片，不碰任何原始素材。
            #
            # folder 模式：
            # delete_source_frames=True 时
            # 删除所选源图。
            # =================================================

            deleted_count = 0

            if (
                not image_mode
                and delete_source_frames
            ):

                for frame in frames:

                    try:

                        frame.unlink()

                        deleted_count += 1

                    except Exception:
                        pass

            duration = (
                frame_count
                / float(fps)
            )

            # =================================================
            # status
            # =================================================

            status_lines = [

                "视频合成成功",

                (
                    "输入模式：IMAGE"
                    if image_mode
                    else "输入模式：Folder"
                ),

                f"源：{source_description}",

                f"输出：{final_output}",

                f"帧数：{frame_count}",

                f"FPS：{fps}",

                f"时长：{duration:.3f}s",

                f"编码器：{encoder}",

                f"Preset：{real_preset}",

                (
                    "音频：已合并"
                    if audio is not None
                    else "音频：无"
                ),

                (
                    "工作流：已写入视频"
                    if save_workflow and metadata_file is not None
                    else "工作流：未保存"
                ),
            ]

            if (
                not image_mode
                and delete_source_frames
            ):

                status_lines.append(
                    f"删除源帧：{deleted_count}"
                )

            status = "\n".join(
                status_lines
            )

            ui = create_video_ui_info(
                final_output,
                status,
            )

            return {

                "ui": ui,

                "result": (

                    trigger,

                    output_image,

                    str(final_output),

                    frame_count,

                    status,
                ),
            }

        except Exception as e:

            status = (
                f"{type(e).__name__}: {e}"
            )

            return {

                "ui": {
                    "text": [status]
                },

                "result": (

                    trigger,

                    output_image,

                    "",

                    0,

                    status,
                ),
            }

        finally:

            # =================================================
            # temp audio
            # =================================================

            if temp_audio_file:

                try:
                    temp_audio_file.unlink(
                        missing_ok=True
                    )
                except Exception:
                    pass

            # =================================================
            # concat
            # =================================================

            if (
                concat_file
                and not keep_concat_file
            ):

                try:
                    concat_file.unlink(
                        missing_ok=True
                    )
                except Exception:
                    pass

            # =================================================
            # workflow metadata 临时文件
            # =================================================

            if metadata_file:

                try:
                    metadata_file.unlink(
                        missing_ok=True
                    )
                except Exception:
                    pass

            # =================================================
            # IMAGE 模式临时 PNG
            # =================================================

            if temp_image_dir:

                try:

                    shutil.rmtree(
                        temp_image_dir,
                        ignore_errors=True,
                    )

                except Exception:
                    pass

# ============================================================
# 清空序列帧目录
# ============================================================

class ClearFolderFrames_design61:

    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {

                "frames_path": (
                    "STRING",
                    {
                        "default": "tmp",
                        "multiline": False,
                        "dynamicPrompts": False,
                    },
                ),

                "extensions": (
                    "STRING",
                    {
                        "default": "png,jpg,jpeg,webp,bmp",
                        "multiline": False,
                        "dynamicPrompts": False,
                    },
                ),

                "create_if_missing": (
                    "BOOLEAN",
                    {
                        "default": True,
                    },
                ),
            },

            "optional": {

                # 可接任意前置 trigger
                "trigger": (
                    ANY_TYPE,
                    {},
                ),
            },
        }

    RETURN_TYPES = (
        ANY_TYPE,
        "STRING",
        "INT",
        "STRING",
    )

    RETURN_NAMES = (
        "trigger",
        "frames_path",
        "deleted_count",
        "status",
    )

    FUNCTION = "run"

    CATEGORY = "FFmpeg/Video"

    # 每次 Queue 都执行
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(
        self,
        frames_path,
        extensions,
        create_if_missing,
        trigger=None,
    ):

        try:

            # =================================================
            # 解析路径
            # =================================================

            target_folder = resolve_frames_path(
                frames_path
            )

            # =================================================
            # 目录不存在
            # =================================================

            if not target_folder.exists():

                if create_if_missing:

                    target_folder.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    status = (
                        "目录原本不存在，已创建。\n"
                        f"路径：{target_folder}\n"
                        "删除帧数：0"
                    )

                    return {
                        "ui": {
                            "text": [status]
                        },

                        "result": (
                            trigger,
                            str(target_folder),
                            0,
                            status,
                        ),
                    }

                else:

                    raise FileNotFoundError(
                        f"目录不存在：{target_folder}"
                    )

            # =================================================
            # 必须是目录
            # =================================================

            if not target_folder.is_dir():

                raise ValueError(
                    f"目标不是目录：{target_folder}"
                )

            # =================================================
            # 解析扩展名
            # =================================================

            allowed_extensions = set()

            for ext in extensions.split(","):

                ext = (
                    ext
                    .strip()
                    .lower()
                    .lstrip(".")
                )

                if ext:

                    allowed_extensions.add(
                        ext
                    )

            if not allowed_extensions:

                raise ValueError(
                    "extensions 不能为空。"
                )

            # =================================================
            # 找出所有序列帧
            # =================================================

            files_to_delete = []

            for file in target_folder.iterdir():

                if not file.is_file():
                    continue

                ext = (
                    file.suffix
                    .lower()
                    .lstrip(".")
                )

                if ext in allowed_extensions:

                    files_to_delete.append(
                        file
                    )

            total_count = len(
                files_to_delete
            )

            # =================================================
            # 删除进度条
            # =================================================

            progress = comfy.utils.ProgressBar(
                max(total_count, 1)
            )

            deleted_count = 0
            failed_files = []

            for index, file in enumerate(
                files_to_delete,
                start=1,
            ):

                try:

                    file.unlink()

                    deleted_count += 1

                except Exception as e:

                    failed_files.append(
                        f"{file.name}: {e}"
                    )

                progress.update_absolute(
                    index,
                    max(total_count, 1),
                )

            # =================================================
            # 没文件时也把进度走完
            # =================================================

            if total_count == 0:

                progress.update_absolute(
                    1,
                    1,
                )

            # =================================================
            # 状态
            # =================================================

            status_lines = [

                "序列帧目录清理完成",

                f"路径：{target_folder}",

                f"匹配文件：{total_count}",

                f"成功删除：{deleted_count}",
            ]

            if failed_files:

                status_lines.append(
                    f"删除失败：{len(failed_files)}"
                )

                # 最多显示前10个错误，避免 status 太长
                status_lines.extend(
                    failed_files[:10]
                )

            status = "\n".join(
                status_lines
            )

            return {

                "ui": {
                    "text": [status]
                },

                "result": (
                    trigger,
                    str(target_folder),
                    deleted_count,
                    status,
                ),
            }

        except Exception as e:

            status = (
                "清空序列帧目录失败：\n"
                f"{type(e).__name__}: {e}"
            )

            return {

                "ui": {
                    "text": [status]
                },

                "result": (
                    trigger,
                    "",
                    0,
                    status,
                ),
            }

# ============================================================
# 注册
# ============================================================

NODE_CLASS_MAPPINGS = {

    "ClearFolderFrames_design61":
        ClearFolderFrames_design61,

    "FolderFramesToVideoFFmpeg_design61":
        FolderFramesToVideoFFmpeg_design61
}


NODE_DISPLAY_NAME_MAPPINGS = {

    "ClearFolderFrames_design61":
        "清空文件夹序列帧",

    "FolderFramesToVideoFFmpeg_design61":
        "文件夹序列帧直接FFmpeg视频"
}