"""Per-run H3 payload normalization and MM-RoPE timeline adaptation.

No ComfyUI class is monkey-patched. ComfyUI builds a normal H3 PackedLayout;
this module then changes only its coordinates in place. Keeping the same
``position_ids`` Tensor identity preserves Sol-Attn's native-H3 span registry.

The preferred Continuum representation is one native H3 ``video`` or
``video_audio`` reference block. Its latent rows are moved onto the beginning
of the target timeline. The older marked-keyframe form remains supported as an
internal compatibility path, but the UI does not emit it.
"""
from __future__ import annotations
from collections.abc import Mapping
import logging
from typing import Any
import torch
from .constants import (
    CONTINUUM_INTEROP_API, CONTINUUM_REFERENCE_METADATA_KEY, FRAME_RESCALE,
    LAYOUT_DEVICE_CACHE_ATTR, LAYOUT_ORIGINAL_TIME_ATTR, LAYOUT_SIGNATURE_ATTR,
    MARK_AUDIO_CONTEXT, MARK_AUDIO_END_FRAME, MARK_AUDIO_OVERHANG,
    MARK_CONTEXT_FRAMES, MARK_VIDEO_CONTEXT, MARK_VIDEO_SLOT,
)
LOG = logging.getLogger("h3_continuum_join")
class LayoutCompatibilityError(RuntimeError): pass

_VIDEO_CHANNELS = 24
_AUDIO_CHANNELS = 32
_VIDEO_PATCH = (1, 2, 2)
_AUDIO_STREAMS = 2

def _reference_metadata(ref:dict[str,Any])->dict[str,Any]:
    value=ref.get(CONTINUUM_REFERENCE_METADATA_KEY); return value if isinstance(value,dict) else {}
def _is_video_context(ref):
    return bool(ref.get(MARK_VIDEO_CONTEXT) or _reference_metadata(ref).get("role")=="video_context")
def _is_audio_context(ref):
    metadata=_reference_metadata(ref)
    return bool(ref.get(MARK_AUDIO_CONTEXT) or metadata.get("role")=="audio_context" or metadata.get("audio_role")=="audio_context")
def payload_has_continuum(payload):
    if not isinstance(payload,dict): return False
    return any(MARK_VIDEO_SLOT in item for item in (payload.get("keyframes") or ())) or any(_is_video_context(item) or _is_audio_context(item) for item in (payload.get("refs") or ()))
def normalize_condition_latents(payload):
    keyframes=list(payload.get("keyframes") or ()); refs=list(payload.get("refs") or ())
    video_latents=[item["latent"] for item in keyframes if item.get("latent") is not None]
    video_latents.extend(item["latent"] for item in refs if item.get("latent") is not None)
    audio_latents=[item["audio_latent"] for item in keyframes if item.get("audio_latent") is not None]
    audio_latents.extend(item["audio_latent"] for item in refs if item.get("audio_latent") is not None)
    payload["cond_video_latents"]=video_latents
    payload["cond_audio_latents"]=audio_latents


def _layout_condition_rows(layout):
    try:
        segments=tuple(layout.segments)
    except Exception as exc:
        raise LayoutCompatibilityError(
            "Core PackedLayout does not expose segments"
        ) from exc
    visual=audio=0
    for item in segments:
        if not isinstance(item,(tuple,list)) or len(item)!=3:
            raise LayoutCompatibilityError(
                "Core PackedLayout segment is not (start, stop, kind)"
            )
        start,stop,kind=int(item[0]),int(item[1]),str(item[2])
        if stop<start:
            raise LayoutCompatibilityError(
                "Core PackedLayout segment has a negative row span"
            )
        rows=stop-start
        if kind in ("cond","ref_img"):
            visual+=rows
        elif kind in ("cond_audio","ref_audio"):
            audio+=rows
    return int(visual),int(audio)


def _layout_signature(layout):
    try:
        signature=tuple(int(value) for value in layout.signature)
    except Exception as exc:
        raise LayoutCompatibilityError(
            "Core PackedLayout does not expose a five-value signature"
        ) from exc
    if len(signature)!=5:
        raise LayoutCompatibilityError(
            "Core PackedLayout signature must be "
            "(text_len, latent_t, latent_h, latent_w, audio_t)"
        )
    text_len,latent_t,latent_h,latent_w,audio_t=signature
    if min(text_len,latent_t,latent_h,latent_w,audio_t)<0:
        raise LayoutCompatibilityError(
            "Core PackedLayout signature contains a negative value"
        )
    if latent_h%_VIDEO_PATCH[1] or latent_w%_VIDEO_PATCH[2]:
        raise LayoutCompatibilityError(
            "Core PackedLayout target latent height/width is not divisible by 2"
        )
    return signature


def _tensor_shape(value,dimensions,label,errors):
    if not torch.is_tensor(value):
        errors.append(f"{label} is missing or is not a Tensor")
        return None
    shape=tuple(int(item) for item in value.shape)
    if len(shape)!=dimensions:
        errors.append(
            f"{label} must have {dimensions} dimensions, got shape={shape}"
        )
        return None
    return shape


def _video_block(*,index,kind,latent,expected_rows,metadata,errors):
    label=f"Visual block #{index} ({kind})"
    block_errors=[]
    shape=_tensor_shape(latent,5,label,block_errors)
    actual_rows=None
    if shape is not None:
        batch,channels,latent_t,latent_h,latent_w=shape
        if batch!=1:
            block_errors.append(f"batch must be 1, got {batch}")
        if channels!=_VIDEO_CHANNELS:
            block_errors.append(
                f"video latent channels must be {_VIDEO_CHANNELS}, got {channels}"
            )
        pt,ph,pw=_VIDEO_PATCH
        if latent_t%pt or latent_h%ph or latent_w%pw:
            block_errors.append(
                "video latent T/H/W is not divisible by the H3 patch size "
                f"{_VIDEO_PATCH}"
            )
        else:
            actual_rows=batch*(latent_t//pt)*(latent_h//ph)*(latent_w//pw)
    if actual_rows is not None and int(expected_rows)!=int(actual_rows):
        block_errors.append(
            f"expected rows {int(expected_rows)} != actual rows {int(actual_rows)}"
        )
    errors.extend(block_errors)
    return {
        "index":int(index),
        "type":str(kind),
        "shape":shape,
        "metadata":dict(metadata),
        "expected_rows":int(expected_rows),
        "actual_rows":None if actual_rows is None else int(actual_rows),
        "errors":tuple(block_errors),
    }


def _audio_block(*,index,kind,latent,expected_rows,metadata,errors):
    label=f"Audio block #{index} ({kind})"
    block_errors=[]
    shape=_tensor_shape(latent,4,label,block_errors)
    actual_rows=None
    if shape is not None:
        batch,channels,streams,latent_t=shape
        if batch!=1:
            block_errors.append(f"batch must be 1, got {batch}")
        if channels!=_AUDIO_CHANNELS:
            block_errors.append(
                f"audio latent channels must be {_AUDIO_CHANNELS}, got {channels}"
            )
        if streams!=_AUDIO_STREAMS:
            block_errors.append(
                f"audio latent streams must be {_AUDIO_STREAMS}, got {streams}"
            )
        actual_rows=streams*latent_t
    if actual_rows is not None and int(expected_rows)!=int(actual_rows):
        block_errors.append(
            f"expected rows {int(expected_rows)} != actual rows {int(actual_rows)}"
        )
    errors.extend(block_errors)
    return {
        "index":int(index),
        "type":str(kind),
        "shape":shape,
        "metadata":dict(metadata),
        "expected_rows":int(expected_rows),
        "actual_rows":None if actual_rows is None else int(actual_rows),
        "errors":tuple(block_errors),
    }


def _format_preflight_failure(report,reason):
    lines=["H3 Continuum PackedLayout preflight failed",""]
    for group,label in (
        (report["visual_blocks"],"Visual"),
        (report["audio_blocks"],"Audio"),
    ):
        for block in group:
            if not block["errors"]:
                continue
            lines.extend((
                f"{label} block #{block['index']}",
                f"type: {block['type']}",
            ))
            metadata=block["metadata"]
            if "resolved_frame_index" in metadata:
                lines.append(
                    f"resolved_frame_index: {metadata['resolved_frame_index']}"
                )
            shape=block["shape"]
            if shape is not None:
                if label=="Visual":
                    b,c,t,h,w=shape
                    lines.append(
                        f"latent shape: B={b} C={c} T={t} H={h} W={w}"
                    )
                else:
                    b,c,ch,t=shape
                    lines.append(
                        f"latent shape: B={b} C={c} streams={ch} T={t}"
                    )
            lines.extend((
                f"expected rows: {block['expected_rows']}",
                f"actual rows: {block['actual_rows']}",
            ))
            if block["actual_rows"] is not None:
                lines.append(
                    "delta: "
                    f"{block['actual_rows']-block['expected_rows']:+d}"
                )
            lines.extend(f"detail: {value}" for value in block["errors"])
            lines.append("")
    lines.extend((
        f"Layout expected visual rows: {report['layout_visual_rows']}",
        f"Actual visual rows: {report['actual_visual_rows']}",
        f"Layout expected audio rows: {report['layout_audio_rows']}",
        f"Actual audio rows: {report['actual_audio_rows']}",
        "",
        str(reason),
        "No tensors were resized, truncated, padded, or replaced.",
        "Sampling was not started.",
    ))
    return "\n".join(lines)


def preflight_packed_layout(payload,*,repair_stale=False):
    """Validate Core H3 condition rows and narrowly repair a stale layout.

    A repair is allowed only when every current keyframe/reference tensor is
    self-consistent with the geometry Core will use to construct a fresh
    PackedLayout.  Tensor geometry is never changed here.
    """
    if not isinstance(payload,dict):
        raise LayoutCompatibilityError("MiniMax H3 payload must be a dict")
    layout=payload.get("layout")
    if layout is None:
        return {
            "status":"layout_unavailable",
            "repaired":False,
            "visual_blocks":(),
            "audio_blocks":(),
        }
    signature=_layout_signature(layout)
    _text_len,_target_t,target_h,target_w,_target_audio_t=signature
    target_frame_rows=(target_h//_VIDEO_PATCH[1])*(target_w//_VIDEO_PATCH[2])
    keyframes=list(payload.get("keyframes") or ())
    refs=list(payload.get("refs") or ())
    visual_blocks=[]; audio_blocks=[]; geometry_errors=[]
    visual_index=audio_index=0
    for keyframe in keyframes:
        if not isinstance(keyframe,Mapping):
            geometry_errors.append("keyframe entry is not a mapping")
            continue
        latent=keyframe.get("latent")
        if latent is not None:
            visual_index+=1
            shape=tuple(int(value) for value in getattr(latent,"shape",()))
            latent_t=shape[2] if len(shape)==5 else 0
            visual_blocks.append(_video_block(
                index=visual_index,
                kind="keyframe",
                latent=latent,
                expected_rows=latent_t*target_frame_rows,
                metadata={
                    "resolved_frame_index":keyframe.get("resolved_frame_index")
                },
                errors=geometry_errors,
            ))
        audio_latent=keyframe.get("audio_latent")
        if audio_latent is not None:
            audio_index+=1
            shape=tuple(int(value) for value in getattr(audio_latent,"shape",()))
            latent_t=shape[-1] if len(shape)==4 else 0
            audio_blocks.append(_audio_block(
                index=audio_index,
                kind="keyframe",
                latent=audio_latent,
                expected_rows=latent_t*_AUDIO_STREAMS,
                metadata={
                    "resolved_frame_index":keyframe.get("resolved_frame_index")
                },
                errors=geometry_errors,
            ))
    for ref in refs:
        if not isinstance(ref,Mapping):
            geometry_errors.append("reference entry is not a mapping")
            continue
        kind=str(ref.get("kind",""))
        if kind=="image":
            visual_index+=1
            latent_h=int(ref.get("latent_h",0)); latent_w=int(ref.get("latent_w",0))
            visual_blocks.append(_video_block(
                index=visual_index,
                kind="ref_image",
                latent=ref.get("latent"),
                expected_rows=(latent_h//2)*(latent_w//2),
                metadata={"latent_t":1,"latent_h":latent_h,"latent_w":latent_w},
                errors=geometry_errors,
            ))
            block=visual_blocks[-1]
            if block["shape"] is not None:
                _b,_c,actual_t,actual_h,actual_w=block["shape"]
                metadata_errors=[]
                if actual_t!=1:
                    metadata_errors.append(
                        f"image reference latent_t must be 1, got {actual_t}"
                    )
                if (actual_h,actual_w)!=(latent_h,latent_w):
                    metadata_errors.append(
                        "reference metadata H/W "
                        f"{(latent_h,latent_w)} != latent H/W {(actual_h,actual_w)}"
                    )
                if metadata_errors:
                    geometry_errors.extend(metadata_errors)
                    block["errors"]=tuple((*block["errors"],*metadata_errors))
        elif kind in ("video","video_audio"):
            visual_index+=1
            latent_t=int(ref.get("latent_t",0)); latent_h=int(ref.get("latent_h",0)); latent_w=int(ref.get("latent_w",0))
            visual_blocks.append(_video_block(
                index=visual_index,
                kind=f"ref_{kind}",
                latent=ref.get("latent"),
                expected_rows=latent_t*(latent_h//2)*(latent_w//2),
                metadata={"latent_t":latent_t,"latent_h":latent_h,"latent_w":latent_w},
                errors=geometry_errors,
            ))
            block=visual_blocks[-1]
            if block["shape"] is not None:
                _b,_c,actual_t,actual_h,actual_w=block["shape"]
                metadata_errors=[]
                if (actual_t,actual_h,actual_w)!=(latent_t,latent_h,latent_w):
                    metadata_errors.append(
                        "reference metadata T/H/W "
                        f"{(latent_t,latent_h,latent_w)} != latent T/H/W "
                        f"{(actual_t,actual_h,actual_w)}"
                    )
                if metadata_errors:
                    geometry_errors.extend(metadata_errors)
                    block["errors"]=tuple((*block["errors"],*metadata_errors))
        elif kind not in ("audio",):
            geometry_errors.append(f"unsupported reference kind: {kind!r}")
        if kind in ("audio","video","video_audio"):
            ref_audio_t=int(ref.get("ref_audio_t",0))
            audio_latent=ref.get("audio_latent")
            if ref_audio_t>0 or audio_latent is not None:
                audio_index+=1
                audio_blocks.append(_audio_block(
                    index=audio_index,
                    kind=f"ref_{kind}",
                    latent=audio_latent,
                    expected_rows=ref_audio_t*_AUDIO_STREAMS,
                    metadata={"ref_audio_t":ref_audio_t},
                    errors=geometry_errors,
                ))
                block=audio_blocks[-1]
                if block["shape"] is not None:
                    actual_t=block["shape"][-1]
                    if actual_t!=ref_audio_t:
                        message=(
                            f"reference metadata audio_t {ref_audio_t} != "
                            f"latent audio_t {actual_t}"
                        )
                        geometry_errors.append(message)
                        block["errors"]=tuple((*block["errors"],message))
    layout_visual_rows,layout_audio_rows=_layout_condition_rows(layout)
    actual_visual_rows=sum(
        int(block["actual_rows"] or 0) for block in visual_blocks
    )
    actual_audio_rows=sum(
        int(block["actual_rows"] or 0) for block in audio_blocks
    )
    report={
        "status":"matched",
        "repaired":False,
        "visual_blocks":tuple(visual_blocks),
        "audio_blocks":tuple(audio_blocks),
        "layout_visual_rows":int(layout_visual_rows),
        "actual_visual_rows":int(actual_visual_rows),
        "layout_audio_rows":int(layout_audio_rows),
        "actual_audio_rows":int(actual_audio_rows),
    }
    if geometry_errors:
        raise LayoutCompatibilityError(_format_preflight_failure(
            report,"Unsafe geometry mismatch."
        ))
    rows_match=(
        layout_visual_rows==actual_visual_rows
        and layout_audio_rows==actual_audio_rows
    )
    if rows_match:
        return report
    if not repair_stale:
        raise LayoutCompatibilityError(_format_preflight_failure(
            report,"PackedLayout rows do not match the current payload."
        ))
    try:
        rebuilt=type(layout)(
            *signature,
            keyframes=keyframes or None,
            refs=refs or None,
        )
    except Exception as exc:
        raise LayoutCompatibilityError(_format_preflight_failure(
            report,
            "PackedLayout is stale, but a fresh layout could not be constructed: "
            f"{type(exc).__name__}: {exc}",
        )) from exc
    rebuilt_visual_rows,rebuilt_audio_rows=_layout_condition_rows(rebuilt)
    if (
        rebuilt_visual_rows!=actual_visual_rows
        or rebuilt_audio_rows!=actual_audio_rows
    ):
        raise LayoutCompatibilityError(_format_preflight_failure(
            report,
            "A fresh PackedLayout still does not match the current payload; "
            "automatic repair is unsafe.",
        ))
    payload["layout"]=rebuilt
    report.update({
        "status":"layout_rebuilt",
        "repaired":True,
        "layout_visual_rows":int(rebuilt_visual_rows),
        "layout_audio_rows":int(rebuilt_audio_rows),
    })
    return report
def _input_device(x):
    device=getattr(x,"device",None)
    if isinstance(device,torch.device): return device
    if hasattr(x,"unbind"):
        try:
            parts=x.unbind()
            if parts:
                d=getattr(parts[0],"device",None)
                if isinstance(d,torch.device): return d
        except Exception: return None
    return None
def materialize_continuum_latents(payload,x,*,debug=False):
    layout=payload.get("layout"); device=_input_device(x)
    if layout is None or device is None or device.type=="cpu": return
    keyframes=[dict(item) for item in (payload.get("keyframes") or ())]; refs=[dict(item) for item in (payload.get("refs") or ())]
    marked=[]
    for i,item in enumerate(keyframes):
        if MARK_VIDEO_SLOT in item and torch.is_tensor(item.get("latent")): marked.append(("keyframe",i,item["latent"]))
    for i,item in enumerate(refs):
        if _is_video_context(item) and torch.is_tensor(item.get("latent")): marked.append(("ref_video",i,item["latent"]))
        if _is_audio_context(item) and torch.is_tensor(item.get("audio_latent")): marked.append(("ref_audio",i,item["audio_latent"]))
    if not marked: return
    cache_key=(str(device),tuple((kind,i,id(t),tuple(t.shape),str(t.dtype)) for kind,i,t in marked))
    cached=getattr(layout,LAYOUT_DEVICE_CACHE_ATTR,None)
    if not isinstance(cached,dict) or cached.get("key")!=cache_key:
        values={(kind,i):t.to(device=device,dtype=torch.float32,non_blocking=True).contiguous() for kind,i,t in marked}
        cached={"key":cache_key,"values":values}; setattr(layout,LAYOUT_DEVICE_CACHE_ATTR,cached)
        if debug:
            total_bytes=sum(t.numel()*t.element_size() for t in values.values())
            LOG.info("Continuum context cached on %s: %d tensors, %.2f MiB",device,len(values),total_bytes/(1024*1024))
    values=cached["values"]
    for i,item in enumerate(keyframes):
        replacement=values.get(("keyframe",i))
        if replacement is not None: item["latent"]=replacement
    for i,item in enumerate(refs):
        replacement=values.get(("ref_video",i))
        if replacement is not None: item["latent"]=replacement
        replacement=values.get(("ref_audio",i))
        if replacement is not None: item["audio_latent"]=replacement
    payload["keyframes"]=keyframes; payload["refs"]=refs

def _single_segment(layout,kind):
    matches=[(int(a),int(b)) for a,b,k in layout.segments if k==kind]
    if len(matches)!=1: raise LayoutCompatibilityError(f"expected one H3 '{kind}' segment, found {len(matches)}")
    return matches[0]
def _map_refs_to_segments(layout,refs):
    available=[(int(a),int(b),str(kind)) for a,b,kind in layout.segments if kind in ("ref_img","ref_audio")]
    cursor=0; result=[]
    def consume(expected):
        nonlocal cursor
        if cursor>=len(available): raise LayoutCompatibilityError(f"layout ended while mapping '{expected}'")
        a,b,kind=available[cursor]; cursor+=1
        if kind!=expected: raise LayoutCompatibilityError(f"reference layout mismatch: expected '{expected}', found '{kind}'")
        return a,b
    for ref in refs:
        kind=ref.get("kind"); mapping={"audio":None,"video":None}
        if kind=="image": mapping["video"]=consume("ref_img")
        elif kind=="audio":
            if int(ref.get("ref_audio_t",0))>0: mapping["audio"]=consume("ref_audio")
        elif kind in ("video","video_audio"):
            if int(ref.get("ref_audio_t",0))>0: mapping["audio"]=consume("ref_audio")
            mapping["video"]=consume("ref_img")
        else: raise LayoutCompatibilityError(f"unsupported H3 reference kind: {kind!r}")
        result.append(mapping)
    if cursor!=len(available): raise LayoutCompatibilityError(f"layout contains {len(available)-cursor} unmapped reference segments")
    return result
def _map_keyframes_to_segments(layout,keyframes):
    available=[(int(a),int(b),str(kind)) for a,b,kind in layout.segments if kind in ("cond","cond_audio")]
    cursor=0; result=[]
    def consume(expected):
        nonlocal cursor
        if cursor>=len(available): raise LayoutCompatibilityError(f"layout ended while mapping keyframe '{expected}'")
        a,b,kind=available[cursor]; cursor+=1
        if kind!=expected: raise LayoutCompatibilityError(f"keyframe layout mismatch: expected '{expected}', found '{kind}'")
        return a,b
    for keyframe in keyframes:
        mapping={"audio":None,"video":None}
        if keyframe.get("latent") is not None: mapping["video"]=consume("cond")
        if keyframe.get("audio_latent") is not None: mapping["audio"]=consume("cond_audio")
        if mapping["video"] is None and mapping["audio"] is None:
            raise LayoutCompatibilityError("H3 keyframe has neither video nor audio latent")
        result.append(mapping)
    if cursor!=len(available): raise LayoutCompatibilityError(f"layout contains {len(available)-cursor} unmapped keyframe segments")
    return result
def _patch_signature(payload,layout):
    return (
        tuple((int(kf.get("resolved_frame_index",-1)),int(kf.get(MARK_VIDEO_SLOT,-1))) for kf in (payload.get("keyframes") or ())),
        tuple((bool(ref.get(MARK_VIDEO_CONTEXT)),tuple(sorted(_reference_metadata(ref).items())),int(ref.get("latent_t",0)),int(ref.get(MARK_CONTEXT_FRAMES,0)),bool(ref.get(MARK_AUDIO_CONTEXT)),int(ref.get("ref_audio_t",0)),float(ref.get(MARK_AUDIO_END_FRAME,0.0)),float(ref.get(MARK_AUDIO_OVERHANG,0.0))) for ref in (payload.get("refs") or ())),
        tuple(getattr(layout,"signature",())), tuple((int(a),int(b),str(k)) for a,b,k in layout.segments),
    )
def _branch_key(transformer_options):
    conds=transformer_options.get("cond_or_uncond"); uuids=transformer_options.get("uuids")
    if conds is None or uuids is None: return ("default",)
    try: cond_values=tuple(int(v) for v in conds); uuid_values=tuple(str(v) for v in uuids)
    except (TypeError,ValueError): return ("default",)
    if not cond_values or len(cond_values)!=len(uuid_values): return ("default",)
    return tuple((cond_values[i],uuid_values[i]) for i in range(len(cond_values)))
def _sampled_tensor_signature(tensor):
    if tensor is None: return None
    value=tensor.detach().to(dtype=torch.float32); flat=value.reshape(-1)
    if flat.numel()<=64: sample=flat
    else:
        indices=torch.linspace(0,flat.numel()-1,64,dtype=torch.long,device=flat.device); sample=flat.index_select(0,indices)
    sample=sample.to("cpu"); finite=bool(torch.isfinite(sample).all().item())
    if not finite: raise LayoutCompatibilityError("Continuum layout/context contains NaN or Inf")
    weights=torch.arange(1,sample.numel()+1,dtype=torch.float32)
    return (tuple(int(part) for part in tensor.shape),str(tensor.dtype),finite,float(sample.mean().item()),float(sample.std(unbiased=False).item()),float((sample*weights).sum().item()))
def _ranges_overlap(left,right): return max(left[0],right[0])<min(left[1],right[1])

def validate_native_continuity_layout(payload,*,transformer_options,branch_baselines):
    layout=payload.get("layout")
    if layout is None: raise LayoutCompatibilityError("Native Continuity payload has no PackedLayout")
    position_ids=getattr(layout,"position_ids",None); segments=tuple((int(start),int(stop),str(kind)) for start,stop,kind in getattr(layout,"segments",()))
    if not torch.is_tensor(position_ids) or position_ids.ndim!=2: raise LayoutCompatibilityError("Native Continuity position_ids are unavailable")
    seq_len=int(getattr(layout,"seq_len",position_ids.shape[0]))
    if seq_len!=int(position_ids.shape[0]): raise LayoutCompatibilityError("PackedLayout seq_len does not match position_ids")
    previous_stop=0
    for start,stop,kind in segments:
        if start<previous_stop or stop<=start or stop>seq_len: raise LayoutCompatibilityError(f"PackedLayout segment {kind!r} has an overlapping or invalid row range")
        previous_stop=stop
    target_audio=[(a,b) for a,b,kind in segments if kind=="audio"]; target_video=[(a,b) for a,b,kind in segments if kind=="video"]
    if len(target_audio)!=1 or len(target_video)!=1: raise LayoutCompatibilityError("Native Continuity requires one target audio and one target video segment")
    audio_range,video_range=target_audio[0],target_video[0]
    if audio_range[1]!=video_range[0] or video_range[1]!=seq_len: raise LayoutCompatibilityError("target segments must be the contiguous packed tail [audio | video]")
    refs=list(payload.get("refs") or ()); mappings=_map_refs_to_segments(layout,refs); context_ranges=[]; context_descriptors=[]
    for ref,mapping in zip(refs,mappings):
        metadata=_reference_metadata(ref)
        if metadata:
            if type(metadata.get("api")) is not int or int(metadata["api"])!=CONTINUUM_INTEROP_API: raise LayoutCompatibilityError("unsupported Continuum reference metadata")
            if _is_video_context(ref) and metadata.get("role")!="video_context": raise LayoutCompatibilityError("Continuum video reference role is invalid")
            if _is_audio_context(ref) and metadata.get("audio_role") not in (None,"audio_context"): raise LayoutCompatibilityError("Continuum audio reference role is invalid")
        if _is_video_context(ref):
            row_range=mapping.get("video")
            if row_range is None: raise LayoutCompatibilityError("Continuum video context has no reference rows")
            context_ranges.append((*row_range,"video_context"))
        if _is_audio_context(ref):
            row_range=mapping.get("audio")
            if row_range is None: raise LayoutCompatibilityError("Continuum audio context has no reference rows")
            context_ranges.append((*row_range,"audio_context"))
        if _is_video_context(ref) or _is_audio_context(ref):
            context_descriptors.append((ref.get("kind"),int(ref.get("latent_t",0)),int(ref.get("latent_h",0)),int(ref.get("latent_w",0)),int(ref.get("ref_audio_t",0)),tuple(sorted(metadata.items())),_sampled_tensor_signature(ref.get("latent")),_sampled_tensor_signature(ref.get("audio_latent"))))
    if not context_ranges: raise LayoutCompatibilityError("Native Continuity payload has no context rows")
    for start,stop,role in context_ranges:
        if _ranges_overlap((start,stop),audio_range) or _ranges_overlap((start,stop),video_range): raise LayoutCompatibilityError(f"{role} rows overlap target rows")
    topology=(tuple(getattr(layout,"signature",())),segments,audio_range,video_range,tuple(context_ranges),tuple(context_descriptors),int(payload.get("frame_count",0) or 0),_sampled_tensor_signature(position_ids))
    options=transformer_options if isinstance(transformer_options,dict) else {}
    if options.get("spectrum_h3_actual") is False: return {"status":"forecast_validated","branch":_branch_key(options)}
    branch=_branch_key(options); baseline=branch_baselines.get(branch)
    if baseline is None: branch_baselines[branch]=topology; return {"status":"baseline_created","branch":branch}
    if baseline!=topology: raise LayoutCompatibilityError(f"Native Continuity topology changed during sampling for branch {branch!r}")
    return {"status":"baseline_matched","branch":branch}

def patch_layout_in_place(payload,*,strict=True,debug=False):
    layout=payload.get("layout")
    if layout is None: raise LayoutCompatibilityError("H3 payload has no prebuilt PackedLayout; use ComfyUI v0.31.0 or newer")
    missing=[name for name in ("position_ids","segments","signature") if not hasattr(layout,name)]
    if missing: raise LayoutCompatibilityError("PackedLayout is missing: "+", ".join(missing))
    position_ids=layout.position_ids
    if not torch.is_tensor(position_ids) or position_ids.ndim!=2 or position_ids.shape[1]<3: raise LayoutCompatibilityError("position_ids must be an [S,3] Tensor")
    signature=_patch_signature(payload,layout)
    if getattr(layout,LAYOUT_SIGNATURE_ATTR,None)==signature: return {"status":"already_patched","position_ids_id":id(position_ids)}
    if not hasattr(layout,LAYOUT_ORIGINAL_TIME_ATTR): setattr(layout,LAYOUT_ORIGINAL_TIME_ATTR,position_ids[:,0].clone())
    original_time=getattr(layout,LAYOUT_ORIGINAL_TIME_ATTR)
    if not torch.is_tensor(original_time) or original_time.shape!=position_ids[:,0].shape: raise LayoutCompatibilityError("stored layout baseline is incompatible")
    position_ids[:,0].copy_(original_time)
    keyframes=list(payload.get("keyframes") or ()); refs=list(payload.get("refs") or ()); keyframe_mappings=_map_keyframes_to_segments(layout,keyframes)
    video_start,video_stop=_single_segment(layout,"video"); _single_segment(layout,"audio")
    text_segments=[(int(a),int(b)) for a,b,kind in layout.segments if kind=="text"]
    if len(text_segments)!=1 or text_segments[0][0]!=0: raise LayoutCompatibilityError("unexpected H3 text segment")
    text_len=text_segments[0][1]; latent_t=int(layout.signature[1]); video_rows=video_stop-video_start
    if latent_t<=0 or video_rows%latent_t: raise LayoutCompatibilityError(f"video rows {video_rows} are incompatible with latent T={latent_t}")
    frame_rows=video_rows//latent_t; target_origin=float(position_ids[video_start,0])
    patched_video_slots=0; patched_keyframe_audio=0
    for keyframe,mapping in zip(keyframes,keyframe_mappings):
        frame_index=int(keyframe.get("resolved_frame_index",0))
        if frame_index<0: raise LayoutCompatibilityError(f"negative H3 keyframe index {frame_index}")
        desired_start=target_origin+FRAME_RESCALE*float(frame_index)
        video_segment=mapping.get("video")
        if MARK_VIDEO_SLOT in keyframe:
            if video_segment is None: raise LayoutCompatibilityError("legacy Continuum video slot has no cond rows")
            start,stop=video_segment
            if stop-start!=frame_rows: raise LayoutCompatibilityError("keyframe rows no longer equal one target slot")
            slot=int(keyframe[MARK_VIDEO_SLOT])
            if not (0<=slot<latent_t): raise LayoutCompatibilityError(f"context slot {slot} is outside target latent T={latent_t}")
            target_row=video_start+slot*frame_rows; position_ids[start:stop].copy_(position_ids[target_row:target_row+frame_rows]); patched_video_slots+=1
        elif video_segment is not None:
            start,stop=video_segment; old_start=float(position_ids[start,0]); position_ids[start:stop,0].add_(desired_start-old_start)
        audio_segment=mapping.get("audio")
        if audio_segment is not None:
            start,stop=audio_segment; old_start=float(position_ids[start,0]); position_ids[start:stop,0].add_(desired_start-old_start); patched_keyframe_audio+=1
    ref_mappings=_map_refs_to_segments(layout,refs); patched_video_refs=0; patched_audio=0
    for ref,mapping in zip(refs,ref_mappings):
        if _is_video_context(ref):
            segment=mapping.get("video")
            if segment is None: raise LayoutCompatibilityError("Continuum video ref has no video segment")
            start,stop=segment; context_t=int(ref.get("latent_t",0)); expected_rows=context_t*frame_rows
            if context_t<=0 or stop-start!=expected_rows: raise LayoutCompatibilityError(f"video-context rows {stop-start} do not match latent T={context_t}")
            if context_t>latent_t: raise LayoutCompatibilityError(f"video context T={context_t} exceeds target latent T={latent_t}")
            position_ids[start:stop].copy_(position_ids[video_start:video_start+expected_rows]); patched_video_refs+=1
        if _is_audio_context(ref):
            segment=mapping.get("audio")
            if segment is None: raise LayoutCompatibilityError("Continuum audio ref has no audio segment")
            start,stop=segment; rt=int(ref.get("ref_audio_t",0))
            if rt<=0 or stop-start!=rt*2: raise LayoutCompatibilityError(f"audio rows {stop-start} do not match {rt} stereo steps")
            end_frame=float(ref.get(MARK_AUDIO_END_FRAME,0.0)); grid_offset=float(ref.get(MARK_AUDIO_OVERHANG,0.0)); desired_end=target_origin+FRAME_RESCALE*end_frame+grid_offset; desired_start=desired_end-float(rt); old_start=float(position_ids[start,0]); position_ids[start:stop,0].add_(desired_start-old_start); patched_audio+=1
    patched_video=patched_video_slots+patched_video_refs
    if patched_video==0:
        message="no marked Continuum video context was found"
        if strict: raise LayoutCompatibilityError(message)
        LOG.warning(message)
    setattr(layout,LAYOUT_SIGNATURE_ATTR,signature)
    if debug: LOG.info("Continuum layout: video_refs=%d legacy_slots=%d keyframe_audio=%d audio_windows=%d origin=%.6f rows=%d pos_id=%s",patched_video_refs,patched_video_slots,patched_keyframe_audio,patched_audio,target_origin,position_ids.shape[0],id(position_ids))
    return {"status":"patched","video_contexts":patched_video,"video_refs":patched_video_refs,"legacy_video_slots":patched_video_slots,"keyframe_audio_windows":patched_keyframe_audio,"audio_windows":patched_audio,"target_origin":target_origin,"position_ids_id":id(position_ids)}
