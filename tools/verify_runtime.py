"""Verify standalone registration, native H3 layout, Finalize and CPU FFmpeg."""
from pathlib import Path
import argparse
import asyncio
import importlib
import hashlib
import json
import sys
import tempfile
import time

parser=argparse.ArgumentParser()
parser.add_argument('--comfy-root',type=Path,required=True)
parser.add_argument('--evidence-root',type=Path,required=True)
args=parser.parse_args()
started=time.perf_counter()
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(args.comfy_root))
import nodes
import torch
from PIL import Image
from comfy.nested_tensor import NestedTensor

before=set(nodes.NODE_CLASS_MAPPINGS)
assert asyncio.run(nodes.load_custom_node(str(root))), 'standalone package import failed'
module=sys.modules[str(root).replace('.', '_x_')]
assert len(module.NODE_CLASS_MAPPINGS)==6
assert set(nodes.NODE_CLASS_MAPPINGS)-before==set(module.NODE_CLASS_MAPPINGS)
assert all(key.endswith('_design61') for key in module.NODE_CLASS_MAPPINGS)
assert all(value.endswith('_design61') for value in module.NODE_DISPLAY_NAME_MAPPINGS.values())
assert all('ComfyUI-H3-Continuum' not in str(getattr(item,'__file__','')) for item in sys.modules.values())
for key,cls in module.NODE_CLASS_MAPPINGS.items():
    assert isinstance(cls.INPUT_TYPES(),dict), key
private=module.__name__+'._h3'
layout=importlib.import_module(private+'.compatibility').run_native_layout_self_test()
state=importlib.import_module(private+'.state')
temporal=importlib.import_module(private+'.temporal')
session=importlib.import_module(private+'.v2.session')
sequence=importlib.import_module(private+'.v3.external_sequence')
frames=124
latent={'samples':NestedTensor((torch.zeros(1,24,temporal.video_latent_t(frames),4,4),torch.zeros(1,32,2,temporal.audio_latent_t(frames))))}
plan=state.make_plan(continuation=False,clip_index=1,total_frames=frames,trim_frames=0,width=64,height=64,context_frames=5,state_capacity_frames=temporal.largest_context_capacity(frames),requested_extend_seconds=frames/24,debug=False)
entry=session.make_chunk_entry(latent=latent,plan=plan,prompt='Synthetic CPU verifier',prompt_hash=hashlib.sha256(b'Synthetic CPU verifier').hexdigest(),seed=123,context_frames=5,motion_score=0.0,reused=False)
assembly=sequence.sequence_outputs((entry,),5.0)[2]
original={'waveform':torch.zeros(1,2,32000*20),'sample_rate':32000}
assembly['_h3_continuum_driving_audio_v1']=original
upstream=args.comfy_root/'custom_nodes'/'ComfyUI-H3-Continuum'
assert asyncio.run(nodes.load_custom_node(str(upstream)))
class Clip:
    def tokenize(self,prompt,**kwargs): return prompt
    def encode_from_tokens_scheduled(self,tokens): return [[torch.ones(1,2,3),{}]]
class VideoVAE:
    def encode(self,image): return torch.ones(1,24,1,image.shape[1]//16,image.shape[2]//16)
class AudioVAE:
    audio_sample_rate=32000
    def encode(self,waveform): return torch.ones(1,32,2,round(waveform.shape[1]/32000*40))
images=nodes.NODE_CLASS_MAPPINGS['H3ContinuumReferenceImagesV39']().pack(reference_use='Per chunk',reference_image_1=torch.ones(1,64,64,3),reference_r1_chunks='2')[0]
audio_bundle=nodes.NODE_CLASS_MAPPINGS['H3ContinuumReferenceAudios']().pack(reference_audio_1={'waveform':torch.zeros(1,2,64000),'sample_rate':32000},reference_audio_vae=AudioVAE())[0]
for size in (64,128):
    result=module.NODE_CLASS_MAPPINGS['H3ContinuumExternalConditioning_design61']().build(clip=Clip(),video_vae=VideoVAE(),prompt='@R1 <Audio 1>',width=size,height=size,length=141,chunks=2,chunk_index=2,reference_images=images,audio_references=audio_bundle)
    refs=result['result'][0][0][1]['minimax_refs']
    assert len([ref for ref in refs if ref['kind']=='image'])==1
    assert [ref['ref_audio_t'] for ref in refs if ref['kind']=='audio']==[80]
assert images.selectors[0]=='2' and len(audio_bundle.sources)==1
finalize=nodes.NODE_CLASS_MAPPINGS['H3ContinuumAssembleSeamV35']()
final=finalize.assemble(images=[torch.zeros(frames,64,64,3)],audio=[{'waveform':torch.ones(1,2,32000*6),'sample_rate':32000}],assembly_plan=[assembly],exact_total_duration=[True],audio_seam=['Off'],video_seam=['Off'],buffer_backend=['RAM'],diagnostics=['Off'])
assert final[0].shape[0]==120 and final[1]['waveform'].shape[-1]==160000
assert torch.count_nonzero(final[1]['waveform'])==0

import imageio_ffmpeg
ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
args.evidence_root.mkdir(parents=True,exist_ok=True)
temporary=Path(tempfile.mkdtemp(prefix='design61-ffmpeg-',dir=args.evidence_root)).resolve()
assert temporary.is_relative_to(args.evidence_root.resolve())
frame_dir=temporary/'frames'
frame_dir.mkdir()
for number,color in [(1,'red'),(2,'green'),(10,'blue'),(11,'white')]:
    Image.new('RGB',(64,64),color).save(frame_dir/f'frame{number}.png')
output=temporary/'encoded.mp4'
cls=module.NODE_CLASS_MAPPINGS['FolderFramesToVideoFFmpeg_design61']
defaults={}
for key,(kind,*options) in cls.INPUT_TYPES()['required'].items():
    defaults[key]=options[0]['default'] if options and 'default' in options[0] else kind[0] if isinstance(kind,list) else ''
defaults.update(frames_path=str(frame_dir),output_path=str(output),ffmpeg_path=ffmpeg,backend='cpu',codec='h264',preset='ultrafast',fps=24,delete_source_frames=False,save_workflow=True)
encoded=cls().run(**defaults,prompt={'test':{'class_type':'FolderFramesToVideoFFmpeg_design61','inputs':{'fps':24,'codec':'h264','backend':'cpu'}}},extra_pnginfo={'workflow':{'test':'synthetic CPU FolderFramesToVideo verification'}})
assert output.is_file(),encoded['result'][-1]
assert encoded['result'][3]==4
count,duration=imageio_ffmpeg.count_frames_and_secs(str(output))
assert count==4,(count,duration)
assert len(list(frame_dir.glob('*.png')))==4
cleared=module.NODE_CLASS_MAPPINGS['ClearFolderFrames_design61']().run(frames_path=str(frame_dir),extensions='png',create_if_missing=False)
assert cleared['result'][2]==4 and not list(frame_dir.glob('*.png'))
print(json.dumps({'result':'PASS','nodes':6,'standalone_core_loader':True,'original_reference_helpers_both_stages':True,'native_layout':layout,'finalize_frames':120,'finalize_original_audio_samples':160000,
 'ffmpeg':{'workflow':defaults,'prompt':'synthetic CPU verification','media_inputs':'four synthetic RGB PNG frames 64x64','model':None,'lora':None,'sampler':None,'steps':None,'chunk_count':1,'duration':duration,'observed_frames':count,'source_clear_count':4,'video':str(output)},
 'gpu_sampling':False,'elapsed_seconds':round(time.perf_counter()-started,3)},ensure_ascii=False))
