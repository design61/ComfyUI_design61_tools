"""design61 personal nodes; only modified features are public."""
from ._h3.v3.external_sampling_nodes import H3ContinuumExternalConditioning_design61, H3ContinuumExternalPrepare_design61
from ._h3.v3.external_sequence_nodes import H3ContinuumExternalSequenceStart_design61, H3ContinuumExternalSequenceEnd_design61
from .folder_frames_to_video import NODE_CLASS_MAPPINGS as folder
WEB_DIRECTORY = "./web"
NODE_CLASS_MAPPINGS = {
    "H3ContinuumExternalSequenceStart_design61": H3ContinuumExternalSequenceStart_design61,
    "H3ContinuumExternalConditioning_design61": H3ContinuumExternalConditioning_design61,
    "H3ContinuumExternalPrepare_design61": H3ContinuumExternalPrepare_design61,
    "H3ContinuumExternalSequenceEnd_design61": H3ContinuumExternalSequenceEnd_design61,
    **folder,
}
NODE_DISPLAY_NAME_MAPPINGS = {'H3ContinuumExternalSequenceStart_design61': 'H3 Continuum External Sequence Start_design61', 'H3ContinuumExternalConditioning_design61': 'H3 Continuum External Conditioning_design61', 'H3ContinuumExternalPrepare_design61': 'H3 Continuum External Prepare_design61', 'H3ContinuumExternalSequenceEnd_design61': 'H3 Continuum External Sequence End_design61', 'ClearFolderFrames_design61': '清空文件夹序列帧_design61', 'FolderFramesToVideoFFmpeg_design61': '文件夹序列帧直接FFmpeg视频_design61'}
for node_id, node in NODE_CLASS_MAPPINGS.items():
    node.CATEGORY = "design61/FFmpeg" if node_id in folder else "design61/H3 Continuum"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
