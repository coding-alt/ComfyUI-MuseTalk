import os
import sys
import uuid
import torch
import torchaudio
import tempfile

import folder_paths
comfy_path = os.path.dirname(folder_paths.__file__)
diffusers_path = folder_paths.get_folder_paths("diffusers")[0]

MuseVCheckPointDir = os.path.join(
    diffusers_path, "TMElyralab/MuseTalk"
)
comfy_musetalk_path = f'{comfy_path}/custom_nodes/ComfyUI-MuseTalk'
sys.path.insert(0, comfy_musetalk_path)

from musetalk_utils.inference import Inference

def get_avatar_list():
    avatar_path = os.path.join(comfy_musetalk_path, "results/avatars")
    return os.listdir(avatar_path)

inference_mode_list = get_avatar_list()

class MuseTalkRun:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "avatar_id":(inference_mode_list,{
                    "default":inference_mode_list[0] if inference_mode_list and inference_mode_list[0] else None
                }),
                "audio":("AUDIO", ),
                "batch_size":("INT",{"default":4}),
                "fps":("INT",{"default":30})
            }
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("audio_save_path", )
    FUNCTION = "run"
    CATEGORY = "MuseTalk"

    def run(self, avatar_id, audio, batch_size, fps):
        try:
            # 创建一个临时文件来保存合并后的音频
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio_file:
                temp_audio_path = temp_audio_file.name

                # 检查并合并所有 waveform 批次
                waveforms = [waveform for waveform in audio["waveform"].cpu()]
                combined_waveform = torch.cat(waveforms, dim=1)

                # 将合并后的音频保存到临时文件
                torchaudio.save(temp_audio_path, combined_waveform, audio["sample_rate"], format="WAV")

            musetalk = Inference(avatar_id, batch_size)
            save_path = f"{comfy_path}/custom_nodes/ComfyUI-MuseTalk/results/avatars/{avatar_id}/vid_output/{str(uuid.uuid4())}.mp4"
            
            state = musetalk.run(temp_audio_path, infer_video_path=save_path, fps=fps, enhance=True)

            if state:
                return save_path
            else:
                return ""
        except Exception as e:
            return str(e)
        finally:
            # 无论成功与否，删除临时音频文件
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

NODE_CLASS_MAPPINGS = {
    "MuseTalkRun":MuseTalkRun
}
