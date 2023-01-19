import io
import gradio as gr
import librosa
import numpy as np
import soundfile
from inference.infer_tool import Svc
from inference import slicer
import logging
import torch

logging.getLogger("numba").setLevel(logging.WARNING)

# 模型及配置文件路径
model_path = "logs/48k/aqua.pth"
config_path = "configs/config.json"
# 模型初始化
svc_model = Svc(model_path, config_path)


# Gradio-WebUI主函数
def app_function(input_audio, trans, device, sr, term):
    # 判断
    if not term:
        return "请阅读并同意《AI阿夸模型使用协议》", None
    if input_audio is None:
        return "请上传音频", None
    if device == "cuda" and not torch.cuda.is_available():
        return "未检测到您的CUDA设备", None

    # 设置设备
    svc_model.dev = device

    # 一些常量
    clean_names = "input_sound"
    slice_db = -40
    spk = 'aqua'
    sr = int(sr)

    # 切片
    sampling_rate, audio = input_audio
    audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio.transpose(1, 0))
    if sampling_rate != sr:
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=sr)
    out_wav_path = f"./raw/{clean_names}.wav"
    soundfile.write(out_wav_path, audio, sr, format="wav")

    chunks = slicer.cut(out_wav_path, db_thresh=slice_db)
    audio_data, audio_sr = slicer.chunks2audio(out_wav_path, chunks)

    # 分片推理并组合
    audio = []
    for (slice_tag, data) in audio_data:
        print(f'#=====segment start, {round(len(data) / audio_sr, 3)}s======')
        length = int(np.ceil(len(data) / audio_sr * svc_model.target_sample))
        raw_path = io.BytesIO()
        soundfile.write(raw_path, data, audio_sr, format="wav")
        raw_path.seek(0)
        if slice_tag:
            print('jump empty segment')
            _audio = np.zeros(length)
        else:
            out_audio, out_sr = svc_model.infer(spk, trans, raw_path)
            _audio = out_audio.cpu().numpy()
        audio.extend(list(_audio))
    out_audio = np.array(audio)

    # 释放内存
    mute_audio_path = "./raw/free_cache.wav"
    mute_chunks = slicer.cut(mute_audio_path, db_thresh=slice_db)
    mute_audio_data, _ = slicer.chunks2audio(out_wav_path, mute_chunks)
    mute_audio = mute_audio_data[0][1]
    raw_path = io.BytesIO()
    soundfile.write(raw_path, mute_audio, audio_sr, format="wav")
    raw_path.seek(0)
    _, _ = svc_model.infer(spk, trans, raw_path)

    # 返回
    return "Success", (48000, out_audio)


# Gradio-WebUI输入
inputs = [
    gr.inputs.Audio(source="upload", label="输入音频"),
    gr.inputs.Number(default=0, label="音调变换"),
    gr.Dropdown(["cpu", "cuda"], value="cpu", label="设备选择"),
    gr.Radio(["24000", "48000"], label="输入音源采样率变换"),
    gr.Checkbox(label="您已阅读并同意《AI阿夸模型使用协议》")
]

# Gradio-WebUI输出
outputs = [
    "text",
    gr.outputs.Audio(type="numpy")
]

# 描述(前言)
des = """
## 在使用此模型前请阅读[AI阿夸模型使用协议](https://huggingface.co/spaces/DoNotSelect/AI-minato/blob/main/terms.md)
"""

# App实例
demo = gr.Interface(
    fn=app_function,
    inputs=inputs,
    outputs=outputs,
    layout="horizontal",
    theme="huggingface",
    description=des,
    allow_flagging="never"
)

# 作为主函数运行
if __name__ == "__main__":
    demo.launch()
