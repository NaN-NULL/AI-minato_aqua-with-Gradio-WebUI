# AI MinatoAqua With Gradio WebUI

## 使用方法
1. 下载soft vc hubert放在hubert目录下[hubert-soft-0d54a1f4.pt](https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt)
2. 下载config文件放在configs目录下[config.json](https://huggingface.co/spaces/DoNotSelect/AI-minato_aqua/resolve/main/configs/config.json)
3. 下载模型文件放在logs/48k目录下[aqua.pth](https://huggingface.co/spaces/DoNotSelect/AI-minato_aqua/resolve/main/logs/48k/aqua.pth)
4. 下载用于释放内存的空白音频文件放在raw目录下[free_cache.wav](https://huggingface.co/spaces/DoNotSelect/AI-minato_aqua/resolve/main/raw/free_cache.wav)
5. 安装依赖
6. 运行app.py
## 基于
+ [so-vits-svc](https://github.com/innnky/so-vits-svc)
+ [vits](https://github.com/jaywalnut310/vits)
+ [hubert](https://github.com/bshall/hubert)
+ [NSF HiFiGAN](https://github.com/openvpi/DiffSinger/tree/refactor/modules/nsf_hifigan)
+ [Gradio](https://gradio.app/)
+ [MinatoAqua](https://www.youtube.com/channel/UC1opHUrw8rvnsadT-iGp7Cg)