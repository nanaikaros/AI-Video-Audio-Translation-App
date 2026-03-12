# AI 音视频 翻译

简介
- 负责音频识别与翻译，输出带ass字幕的mkv格式视频。

主要功能
- 音频识别（本地模型）
- 翻译（本地模型）
- 支持输入 SRT、输出 SRT/ASS
- 多线程、进度与日志
- 提供 CLI 或可部署为后端服务的可执行文件

目录建议
- /models/   -> 存放模型文件（ggml / gguf）
- /bin/      -> 编译生成的可执行文件或服务二进制
- /samples/  -> 示例输入文件
- /output/   -> 处理后字幕/视频输出
- config.json -> 默认运行参数

先决条件（macOS）
- ffmpeg：brew install ffmpeg
- 必要的构建工具（取决于项目语言，如 gcc/clang、CMake、make、go、cargo）：请按项目 README 的构建说明安装
- 足够的磁盘与内存用于模型加载

模型与资源放置
- 将模型文件放入 /models/，示例：
  - /models/ggml-large-v3.bin
  - /models/HY-MT1.5-1.8B-GGUF.gguf

配置示例（config.json）
```json
{
  "audioModel": "./models/ggml-large-v3.bin",
  "translateModel": "./models/HY-MT1.5-1.8B-GGUF.gguf",
  "inputVideo": "./samples/input.mp4",
  "inputAudio": "./samples/input.wav",
  "inputSrt": "",
  "outputSrt": "./output/output.srt",
  "outputAss": "./output/output.ass",
  "threads": 4,
  "targetLang": "zh-CN",
  "silentLog": true
}
```

安装与运行（示例）
1. 切换到项目目录：
   ```bash
   cd /test
   ```
2. 若项目有构建脚本，编译生成二进制（示例）：
   ```bash
   # 示例：根据项目实际构建命令替换
   mkdir -p build && cd build
   cmake .. && make -j4
   ```
3. 运行 CLI / 服务（示例）：
   ```bash
   ./build/my_app -v [video path] -w [whisper model] -m [translation model] -t [thread] -o [output video path]
   ```

注意事项
- 模型体积大且占内存，确保目标机器资源充足。
- 使用 Intl 或后端本地化方案处理时间/数字格式；若需要多语言 UI，可由独立前端对接此后端。
- 若翻译质量不足，尝试替换或微调翻译模型。
