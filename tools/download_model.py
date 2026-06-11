import os

# 设置 Hugging Face 国内加速镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

# 定位本地目录
local_dir = os.path.join(os.getcwd(), "models", "opus-mt-zh-en")
os.makedirs(local_dir, exist_ok=True)

print(f"正在从国内镜像源下载模型到: {local_dir}")
print("请耐心等待进度条走完...")

try:
    snapshot_download(
        repo_id="Helsinki-NLP/opus-mt-zh-en",
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    print("\n🎉 下载成功！现在里面应该有所有实体文件了。")
except Exception as e:
    print(f"\n❌ 下载失败，错误原因: {e}")