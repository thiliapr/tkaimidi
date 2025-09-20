# Tk AI MIDI
## 简介
这是一个自由的 MIDI 生成软件，主要使用[mido](https://github.com/mido/mido/)来解析音乐，[pytorch](https://github.com/pytorch/pytorch/)来机器学习，旨在通过深度学习技术生成音乐作品。

## License
![GNU AGPL Version 3 Logo](https://www.gnu.org/graphics/agplv3-with-text-162x68.png)

tkaimidi 是自由软件，遵循`Affero GNU 通用公共许可证第 3 版或任何后续版本`。你可以自由地使用、修改和分发该软件，但不提供任何明示或暗示的担保。有关详细信息，请参见 [Affero GNU 通用公共许可证](https://www.gnu.org/licenses/agpl-3.0.html)。

## 安装依赖
### CPU 用户
```bash
pip install -r requirements.txt
```

### CUDA 用户（Nvidia 显卡）
```bash
nvidia-smi  # 查看 CUDA 版本
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
```

> [!TIP]
> 根据`nvidia-smi`的输出`CUDA Version`把`cu128`换成你自己的 CUDA 版本，比如输出`CUDA Version: 12.1`就把`cu128`替换为`cu126`  
> 具体来说，PyTorch 的CUDA是向下兼容的，所以选择时只需要选择比自己的 CUDA 版本小一点的版本就行了。  
> 比如 PyTorch 提供了三个版本: `12.6, 12.8, 12.9`，然后你的 CUDA 版本是`12.7`，那么就选择`12.8`（因为官方提供的`12.6` < 你的`12.7` < 官方提供的`12.8`）


## 使用示例
```bash
# 下载数据集
mkdir train_data valid_data
curl -Lo takara-midi.zip https://www.kaggle.com/api/v1/datasets/download/yigk4out/takara-midi
curl -Lo lmd_full.tar.gz http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz

# 解压数据集，这里不作演示，我们假设数据集解压到了 /path/to/dataset/takara-midi 和 /path/to/dataset/lmd_full 里
...

# 准备训练集和验证集
python3 prepare_fast_dataset.py /path/to/dataset /path/to/fast_dataset train:9 val:1

# 初始化检查点
python3 init_checkpoint.py /path/to/ckpt

# 训练模型（要训练多少个 Epoch、检查点路径）
python3 train.py 15 /path/to/ckpt -t /path/to/fast_dataset/train.npz -v /path/to/fast_dataset/val.npz

# 生成 1000 帧音乐并保存
python3 generate.py /path/to/ckpt 1000 /path/to/output.mid
```

**注意事项**
- 请在命令行输入`python3 file.py --help`获得帮助

## 文档
文档是不可能写的，这辈子都不可能写的。经验表明，写了文档只会变成“代码一天一天改，文档一年不会动”的局面，反而误导人。

所以我真心推荐：有什么事直接看代码（代码的注释和函数的文档还是会更新的），或者复制代码问ai去吧（记得带上下文）。

## 贡献与开发
欢迎提出问题、改进或贡献代码。如果有任何问题或建议，您可以在 GitHub 上提 Issues，或者直接通过电子邮件联系开发者。

## 联系信息
如有任何问题或建议，请联系项目维护者 thiliapr。
- Email: thiliapr@tutanota.com

# 无关软件本身的广告
## Join the Blue Ribbon Online Free Speech Campaign!
![Blue Ribbon Campaign Logo](https://www.eff.org/files/brstrip.gif)

支持[Blue Ribbon Online 言论自由运动](https://www.eff.org/pages/blue-ribbon-campaign)！  
你可以通过向其[捐款](https://supporters.eff.org/donate)以表示支持。

## 支持自由软件运动
为什么要自由软件: [GNU 宣言](https://www.gnu.org/gnu/manifesto.html)

你可以通过以下方式支持自由软件运动:
- 向非自由程序或在线敌服务说不，哪怕只有一次，也会帮助自由软件。不和其他人使用它们会帮助更大。进一步，如果你告诉人们这是在捍卫自己的自由，那么帮助就更显著了。
- [帮助 GNU 工程和自由软件运动](https://www.gnu.org/help/help.html)
- [向 FSF 捐款](https://www.fsf.org/about/ways-to-donate/)