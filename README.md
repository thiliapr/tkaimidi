# Tk AI MIDI
## 简介
这是一个自由的MIDI生成软件，主要使用[mido](https://github.com/mido/mido/)来解析音乐，[pytorch](https://github.com/pytorch/pytorch/)来机器学习，旨在通过深度学习技术生成音乐作品。

## License
![GNU AGPL Version 3 Logo](https://www.gnu.org/graphics/agplv3-with-text-162x68.png)

tkaimidi 是自由软件，遵循`Affero GNU 通用公共许可证第 3 版或任何后续版本`。你可以自由地使用、修改和分发该软件，但不提供任何明示或暗示的担保。有关详细信息，请参见 [Affero GNU 通用公共许可证](https://www.gnu.org/licenses/agpl-3.0.html)。

## 安装依赖
```bash
pip install -r requirements.txt
```

## 使用示例
```bash
# 这里演示的是大致流程，实际可能需要调整，不过一般照着这个来就行了
# 下载并解压数据集
mkdir train_data valid_data
curl -Lo takara-midi.zip https://www.kaggle.com/api/v1/datasets/download/yigk4out/takara-midi
curl -Lo lmd_full.tar.gz http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz
unzip takara-midi.zip -d valid_data/
cd train_data
tar -xf ../lmd_full.tar.gz

# 提取数据，如果训练多次或重新训练分词器，这样就不用多次加载原始数据，节省时间
python3 extract.py train_data train_optimized_data
python3 extract.py valid_data valid_optimized_data

# 训练分词器（检查点保存到哪里）
python3 tokenizer.py ckpt -t train_optimized_data -v valid_optimized_data

# 训练模型（要训练多少个 Epoch、检查点保存到哪里）
python3 train.py 15 ckpt -t train_optimized_data -v valid_optimized_data

# 生成音乐并保存为 MIDI
python3 generate.py ckpt output.mid
```

**注意事项**
- 值得注意的是，如果你不重新训练分词器，而是从已有检查点复制分词器过来的话，直接运行`train.py`。在这种情况下，检查点最好不要包含`optimizer.pth`、`model.pth`和`metrics.json`（根据经验，如果检查点包含不符合现在参数的模型和检查点的话，可能引发异常）
- 更进一步，如果你不重新训练分词器，而且只运行`python3 train.py`一次，则不必先提取数据，即直接运行`python3 train.py 15 ckpt -t train_data -v valid_data`。（在这种情况下，先提取数据再训练和直接训练所耗费时间几乎是相等的。不过我个人还是建议不要这样做，毕竟万一你又不满意模型效果想重新训练了呢 :)）
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