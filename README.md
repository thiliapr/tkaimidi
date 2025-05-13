# Tk AI MIDI
## 简介
这是一个MIDI生成模型，旨在通过深度学习技术生成音乐作品。

## License
![GNU AGPL Version 3 Official Logo](https://www.gnu.org/graphics/agplv3-with-text-162x68.png)

TkTransl 是自由软件，遵循`Affero GNU 通用公共许可证第 3 版或任何后续版本`。你可以自由地使用、修改和分发该软件，但不提供任何明示或暗示的担保。有关详细信息，请参见 [Affero GNU 通用公共许可证](https://www.gnu.org/licenses/agpl-3.0.html)。

## 安装与依赖
```bash
pip install -r requirements.txt
```

## 使用示例
```bash
python3 tokenizer.py ckpt -t train_data -v valid_data  # 训练分词器
python3 train.py 20 ckpt -t train_data -v valid_data  # 训练模型
python3 generate.py ckpt -l 1024  # 生成
```

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