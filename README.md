# Tk AI MIDI
## 简介
这是一个基于`TransformerEncoder`的MIDI生成模型。该模型利用`Transformer`架构的编码器部分。

## License
![GNU AGPL Version 3 Official Logo](https://www.gnu.org/graphics/agplv3-with-text-162x68.png)

本项目采用[GNU AGPLv3 or later](https://www.gnu.org/licenses/agpl-3.0.html)许可证。您可以自由使用、修改和分发本项目的代码，但必须在相同许可证或其任何后续版本下进行。

## 联系方式
- Email: thiliapr@tutanota.com

## 各个文件的作用
### `model.py`
模型的结构代码。还有保存、加载模型和恢复训练所需的数据的工具函数。
```mermaid
graph TD
    subgraph MidiNet:
        direction TB
        Embedding[嵌入层] --> PositionalEncoding[位置编码] --> TransformerEncoderLayers[Transformer编码器] --> Linear[全连接层]
    end
```

### `utils.py`
一些训练和推理会用到的工具，包括转换MIDI为音符列表、规范化音符数据、清理缓存的函数

### `generate.py`
以下为生成MIDI的流程
```mermaid
graph TD
    OriginalInput[输入原始提示] --> NormData[规范化数据] -->|作为模型输入的预处理提示| ModelForward[模型前向传播] -->|下一个音符的概率分布| Multinomial[根据概率选择一个] --> AppendToPrompt[添加到提示] --> IsStop{是否生成足够音符}
    IsStop -->|是| ConvertToTrack[将模型输出转化为MIDI轨道] --> WriteToFile[写入文件]
    IsStop -->|否| ModelForward
```

### `train.py`
训练用的函数、数据集。

## 无关软件本身的广告
### Join the Blue Ribbon Online Free Speech Campaign!
![Blue Ribbon Campaign Logo](https://www.eff.org/files/brstrip.gif)

支持[Blue Ribbon Online 言论自由运动](https://www.eff.org/pages/blue-ribbon-campaign)！  
你可以通过向其[捐款](https://supporters.eff.org/donate)以表示支持。

## 支持自由软件运动
为什么要自由软件: [GNU 宣言](https://www.gnu.org/gnu/manifesto.html)

你可以通过以下方式支持自由软件运动:
- 向非自由程序或在线敌服务说不，哪怕只有一次，也会帮助自由软件。不和其他人使用它们会帮助更大。进一步，如果你告诉人们这是在捍卫自己的自由，那么帮助就更显著了。
- [帮助 GNU 工程和自由软件运动](https://www.gnu.org/help/help.html)
- [向 FSF 捐款](https://www.fsf.org/about/ways-to-donate/)