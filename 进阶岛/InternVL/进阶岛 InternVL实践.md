
| 任务类型   | 任务描述                                                                 |
| ------ | -------------------------------------------------------------------- |
| 闯关任务 1 | 理解多模态大模型的常见设计模式，可以大概讲出多模态大模型的工作原理。                                   |
| 闯关任务 2 | 了解 InternVL2 的设计模式，可以大概描述 InternVL2 的模型架构和训练流程。                      |
| 闯关任务 3 | 了解 LMDeploy 部署多模态大模型的核心代码，并运行提供的 gradio 代码，在 UI 界面体验与 InternVL2 的对话。 |
| 闯关任务 4 | 了解 XTuner，并利用给定数据集微调 InternVL2-2B 后，再次启动 UI 界面，体验模型美食鉴赏能力的变化。        |
| 闯关任务 5 | 将训练好的模型上传到 Hugging Face 或 ModelScope 上，模型名称包含 InternVL 关键词（优秀学员必做）   |

## 多模态大模型简介（闯关任务 1）

> 理解多模态大模型的常见设计模式，可以大概讲出多模态大模型的工作原理。

**多模态大语言模型**（Multimodal Large Language Model,  MLLM）是指能够处理和融合多种不同类型数据（如文本、图像、音频、视频等）的大型人工智能模型。这些模型通常基于深度学习技术，能够理解和生成多种模态的数据，从而在各种复杂的应用场景中表现出强大的能力。

## 1. 常见设计模式

多模态大模型研究的一个关键点是不同模态特征空间的对齐。常见的多模态融合模式有 Q-former 和 MLP

### 1) Q-former

BLIP-2 提出了 Q-former，是多模态大模型领域最早最有影响力的工作之一。类似之前经典的多模态模型的双塔设计结构，Q-former 架构中两个塔之间通过self attention来进行参数的共享，起到一定的模态融合的作用。Feed Forward 层（FFN），不共享参数，类似于MOE中的那个专家模块，处理模态的差异化信息。
![[Pasted image 20241229144158.png]]
Q former学习三个loss。第一个是图文匹配loss，第二个是基于图像的文本生成loss以及图文的对比学习loss。通过这三个图文任务来优化这个多模态模型的对齐效果。

去年爆火的MiniGPT4 便采用 Q-former 进行多模态对齐

![[Pasted image 20241229150025.png]]

### 2) MLP

#### LLaVA

LLaVA 的想法比 Q-Former 简单很多，就是把 CLIP 的 Vision Encoder 用一个线性层（或者 MLP）变换后对齐到文本表示中，对齐的时候甚至只学线性层，但是效果却很好。

LLaVA 的设计非常简单，仅仅使用简单的线性层将图像特征投影到文本空间。参数少。

![[Pasted image 20241229150532.png]]

#### LLaVA-NeXT

LLaVA-NeXT 在 LLaVA1.5 的基础上，将图片分块后分别编码，这样可以支持更高分辨率。同时将整体图像 resize 到规定尺寸编码，保留全局信息。

![[Pasted image 20241229150626.png]]

**为什么用 Q-Former 的变少了**   
① **收敛速度慢**：Q-Former 的参数量较大（例如 BLIP-2 中的 100M 参数），导致其在训练过程中收敛速度较慢。相比之下，MLP 作为 connector 的模型（如 LLaVA-1.5）在相同设置下能够更快地收敛，并且取得更好的性能。  
② **性能收益不明显**：在数据量和计算资源充足的情况下，Q-Former 并没有展现出明显的性能提升。即使通过增加数据量和计算资源，Q-Former 的性能提升也并不显著，无法超越简单的 MLP 方案。

  
**为什么大家都用 LLaVA**  
①**更强的 baselinesetting**：LLaVA-1.5 通过改进训练数据，在较少的数据量和计算资源下取得了优异的性能，成为了一个强有力的 baseline。相比之下，BLIP2 的后续工作 lnstructBLIP 在模型结构上的改进并未带来显著的性能提升，且无法推广至多轮对话。  
②**模型结构的简洁性**：LLaVA 系列采用了最简洁的模型结构，而后续从模型结构上进行改进的工作并未取得明显的效果。这表明，在当前的技术和数据条件下，简洁的模型结构可能更为有效。

## 2. 工作原理

![[Pasted image 20241229154247.png]]

# InternVL2 简介（闯关任务 2）

> 了解 InternVL2 的设计模式，可以大概描述 InternVL2 的模型架构和训练流程。


InternVL2 是一款由上海人工智能实验室 OpenGVLab 发布的多模态大模型，其设计模式和模型架构以及训练流程都体现了多模态融合和深度学习的先进理念。

### 1. 设计模式

InternVL2 采取了LLaVA 式架构设计 (ViT-MLP-LLM):

- InternLM2-20B
- InternViT-6B
- MLP

### 2. 模型架构

![[Pasted image 20241229154434.png]]

#### 1) Dynamic High Resolution

InternVL 独特的预处理模块：动态高分辨率，是为了让 ViT 模型能够尽可能获取到更细节的图像信息，提高视觉特征的表达能力。

![[Pasted image 20241229154508.png]]
- Pre-defined Aspect Ratios: 考虑到计算资源，设置最多 12 个 tile，就有 35 种长宽比的排列组合 (m\*n, m, n≤12; 12+6+4+3+2+2+6)。
- Match and split: 选择最接近的长宽比，resize 过去，切片成 448*448 的 tiles。
- Thumbnail: 某些任务需要全局信息，为了更好的感知全局信息，把原图 resize 到 448*448，一块喂给 LLM。

#### 2) InternViT
下图为 InternVL 的训练流程。与传统的监督学习或CLIP的对比学习方法不同，InternVL 做了两方面的改进。一是 InternVL 增大了视觉编码器的参数量；二是，虽然InternVL也使用了类似CLIP的对比学习方法，但其训练方式有所不同：在CLIP中，对比学习训练完成后，通常将视觉编码器用于多模态大模型，而文本编码器则被丢弃；而在InternVL的训练过程中，视觉编码器与大语言模型（LLM）直接对齐，LLM替代了传统的文本编码器的位置。在后续的生成任务中，InternVL可以直接使用经过对齐的LLM。由于LLM在预训练阶段就已实现了自然对齐，因此后续的对齐适配效果会更好。
![[Pasted image 20241229151114.png]]

InternViT-6B-448px-V1.2 (InternVL 中的对 ViT 模块的修改)
- 在实验中发现，倒数第四层特征最有用，砍掉后三层，共 45 层
- 分辨率从 224 扩展到 448
- 与 LLM 联合训练时，在 captioning 和 OCR 数据集上训练
- 获取高分辨率和 OCR 能力

InternViT-6B-448px-V1.5 (InternVL2 对 InternViT 模块做了如下升级)
- 动态分辨率（类似 LLaVA-NeXT），最多 12 个 tile
- 更高质量的数据

#### 3) Pixel Shuffle

Pixel Shuffle 在超分任务中是一个常见的操作，PyTorch 中有官方实现，即 nn.PixelShuffle (upscale_factor) 该类的作用就是将一个 tensor 中的元素值进行重排列，假设 tensor 维度为\[B, C, H, W\], PixelShuffle 操作不仅可以改变 tensor 的通道数，也会改变特征图的大小。

![[Pasted image 20241229155243.png]]
Why: 
- 对于 448×448 像素的图像，若 patch 大小设置为 14×14，则会得到 32×32=1024 个 patch，相当于视觉模型需要处理 1024 个 token。这种设置会导致信息冗余，消耗大量计算资源，不利于处理较长的多模态上下文。

What: 
- Pixel shuffle 技术源自超分辨率领域，它通过将不同通道的特征重新排列组合到一个通道上，实现特征图的上采样。具体来说，它将形状为$(N, C \times r^2, H, W)$ 的特征图转换为$(N, C, H \times r, W \times r)$，其中$r$ 是上采样因子。

How: 
- 在此案例中，将采样因子$r$ 设为 0.5，可以将原本尺寸为$4096 \times 0.5 \times 0.5$（即 32×32）的图像特征转换为$4096 \times 32 \times 0.5 \times 32 \times 0.5$，实现下采样至 256 个 token。

#### 4) Multitask output

- 利用 VisionLLMv2 的技术，初始化了一些任务特化 embedding（图像生成、分割、检测），添加了一些任务路由 token
- 训练下游任务特化 embedding，生成路由 token 时，把任务 embedding 拼在路由 embedding 后面，送给 LLM 拿到 hidden_state
- 把 hidden_state 送给路由到的解码器中，生成图像/bounding box/masks
![[Pasted image 20241229155610.png]]

### 3. 训练流程

第一阶段：训练 MLP，用高质量预训练数据（各种视觉任务）
第二阶段：ViT+MLP+LLM 联合训练，用高质量视觉-文本指令任务

![[Pasted image 20241229160009.png]]


# 1.环境配置

## 1.1.训练环境配置

新建虚拟环境并进入:

```Bash
conda create --name xtuner-env python=3.10 -y
conda activate xtuner-env
```

"xtuner-env"为训练环境名，可以根据个人喜好设置，在本教程中后续提到训练环境均指"xtuner-env"环境。

安装与deepspeed集成的xtuner和相关包：

```Bash
pip install xtuner==0.1.23 timm==1.0.9
pip install 'xtuner[deepspeed]'
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.39.0 tokenizers==0.15.2 peft==0.13.2 datasets==3.1.0 accelerate==1.2.0 huggingface-hub==0.26.5
```

我们输入下面命令检查一下是否安装成功。
```bash
pip list | grep -E "xtuner|deepspeed|timm|torch|transformers|tokenizers|peft|datasets|accelerate|huggingface-hub"
```
![[Pasted image 20241227201658.png]]
## 1.2.推理环境配置

配置推理所需环境，这里的环境和原先的 LMDeploy 章节中有些区别，我们可以重新建一个环境：
```Bash
conda create -n lmdeploy-vl python=3.10 -y
conda activate lmdeploy-vl
pip install lmdeploy==0.6.1 gradio==4.44.1 timm==1.0.9
```

"lmdeploy-vl"为推理使用环境名。我们用下面的命令查看安装情况：
```bash
pip list | grep -E "lmdeploy|gradio|timm"
```
![[Pasted image 20241229123111.png]]

# 2.LMDeploy部署（闯关任务 3）

> 了解 LMDeploy 部署多模态大模型的核心代码，并运行提供的 gradio 代码，在 UI 界面体验与 InternVL2 的对话。
## 2.1.LMDeploy基本用法介绍

我们主要通过`pipeline.chat` 接口来构造多轮对话管线，核心代码为：

```Python
## 1.导入相关依赖包
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl import load_image

## 2.使用你的模型初始化推理管线
model_path = "your_model_path"
pipe = pipeline(model_path,
                backend_config=TurbomindEngineConfig(session_len=8192))
                
## 3.读取图片（此处使用PIL读取也行）
image = load_image('your_image_path')

## 4.配置推理参数
gen_config = GenerationConfig(top_p=0.8, temperature=0.8)
## 5.利用 pipeline.chat 接口 进行对话，需传入生成参数
sess = pipe.chat(('describe this image', image), gen_config=gen_config)
print(sess.response.text)
## 6.之后的对话轮次需要传入之前的session，以告知模型历史上下文
sess = pipe.chat('What is the woman doing?', session=sess, gen_config=gen_config)
print(sess.response.text)
```

lmdeploy推理的核心代码如上注释所述。

## 2.2.网页应用部署体验

我们可以使用UI界面先体验与InternVL对话：

拉取本教程的github仓库[https://github.com/Control-derek/InternVL2-Tutorial.git](https://github.com/Control-derek/InternVL2-Tutorial.git)：

```Bash
git clone https://github.com/Control-derek/InternVL2-Tutorial.git
cd InternVL2-Tutorial
```

demo.py文件中，MODEL_PATH处传入InternVL2-2B的路径，如果使用的是InternStudio的开发机则无需修改，否则改为模型路径。

![[Pasted image 20241229123303.png]]

启动demo:

```Bash
conda activate lmdeploy
python demo.py
```

上述命令请在vscode下运行，因为vscode自带端口转发，可以把部署在服务器上的网页服务转发到本地。
![[Pasted image 20241229124042.png]]
![[Pasted image 20241229124124.png]]
启动后，CTRL+鼠标左键点进 `http://127.0.0.1:1096/` 这个链接或者复制链接到浏览器
![[Pasted image 20241229124142.png]]


会看到如下界面：

点击**Start Chat**即可开始聊天，下方**食物快捷栏**可以快速输入图片，**输入示例**可以快速输入文字。输入完毕后，按enter键即可发送。InternVL 好像不认识龙井虾仁诶
![[Pasted image 20241229133531.png]]

## 2.3.可能遇到棘手bug的解决

如果输入多张图，或者开多轮对话时报错：
```bash
RuntimeError: Current event loop is different from the one bound to loop task!
```
![[Pasted image 20241229124759.png]]
![[Pasted image 20241229125344.png]]
可以参考github的issue[https://github.com/InternLM/lmdeploy/issues/2101](https://github.com/InternLM/lmdeploy/issues/2101)：

<div align="center">
  <img width="750" alt="" src="https://github.com/user-attachments/assets/da205682-b51e-4e4c-8fab-07d2e42a3399">
</div>

屏蔽报错的 `/root/.conda/envs/lmdeploy-vl/lib/python3.10/site-packages/lmdeploy/vl/engine.py` 的126，127行，添加 `self._create_event_loop_task()` 后，即可解决上面报错。

![[Pasted image 20241229125916.png]]

# 3.XTuner微调实践（闯关任务 4）

> 了解 XTuner，并利用给定数据集微调 InternVL2-2B 后，再次启动 UI 界面，体验模型美食鉴赏能力的变化。
## 3.1.准备基本配置文件

在InternStudio开发机的`/root/xtuner`路径下，即为开机自带的xtuner，先进入工作目录并激活训练环境：

```Bash
cd /root/xtuner
conda activate xtuner-env  # 或者是你自命名的训练环境
```

如果没有该路径，可以从GitHub上克隆一个：

```Bash
cd /root
git clone https://github.com/InternLM/xtuner.git
conda activate xtuner-env
```
![[Pasted image 20241229134420.png]]
原始internvl的微调配置文件在路径`./xtuner/configs/internvl/v2`下，假设上面克隆的仓库在/`root/InternVL2-Tutorial`,复制配置文件到目标目录下：

```Bash
cp /root/InternVL2-Tutorial/xtuner_config/internvl_v2_internlm2_2b_lora_finetune_food.py /root/xtuner/xtuner/configs/internvl/v2/internvl_v2_internlm2_2b_lora_finetune_food.py
```

## 3.2.配置文件参数解读

在第一部分的设置中，有如下参数：

- `path`: 需要微调的模型路径，在InternStudio环境下，无需修改。
- `data_root`: 数据集所在路径。
- `data_path`: 训练数据文件路径。
- `image_folder`: 训练图像根路径。
- `prompt_temple`: 配置模型训练时使用的聊天模板、系统提示等。使用与模型对应的即可，此处无需修改。
- `max_length`: 训练数据每一条最大token数。
- `batch_size`: 训练批次大小，可以根据显存大小调整。
- `accumulative_counts`: 梯度累积的步数，用于模拟较大的batch_size，在显存有限的情况下，提高训练稳定性。
- `dataloader_num_workers`: 指定数据集加载时子进程的个数。
- `max_epochs`:训练轮次。
- `optim_type`:优化器类型。
-  `lr`: 学习率
- `betas`: Adam优化器的beta1, beta2
- `weight_decay`: 权重衰减，防止训练过拟合用
- `max_norm`: 梯度裁剪时的梯度最大值
- `warmup_ratio`: 预热比例，前多少的数据训练时，学习率将会逐步增加。
- `save_steps`: 多少步存一次checkpoint
- `save_total_limit`: 最多保存几个checkpoint，设为-1即无限制

![[Pasted image 20241229134603.png]]

LoRA相关参数：

![[Pasted image 20241229134629.png]]

- `r`: 低秩矩阵的秩，决定了低秩矩阵的维度。
- `lora_alpha` 缩放因子，用于调整低秩矩阵的权重。
- `lora_dropout`  dropout 概率，以防止过拟合。

如果想断点重训，可以在最下面传入参数：

![[Pasted image 20241229134719.png]]

把这里的`load_from`传入你想要载入的checkpoint，并设置`resume=True`即可断点重续。

## 3.3.数据集下载

我们采用的是FoodieQA数据集，这篇文章中了2024EMNLP的主会，其引用信息如下：

```
@article{li2024foodieqa,
  title={FoodieQA: A Multimodal Dataset for Fine-Grained Understanding of Chinese Food Culture},
  author={Li, Wenyan and Zhang, Xinyu and Li, Jiaang and Peng, Qiwei and Tang, Raphael and Zhou, Li and Zhang, Weijia and Hu, Guimin and Yuan, Yifei and S{\o}gaard, Anders and others},
  journal={arXiv preprint arXiv:2406.11030},
  year={2024}
}
```

FoodieQA 是一个专门为研究中国各地美食文化而设计的数据集。它包含了大量关于食物的图片和问题，帮助多模态大模型更好地理解不同地区的饮食习惯和文化特色。这个数据集的推出，让我们能够更深入地探索和理解食物背后的文化意义。

**可以通过`3.2.a.`和`3.2.b.`两种方式获取数据集**，根据获取方式的不同，可能需要修改配置文件中的`data_root`变量为你数据集的路径：

![[Pasted image 20241229134831.png]]

### 3.3.a.通过huggingface下载

有能力的同学，建议去huggingface下载此数据集：[https://huggingface.co/datasets/lyan62/FoodieQA](https://huggingface.co/datasets/lyan62/FoodieQA)。该数据集为了防止网络爬虫污染测评效果，需要向提交申请后下载使用。

由于申请的与huggingface账号绑定，需要在命令行登录huggingface后直接在服务器上下载：

```Bash
huggingface-cli login
```

然后在这里输入huggingface的具有`read`权限的token即可成功登录。

![[Pasted image 20241229135233.png]]

再使用命令行下载数据集：

```Bash
huggingface-cli download --repo-type dataset --resume-download lyan62/FoodieQA --local-dir /root/huggingface/FoodieQA --local-dir-use-symlinks False
```

当然需要等仓库所有者许可之后才能下载
![[Pasted image 20241229135436.png]]
否则会有报错
```
huggingface_hub.errors.GatedRepoError: 403 Client Error. (Request ID: Root=1-6770e3bc-0f5479ee52fdbb1719d7893d;af389d37-31ee-444c-bfce-c209c16bf81e)

Cannot access gated repo for url https://hf-mirror.com/datasets/lyan62/FoodieQA/resolve/df1038377a5cec73cfe9c2af0433b7681a267cbe/.gitattributes.
Your request to access dataset lyan62/FoodieQA is awaiting a review from the repo authors.
```
如果觉得上述过程麻烦，可以用浏览器下载后，再上传服务器即可😊

由于原始数据集格式不符合微调需要格式，需要处理方可使用，在`InternVL2-Tutorial`下，运行：

```Bash
python process_food.py
```

即可把数据处理为XTuner所需格式。**注意查看 `input_path` 和 `output_path` 变量与自己下载路径的区别。**

需要修改第 2 、3 行，否则会找不到文件，不过书生大模型本地 share 目录下已经是处理好的 json 了不需要修改。
![[Pasted image 20241229135901.png]]
### 3.3.b.利用share目录下处理好的数据集

由于该数据集即需要登录huggingface的方法，又需要申请，下完还需要自己处理，因此我把**处理后**的文件放在开发机的`/root/share/datasets/FoodieQA`路径下了。

## 3.4.开始微调🐱🏍

运行命令，开始微调：

```Bash
xtuner train internvl_v2_internlm2_2b_lora_finetune_food --deepspeed deepspeed_zero2
```

![[Pasted image 20241229143828.png]]

如果报错如：keyerror或者Filenotfound之类的，可能是XTuner没识别到新写的配置文件，需要指定配置文件的完整路径：

```Bash
xtuner train /root/xtuner/xtuner/configs/internvl/v2/internvl_v2_internlm2_2b_lora_finetune_food.py --deepspeed deepspeed_zero2
```

把 `/root/xtuner/xtuner/configs/internvl/v2/internvl_v2_internlm2_2b_lora_finetune_food.py` 换成自己配置文件的路径即可。

看到有日志输出，即为启动成功：
![[Pasted image 20241229154125.png]]
微调后，把模型checkpoint的格式转化为便于测试的格式：

```Bash
python xtuner/configs/internvl/v1_5/convert_to_official.py xtuner/configs/internvl/v2/internvl_v2_internlm2_2b_lora_finetune_food.py ./work_dirs/internvl_v2_internlm2_2b_lora_finetune_food/iter_640.pth ./work_dirs/internvl_v2_internlm2_2b_lora_finetune_food/lr35_ep10/
```

如果修改了超参数，`iter_xxx.pth` 需要修改为对应的想要转的checkpoint。 `./work_dirs/internvl_v2_internlm2_2b_lora_finetune_food/lr35_ep10/` 为转换后的模型checkpoint保存的路径。

同样的问题，如果出现 `FileNotFoundError` 报错：
```
FileNotFoundError: Cannot find ./work_dirs/internvl_v2_internlm2_2b_lora_finetune_food/iter_640.pth
```
说明没识别到新生成的权重，写绝对地址就可以了
```
python xtuner/configs/internvl/v1_5/convert_to_official.py xtuner/configs/internvl/v2/internvl_v2_internlm2_2b_lora_finetune_food.py /root/work_dirs/internvl_v2_internlm2_2b_lora_finetune_food/iter_640.pth ./work_dirs/internvl_v2_internlm2_2b_lora_finetune_food/lr35_ep10/
```
然后就可以正常跑了
![[Pasted image 20241229180436.png]]
![[Pasted image 20241229180647.png]]
# 4.与AI美食家玩耍🎉

修改MODEL_PATH为刚刚转换后保存的模型路径：
```diff
- MODEL_PATH = "/root/share/new_models/OpenGVLab/InternVL2-2B"
+ MODEL_PATH = "/root/xtuner/work_dirs/internvl_v2_internlm2_2b_lora_finetune_food/lr35_ep10"
```

![[Pasted image 20241229180845.png]]

就像在第2节中做的那样，启动网页应用：

```Bash
cd /root/InternVL2-Tutorial
conda activate lmdeploy
python demo.py
```

部分case展示：
还记得么，微调之前 InternVL 居然告诉我这是虾仁炒青菜，差点没笑死我了
![[Pasted image 20241229133532.png]]

微调之后，终于认出来是龙井虾仁了，好耶~~ 但是怎么回答得怎么冷漠。。。真是惜字如金，啧啧
![[Pasted image 20241229191252.png]]


# 模型部署（闯关任务 5）

> 将训练好的模型上传到 Hugging Face 或 ModelScope 上，模型名称包含 InternVL 关键词（优秀学员必做）

由于模型文件比较大，所以先下载 lfs 工具，然后克隆新建的模型仓库，把 merged 文件夹下的所有文件移动到仓库文件夹下（由于文件较大，cp 拷贝比较慢，所以使用 mv 移动文件），然后 add、commit、push 三步走，由于文件较大 git add 较慢，需要耐心等待
```
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install git-lfs
git clone https://hf-mirror.com/freshlittlelemon/InternVL-test
rsync -avz /root/xtuner/work_dirs/internvl_v2_internlm2_2b_lora_finetune_food/lr35_ep10/ InternVL-test/
cd InternVL-test/
git add .
git commit -m "add:InternVL-test"
git push
```
![[Pasted image 20241229193224.png]]
![[Pasted image 20241229200406.png]]
模型权重文件太大了
![[Pasted image 20241229201620.png]]
用 lfs track 一下
```
git lfs track "model.safetensors"
git add .gitattributes
```
![[Pasted image 20241229202327.png]]


ok，模型上传成功
![[Pasted image 20241229202235.png]]
居然还能点开看里面的详细内容，不错不错
![[Pasted image 20241229202456.png]]