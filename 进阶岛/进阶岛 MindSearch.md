| 任务类型 | 任务描述                               |
| ---- | ---------------------------------- |
| 闯关任务 | 部署MindSearch到 hugging face Spaces上 |

# 闯关任务

## 打开codespace主页，选择Blank模板进行创建

![[Pasted image 20241218222827.png]]

## 创建conda环境隔离并安装依赖

如果只针对于这个实验的话，其实在codespace里面不用单独创建conda环境。但是隔离是一个好习惯，因此我们还是创建一个相应的虚拟环境来隔离

```bash
conda create -n mindsearch python=3.10 -y
conda init
```

如果是新建的codespace，在第一次创建conda环境时，需要conda init，**再另启一个终端并activate** 
![[Pasted image 20241218223041.png]]

重启终端之后我们输入：
```bash
conda activate mindsearch
cd /workspaces/codespaces-blank
git clone https://github.com/InternLM/MindSearch.git && cd MindSearch && git checkout ae5b0c5
pip install -r requirements.txt
```

![[Pasted image 20241218223301.png]]

## 获取硅基流动API KEY

因为要使用硅基流动的 API Key，所以接下来便是注册并获取 API Key 了。
首先，我们打开它的[登录界面](https://account.siliconflow.cn/login)来注册硅基流动的账号（如果注册过，则直接登录即可）。
在完成注册后，打开[api key页面](https://cloud.siliconflow.cn/account/ak)来准备 API Key。首先创建新 API 密钥，然后点击密钥进行复制，以备后续使用。

![[Pasted image 20241219194843.png]]


## 启动MindSearch

### 启动后端

由于硅基流动 API 的相关配置已经集成在了 MindSearch 中，所以我们在一个终端A中可以直接执行下面的代码来启动 MindSearch 的后端。

```bash
export SILICON_API_KEY=<上面复制的API KEY>
conda activate mindsearch

# 进入你clone的项目目录
cd /workspaces/codespaces-blank/MindSearch
python -m mindsearch.app --lang cn --model_format internlm_silicon --search_engine DuckDuckGoSearch --asy
```

- --lang: 模型的语言，en 为英语，cn 为中文。
- --model_format: 模型的格式。
  - internlm_silicon 为 InternLM2.5-7b-chat 在硅基流动上的API模型
- --search_engine: 搜索引擎。
  - DuckDuckGoSearch 为 DuckDuckGo 搜索引擎。
  - BingSearch 为 Bing 搜索引擎。
  - BraveSearch 为 Brave 搜索引擎。
  - GoogleSearch 为 Google Serper 搜索引擎。
  - TencentSearch 为 Tencent 搜索引擎。
![[Pasted image 20241220221745.png]]

随后会弹出一个后端的网页
![[Pasted image 20241218225043.png]]
### 启动前端

在后端启动完成后，我们**新开一个终端**运行如下命令来启动 MindSearch 的前端:

```bash
conda activate mindsearch
# 进入你clone的项目目录
cd /workspaces/codespaces-blank/MindSearch
python frontend/mindsearch_gradio.py
```

![[Pasted image 20241218225458.png]]

前后端都启动后，我们应该可以看到github自动为这两个进程做端口转发:
![[Pasted image 20241220101303.png]]


如果启动前端后没有自动打开前端页面的话，我们可以手动用7882的端口转发地址打开gradio的前端页面~
然后就可以体验MindSearch gradio版本啦~
![[Pasted image 20241218225427.png]]
比如向其询问："Find legal precedents in contract law." 等待一段时间后，会在页面上输出它的结果。

![[Pasted image 20241218225729.png]]


在这一步中，可能终端会打印报错信息，但是只要前端页面上没有出现报错就行。

![[Pasted image 20241220103106.png]]
如果前端页面上出现错误并终止，那么可能是 MindSearch 中 searcher 模块的问题。
![[Pasted image 20241220214023.png]]在上面的例子中我们使用的是 DuckDuckGoSearch，因此你也可以尝试其他的搜索引擎 API。如我们可以替换为 BingSearch 或者 TencentSearch 进行尝试。
```bash
# BingSearch
python -m mindsearch.app --lang cn --model_format internlm_silicon --search_engine BingSearch --asy
# TencentSearch
# python -m mindsearch.app --lang cn --model_format internlm_silicon --search_engine TencentSearch --asy
```

成功~~
![[Pasted image 20241220214354.png]]

## 部署到自己的 HuggingFace Spaces上

在之前课程的学习中，已经尝试过将模型或者应用上传/部署到hugging face上过了。在这里我们使用一种更简单的方法，它就像克隆一样，无需编写代码即可部署自己的Spaces应用~

首先我们找到InternLM官方部署的[MindSearch Spaces应用](https://huggingface.co/spaces/internlm/MindSearch)

### 选择配置

在该页面的右上角，选择Duplicate this Space

![[Pasted image 20241220103506.png]]


选择如下配置后，即可Duplicate Space
- Space Hardware选择第一条，即**Free的2vCPU**即可
- 填写好SILICON_API_KEY，即上面提到的硅基流动的API KEY


![[Pasted image 20241220103623.png]]


### 测试结果

等待Spaces应用启动，当启动好后上方会显示绿色的**running**标志，这时我们可以输入 input 进行测试了，我们可以在 Sapces 应用页面的输入框中输入以下内容：
```
# input
What are the top 10 e-commerce websites?
```
![[Pasted image 20241220111037.png]]
此时如果出现和之前在 Codespace 一样的在前端出现“错误”字样的问题时，打开 log 页面，`Ctrl+F` 搜索 duckduckgo，可以发现正是搜索引擎 DuckDuckGo 免费 api 请求次数达到调用限制了
```
Failed to get search results from DuckDuckGo after retires.
...
raise RatelimitException(f"{resp.url} {resp.status_code} Ratelimit")
duckduckgo_search.exceptions.RatelimitException: https://duckduckgo.com 202 Ratelimit
```
如下图所示：
![[Pasted image 20241220110142.png]]
![[Pasted image 20241220110032.png]]

此时在页面右上角选择Restart Space，**待到重启完成后（显示绿色running标志后）再刷新一下网页页面**，**有一定概率成功**，再次测试结果如下~

![[Pasted image 20241220105203.png]]


至此，我们就完成了MindSearch在Hugging Face上面的部署。
[MindSearch - a Hugging Face Space by freshlittlelemon](https://huggingface.co/spaces/freshlittlelemon/MindSearch)

