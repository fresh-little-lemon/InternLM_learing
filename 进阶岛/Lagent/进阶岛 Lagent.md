| 任务类型   | 任务描述                                      |
| ------ | ----------------------------------------- |
| 闯关任务 1 | 使用 Lagent 复现文档中 “制作一个属于自己的Agent”          |
| 闯关任务 2 | 使用 Lagent 复现文档中 “Multi-Agents博客写作系统的搭建”   |
| 可选任务   | 将自己的Agent部署到 Hugging Face 或 ModelScope 平台 |

# 闯关任务 1

## 环境配置

首先来为 Lagent 配置一个可用的环境。
```python
# 创建环境
conda create -n lagent python=3.10 -y
# 激活环境
conda activate lagent
# 安装 torch
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
# 安装其他依赖包
pip install termcolor==2.4.0
pip install streamlit==1.39.0
pip install class_registry==2.1.2
pip install datasets==3.1.0
```
![[Pasted image 20241218220957.png]]
![[Pasted image 20241218221639.png]]
![[Pasted image 20241218222522.png]]
![[Pasted image 20241218223633.png]]

接下来，我们通过源码安装的方式安装 lagent。
```cmd
# 创建目录以存放代码
mkdir -p /root/agent_camp4
cd /root/agent_camp4
git clone https://github.com/InternLM/lagent.git
cd lagent && git checkout e304e5d && pip install -e . && cd ..
pip install griffe==0.48.0
```
![[Pasted image 20241218224109.png]]
![[Pasted image 20241218230803.png]]

## Lagent框架中Agent的使用

创建一个代码example，创建`agent_api_web_demo.py`，在里面实现Web Demo：
```cmd
conda activate lagent
cd /root/agent_camp4/lagent/examples
touch agent_api_web_demo.py
```

首先体验一下，让LLM调用Arxiv文献检索这个工具：

在 `agent_api_web_demo.py` 中写入教程中提供的代码，然后利用 `Streamlit` 启动Web服务，在终端中记得先将获取的API密钥写入环境变量，然后再输入启动命令：
```cmd
export token='your_token_here'
streamlit run agent_api_web_demo.py
```

可以看到页面如下：
![[Pasted image 20241219170317.png]]

此时有概率出现如下报错
```
File "/root/.conda/envs/lagent/lib/python3.10/site-packages/streamlit/runtime/scriptrunner/exec_code.py", line 88, in exec_func_with_error_handling result = func()File "/root/.conda/envs/lagent/lib/python3.10/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 579, in code_to_exec exec(code, module.__dict__)File "/root/agent_camp4/lagent/examples/agent_api_web_demo.py", line 195, in <module> main()File "/root/agent_camp4/lagent/examples/agent_api_web_demo.py", line 184, in main res = agent(user_input, session_id=0)File "/root/agent_camp4/lagent/lagent/agents/agent.py", line 82, in __call__ response_message = self.forward(File "/root/agent_camp4/lagent/lagent/agents/stream.py", line 122, in forward raise RuntimeError(f'No available {tool_type} executor')
```
![[Pasted image 20241218231542.png]]
**解决方案是，将 bash 终端关闭后重启即可恢复正常。**

可以尝试进行几轮简单的对话，并让其搜索文献，会发现大模型现在尽管有比较好的对话能力，但是并不能帮我们准确的找到文献，**例如输入指令“帮我搜索一下最新版本的MindSearch论文”**，会发现 LLM 只是给出了如何查询的步骤并没有实际查询的结果：
![[Pasted image 20241219154317.png]]

现在**将ArxivSearch插件选择上**，再次输入指令“帮我搜索一下最新版本的MindSearch论文”，可以看到，通过调用外部工具，大模型成功理解了我们的任务，得到了我们需要的文献：
![[Pasted image 20241219154416.png]]

## 制作一个属于自己的Agent

接下来以实时天气查询为例子，通过调用和风天气API，基于 Lagent 框架实现一个自己的 Agent工具。

首先，为了使用和风天气的 API 服务，**需要获取一个 API Key**。

（1）访问 [和风天气 API 文档](https://dev.qweather.com/docs/api/)（需要注册账号）。

（2）点击页面右上角的“控制台”。
![[Pasted image 20241219154845.png]]

（3）在控制台中，点击左侧的“项目管理”，然后点击右上角“创建项目”。
![[Pasted image 20241219154912.png]]

（4）输入项目名称（可以使用“Lagent”），选择免费订阅，并在凭据设置中创建新的凭据。
![[Pasted image 20241219154949.png]]
![[Pasted image 20241219155034.png]]

（5）创建凭据页面选择 API KEY，并填写凭据名称
![[Pasted image 20241219155117.png]]

（6）创建凭据成功，将申请的 API Key 在终端中输入进去：**
```cmd
export weather_token='your_token_here'
```
![[Pasted image 20241219170044.png]]

接着，我们需要在`laegnt/actions`文件夹下面创建一个天气查询的工具程序。
```cmd
conda activate lagent
cd /root/agent_camp4/lagent/lagent/actions
touch weather_query.py
```

复制教程中的代码到创建好的 `weather_query.py` 中
![[Pasted image 20241219170450.png]]

在`/root/agent_camp4/lagent/lagent/actions/__init__.py`中加入下面的代码，用以初始化`WeatherQuery`方法：
```diff
+ from .weather_query import WeatherQuery
__all__ = [
    'BaseAction', 'ActionExecutor', 'AsyncActionExecutor', 'InvalidAction',
    'FinishAction', 'NoAction', 'BINGMap', 'AsyncBINGMap', 'ArxivSearch',
    'AsyncArxivSearch', 'GoogleSearch', 'AsyncGoogleSearch', 'GoogleScholar',
    'AsyncGoogleScholar', 'IPythonInterpreter', 'AsyncIPythonInterpreter',
    'IPythonInteractive', 'AsyncIPythonInteractive',
    'IPythonInteractiveManager', 'PythonInterpreter', 'AsyncPythonInterpreter',
    'PPT', 'AsyncPPT', 'WebBrowser', 'AsyncWebBrowser', 'BaseParser',
-   'JsonParser', 'TupleParser', 'tool_api' 
+   'JsonParser', 'TupleParser', 'tool_api', 'WeatherQuery'
]
```
![[Pasted image 20241219155554.png]]

接下来，修改 Web Demo 脚本来集成自定义的 `WeatherQuery` 插件。
打开`agent_api_web_demo.py`, 修改内容如下，目的是将该工具注册进大模型的插件列表中，使得其可以知道。
```diff
- from lagent.actions import ArxivSearch
+ from lagent.actions import ArxivSearch, WeatherQuery
- # 初始化插件列表
-        action_list = [
-            ArxivSearch(),
-       ]
+        action_list = [
+            ArxivSearch(),
+            WeatherQuery(),
+       ]
```
![[Pasted image 20241219160903.png]]

**再次启动Web程序，`streamlit run agent_api_web_demo.py`。**
可以看到左侧的插件栏多了天气查询插件。
![[Pasted image 20241219164916.png]]

我们首先**输入命令“帮我查询一下上海现在的天气”**，可以看到模型无法知道现在的实时天气情况。
![[Pasted image 20241219163350.png]]
**笑不活了，现在的上海冷得刮刮抖，结果 LLM 说最高气温 30 摄氏度，还说要注意防晒，简直胡言乱语。**

现在，我们**将2个插件同时勾选上**，用以说明模型具备识别调用不同工具的能力，什么任务对应什么工具来解决。

这次我们重新**输入命令“帮我查询一下上海现在的天气”。** 现在，大模型通过天气查询的API准确地完成了这个任务：
![[Pasted image 20241219163230.png]]

可以看到勾选上插件的 lagent 不仅仅给出了和气象局完全一致的天气信息，并且甚至还给出了调用 api 的时间信息，还挺好的
![[Pasted image 20241219162515.png]]
此时如果我们再次询问，让其搜索文献，可以看到，模型具备了根据任务情况调用不同工具的能力。
![[Pasted image 20241219163300.png]]

# 闯关任务 2
## Multi-Agents博客写作系统的搭建

在这一节中，我们将使用 **Lagent** 来构建一个多智能体系统 (**Multi-Agent System**)，展示如何协调不同的智能代理完成内容生成和优化的任务。我们的多智能体系统由两个主要代理组成：

（1）**内容生成代理**：负责根据用户的主题提示生成一篇结构化、专业的文章或报告。

（2）**批评优化代理**：负责审阅生成的内容，指出不足，推荐合适的文献，使文章更加完善。

Multi-Agents博客写作系统的流程图如下：

![[Pasted image 20241219174359.png]]


首先，创建一个新的 Python 文件 `multi_agents_api_web_demo.py`，并进入 `lagent` 环境：

```bash
conda activate lagent
cd /root/agent_camp4/lagent/examples
touch multi_agents_api_web_demo.py
```

将下面的代码填入`multi_agents_api_web_demo.py`:

运行 `streamlit run multi_agents_api_web_demo.py`，启动Web服务
![[Pasted image 20241219171913.png]]
输入话题，比如`Semi-Supervised Learning`：

可以看到，Multi-Agents博客写作系统正在按照下面的3步骤，生成、批评和完善内容。

**Step 1**：写作者根据用户输入生成初稿。

**Step 2**：批评者对初稿进行评估，提供改进建议和文献推荐（通过关键词触发 Arxiv 文献搜索）。

**Step 3**：写作者根据批评意见对内容进行改进。

输入一个感兴趣的话题：
![[Pasted image 20241219174007.png]]



第一步生成的结果：

![[Pasted image 20241219174113.png]]

第二步批评和文献检索的结果：

![[Pasted image 20241219174207.png]]

第三步最后完善的内容，可以看到其中包括了检索得到的文献，使得博客内容更加具有可信度。

![[Pasted image 20241219174312.png]]

完整的视频如下：


# 可选任务
> 将自己的 Agent 部署到 Hugging Face 或 ModelScope 平台

首先在 `/root/agent_camp4/lagent/requirements.txt` 文件中添加 python 包
```
torch==2.1.2
torchvision==0.16.2
torchaudio==2.1.2
termcolor==2.4.0
streamlit==1.39.0
class_registry==2.1.2
datasets==3.1.0
griffe==0.48.0
```
![[Pasted image 20241219175535.png]]

然后我尝试通过将 `agent_camp4/lagent` 文件夹下除了 github 配置文件 `.github`，`.gitignore` 外的所有文件拷贝到自己新建的 huggingface Spaces 仓库下，一并提交
```
rsync -av -o -t --exclude='.git*' /root/agent_camp4/lagent/ /root/Lagent/
```

其中在 ` git push ` 的过程中出现了如下报错
```bash
(lagent) root@intern-studio-50193904:~/Lagent# git push
Missing or invalid credentials.
Error: connect ECONNREFUSED /tmp/vscode-git-1ec1b9ad1c.sock
    at PipeConnectWrap.afterConnect [as oncomplete] (node:net:1611:16) {
  errno: -111,
  code: 'ECONNREFUSED',
  syscall: 'connect',
  address: '/tmp/vscode-git-1ec1b9ad1c.sock'
}
Missing or invalid credentials.
Error: connect ECONNREFUSED /tmp/vscode-git-1ec1b9ad1c.sock
    at PipeConnectWrap.afterConnect [as oncomplete] (node:net:1611:16) {
  errno: -111,
  code: 'ECONNREFUSED',
  syscall: 'connect',
  address: '/tmp/vscode-git-1ec1b9ad1c.sock'
}
remote: Invalid username or password.
fatal: Authentication failed for 'https://hf-mirror.com/spaces/freshlittlelemon/Lagent/'
```
![[Pasted image 20241219181613.png]]

解决方案是打开设置搜索 `git.terminal.Authentication`，然后把下图的勾选取消
![[Pasted image 20241219181540.png]]

就可以正常上传了，然而我忽略了一个两个问题：一是原先 `agent_camp4/lagent` 文件夹下的 `README.md` 覆盖了我的 huggingface Spaces 仓库下的 `README.md` 文件，导致 `Configuration error`

![[Pasted image 20241219182156.png]]

二是 `agent_camp4/lagent` 文件夹下并没有 `app.py`，因此需要做一个 `app.py` 的入口文件
![[Pasted image 20241223085618.png]]

我们干脆将 `app.py` 做成一个多页面的 streamlit 首页实现对天气查询小助手和博客写作小助手两个 agent 的导航，代码如下
```python
import streamlit as st
import os
import runpy
st.set_page_config(layout="wide", page_title="My Multi-Page App")
def set_env_variable(key, value):
    os.environ[key] = value
def home_page():
    st.header("欢迎来到首页")
    # 设置输入框为隐私状态
    token = st.text_input("请输入浦语token:", type="password", key="token")
    weather_token = st.text_input("请输入和风天气token:", type="password", key="weather_token")
    if st.button("保存并体验agent"):
        if token and weather_token:
            set_env_variable("token", token)  # 设置环境变量为 'token'
            set_env_variable("weather_token", weather_token)  # 设置环境变量为 'weather_token'
            st.session_state.token_entered = True
            st.rerun()
        else:
            st.error("请输入所有token")
if 'token_entered' not in st.session_state:
    st.session_state.token_entered = False
if not st.session_state.token_entered:
    home_page()
else:
    # 动态加载子页面
    page = st.sidebar.radio("选择页面", ["天气查询助手", "博客写作助手"])
    if page == "天气查询助手":
        runpy.run_path("examples/agent_api_web_demo.py", run_name="__main__")
    elif page == "博客写作助手":
        runpy.run_path("examples/multi_agents_api_web_demo.py", run_name="__main__")
```

此外由于 streamlit 要求一个页面内只能有一个 `st.set_page_config()` 函数，因此需要把 `agent_api_web_demo.py` 和 `multi_agents_api_web_demo.py` 中的相应代码注释掉

`agent_api_web_demo.py` 第 49~53 行和第 136~140 行
![[Pasted image 20241223090619.png]]
![[Pasted image 20241223090707.png]]
 `multi_agents_api_web_demo.py` 第 153 行
![[Pasted image 20241223090819.png]]

然后我们将 `agent_camp4/lagent` 文件夹下需要的文件拷贝到自己新建的 huggingface Spaces 仓库下
```
rsync -av -o -t --exclude='. git*' --exclude='README. md' /root/agent_camp4/lagent/ /root/Lagent/
git add .
git commit -m "Add files"
git push
```

此时出现了环境错误，docker 找不到 `requirements/optional.txt` 文件
![[Pasted image 20241223085959.png]]
我们将其手动添加至 `requirements.txt` 中
![[Pasted image 20241223091226.png]]

## huggingface 部署成功~~
![[Pasted image 20241223092958.png]]
![[Pasted image 20241223092847.png]]
![[Pasted image 20241223092907.png]]
![[Pasted image 20241223092818.png]]

