
| 任务类型   | 任务内容                                 |
| ------ | ------------------------------------ |
| 闯关任务 1 | Leetcode 383 (笔记中提交代码与 leetcode 提交通过截图) |
| 闯关任务 2 | VScode 连接 InternStudio debug 笔记      |
| 可选任务   | pip 安装到指定目录                          |

## 闯关任务 1

> Leetcode 383 (笔记中提交代码与 leetcode 提交通过截图)

![InternLM_learing/images/屏幕截图 2024-11-21 215923.png at main · fresh-little-lemon/InternLM_learing (github.com)](https://github.com/fresh-little-lemon/InternLM_learing/blob/main/images/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-11-21%20215923.png)

### 代码如下：
``` python
from collections import Counter

class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        return Counter(ransomNote) <= Counter(magazine)

s = Solution()

print("示例 1：")
ransomNote, magazine = "a", "b"
print(s.canConstruct(ransomNote=ransomNote, magazine=magazine))

print("示例 2：")
ransomNote, magazine = "aa", "ab"
print(s.canConstruct(ransomNote=ransomNote, magazine=magazine))

print("示例 3：")
ransomNote, magazine = "aa", "aab"
print(s.canConstruct(ransomNote=ransomNote, magazine=magazine))
```

## 闯关任务 2

> VScode 连接 InternStudio debug 笔记

任务描述：下面是一段调用书生浦语 API 实现将非结构化文本转化成结构化 json 的例子，其中有一个小 bug 会导致报错。请大家自行通过 debug 功能定位到报错原因。

**TIPS**:
- 打断点查看下 LLM 返回的文本结果。造成本 bug 的原因与 LLM 的输出有关，学有余力的同学可以尝试修正这个 BUG。
- 作业提交时需要有 debug 过程的图文笔记，必须要有**打断点在 debug 中看到 `res` 变量的值的截图。**
- **避免将 api_key 明文写在程序中！！！** 本段 demo 为了方便大家使用 debug 所以将 api_key 明文写在代码中，这是一种极其不可取的行为!


首先，通过文件读取的方式传入 api_key，以免暴露明文
``` python
with open('apikey.txt', 'r', encoding='utf-8') as file:
    api_key = file.readline().strip()
```

运行代码，发现第 33 行报 JSONDecodeError 错误，猜测 json 解析出问题
![InternLM_learing/images/屏幕截图 2024-11-23 161845.png at main · fresh-little-lemon/InternLM_learing (github.com)](https://github.com/fresh-little-lemon/InternLM_learing/blob/main/images/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-11-23%20161845.png)

因此使用 VScode 调试工具对第 32 行和第 33 行设置断点
![InternLM_learing/images/屏幕截图 2024-11-23 162106.png at main · fresh-little-lemon/InternLM_learing (github.com)](https://github.com/fresh-little-lemon/InternLM_learing/blob/main/images/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-11-23%20162106.png)

发现第 32 行可以正常调用书生大模型，但返回值与正常的 json 格式不一致，猜测由于书生大模型返回的值实际为 `markdown` 格式，如下：
``` markdown
res = "```json\n{\n \"key\": \"value\"\n}\n```"
```

故出现以 ` ```json\n` 开头，以 ` ``` ` 结尾的现象，因此编写如下代码去除 markdown 格式标记
```python
if res.startswith("```json"):
    res = res.strip("```json").strip("```").strip()
```
![InternLM_learing/images/屏幕截图 2024-11-23 162106.png at main · fresh-little-lemon/InternLM_learing (github.com)](https://github.com/fresh-little-lemon/InternLM_learing/blob/main/images/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-11-23%20162548.png)

最终成功输出
![InternLM_learing/images/屏幕截图 2024-11-23 162106.png at main · fresh-little-lemon/InternLM_learing (github.com)](https://github.com/fresh-little-lemon/InternLM_learing/blob/main/images/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-11-23%20162626.png)


## 可选任务 

> pip 安装到指定目录

任务描述：使用 VScode 连接开发机后使用 `pip install -t` 命令安装一个 numpy 到看开发机 `/root/myenvs` 目录下，并成功在一个新建的 python 文件中引用。

成功运行
![InternLM_learing/images/屏幕截图 2024-11-23 162106.png at main · fresh-little-lemon/InternLM_learing (github.com)](https://github.com/fresh-little-lemon/InternLM_learing/blob/main/images/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-11-23%20172747.png)

但此时如果在代码中添加一个不在 `/root/myenvs` 目录下的 python 库，如 `import pandas as pd` 就会报错：
![InternLM_learing/images/屏幕截图 2024-11-23 162106.png at main · fresh-little-lemon/InternLM_learing (github.com)](https://github.com/fresh-little-lemon/InternLM_learing/blob/main/images/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-11-23%20172819.png)
