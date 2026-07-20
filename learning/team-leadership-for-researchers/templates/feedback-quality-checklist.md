# 向下反馈质量自检表

> 配套 L2、`src/feedback_quality.py`。四个维度对应一次给junior的反馈必须具备的要素, 每维给出具体自问。写完用 `feedback_quality.py` 的 `audit()` 自检有没有缺分数/缺依据。

## 反馈对象: __________ (mentee/intern姓名或代号)

## 1. 具体性 (specificity)

有没有点名了具体的代码行/实验设定/会议发言, 而不是"做得不错""再仔细点"这类没有主语的评价?

_填写(打分1-5 + 依据):_
```


```

## 2. 可执行性 (actionability)

对方听完这句反馈, 知不知道接下来具体要做什么动作? 如果答案是"知道大概方向但不知道具体做什么", 这一项就不合格。

_填写(打分1-5 + 依据):_
```


```

## 3. 平衡性 (balance)

这次反馈里认可和改进建议的比例是否合理? 注意: 不是"表扬三明治"式的机械公式, 是认可的部分和改进的部分**各自都要具体**(参照第1项标准检验), 而不是凑数字。

_填写(打分1-5 + 依据):_
```


```

## 4. 时效性 (timing)

问题发生之后隔了多久才说? 当天/当场给出的反馈, 对方还记得完整上下文; 拖得越久, 对方复盘成本越高。

_填写(打分1-5 + 依据):_
```


```

---

## 自检

```python
from feedback_quality import blank_feedback, audit, render

fb = blank_feedback("填入mentee姓名")
fb["scores"]["specificity"] = {"score": 0, "note": "..."}
fb["scores"]["actionability"] = {"score": 0, "note": "..."}
fb["scores"]["balance"] = {"score": 0, "note": "..."}
fb["scores"]["timing"] = {"score": 0, "note": "..."}

print(render(fb))
chk = audit(fb)
print("✅ 反馈质量合格" if chk["ready"] else chk["issues"])
```

(填完四维之后, 用上面的代码跑一遍自检——`ready` 为真才说明每个维度都有分数和具体依据; 如果某一维只填了分数没写依据, 说明这个分数是拍脑袋打的, 回去补上具体的观察依据。)
