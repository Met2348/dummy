"""Networking Mastery 共享数据结构：DeepPoint（讲解+追问链）+ ScenarioPoint（讲解+场景判断）
+ 打分/抽题工具。

与 software-engineering-mastery/design-patterns-mastery/database-mastery 同名文件是同一套已验证
设计的独立副本（各 mastery track 自成一体，不跨 track 相互 import）。DeepPoint/ScenarioPoint 都
带 `explain` 字段：因为计算机网络对用户是完全没系统学过的领域，每个知识点先给一段讲仔细的系统性
讲解（是什么/为什么/怎么用/常见误区），再接三层追问链或场景判断，兼顾"系统学会"和"面试接得住"
两个目标。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeepPoint:
    id: str
    cat: str
    trigger: str
    explain: str
    chain: tuple[tuple[str, str, tuple[str, ...]], ...]
    pitfall: str
    real_world_link: str = ""


def categories(bank: list[DeepPoint]) -> list[str]:
    seen: list[str] = []
    for dp in bank:
        if dp.cat not in seen:
            seen.append(dp.cat)
    return seen


def grade_chain(dp: DeepPoint, answers: list[str]) -> list[float]:
    scores: list[float] = []
    for i, (_question, _ref, keys) in enumerate(dp.chain):
        if i >= len(answers) or not keys:
            scores.append(0.0)
            continue
        ans = answers[i].lower()
        hit = sum(1 for k in keys if k.lower() in ans)
        scores.append(hit / len(keys))
    return scores


def drill(bank: list[DeepPoint], cat: str | None = None, n: int | None = None) -> list[DeepPoint]:
    pool = [dp for dp in bank if cat is None or dp.cat == cat]
    return pool if n is None else pool[:n]


@dataclass(frozen=True)
class ScenarioPoint:
    id: str
    cat: str
    trigger: str
    explain: str
    rubric: tuple[str, ...]
    trap: str
    real_world_link: str = ""


def grade_scenario(sp: ScenarioPoint, answer: str) -> float:
    if not sp.rubric:
        return 0.0
    ans = answer.lower()
    hit = sum(1 for k in sp.rubric if k.lower() in ans)
    return hit / len(sp.rubric)


def _self_test() -> None:
    sample = [
        DeepPoint(
            id="dp-test-01",
            cat="test",
            trigger="你说TCP三次握手，为什么不是两次？",
            explain="TCP三次握手（Three-Way Handshake）的目的是让通信双方都确认彼此的发送和接收能力"
                    "正常，并同步初始序列号（ISN）。如果只用两次握手，客户端发送连接请求、服务端确认，"
                    "服务端就无法确认客户端是否收到了自己的确认包——也就是服务端不知道自己的发送能力和"
                    "客户端的接收能力是否正常。经典的场景是：客户端发的第一个连接请求因为网络延迟滞留，"
                    "客户端超时后重发并建立连接、传输完毕断开，此时那个滞留的旧请求才到达服务端，如果"
                    "只有两次握手服务端会误认为这是一个新连接请求并建立连接等待数据，白白浪费资源。",
            chain=(
                ("两次握手会有什么具体问题？", "服务端无法确认客户端是否收到了自己的确认包也就是不知道自己的发送能力是否正常", ("服务端无法确认客户端是否收到了自己的确认包", "不知道自己的发送能力是否正常")),
                ("能举一个两次握手会出错的具体场景吗？", "旧的连接请求因网络延迟滞留后到达导致服务端误认为新连接请求并建立连接等待数据浪费资源", ("旧的连接请求因网络延迟滞留后到达", "服务端误认为新连接请求", "浪费资源")),
                ("三次握手能完全杜绝这类问题吗？", "不能完全杜绝，只能大幅降低概率，配合序列号和超时重传机制来兜底", ("不能完全杜绝", "序列号和超时重传机制来兜底")),
            ),
            pitfall="很多人只会背'三次握手'这个名词，答不出两次握手具体会出什么问题。",
            real_world_link="",
        )
    ]
    assert len(sample[0].chain) >= 3
    assert len(sample[0].explain) >= 100, len(sample[0].explain)
    assert categories(sample) == ["test"]
    scores = grade_chain(sample[0], ["服务端无法确认客户端是否收到了自己的确认包，不知道自己的发送能力是否正常", "旧的连接请求因网络延迟滞留后到达，服务端误认为新连接请求，浪费资源"])
    assert scores[0] == 1.0, scores
    assert scores[1] == 1.0, scores
    assert drill(sample, cat="test", n=1) == sample

    sp_sample = [
        ScenarioPoint(
            id="sc-test-01",
            cat="test",
            trigger="如果线上服务突然延迟飙升，你会怎么分层排查网络问题？",
            explain="网络延迟排查通常按分层思路推进：先看DNS解析是否异常（用dig/nslookup），再看TCP"
                    "连接建立是否正常（三次握手耗时），再看应用层协议本身（HTTP响应时间），最后结合"
                    "抓包工具（tcpdump/Wireshark）定位具体在哪一层耗时异常，避免一上来就盲目怀疑某一层。",
            rubric=("先排查DNS解析是否异常", "再看TCP连接建立耗时", "结合抓包工具定位具体层级"),
            trap="只会说'看日志'，说不清具体先看哪一层、用什么工具",
            real_world_link="",
        )
    ]
    assert len(sp_sample[0].explain) >= 100, len(sp_sample[0].explain)
    assert categories(sp_sample) == ["test"]
    sp_score = grade_scenario(sp_sample[0], "先排查DNS解析是否异常，再看TCP连接建立耗时")
    assert sp_score == 2 / 3, sp_score
    assert grade_scenario(sp_sample[0], "") == 0.0
    print("[PASS] deep_common: DeepPoint/grade_chain/drill/ScenarioPoint/grade_scenario(含explain字段) 自检通过")


if __name__ == "__main__":
    _self_test()
