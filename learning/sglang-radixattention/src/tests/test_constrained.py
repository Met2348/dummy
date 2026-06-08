"""Tests for grammar FSM + constrained sampler + jump-forward."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from grammar_fsm import compile_literal, compile_digits_n, compile_concat
from constrained_sampler import ConstrainedSampler
from jump_forward import jump_forward


def test_literal_fsm_accepts_exact():
    fsm = compile_literal("hello")
    assert fsm.accepts("hello")
    assert not fsm.accepts("hellp")
    assert not fsm.accepts("hell")


def test_digits_n_accepts_n_digits():
    fsm = compile_digits_n(4)
    assert fsm.accepts("1234")
    assert not fsm.accepts("12345")
    assert not fsm.accepts("12a4")


def test_concat_works():
    fsm = compile_concat(compile_literal("v"), compile_digits_n(2))
    assert fsm.accepts("v12")
    assert not fsm.accepts("v1")
    assert not fsm.accepts("vab")


def test_constrained_sampler_blocks_illegal_tokens():
    fsm = compile_concat(compile_literal("v"), compile_digits_n(2))
    vocab = ["a", "v", "1", "2", "12", "v12"]
    s = ConstrainedSampler(fsm=fsm, vocab=vocab)
    mask = s.legal_token_mask()
    # Only "v" and "v12" should be legal at the start
    assert mask[vocab.index("v")] is True
    assert mask[vocab.index("v12")] is True
    assert mask[vocab.index("a")] is False
    assert mask[vocab.index("1")] is False


def test_constrained_sampler_commit_advances():
    fsm = compile_concat(compile_literal("v"), compile_digits_n(2))
    vocab = ["v", "12"]
    s = ConstrainedSampler(fsm=fsm, vocab=vocab)
    assert s.commit(0)        # 'v'
    assert s.commit(1)        # '12'
    assert s.at_accept()


def test_jump_forward_skips_unique_path():
    """Literal-only FSM: every char forced; jump should consume all of it."""
    fsm = compile_literal('{"name":"')
    forced, new_state = jump_forward(fsm, state=0)
    assert forced == '{"name":"'
    assert new_state in fsm.accept


def test_jump_forward_stops_at_branch():
    """At a branching digit position, jump cannot proceed (10 choices)."""
    fsm = compile_concat(compile_literal("v"), compile_digits_n(1))
    forced, new_state = jump_forward(fsm, state=0)
    assert forced == "v"             # 'v' forced, but next is any digit
    # state advanced past the literal 'v'
    assert new_state == 1
