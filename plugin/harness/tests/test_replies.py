import pytest

from harness.replies import parse


@pytest.mark.parametrize("text", ["continue", "Continue!", "继续", "好的", "OK", "go ahead", "  yes. "])
def test_approve(text):
    assert parse(text).kind == "approve"


@pytest.mark.parametrize("text", ["cancel", "取消", "算了", "no"])
def test_cancel(text):
    assert parse(text).kind == "cancel"


@pytest.mark.parametrize("text,cap", [("cap $8", 8), ("上限 $5", 5), ("$12.5", 12.5), ("budget 20", 20), ("上限改成 $6", 6)])
def test_cap(text, cap):
    r = parse(text)
    assert r.kind == "cap" and r.cap_usd == cap


@pytest.mark.parametrize("text", ["好的，不过先只改第一个文件然后告诉我", "continue but only the first file please", "hello", ""])
def test_other(text):
    assert parse(text).kind == "other"
