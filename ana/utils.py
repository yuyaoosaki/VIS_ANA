import re


def __remove_capture(old_pattern):
    return re.sub(r'(?<!\\)\((?!\?)', '(?:', old_pattern)


def replace(text, pattern, replacement):
    """
    Parameters
    ----------
    pattern: str
        検索正規表現

    replacement: str or callable
        str の場合はその値で置換する。
        callable の場合はその実行結果で置換する。

    Returns
    -------
    str
    """
    if isinstance(replacement, str):
        return re.sub(pattern, replacement, text)
    pattern = __remove_capture(pattern)
    arr = re.split('(' + pattern + ')', text)
    text = ''
    for i, v in enumerate(arr):
        if i % 2:
            text += replacement(v)
        else:
            text += v
    return text
