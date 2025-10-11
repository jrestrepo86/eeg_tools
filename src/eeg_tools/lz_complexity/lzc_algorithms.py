"""
Lempel-Ziv Complexity Algorithms

This module provides implementations of the Lempel-Ziv complexity measures:

- lzc76:
@article{4015609,
  author={Hu, Jing and Gao, Jianbo and Principe, Jose C.},
  journal={IEEE Transactions on Biomedical Engineering},
  title={Analysis of Biomedical Signals by the Lempel-Ziv Complexity: the Effect of
  Finite Data Size},
  doi={10.1109/TBME.2006.883825}}

Juan Felipe Restrepo <juan.restrepo@uner.edu.ar>
"""


def lz_complexity_76_scheme1(s: str) -> int:
    """
    Computes Lempel-Ziv 76 complexity using Scheme 1 parsing.

    Parameters:
    S (array-like): Input sequence.

    Returns:
    int: Complexity count.
    """
    n = len(s)
    if n == 0:
        return 0

    complexity = 1
    i = 0
    j = 1
    v = 1

    while j + v <= n:
        current_sub = s[j : j + v]
        history = s[: i + v]

        if current_sub not in history:
            complexity += 1
            i = j + v - 1
            j = i + 1
            v = 1
        else:
            v += 1

        # Handle end of sequence
        if j + v > n and v > 1:
            complexity += 1

    return complexity


def lz_complexity_76_scheme2(s: str) -> int:
    if not s:
        return 0
    phrases = [s[0]]
    dictionary = set(phrases)
    current_pos = 1
    n = len(s)
    while current_pos < n:
        max_l = n - current_pos
        found = False
        for i in range(1, max_l + 1):
            if i == 1:
                candidate = s[current_pos]
                if candidate not in dictionary:
                    phrases.append(candidate)
                    dictionary.add(candidate)
                    current_pos += i
                    found = True
                    break
            else:
                end_pos = current_pos + i
                if end_pos > n:
                    continue
                candidate = s[current_pos:end_pos]
                prefix = candidate[:-1]
                if prefix in dictionary and candidate not in dictionary:
                    phrases.append(candidate)
                    dictionary.add(candidate)
                    current_pos += i
                    found = True
                    break
        if not found:
            # Handle remaining characters by checking each possible extension
            remaining = s[current_pos:]
            for i in range(1, len(remaining) + 1):
                candidate = remaining[:i]
                if i == 1 and candidate not in dictionary:
                    phrases.append(candidate)
                    dictionary.add(candidate)
                    current_pos += i
                    found = True
                    break
                elif i > 1:
                    prefix = candidate[:-1]
                    if prefix in dictionary and candidate not in dictionary:
                        phrases.append(candidate)
                        dictionary.add(candidate)
                        current_pos += i
                        found = True
                        break
            if not found:
                # Append the remaining as individual characters if not found (should not happen for valid input)
                for c in remaining:
                    if c not in dictionary:
                        phrases.append(c)
                        dictionary.add(c)
                        current_pos += 1
                break
    return len(phrases)
