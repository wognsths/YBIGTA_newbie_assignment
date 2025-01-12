from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Iterable


"""
TODO:
- Trie.push 구현하기
- (필요할 경우) Trie에 추가 method 구현하기
"""


T = TypeVar("T")


@dataclass
class TrieNode(Generic[T]):
    body: Optional[T] = None
    children: list[int] = field(default_factory=lambda: [])
    is_end: bool = False


class Trie(list[TrieNode[T]]):
    def __init__(self) -> None:
        super().__init__()
        self.append(TrieNode(body=None))

    def push(self, seq: Iterable[T]) -> None:
        """
        seq: T의 열 (list[int]일 수도 있고 str일 수도 있고 등등...)

        action: trie에 seq을 저장하기
        """
        node = 0
        for c in seq:
            idx = ord(c) - ord('A')
            
            if len(self[node].children) == 0:
                self[node].children = [-1] * 26

            if self[node].children[idx] == -1:
                self[node].children[idx] = len(self)
                self.append(TrieNode(body=None))
            node = self[node].children[idx]
        self[node].is_end = True

    def solve(self, get_factorial, MOD, node: int = 0) -> int:
        children_count = 0
        if len(self[node].children) > 0:
            for nxt in self[node].children:
                if nxt != -1:
                    children_count += 1
        group_size = children_count + (1 if self[node].is_end else 0)

        ret = get_factorial(group_size) % MOD

        if len(self[node].children) > 0:
            for nxt in self[node].children:
                if nxt != -1:
                    ret = (ret * self.solve(get_factorial, MOD, nxt)) % MOD

        return ret