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


import sys


"""
TODO:
- 일단 Trie부터 구현하기
- main 구현하기

힌트: 한 글자짜리 자료에도 그냥 str을 쓰기에는 메모리가 아깝다...
"""


def main() -> None:
    
    input = sys.stdin.readline
    MOD = 10 ** 9 + 7

    MAX_factorial = 30
    factorial = [1] * (MAX_factorial + 1)
    for i in range(1, MAX_factorial + 1):
        factorial[i] = (factorial[i - 1] * i) % MOD
    
    def get_factorial(x: int) -> int:
        return factorial[x]
    
    def get_len(a: str, b: str) -> int:
        i = 0
        while i < len(a) and i < len(b) and a[i] == b[i]:
            i += 1
        return i
    
    N = int(input().strip())
    names = [input().strip() for _ in range(N)]
    names.sort()
    trie = Trie()

    prev = 0

    for i in range(N):
        if i == N - 1:
            cur = len(names[i])
        else:
            cur = get_len(names[i], names[i+1])
        
        limit_len = max(prev, cur) + 1
        trie.push(names[i][:limit_len])

        prev = cur


    answer = trie.solve(get_factorial, MOD)
    print(answer)


if __name__ == "__main__":
    main()