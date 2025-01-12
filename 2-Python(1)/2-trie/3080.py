from lib import Trie
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