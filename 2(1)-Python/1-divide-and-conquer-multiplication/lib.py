from __future__ import annotations
import copy


"""
TODO:
- __setitem__ 구현하기
- __pow__ 구현하기 (__matmul__을 활용해봅시다)
- __repr__ 구현하기
"""


class Matrix:
    MOD = 1000

    def __init__(self, matrix: list[list[int]]) -> None:
        self.matrix = matrix

    @staticmethod
    def full(n: int, shape: tuple[int, int]) -> Matrix:
        return Matrix([[n] * shape[1] for _ in range(shape[0])])

    @staticmethod
    def zeros(shape: tuple[int, int]) -> Matrix:
        return Matrix.full(0, shape)

    @staticmethod
    def ones(shape: tuple[int, int]) -> Matrix:
        return Matrix.full(1, shape)

    @staticmethod
    def eye(n: int) -> Matrix:
        matrix = Matrix.zeros((n, n))
        for i in range(n):
            matrix[i, i] = 1
        return matrix

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.matrix), len(self.matrix[0]))

    def clone(self) -> Matrix:
        return Matrix(copy.deepcopy(self.matrix))

    def __getitem__(self, key: tuple[int, int]) -> int:
        return self.matrix[key[0]][key[1]]

    def __setitem__(self, key: tuple[int, int], value: int) -> None:
        """
        matrix[i][j] = 값 형태를 matrix[i, j] = 값 형태로 가져오기 위한 방법
        Modulo 연산 적용
        """
        self.matrix[key[0]][key[1]] = value % self.MOD

    def __matmul__(self, matrix: Matrix) -> Matrix:
        x, m = self.shape
        m1, y = matrix.shape
        assert m == m1

        result = self.zeros((x, y))

        for i in range(x):
            for j in range(y):
                for k in range(m):
                    result[i, j] += self[i, k] * matrix[k, j]

        return result

    def __pow__(self, n: int) -> Matrix:
        """
        분할 정복 방식을 이용한 행렬 곱.
        ex) 2^16에 관한 문제에 대해서,
            2 * 2 -> 4 = 2^2
            2^2 * 2^2 -> 2^4
            2^4 * 2^4 -> 2^8
            2^8 * 2^8 -> 2^16
            으로, 2를 15번 곱하는 방식과 다르게 연산 4번으로 해결 가능
            => O(n)의 시간복잡도를 O(log n)으로 해결
        
        Args:
            - n (int): 행렬의 거듭제곱 수
        Returns:
            - Matrix: 행렬을 거듭제곱한 결과
                - n = 0인 경우 항등행렬(Matrix.eye())을 반환
        """
        if n == 0: return Matrix.eye(self.shape[0])

        if n == 1:
            clone_matrix = self.clone()
            for i in range(clone_matrix.shape[0]):
                for j in range(clone_matrix.shape[1]):
                    clone_matrix[i, j] %= self.MOD
            return clone_matrix
        half = self ** (n // 2)
        squared_matrix = half @ half
        if n % 2 == 0:
            return squared_matrix
        else:
            return squared_matrix @ self

    def __repr__(self) -> str:
        """
        행렬 객체를 출력 시 보기 좋게 출력하는 방법
            - ex) 행렬 [[1, 0], [0, 1]] 출력 시
                    1 0
                    0 1 반환
        """
        back = '\n'.join([' '.join(map(str, row)) for row in self.matrix])
        return back