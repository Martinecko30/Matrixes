class Matrix:
    def __init__(self, array: list[list[int | float]]) -> None:
        if not isinstance(array, list):
            raise TypeError("Matrix must be a list")

        if not array:
            raise ValueError("Matrix must not be empty")

        # Complete checking of if array is
        # list[list[int | float]] is made by ChatGPT4
        # ===
        for sublist in array:
            if not isinstance(sublist, list):
                raise TypeError("Matrix must be a list")

            if not sublist:
                raise ValueError("Matrix must not be empty")

            if not all(isinstance(item, (int, float)) for item in sublist):
                raise TypeError("Matrix must be a list")

        # ===

        self.m = len(array)
        self.n = len(array[0])
        self.matrix = array

    @staticmethod
    def zero_matrix(height: int, width: int) -> 'Matrix':
        return Matrix([[0] * width for _ in range(height)])

        # Modified from StackOverflow
    @staticmethod
    def identity_matrix(side: int) -> 'Matrix':
        return Matrix(
            [
                [0] * i +
                [1] + [0] *
                (side - i - 1) for i in range(side)
            ]
        )

    def __str__(self) -> str:
        return '\n'.join(
            [
                ' '.join(
                    [f'{col}' for col in rows]
                ) for rows in self.matrix
            ]
        )

    def __getitem__(self, tup: tuple[int, int]) -> int | float:
        return self.matrix[tup[0] - 1][tup[1] - 1]

    def __setitem__(self,
                    tup: tuple[int, int],
                    new_value: int | float) -> None:
        self.matrix[tup[0] - 1][tup[1] - 1] = new_value

    def __delitem__(self, tup: tuple[int, int]) -> None:
        del self.matrix[tup[0] - 1][tup[1] - 1]

    def transposition(self) -> 'Matrix':
        return Matrix([list(row) for row in zip(*self.matrix)])

    def get_info(self) -> tuple[tuple[int, int], bool, bool, bool, bool, bool]:
        return ((self.m, self.n),
                self.m == self.n,
                self.is_symmetric(),
                self.is_skew_symmetric(),
                self.is_echelon(),
                self.is_diagonal())

    def __eq__(self, other_matrix: object) -> bool:
        """ Pretizeni operatoru ==; tzn jestli se dve matice rovnaji """
        if not isinstance(other_matrix, Matrix):
            return False

        if self.m != other_matrix.m or self.n != other_matrix.n:
            return False

        if self is other_matrix:
            return True

        if self.matrix == other_matrix.matrix:
            return True

        return False

    def __ne__(self, other_matrix: object) -> bool:
        return not self == other_matrix

    def __add__(self, other_matrix: 'Matrix') -> 'Matrix':
        for row in range(self.m):
            for col in range(self.n):
                self[row, col] = self[row, col] + other_matrix[row, col]

        return self

    def __sub__(self, other_matrix: 'Matrix') -> 'Matrix':
        for row in range(self.m):
            for col in range(self.n):
                self[row, col] = self[row, col] - other_matrix[row, col]

        return self

    def __mul__(self, other_matrix: 'Matrix') -> 'Matrix':
        if self.n != other_matrix.m:
            raise ValueError(
                "Number of columns in first matrix must \
                match with number of rows in second matrix."
            )

        result = Matrix.zero_matrix(self.m, other_matrix.n)
        for i in range(self.m):
            for j in range(other_matrix.n):
                result[i, j] = sum(
                    self[i, k] * other_matrix[k, j] for k in range(self.n)
                )

        return result

    def __rmul__(self, constant: int | float) -> 'Matrix':
        for row in range(self.m):
            for col in range(self.n):
                self[row, col] = self[row, col] * constant

        return self

    def determinant(self) -> int | float:
        if self.m != self.n:
            raise ValueError(
                "Matrix must be have sides the rows and columns the same size!"
            )

        def det_2x(matrix: Matrix) -> int | float:
            return ((matrix[1, 1] * matrix[2, 2]) -
                    (matrix[1, 2] * matrix[2, 1]))

        if self.n == 1:
            return self[1, 1]

        if self.n == 2:
            return det_2x(self)

        determinant = 0
        for col in range(1, self.n + 1):
            minor = Matrix([
                [
                    self[i, j] for j in range(1, self.n + 1) if j != col
                ] for i in range(2, self.m + 1)
            ])

            determinant += (
                    ((-1) ** (1 + col)) *
                    self[1, col] * minor.determinant()
            )

        return determinant

    def inverse(self) -> 'Matrix':
        if self.m != self.n:
            raise ValueError(
                "Matrix must be have sides the rows and columns the same size!"
            )

        determ = self.determinant()
        if determ == 0:
            raise ValueError(
                "Matrix is singular and cannot be inverted."
            )

        n = self.n
        adjugate = Matrix.zero_matrix(n, n)

        for i in range(1, n + 1):
            for j in range(1, n + 1):
                minor = Matrix([
                    [self[x, y] for y in range(1, n + 1) if y != j]
                    for x in range(1, n + 1) if x != i
                ])
                adjugate[j, i] = ((-1) ** (i + j)) * minor.determinant()

        return (1 / determ) * adjugate

    def is_symmetric(self) -> bool:
        return self.transposition() == self

    def is_skew_symmetric(self) -> bool:
        if self.m != self.n:
            return False

        for i in range(self.m):
            for j in range(self.n):
                if i == j:
                    if self[(i, j)] != 0:
                        return False
                    if self[(i, j)] != -self[(j, i)]:
                        return False
        return True

    def is_echelon(self):
        leading_zeros = 0

        for row in self.matrix:
            count = 0
            while count < self.n and row[count] == 0:
                count += 1

            if count < leading_zeros:
                return False

            leading_zeros = count + 1

        return True

    def is_diagonal(self):
        if self.m != self.n:
            return False

        for i in range(self.m):
            for j in range(self.n):
                if i != j and self.matrix[i][j] != 0:
                    return False

        return True


class Matrix3D:
    def __init__(self, array: list[list[list[int]]]) -> None:
        if not isinstance(array, list):
            raise TypeError("Matrix must be a list")

        if not array:
            raise ValueError("Matrix must not be empty")

        for sublist in array:
            if not isinstance(sublist, list):
                raise TypeError("Matrix must be a list")

            for subsublist in sublist:
                if not isinstance(subsublist, list):
                    raise TypeError("Matrix must be a list")

                if not subsublist:
                    raise ValueError("Matrix must not be empty")

                if not all(
                        isinstance(item, (int, float)) for item in subsublist
                ):
                    raise TypeError("Matrix must be a list")

        self.m = len(array)
        self.n = len(array[0])
        self.k = len(array[0][0])
        self.matrix = array

    def __eq__(self, other_matrix: 'Matrix3D') -> bool:
        """ Pretizeni operatoru ==; tzn jestli se dve 3D matice rovnaji """
        if not isinstance(other_matrix, Matrix3D):
            return False

        if (self.m != other_matrix.m or
                self.n != other_matrix.n or
                self.k != other_matrix.k):
            return False

        if self is other_matrix:
            return True

        if self.matrix == other_matrix.matrix:
            return True

        return False

    def __ne__(self, other_matrix: 'Matrix3D') -> bool:
        return not self == other_matrix

    def __getitem__(self, tup: tuple[int, int, int]) -> int | float:
        return self.matrix[tup[2] - 1][tup[0] - 1][tup[1] - 1]

    def determinant_3d(self) -> int:
        if self.m != self.n != self.k:
            raise ValueError(
                "Matrix must be have the rows, columns and depth same size!"
            )

        def det_2x2(matrix: Matrix3D) -> int | float:
            return ((matrix[1, 1, 1] * matrix[2, 2, 2]) -
                    (matrix[1, 2, 2] * matrix[2, 1, 1]) +
                    (matrix[2, 2, 1] * matrix[1, 1, 2]) -
                    (matrix[1, 2, 1] * matrix[2, 1, 2]))

        if self.n == 1:
            return self[1, 1, 1]

        if self.n == 2:
            return det_2x2(self)

        determinant = 0
        for row in range(1, self.m + 1):
            for col in range(1, self.n + 1):
                minor = Matrix3D(
                    [
                        [
                            [
                                self[i, j, k]
                                for j in range(1, self.n + 1) if j != col
                            ] for i in range(1, self.m + 1) if i != row
                        ] for k in range(2, self.k + 1)
                    ]
                )

                determinant += (
                        ((-1) ** (1 + row + col)) *
                        self[row, col, 1] * minor.determinant_3d()
                )

        return determinant
