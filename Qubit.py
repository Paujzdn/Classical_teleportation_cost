import numpy as np

class Qubit:
    def __init__(self, vector=None):
        if vector is not None:
            self.alpha, self.beta = vector
            self.normalize()
        else:
            self.initialize_qubit()

    def initialize_qubit(self):
        random_matrix = self.generate_random_unitary_matrix()
        self.alpha, self.beta = random_matrix[:, 0]
        self.normalize()

    def generate_random_unitary_matrix(self):
        while True:
            real_matrix = np.random.normal(0, 1, (2, 2))
            imag_matrix = 1j * np.random.normal(0, 1, (2, 2))
            unitary_matrix, _ = np.linalg.qr(real_matrix + imag_matrix)
            if np.allclose(np.eye(2), np.dot(unitary_matrix, unitary_matrix.conj().T)):
                return unitary_matrix

    def normalize(self):
        norm = np.sqrt(np.abs(self.alpha) ** 2 + np.abs(self.beta) ** 2)
        self.alpha /= norm
        self.beta /= norm

    def ket(self):
        return np.array([self.alpha, self.beta])

    def matrix(self):
        return np.outer(self.ket(), np.conjugate(self.ket()))

    def bloch(self):
        dm = self.matrix()
        return [np.real(np.trace(np.matmul(dm, i))) for i in self.pauli_matrices()]

    @staticmethod
    def density_to_bloch(dm):
        sigmas = Qubit.pauli_matrices()
        return [np.real(np.trace(np.matmul(dm, i))) for i in sigmas]

    @staticmethod
    def pauli_matrices():
        sigma_x = np.array([[0, 1], [1, 0]])
        sigma_y = np.array([[0, -1j], [1j, 0]])
        sigma_z = np.array([[1, 0], [0, -1]])
        return [sigma_x, sigma_y, sigma_z]

    def display_ket(self):
        return '{} |0> + {} |1>'.format(self.alpha, self.beta)

    @classmethod
    def from_eigenvectors(cls, eigenvectors):
        q1 = cls(eigenvectors[:, 0])
        q2 = cls(eigenvectors[:, 1])
        return q1, q2
