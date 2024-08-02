from utils import get_subspace_indices_with_env_qubits
import pytest


@pytest.mark.parametrize(
    "indices, n_sys_qubits, n_env_qubits, correct_indices",
    (
        (
            [
                [0, 9],
                4,
                2,
                [0, 16, 32, 48, 9, 25, 41, 57],
            ],
        ),
    ),
)
def test_get_subspace_indices_with_env_qubits(
    indices, n_sys_qubits, n_env_qubits, correct_indices
):
    new_indices = get_subspace_indices_with_env_qubits(
        indices=indices, n_sys_qubits=n_sys_qubits, n_env_qubits=n_env_qubits
    )

    print(new_indices)
    assert set(new_indices) == set(correct_indices)


test_get_subspace_indices_with_env_qubits(
    [0, 9],
    4,
    2,
    [0, 16, 32, 48, 9, 25, 41, 57],
)
