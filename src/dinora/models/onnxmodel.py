import pathlib

import chess
import numpy as np
import numpy.typing as npt
import onnxruntime

from dinora.encoders.board_representation import board_to_tensor
from dinora.models import search_weights
from dinora.models.nnwrapper import NNWrapper

npf32 = npt.NDArray[np.float32]
DEFAULT_WEIGHTS_FILENAME = "alphanet_classic.ckpt.onnx"


class OnnxModel(NNWrapper):
    def __init__(self, weights: pathlib.Path | None = None, device: str | None = None):
        if device is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif device == "cuda":
            providers = ["CUDAExecutionProvider"]
        else:
            raise ValueError(f"Device '{device}' is not supported")

        if weights is None:
            weights = search_weights(DEFAULT_WEIGHTS_FILENAME)

        self.ort_session = onnxruntime.InferenceSession(weights, providers=providers)

    def name(self) -> str:
        return "Onnx"

    def raw_outputs(self, board: chess.Board) -> tuple[npf32, npf32]:
        """
        NN outputs as numpy arrays
        priors numpy of shape (1880,)
        state value of shape (1,)
        """
        flip = not board.turn
        board_tensor = board_to_tensor(board, flip)
        raw_policy, raw_value = self.ort_session.run(
            None, {"input": board_tensor.reshape(1, 18, 8, 8)}
        )
        return raw_policy[0], raw_value[0]

    def raw_batch_outputs(self, boards: list[chess.Board]) -> tuple[npf32, npf32]:
        batch = []
        for board in boards:
            batch.append(board_to_tensor(board, not board.turn))

        batch_np = np.array(batch)
        raw_policy, raw_value = self.ort_session.run(None, {"input": batch_np})

        return raw_policy, raw_value

    def reset(self) -> None:
        pass
