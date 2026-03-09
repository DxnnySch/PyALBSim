from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt

FloatDType = np.float32 | np.float64

Vector3 = Annotated[npt.NDArray[FloatDType], Literal[3]]
Matrix3 = Annotated[npt.NDArray[FloatDType], Literal[3, 3]]
Array = Annotated[npt.NDArray[FloatDType], Literal["N"]]
IntArray = Annotated[npt.NDArray[np.int32], Literal["N"]]
BoolArray = Annotated[npt.NDArray[np.bool_], Literal["N"]]
MatrixN = Annotated[npt.NDArray[FloatDType], Literal["N", "N"]]
Vector3Array = Annotated[npt.NDArray[FloatDType], Literal["N", 3]]
Matrix3Array = Annotated[npt.NDArray[FloatDType], Literal["N", 3, 3]]
