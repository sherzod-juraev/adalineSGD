from fastapi import APIRouter, status, Body
from .scheme import AdalineSGDIn, AdalineSGDOut, AdalineSGDPredict
from adalineSGD import AdalineSGD

modules_router = APIRouter()
adaline_SGD = AdalineSGD(
    n_iter=1_0000
)


@modules_router.post(
    '/',
    summary='AdalineSGD fit',
    status_code=status.HTTP_200_OK,
    response_model=AdalineSGDOut
)
async def model_fit(
        adalineSGD_scheme: AdalineSGDIn = Body()
) -> AdalineSGDOut:
    adalineSGD_out = AdalineSGDOut(
        fit=adaline_SGD.fit(adalineSGD_scheme.X, adalineSGD_scheme.y),
        w_=adaline_SGD.w_.tolist()
    )
    return adalineSGD_out


@modules_router.post(
    '/predict',
    summary='AdalineSGD predict',
    status_code=status.HTTP_200_OK,
    response_model=int
)
async def model_predict(
        adalineSGD_scheme: AdalineSGDPredict
) -> int:
    return adaline_SGD.predict(adalineSGD_scheme.X)