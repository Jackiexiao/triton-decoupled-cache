# triton-decoupled-cache
reproduce bug for triton inference server with python decouple cache

issue: https://github.com/triton-inference-server/server/issues/4362

reproduce step
1. `bash run.sh` to start triton server
2. `python client.py`

get result
```
response output count mismatch
```
