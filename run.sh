base_dir=`pwd`
triton_version=22.05

docker run  --rm \
-p8000:8000 -p8001:8001 -p8002:8002 \
--ipc=host \
-v ${base_dir}/decoupled_cache:/models \
--name triton_server \
nvcr.io/nvidia/tritonserver:${triton_version}-py3 \
tritonserver \
--model-repository=/models --model-control-mode=explicit --load-model=* --response-cache-byte-size=1048000 

