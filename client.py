import random
import sys
from functools import partial
from black import main
import numpy as np
import queue
import json
import time

from tritonclient.utils import *
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient

def trigger_cache(url="localhost:8000", enable=True):
    with httpclient.InferenceServerClient(url=url) as triton_client:
        model_indexs = triton_client.get_model_repository_index()
        for model_index in model_indexs:
            model_name = model_index['name']
            model_config = triton_client.get_model_config(model_name)
            model_config['response_cache'] = {}
            model_config['response_cache']['enable'] = enable
            triton_client.load_model(model_name, config=json.dumps(model_config))


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


def rand_int_string(length: int = 15) -> str:
    return ''.join([str(random.randint(0, 9)) for i in range(length)])

def send_request():
    model_name = "repeat"

    repeat_num = 5
    inputs = [grpcclient.InferInput('IN', [1], "INT32")]
    outputs = [grpcclient.InferRequestedOutput('OUT')]
    user_data = UserData()
    with grpcclient.InferenceServerClient(
        url="localhost:8001", verbose=False
    ) as triton_client:
        triton_client.start_stream(callback=partial(callback, user_data))

        in_data = np.array([repeat_num], dtype=np.int32)
        inputs[0].set_data_from_numpy(in_data)

        request_id = rand_int_string()
        triton_client.async_stream_infer(
            model_name=model_name, inputs=inputs, request_id=request_id, outputs=outputs
        )

        recv_count = 0
        expected_count = repeat_num
        while recv_count < expected_count:
            data_item = user_data._completed_requests.get(timeout=2)
            if type(data_item) == InferenceServerException:
                print(data_item)
                raise data_item
            else:
                print(data_item.as_numpy('OUT'))
            recv_count += 1
        print('=======end=======')


# trigger_cache(enable=False)
# time.sleep(2)
# send_request()

# trigger_cache(enable=True)
# time.sleep(3)
send_request()

