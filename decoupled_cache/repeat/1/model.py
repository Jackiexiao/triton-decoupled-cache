import json
import time
import numpy as np
from typing import List

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        using_decoupled = pb_utils.using_decoupled_model_transaction_policy(
            model_config
        )
        if not using_decoupled:
            raise pb_utils.TritonModelException(
                """the model `{}` can generate any number of responses per request,
                enable decoupled transaction policy in model configuration :o
                serve this model""".format(
                    args['model_name']
                )
            )

        # Get OUT configuration
        out_config = pb_utils.get_output_config_by_name(model_config, "OUT")

        # Convert Triton types to numpy types
        self.out_dtype = pb_utils.triton_string_to_numpy(out_config['data_type'])

    def execute(self, requests):
        # This model does not support batching, so 'request_count' should always
        # be 1.
        if len(requests) != 1:
            raise pb_utils.TritonModelException(
                "unsupported batch size " + len(requests)
            )

        in_input = pb_utils.get_input_tensor_by_name(requests[0], 'IN').as_numpy()
        response_sender = requests[0].get_response_sender()

        for i in range(in_input[0]):
            time.sleep(0.1)

            out_tensor = pb_utils.Tensor(
                'OUT', np.array([[i], [i], [i]], dtype=self.out_dtype)
            )

            response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            response_sender.send(response)

        response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
        return None

    def finalize(self):
        print('Finalize invoked')
