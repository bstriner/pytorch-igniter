
import numpy
from io import BytesIO
import torch
from scipy.io import wavfile

def input_fn(request_body, request_content_type):
    if request_content_type in ['audio/wav', 'audio/wave', 'audio/x-wav']:
        fs, data = wavfile.read(BytesIO(request_body))
        #data = torch.from_numpy(data)
        return fs, data
    else:
        raise ValueError(
            "Unsupported content type: {}".format(request_content_type))
