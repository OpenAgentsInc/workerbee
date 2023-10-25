import base64
import llama_cpp

GGML_TYPE_MAP = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
}

GGML_INVERSE_MAP={v.lower():k for k, v in GGML_TYPE_MAP.items()}

def b64enc(byt):
    return base64.urlsafe_b64encode(byt).decode()


def b64dec(str_):
    return base64.urlsafe_b64decode(str_)


def quantize_gguf(fil, level):
    out = fil + "." + level
    fil = fil.encode("utf-8")
    out = out.encode("utf-8")
    level = GGML_INVERSE_MAP[level.lower()]
    qp = llama_cpp.llama_model_quantize_default_params()
    qp.ftype = level
    return_code = llama_cpp.llama_model_quantize(fil, out, qp)
    if return_code != 0:
        raise RuntimeError("Failed to quantize model")
    return out
