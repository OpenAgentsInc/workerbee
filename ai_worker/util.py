import base64


def b64enc(byt):
    return base64.urlsafe_b64encode(byt).decode()


def b64dec(str_):
    return base64.urlsafe_b64decode(str_)


