import base64
import os
from hashlib import sha256
from typing import Optional, Union

import coincurve as secp256k1

"""Minimalist pub/priv key classes for signing and verification based on coincurve"""


class PublicKey:
    def __init__(self,
                 raw_bytes: Union[bytes, "PrivateKey", secp256k1.keys.PublicKey, secp256k1.keys.PublicKeyXOnly, str]):
        """
        :param raw_bytes: The formatted public key.
        :type data: bytes, private key to copy, or hex str
        """
        if isinstance(raw_bytes, PrivateKey):
            self.raw_bytes = raw_bytes.public_key.raw_bytes
        elif isinstance(raw_bytes, secp256k1.keys.PublicKey):
            self.raw_bytes = raw_bytes.format(compressed=True)[2:]
        elif isinstance(raw_bytes, secp256k1.keys.PublicKeyXOnly):
            self.raw_bytes = raw_bytes.format()
        elif isinstance(raw_bytes, str):
            self.raw_bytes = bytes.fromhex(raw_bytes)
        else:
            self.raw_bytes = raw_bytes

    def hex(self) -> str:
        return self.raw_bytes.hex()

    def verify(self, sig: str, message: bytes) -> bool:
        pk = secp256k1.PublicKeyXOnly(self.raw_bytes)
        return pk.verify(bytes.fromhex(sig), message)

    @classmethod
    def from_hex(cls, hex_: str, /) -> 'PublicKey':
        return cls(bytes.fromhex(hex_))

    def __repr__(self):
        pubkey = self.hex()
        return f'PublicKey({pubkey[:10]}...{pubkey[-10:]})'

    def __eq__(self, other):
        return isinstance(other, PublicKey) and self.raw_bytes == other.raw_bytes

    def __hash__(self):
        return hash(self.raw_bytes)

    def __str__(self):
        """Return public key in hex form
        :return: string
        :rtype: str
        """
        return self.hex()

    def __bytes__(self):
        """Return raw bytes
        :return: Raw bytes
        :rtype: bytes
        """
        return self.raw_bytes


class PrivateKey:
    def __init__(self, raw_secret: Optional[bytes] = None) -> None:
        """
        :param raw_secret: The secret used to initialize the private key.
                           If not provided or `None`, a new key will be generated.
        :type raw_secret: bytes
        """
        if raw_secret is not None:
            self.raw_secret = raw_secret
        else:
            self.raw_secret = os.urandom(32)

        sk = secp256k1.PrivateKey(self.raw_secret)
        self.public_key = PublicKey(sk.public_key_xonly)

    @classmethod
    def from_hex(cls, hex_: str, /):
        """Load a PrivateKey from its hex form."""
        return cls(bytes.fromhex(hex_))

    def __hash__(self):
        return hash(self.raw_secret)

    def __eq__(self, other):
        return isinstance(other, PrivateKey) and self.raw_secret == other.raw_secret

    def hex(self) -> str:
        return self.raw_secret.hex()

    def sign(self, message: bytes, aux_randomness: bytes = b'') -> str:
        sk = secp256k1.PrivateKey(self.raw_secret)
        return sk.sign_schnorr(message, aux_randomness).hex()

    def __repr__(self):
        pubkey = self.public_key.hex()
        return f'PrivateKey({pubkey[:10]}...{pubkey[-10:]})'

    def __str__(self):
        """Return private key in hex form
        :return: hex string
        :rtype: str
        """
        return self.hex()

    def __bytes__(self):
        """Return raw bytes
        :return: Raw bytes
        :rtype: bytes
        """
        return self.raw_secret


def test_cp():
    pk = PrivateKey()
    pk2 = PrivateKey(pk.raw_secret)
    assert pk == pk2


def test_fromhex():
    pk = PrivateKey()
    pk2 = PrivateKey.from_hex(pk.hex())
    assert pk == pk2


def test_sig():
    pk = PrivateKey()
    pub = pk.public_key
    sig = pk.sign(b'1' * 32)
    assert pub.verify(sig, b'1' * 32)
