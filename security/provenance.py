import json, hashlib
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
    from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, NoEncryption, PublicFormat
    CRYPTO_OK = True
except Exception:
    CRYPTO_OK = False

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def sign_manifest(data_dict, priv_key=None):
    payload = json.dumps(data_dict, sort_keys=True).encode()
    if not CRYPTO_OK or priv_key is None:
        return {"hash": hashlib.sha256(payload).hexdigest(), "sig": None}
    sig = priv_key.sign(payload, ec.ECDSA(hashes.SHA256()))
    return {"hash": hashlib.sha256(payload).hexdigest(), "sig": sig.hex()}

def new_keypair():
    if not CRYPTO_OK:
        return None, None
    private_key = ec.generate_private_key(ec.SECP256R1())
    public_key = private_key.public_key()
    return private_key, public_key

def merkle_root(hashes):
    hs = [bytes.fromhex(h) for h in hashes]
    if not hs:
        return None
    level = hs
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            a = level[i]
            b = level[i+1] if i+1 < len(level) else a
            nxt.append(hashlib.sha256(a+b).digest())
        level = nxt
    return level[0].hex()
