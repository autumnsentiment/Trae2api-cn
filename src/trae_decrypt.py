"""trae_decrypt.py - Trae "tc" encrypted storage format decryption

Decrypt auth data from Trae CN / SOLO CN storage.json.
Data format: [6B Header][32B RandomBytes][N EncryptedData]
Decrypted:  [64B SHA-512 Hash][PKCS7 padded plaintext JSON]

Key derivation: SHA-512(RandomBytes) -> XOR salt -> SHA-512 -> Key(16B) + IV(16B)
"""

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Optional

# Four hard-coded 64-byte salts (from Trae CN frontend JS reverse engineering)
SALT_A = bytes([
    82, 9, 106, 213, 48, 54, 165, 56, 191, 64, 163, 158, 129, 243, 215, 251,
    124, 227, 57, 130, 155, 47, 255, 135, 52, 142, 67, 68, 196, 222, 233, 203,
    84, 123, 148, 50, 166, 194, 35, 61, 238, 76, 149, 11, 66, 250, 195, 78,
    8, 46, 161, 102, 40, 217, 36, 178, 118, 91, 162, 73, 109, 139, 209, 37,
])

SALT_B = bytes([
    31, 221, 168, 51, 136, 7, 199, 49, 177, 18, 16, 89, 39, 128, 236, 95,
    96, 81, 127, 169, 25, 181, 74, 13, 45, 229, 122, 159, 147, 201, 156, 239,
    160, 224, 59, 77, 174, 42, 245, 176, 200, 235, 187, 60, 131, 83, 153, 97,
    23, 43, 4, 126, 186, 119, 214, 38, 225, 105, 20, 99, 85, 33, 12, 125,
])

SALT_C = bytes([
    191, 192, 216, 250, 122, 246, 220, 97, 31, 254, 98, 27, 8, 72, 71, 176,
    135, 99, 96, 18, 127, 101, 203, 104, 211, 102, 191, 125, 37, 72, 150, 156,
    51, 229, 121, 35, 17, 153, 141, 177, 110, 131, 150, 128, 172, 255, 254, 6,
    18, 140, 55, 62, 236, 249, 135, 64, 135, 12, 117, 4, 89, 149, 168, 209,
])

SALT_D = bytes([
    246, 204, 26, 232, 232, 70, 129, 109, 223, 146, 169, 242, 23, 241, 105, 145,
    50, 196, 165, 42, 254, 120, 3, 54, 244, 207, 209, 85, 53, 6, 138, 106,
    175, 148, 31, 204, 186, 186, 165, 182, 87, 142, 49, 10, 39, 110, 26, 154,
    86, 56, 173, 125, 18, 64, 198, 225, 99, 99, 83, 82, 191, 134, 76, 170,
])

STORAGE_KEY = "iCubeAuthInfo://icube.cloudide"


def _xor_bytes(a: bytes, b: bytes, length: int) -> bytes:
    return bytes(a[i] ^ b[i] for i in range(length))


def _detect_enc_type(header: bytes) -> str:
    """Detect encryption type."""
    # AES: 0x74 0x63 0x05 0x10 0x00 0x00 ("tc" prefix)
    if header[:2] == b"tc" and header[2:6] == b"\x05\x10\x00\x00":
        return "AES"
    # AES_PRIVATE: 18 57 32 32 2 3
    if header[:6] == bytes([18, 57, 32, 32, 2, 3]):
        return "AES_PRIVATE"
    return "UNKNOWN"


def _derive_key_and_iv(random_bytes: bytes, enc_type: str) -> tuple[bytes, bytes]:
    """Derive AES-128-CBC key and IV."""
    if enc_type == "AES_PRIVATE":
        salt = _xor_bytes(SALT_C, SALT_D, 64)
    else:
        salt = _xor_bytes(SALT_A, SALT_B, 64)

    # SHA-512(RandomBytes) -> hashOfRandom (64 bytes)
    hash_of_random = hashlib.sha512(random_bytes).digest()

    # SHA-512(hashOfRandom + salt) -> finalHash (64 bytes)
    final_hash = hashlib.sha512(hash_of_random + salt).digest()

    # First 16 bytes are the AES key, next 16 are the IV
    return final_hash[:16], final_hash[16:32]


def _pkcs7_unpad(data: bytes) -> bytes:
    if not data:
        return data
    pad = data[-1]
    if 1 <= pad <= 16 and data.endswith(bytes([pad]) * pad):
        return data[:-pad]
    return data


def decrypt_storage_value(base64_value: str) -> str:
    """Decrypt a single tc-format encrypted value, returning plaintext string."""
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    buffer = base64.b64decode(base64_value)

    # Parse structure: [6B Header][32B RandomBytes][N EncryptedData]
    header = buffer[:6]
    random_bytes = buffer[6:38]
    encrypted_data = buffer[38:]

    enc_type = _detect_enc_type(header)
    if enc_type == "UNKNOWN":
        raise ValueError(f"Unknown encryption type, header={header.hex()}")

    aes_key, iv = _derive_key_and_iv(random_bytes, enc_type)

    # AES-128-CBC decrypt
    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
    decryptor = cipher.decryptor()
    decrypted = decryptor.update(encrypted_data) + decryptor.finalize()

    # Validate hash: [64B SHA-512 Hash][PKCS7 padded plaintext JSON]
    stored_hash = decrypted[:64]
    body = _pkcs7_unpad(decrypted[64:])
    computed_hash = hashlib.sha512(body).digest()

    if stored_hash != computed_hash:
        raise ValueError("Hash verification failed - decryption may be incorrect")

    return body.decode("utf-8")


def decrypt_auth_data(data_dir: str | Path) -> dict:
    """Decrypt auth data from a Trae storage.json directory."""
    storage_path = Path(data_dir) / "globalStorage" / "storage.json"

    if not storage_path.exists():
        raise FileNotFoundError(f"storage.json not found at: {storage_path}")

    storage = json.loads(storage_path.read_text("utf-8"))
    encrypted_auth = storage.get(STORAGE_KEY)

    if not encrypted_auth:
        raise KeyError(f"{STORAGE_KEY} not found in storage.json")

    # Plain JSON (SG international edition)
    if encrypted_auth.strip().startswith("{"):
        return json.loads(encrypted_auth)

    # tc encrypted format
    decrypted = decrypt_storage_value(encrypted_auth)
    return json.loads(decrypted)


def get_trae_cn_data_dir() -> str:
    """Trae CN (China) data directory."""
    appdata = os.environ.get("APPDATA", "")
    if not appdata:
        appdata = str(Path.home() / "AppData" / "Roaming")
    return str(Path(appdata) / "Trae CN" / "User")


def get_trae_solo_cn_data_dir() -> str:
    """TRAE SOLO CN data directory."""
    appdata = os.environ.get("APPDATA", "")
    if not appdata:
        appdata = str(Path.home() / "AppData" / "Roaming")
    return str(Path(appdata) / "TRAE SOLO CN" / "User")


def get_trae_sg_data_dir() -> str:
    """Trae international edition data directory."""
    appdata = os.environ.get("APPDATA", "")
    if not appdata:
        appdata = str(Path.home() / "AppData" / "Roaming")
    return str(Path(appdata) / "Trae" / "User")


def get_trae_solo_sg_data_dir() -> str:
    """TRAE SOLO international edition data directory."""
    appdata = os.environ.get("APPDATA", "")
    if not appdata:
        appdata = str(Path.home() / "AppData" / "Roaming")
    return str(Path(appdata) / "TRAE SOLO" / "User")


def try_auto_discover() -> tuple[Optional[dict], Optional[str]]:
    """Iterate all known editions, returning the first usable (auth_data, edition_name)."""
    editions = [
        ("cn", get_trae_cn_data_dir()),
        ("solo", get_trae_solo_cn_data_dir()),
        ("sg", get_trae_sg_data_dir()),
        ("solo-sg", get_trae_solo_sg_data_dir()),
    ]
    for edition, data_dir in editions:
        try:
            auth = decrypt_auth_data(data_dir)
            return auth, edition
        except Exception:
            continue
    return None, None
