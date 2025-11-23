"""
Homomorphic Encryption Benchmark Suite with TenSEAL

This script benchmarks:
  1. Context serialization size for CKKS and BFV (different keys and params)
  2. Ciphertext size and overhead for encrypted vectors
  3. Precision behavior across value ranges and operations
  4. A ResNet18 encrypt plus decrypt workload for timing

You can run it as a normal Python script:

    python homomorphic_benchmarks.py

Requirements (install via pip before running):
    pip install tenseal numpy tabulate torch torchvision pandas
"""

import math
import random
import time

import numpy as np
import pandas as pd
import tenseal as ts
import tenseal.sealapi as seal  # kept in case you want to extend with SEAL directly

from tabulate import tabulate
import torch
import torch.nn as nn
from torchvision import models


# ======================================================================
# Utility helpers
# ======================================================================

def convert_size(size_bytes: int) -> str:
    """
    Convert a size in bytes to a human readable string.
    """
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "{} {}".format(s, size_name[i])


def decrypt(enc):
    """
    Helper to decrypt a TenSEAL encrypted vector with its context.
    """
    return enc.decrypt()


# String maps for readable tables
ENC_TYPE_STR = {
    ts.ENCRYPTION_TYPE.SYMMETRIC: "symmetric",
    ts.ENCRYPTION_TYPE.ASYMMETRIC: "asymmetric",
}

SCHEME_STR = {
    ts.SCHEME_TYPE.CKKS: "ckks",
    ts.SCHEME_TYPE.BFV: "bfv",
}


# ======================================================================
# Benchmark 1: Context serialization sizes
# ======================================================================

def benchmark_context_sizes():
    """
    Benchmark the size of serialized TenSEAL contexts
    for different schemes, parameters, and saved keys.

    Returns:
        list of lists (rows for a table)
    """
    ctx_size_benchmarks = [
        [
            "Encryption Type",
            "Scheme Type",
            "Polynomial modulus",
            "Coefficient modulus sizes",
            "Saved keys",
            "Context serialized size",
        ]
    ]

    for enc_type in [ts.ENCRYPTION_TYPE.SYMMETRIC, ts.ENCRYPTION_TYPE.ASYMMETRIC]:
        # CKKS parameter sets
        ckks_param_sets = [
            (8192, [40, 21, 21, 21, 21, 21, 21, 40]),
            (8192, [40, 20, 40]),
            (8192, [20, 20, 20]),
            (8192, [17, 17]),
            (4096, [40, 20, 40]),
            (4096, [30, 20, 30]),
            (4096, [20, 20, 20]),
            (4096, [19, 19, 19]),
            (4096, [18, 18, 18]),
            (4096, [18, 18]),
            (4096, [17, 17]),
            (2048, [20, 20]),
            (2048, [18, 18]),
            (2048, [16, 16]),
        ]

        for poly_mod, coeff_mod_bit_sizes in ckks_param_sets:
            context = ts.context(
                scheme=ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=poly_mod,
                coeff_mod_bit_sizes=coeff_mod_bit_sizes,
                encryption_type=enc_type,
            )
            context.generate_galois_keys()
            context.generate_relin_keys()

            # Serialize with all keys
            ser = context.serialize(
                save_public_key=True,
                save_secret_key=True,
                save_galois_keys=True,
                save_relin_keys=True,
            )
            ctx_size_benchmarks.append([
                ENC_TYPE_STR[enc_type],
                SCHEME_STR[ts.SCHEME_TYPE.CKKS],
                poly_mod,
                coeff_mod_bit_sizes,
                "all",
                convert_size(len(ser)),
            ])

            # Public key only (for asymmetric)
            if enc_type is ts.ENCRYPTION_TYPE.ASYMMETRIC:
                ser = context.serialize(
                    save_public_key=True,
                    save_secret_key=False,
                    save_galois_keys=False,
                    save_relin_keys=False,
                )
                ctx_size_benchmarks.append([
                    ENC_TYPE_STR[enc_type],
                    SCHEME_STR[ts.SCHEME_TYPE.CKKS],
                    poly_mod,
                    coeff_mod_bit_sizes,
                    "public key",
                    convert_size(len(ser)),
                ])

            # Secret key only
            ser = context.serialize(
                save_public_key=False,
                save_secret_key=True,
                save_galois_keys=False,
                save_relin_keys=False,
            )
            ctx_size_benchmarks.append([
                ENC_TYPE_STR[enc_type],
                SCHEME_STR[ts.SCHEME_TYPE.CKKS],
                poly_mod,
                coeff_mod_bit_sizes,
                "secret key",
                convert_size(len(ser)),
            ])

            # Galois keys only
            ser = context.serialize(
                save_public_key=False,
                save_secret_key=False,
                save_galois_keys=True,
                save_relin_keys=False,
            )
            ctx_size_benchmarks.append([
                ENC_TYPE_STR[enc_type],
                SCHEME_STR[ts.SCHEME_TYPE.CKKS],
                poly_mod,
                coeff_mod_bit_sizes,
                "galois keys",
                convert_size(len(ser)),
            ])

            # Relinearization keys only
            ser = context.serialize(
                save_public_key=False,
                save_secret_key=False,
                save_galois_keys=False,
                save_relin_keys=True,
            )
            ctx_size_benchmarks.append([
                ENC_TYPE_STR[enc_type],
                SCHEME_STR[ts.SCHEME_TYPE.CKKS],
                poly_mod,
                coeff_mod_bit_sizes,
                "relin keys",
                convert_size(len(ser)),
            ])

            # No keys
            ser = context.serialize(
                save_public_key=False,
                save_secret_key=False,
                save_galois_keys=False,
                save_relin_keys=False,
            )
            ctx_size_benchmarks.append([
                ENC_TYPE_STR[enc_type],
                SCHEME_STR[ts.SCHEME_TYPE.CKKS],
                poly_mod,
                coeff_mod_bit_sizes,
                "none",
                convert_size(len(ser)),
            ])

        # BFV parameter sets
        bfv_param_sets = [
            (8192, [40, 21, 21, 21, 21, 21, 21, 40]),
            (8192, [40, 21, 21, 21, 21, 21, 40]),
            (8192, [40, 21, 21, 21, 21, 40]),
            (8192, [40, 21, 21, 21, 40]),
            (8192, [40, 21, 21, 40]),
            (8192, [40, 20, 40]),
            (4096, [40, 20, 40]),
            (4096, [30, 20, 30]),
            (4096, [20, 20, 20]),
            (4096, [19, 19, 19]),
            (4096, [18, 18, 18]),
            (2048, [20, 20]),
        ]

        for poly_mod, coeff_mod_bit_sizes in bfv_param_sets:
            context = ts.context(
                scheme=ts.SCHEME_TYPE.BFV,
                poly_modulus_degree=poly_mod,
                plain_modulus=786433,
                coeff_mod_bit_sizes=coeff_mod_bit_sizes,
                encryption_type=enc_type,
            )
            context.generate_galois_keys()
            context.generate_relin_keys()

            # All keys
            ser = context.serialize(
                save_public_key=True,
                save_secret_key=True,
                save_galois_keys=True,
                save_relin_keys=True,
            )
            ctx_size_benchmarks.append([
                ENC_TYPE_STR[enc_type],
                SCHEME_STR[ts.SCHEME_TYPE.BFV],
                poly_mod,
                coeff_mod_bit_sizes,
                "all",
                convert_size(len(ser)),
            ])

            if enc_type is ts.ENCRYPTION_TYPE.ASYMMETRIC:
                ser = context.serialize(
                    save_public_key=True,
                    save_secret_key=False,
                    save_galois_keys=False,
                    save_relin_keys=False,
                )
                ctx_size_benchmarks.append([
                    ENC_TYPE_STR[enc_type],
                    SCHEME_STR[ts.SCHEME_TYPE.BFV],
                    poly_mod,
                    coeff_mod_bit_sizes,
                    "public key",
                    convert_size(len(ser)),
                ])

            ser = context.serialize(
                save_public_key=False,
                save_secret_key=True,
                save_galois_keys=False,
                save_relin_keys=False,
            )
            ctx_size_benchmarks.append([
                ENC_TYPE_STR[enc_type],
                SCHEME_STR[ts.SCHEME_TYPE.BFV],
                poly_mod,
                coeff_mod_bit_sizes,
                "secret key",
                convert_size(len(ser)),
            ])

            ser = context.serialize(
                save_public_key=False,
                save_secret_key=False,
                save_galois_keys=True,
                save_relin_keys=False,
            )
            ctx_size_benchmarks.append([
                ENC_TYPE_STR[enc_type],
                SCHEME_STR[ts.SCHEME_TYPE.BFV],
                poly_mod,
                coeff_mod_bit_sizes,
                "galois keys",
                convert_size(len(ser)),
            ])

            ser = context.serialize(
                save_public_key=False,
                save_secret_key=False,
                save_galois_keys=False,
                save_relin_keys=True,
            )
            ctx_size_benchmarks.append([
                ENC_TYPE_STR[enc_type],
                SCHEME_STR[ts.SCHEME_TYPE.BFV],
                poly_mod,
                coeff_mod_bit_sizes,
                "relin keys",
                convert_size(len(ser)),
            ])

            ser = context.serialize(
                save_public_key=False,
                save_secret_key=False,
                save_galois_keys=False,
                save_relin_keys=False,
            )
            ctx_size_benchmarks.append([
                ENC_TYPE_STR[enc_type],
                SCHEME_STR[ts.SCHEME_TYPE.BFV],
                poly_mod,
                coeff_mod_bit_sizes,
                "none",
                convert_size(len(ser)),
            ])

    return ctx_size_benchmarks


# ======================================================================
# Benchmark 2: Ciphertext sizes and overhead
# ======================================================================

def benchmark_ciphertext_sizes():
    """
    Benchmark ciphertext serialization size for CKKS and BFV
    and compute the overhead compared to plain data.

    Returns:
        list of lists (rows for a table)
    """
    # Sample dummy vector to simulate network payload
    data = [random.uniform(-10, 10) for _ in range(10 ** 3)]
    network_data = bytes(str(data), encoding="utf8")
    print("Plain data size approximately {}".format(convert_size(len(network_data))))

    enc_type = ts.ENCRYPTION_TYPE.ASYMMETRIC

    ct_size_benchmarks = [
        [
            "Encryption Type",
            "Scheme Type",
            "Polynomial modulus",
            "Coefficient modulus sizes",
            "Precision",
            "Ciphertext serialized size",
            "Encryption increase ratio",
        ]
    ]

    # CKKS ciphertext sizes
    ckks_settings = [
        (8192, [60, 40, 60], 40),
        (8192, [40, 21, 21, 21, 21, 21, 21, 40], 40),
        (8192, [40, 21, 21, 21, 21, 21, 21, 40], 21),
        (8192, [40, 20, 40], 40),
        (8192, [20, 20, 20], 38),
        (8192, [60, 60], 38),
        (8192, [40, 40], 38),
        (8192, [17, 17], 15),
        (4096, [40, 20, 40], 40),
        (4096, [30, 20, 30], 40),
        (4096, [20, 20, 20], 38),
        (4096, [19, 19, 19], 35),
        (4096, [18, 18, 18], 33),
        (4096, [30, 30], 25),
        (4096, [25, 25], 20),
        (4096, [18, 18], 16),
        (4096, [17, 17], 15),
        (2048, [20, 20], 18),
        (2048, [18, 18], 16),
        (2048, [16, 16], 14),
    ]

    for poly_mod, coeff_mod_bit_sizes, prec in ckks_settings:
        context = ts.context(
            scheme=ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_mod,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes,
            encryption_type=enc_type,
        )
        scale = 2 ** prec
        ckks_vec = ts.ckks_vector(context, data, scale)

        enc_network_data = ckks_vec.serialize()
        ratio = round(len(enc_network_data) / len(network_data), 2)

        ct_size_benchmarks.append([
            ENC_TYPE_STR[enc_type],
            SCHEME_STR[ts.SCHEME_TYPE.CKKS],
            poly_mod,
            coeff_mod_bit_sizes,
            "2**{}".format(prec),
            convert_size(len(enc_network_data)),
            ratio,
        ])

    # BFV ciphertext sizes
    bfv_settings = [
        (8192, [40, 21, 21, 21, 21, 21, 21, 40]),
        (8192, [40, 21, 21, 21, 21, 21, 40]),
        (8192, [40, 21, 21, 21, 21, 40]),
        (8192, [40, 21, 21, 21, 40]),
        (8192, [40, 21, 21, 40]),
        (8192, [40, 20, 40]),
        (4096, [40, 20, 40]),
        (4096, [30, 20, 30]),
        (4096, [20, 20, 20]),
        (4096, [19, 19, 19]),
        (4096, [18, 18, 18]),
        (2048, [20, 20]),
    ]

    for poly_mod, coeff_mod_bit_sizes in bfv_settings:
        context = ts.context(
            scheme=ts.SCHEME_TYPE.BFV,
            poly_modulus_degree=poly_mod,
            plain_modulus=786433,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes,
            encryption_type=enc_type,
        )
        vec = ts.bfv_vector(context, data)
        enc_network_data = vec.serialize()
        ratio = round(len(enc_network_data) / len(network_data), 2)

        ct_size_benchmarks.append([
            ENC_TYPE_STR[enc_type],
            SCHEME_STR[ts.SCHEME_TYPE.BFV],
            poly_mod,
            coeff_mod_bit_sizes,
            "-",
            convert_size(len(enc_network_data)),
            ratio,
        ])

    return ct_size_benchmarks


# ======================================================================
# Benchmark 3: Precision behavior across ranges and ops
# ======================================================================

def values_close(x_list, y_list, tol):
    """
    Simple approximate equality check between two lists.
    """
    if len(x_list) != len(y_list):
        return False
    for x, y in zip(x_list, y_list):
        if abs(x - y) > tol:
            return False
    return True


def benchmark_precision_ranges():
    """
    Evaluate how precision behaves across value ranges
    for CKKS encryption, decryption, addition, and multiplication.

    Returns:
        list of rows for a table.
    """
    enc_type = ts.ENCRYPTION_TYPE.ASYMMETRIC
    results = [
        [
            "Value range",
            "Polynomial modulus",
            "Coefficient modulus sizes",
            "Precision",
            "Operation",
            "Status",
        ]
    ]

    # Encryption and sum behavior
    for data_pow in [-1, 0, 1, 5, 11, 21, 41, 51]:
        data = [random.uniform(2 ** data_pow, 2 ** (data_pow + 1))]
        val_str = "[2^{} - 2^{}]".format(data_pow, data_pow + 1)

        ckks_settings = [
            (2 ** 14, [50, 50, 50, 50, 50, 50, 50, 50], 50),
        ]

        for poly_mod, coeff_mod_bit_sizes, prec in ckks_settings:
            context = ts.context(
                scheme=ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=poly_mod,
                coeff_mod_bit_sizes=coeff_mod_bit_sizes,
                encryption_type=enc_type,
            )
            scale = 2 ** prec

            try:
                ckks_vec = ts.ckks_vector(context, data, scale)
            except BaseException:
                results.append([
                    val_str,
                    poly_mod,
                    coeff_mod_bit_sizes,
                    "2**{}".format(prec),
                    "encrypt",
                    "encryption failed",
                ])
                continue

            decrypted = decrypt(ckks_vec)

            # try to find the best precision where decrypted ~ data
            for dec_prec in reversed(range(prec)):
                if values_close(decrypted, data, tol=2 ** -dec_prec):
                    results.append([
                        val_str,
                        poly_mod,
                        coeff_mod_bit_sizes,
                        "2**{}".format(prec),
                        "encrypt",
                        "decryption precision 2 ** {}".format(-dec_prec),
                    ])
                    break

            ckks_sum = ckks_vec + ckks_vec
            decrypted_sum = decrypt(ckks_sum)
            target_sum = [data[0] + data[0]]

            for dec_prec in reversed(range(prec)):
                if values_close(decrypted_sum, target_sum, tol=2 ** -dec_prec):
                    results.append([
                        val_str,
                        poly_mod,
                        coeff_mod_bit_sizes,
                        "2**{}".format(prec),
                        "sum",
                        "decryption precision 2 ** {}".format(-dec_prec),
                    ])
                    break

    # Multiplication behavior with more varied settings
    for data_pow in [-1, 0, 1, 5, 11, 21, 41, 51]:
        data = [random.uniform(2 ** data_pow, 2 ** (data_pow + 1))]
        val_str = "[2^{} - 2^{}]".format(data_pow, data_pow + 1)

        ckks_settings = [
            (8192, [60, 40, 40, 60], 40),
            (8192, [40, 21, 21, 40], 40),
            (8192, [40, 21, 21, 40], 21),
            (8192, [40, 20, 20, 40], 40),
            (8192, [20, 20, 20], 38),
            (4096, [40, 20, 40], 40),
            (4096, [30, 20, 30], 40),
            (4096, [20, 20, 20], 38),
            (4096, [19, 19, 19], 35),
            (4096, [18, 18, 18], 33),
            (4096, [30, 30, 30], 25),
            (4096, [25, 25, 25], 20),
            (4096, [18, 18, 18], 16),
            (2048, [18, 18, 18], 16),
        ]

        for poly_mod, coeff_mod_bit_sizes, prec in ckks_settings:
            context = ts.context(
                scheme=ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=poly_mod,
                coeff_mod_bit_sizes=coeff_mod_bit_sizes,
                encryption_type=enc_type,
            )
            scale = 2 ** prec

            try:
                ckks_vec = ts.ckks_vector(context, data, scale)
            except BaseException:
                continue

            try:
                ckks_mul = ckks_vec * ckks_vec
            except BaseException:
                results.append([
                    val_str,
                    poly_mod,
                    coeff_mod_bit_sizes,
                    "2**{}".format(prec),
                    "mul",
                    "failed",
                ])
                continue

            decrypted_mul = decrypt(ckks_mul)
            target_mul = [data[0] * data[0]]

            for dec_prec in reversed(range(prec)):
                if values_close(decrypted_mul, target_mul, tol=2 ** -dec_prec):
                    results.append([
                        val_str,
                        poly_mod,
                        coeff_mod_bit_sizes,
                        "2**{}".format(prec),
                        "mul",
                        "decryption precision 2 ** {}".format(-dec_prec),
                    ])
                    break

    return results


# ======================================================================
# Benchmark 4: ResNet18 encrypt and decrypt workload
# ======================================================================

def encrypt_decrypt_resnet(model: nn.Module, context: ts.Context, scale: float) -> bool:
    """
    Encrypt and decrypt all parameters of a ResNet model using CKKS vectors.
    Returns True if everything succeeds, False if any layer fails.
    """
    for param in model.parameters():
        flat_param = param.data.numpy().flatten().tolist()
        try:
            ckks_vec = ts.ckks_vector(context, flat_param, scale)
            _ = ckks_vec.decrypt()
        except BaseException:
            return False
    return True


def benchmark_resnet_encryption():
    """
    Sweep over polynomial degrees and modulus bit sizes
    and measure the time to encrypt and decrypt ResNet18 parameters.

    Returns:
        list of rows for a table.
    """
    print("Loading ResNet18 model from torchvision...")
    resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet18.eval()

    ct_size_benchmarks = [
        ["Polynomial modulus", "Coefficient modulus sizes", "Precision", "Operation", "Time (s)"]
    ]

    # q values from 25 to 60 with step of 5
    q_values = list(range(25, 65, 5))
    # N values: 2^13, 2^14, 2^15
    N_values = [2 ** 13, 2 ** 14, 2 ** 15]

    for poly_mod in N_values:
        for q in q_values:
            coeff_mod_bit_sizes = [q] * 4
            prec = q

            context = ts.context(
                scheme=ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=poly_mod,
                coeff_mod_bit_sizes=coeff_mod_bit_sizes,
            )
            scale = 2 ** prec

            try:
                start_time = time.time()
                success = encrypt_decrypt_resnet(resnet18, context, scale)
                end_time = time.time()

                elapsed_time = end_time - start_time if success else None
                ct_size_benchmarks.append([
                    poly_mod,
                    coeff_mod_bit_sizes,
                    "2**{}".format(prec),
                    "encrypt-decrypt",
                    elapsed_time if success else "failed",
                ])
            except Exception:
                ct_size_benchmarks.append([
                    poly_mod,
                    coeff_mod_bit_sizes,
                    "2**{}".format(prec),
                    "encrypt-decrypt",
                    "failed",
                ])

    return ct_size_benchmarks


# ======================================================================
# Main entry point
# ======================================================================

def main():
    print("\n=== Benchmark 1: Context serialization sizes ===")
    ctx_rows = benchmark_context_sizes()
    print(tabulate(ctx_rows[1:], headers=ctx_rows[0], tablefmt="github"))

    print("\n=== Benchmark 2: Ciphertext sizes and overhead ===")
    ct_rows = benchmark_ciphertext_sizes()
    print(tabulate(ct_rows[1:], headers=ct_rows[0], tablefmt="github"))

    print("\n=== Benchmark 3: Precision ranges and operations ===")
    prec_rows = benchmark_precision_ranges()
    print(tabulate(prec_rows[1:], headers=prec_rows[0], tablefmt="github"))

    print("\n=== Benchmark 4: ResNet18 encrypt and decrypt timing ===")
    resnet_rows = benchmark_resnet_encryption()
    df_resnet = pd.DataFrame(resnet_rows[1:], columns=resnet_rows[0])
    print(df_resnet.head())

    # Optional: save results to CSV files for later analysis
    pd.DataFrame(ctx_rows[1:], columns=ctx_rows[0]).to_csv("context_sizes.csv", index=False)
    pd.DataFrame(ct_rows[1:], columns=ct_rows[0]).to_csv("ciphertext_sizes.csv", index=False)
    pd.DataFrame(prec_rows[1:], columns=prec_rows[0]).to_csv("precision_ranges.csv", index=False)
    df_resnet.to_csv("resnet_benchmarks.csv", index=False)

    print("\nCSV files written: context_sizes.csv, ciphertext_sizes.csv, precision_ranges.csv, resnet_benchmarks.csv")


if __name__ == "__main__":
    main()
