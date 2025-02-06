import os
import argparse
import sys
import numpy as np
from safetensors import safe_open
from concurrent.futures import ThreadPoolExecutor, as_completed


def process_tensor(key: str, tensor: np.ndarray):
    """
    Process a single tensor: prepare name bytes, shape, dtype string, and raw data bytes.
    Returns a tuple with all information.
    """
    name_bytes = key.encode("utf-8")
    shape = tensor.shape
    # Build shape list as a list of ints
    shape_list = [int(dim) for dim in shape]
    dtype_str = str(tensor.dtype)
    dtype_bytes = dtype_str.encode("utf-8")
    tensor_bytes = tensor.tobytes()
    return (key, name_bytes, shape_list, dtype_bytes, tensor_bytes)


def convert_safetensors_to_gguf(safetensor_path: str, gguf_path: str, threads: int):
    """
    Converts a safetensors model file to a simplified GGUF format.

    The output GGUF file structure:
      - 4-byte magic header ("gguf")
      - 4-byte version number (int32, here set to 1)
      - 4-byte integer for the number of tensors
      - For each tensor (processed in sorted order by key):
          - 4-byte integer: length of tensor name, then name bytes.
          - 4-byte integer: number of dimensions, then each dimension (int32).
          - 4-byte integer: length of the dtype string, then the dtype string in UTF-8.
          - 8-byte integer: length of tensor data in bytes, then the raw tensor data.

    The conversion of individual tensors is done concurrently using the specified number of threads.
    """
    print(f"Loading safetensors model from: {safetensor_path}")
    tensors = {}
    try:
        with safe_open(safetensor_path, framework="numpy") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    except Exception as e:
        print(f"Error loading safetensors file: {e}")
        sys.exit(1)

    # Process tensors concurrently if threads > 1; otherwise sequentially.
    tensor_items = []
    keys = sorted(tensors.keys())
    if threads > 1:
        print(f"Processing tensors concurrently using {threads} threads...")
        with ThreadPoolExecutor(max_workers=threads) as executor:
            future_to_key = {
                executor.submit(process_tensor, key, tensors[key]): key for key in keys
            }
            for future in as_completed(future_to_key):
                try:
                    result = future.result()
                    tensor_items.append(result)
                except Exception as exc:
                    print(
                        f"Tensor {future_to_key[future]} generated an exception: {exc}"
                    )
    else:
        print("Processing tensors sequentially...")
        for key in keys:
            tensor_items.append(process_tensor(key, tensors[key]))

    # Ensure the tensors are written in sorted order by key.
    tensor_items.sort(key=lambda x: x[0])
    num_tensors = len(tensor_items)
    print(f"Found {num_tensors} tensors. Converting to GGUF format...")

    try:
        with open(gguf_path, "wb") as out:
            # Write magic header and version number
            out.write(b"gguf")  # 4 bytes: magic
            out.write((1).to_bytes(4, byteorder="little"))  # 4 bytes: version number

            # Write number of tensors as int32
            out.write(num_tensors.to_bytes(4, byteorder="little"))

            # Write each tensor's data
            for key, name_bytes, shape_list, dtype_bytes, tensor_bytes in tensor_items:
                # Write tensor name: length (int32) then name bytes
                out.write(len(name_bytes).to_bytes(4, byteorder="little"))
                out.write(name_bytes)

                # Write shape: number of dims (int32) then each dimension (int32)
                out.write(len(shape_list).to_bytes(4, byteorder="little"))
                for dim in shape_list:
                    out.write(int(dim).to_bytes(4, byteorder="little"))

                # Write dtype: length (int32) + dtype bytes
                out.write(len(dtype_bytes).to_bytes(4, byteorder="little"))
                out.write(dtype_bytes)

                # Write tensor data: length (int64) then raw bytes
                out.write(len(tensor_bytes).to_bytes(8, byteorder="little"))
                out.write(tensor_bytes)

        print(f"Conversion complete. GGUF file saved to: {gguf_path}")
    except Exception as e:
        print(f"Error writing GGUF file: {e}")
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a safetensors model file to a GGUF file using pip libraries with threading options."
    )
    parser.add_argument(
        "--safetensor_file",
        type=str,
        required=True,
        help="Path to the input safetensors model file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the converted GGUF model file.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of threads to use for processing tensors concurrently.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Check if input file exists
    if not os.path.exists(args.safetensor_file):
        print(f"Input safetensors file not found: {args.safetensor_file}")
        sys.exit(1)

    convert_safetensors_to_gguf(
        safetensor_path=args.safetensor_file,
        gguf_path=args.output_file,
        threads=args.threads,
    )


if __name__ == "__main__":
    main()
