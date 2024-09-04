import torch
import sys

output_file = "checkpoint_basic_information.log"


def load_checkpoint(checkpoint_path, use_gpu=True):
    # 加载 checkpoint 文件到 GPU 或 CPU
    map_location = "cuda:0" if use_gpu and torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    return checkpoint


def print_checkpoint_structure(checkpoint):
    state_dict = checkpoint["state_dict"]
    with open(output_file, "w") as f:
        original_stdout = sys.stdout
        sys.stdout = f

        # 打印 checkpoint 的键
        print("Checkpoint keys:", checkpoint.keys())

        # 获取 state_dict

        print("\nBasic Information: ", state_dict)
        # 打印每层的键和形状
        print("\nModel State Dict:")
        for key, value in state_dict.items():
            print(f"Layer: {key} | Shape: {value.shape}")
        sys.stdout = original_stdout


def analyze_attention_layers(state_dict):
    # 分析注意力层
    attention_layers = {k: v for k, v in state_dict.items() if "attn" in k}
    print("\nAttention Layers:")
    for key, value in attention_layers.items():
        print(f"Attention Layer: {key} | Shape: {value.shape}")


def main():
    checkpoint_path = "/scratch/project_2007023/boris/missing_aware_prompts/missing_aware_prompts/result/mmimdb_input_trainm_t_testm_t_seed0_from_vilt_200k_mlm_itm/version_1/checkpoints/last.ckpt"
    checkpoint = load_checkpoint(checkpoint_path, use_gpu=True)
    print_checkpoint_structure(checkpoint)
    analyze_attention_layers(checkpoint["state_dict"])


if __name__ == "__main__":
    main()
