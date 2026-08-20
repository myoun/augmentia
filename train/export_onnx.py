import argparse
from pathlib import Path

import onnx
import torch

from model import AugmentNet


def export_onnx(model_path, onnx_path, blocks=6, hidden_dim=128):
    output_path = Path(onnx_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = AugmentNet(num_blocks=blocks, hidden_dim=hidden_dim)
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f'Loaded weights from {model_path}')
    else:
        print('Exporting with initialized weights')
    model.eval()

    dummy_spatial = torch.zeros(1, 64, 8, 8, dtype=torch.float32)
    dummy_global = torch.zeros(1, 128, dtype=torch.float32)

    torch.onnx.export(
        model,
        (dummy_spatial, dummy_global),
        output_path,
        input_names=['spatial', 'global'],
        output_names=['policy', 'value', 'logits'],
        dynamic_axes={
            'spatial': {0: 'batch_size'},
            'global': {0: 'batch_size'},
            'policy': {0: 'batch_size'},
            'value': {0: 'batch_size'},
            'logits': {0: 'batch_size'},
        },
        opset_version=17,
    )

    model_proto = onnx.load(output_path)
    model_proto.ir_version = 9
    onnx.save(model_proto, output_path)
    print(f'Successfully exported ONNX model with IR version 9 -> {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='', help='Path to PyTorch .pt model')
    parser.add_argument(
        '--onnx',
        type=str,
        default='artifacts/v2/augment_net.onnx',
        help='Output ONNX file path',
    )
    parser.add_argument('--blocks', type=int, default=6, help='Number of ResNet blocks')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Channels in hidden layers')
    args = parser.parse_args()
    export_onnx(args.model, args.onnx, args.blocks, args.hidden_dim)
