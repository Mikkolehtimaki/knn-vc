"""Speech conversion."""
import argparse
import json

import torch
import torchaudio
from scipy.io import wavfile


def main(
    speaker_source_wav: str,
    target_wavs: list[str],
    out_path: str,
    hifigan_weights: str = None,
    device: str = "cpu",
):
    """Convert speech from one speaker to another.

    speaker_source_wav is the sound that needs to be converted.
    target_wavs is a list of target speakers to convert to.

    Sound must be mono 16kHz.
    """
    device = torch.device(device)

    knn_vc = torch.hub.load(
        "bshall/knn-vc",
        "knn_vc",
        prematched=True,
        trust_repo=True,
        pretrained=True,
        device=device,
    )

    if hifigan_weights:
        from hifigan.utils import AttrDict
        from hifigan.models import Generator as HiFiGAN

        with open("hifigan/config_v1_wavlm.json") as f:
            data = f.read()

        json_config = json.loads(data)
        h = AttrDict(json_config)

        print("Loading local hifigan weights")
        state_dict_g = torch.load(hifigan_weights, map_location=device)

        generator = HiFiGAN(h).to(device)
        generator.load_state_dict(state_dict_g["generator"])
        generator.eval()
        generator.remove_weight_norm()
        knn_vc.hifigan = generator

    query_seq = knn_vc.get_features(speaker_source_wav)
    matching_set = knn_vc.get_matching_set(target_wavs)

    out_wav = knn_vc.match(query_seq, matching_set, topk=4)
    # out_wav is (T,) tensor converted 16kHz output wav using k=4 for kNN.

    # Save to file
    wavfile.write(out_path, 16000, out_wav.numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert speech from one speaker to another."
    )
    parser.add_argument("-s", "--speaker_source_wav", required=True, type=str)
    parser.add_argument("-t", "--target_wavs", required=True, nargs="+", type=str)
    parser.add_argument("-o", "--out_path", required=True, type=str)
    parser.add_argument("-w", "--hifigan_weights", default=None, type=str)
    parser.add_argument("-d", "--device", default="cpu", type=str)

    args = parser.parse_args()
    main(
        args.speaker_source_wav,
        args.target_wavs,
        args.out_path,
        args.hifigan_weights,
        args.device,
    )
