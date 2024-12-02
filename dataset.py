import torch
import pathlib
import json
import soundfile as sf
import numpy as np
import scipy.io


class MyCustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []
        for path_iter in sorted(root_dir.glob('*')):
            if not path_iter.is_dir():
                continue
            mix_file = None
            sim_file = None
            loc_file = None

            # iterate over the wav files of a given path_iter
            for audio_file in path_iter.glob('*.wav'):
                if audio_file.stem[:3] == 'mix':
                    mix_file = audio_file
                    sim_file = str(audio_file.parent) + "/sim" + audio_file.stem[3:] + ".wav"
                    loc_file = str(audio_file.parent) + "/vector" + audio_file.stem[3:] + ".mat"
                    self.samples.append((mix_file,sim_file,loc_file))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # get the paths of the mix and the reference audio signals
        mix_file, sim_file, loc_file = self.samples[idx]
        # load the mix signal
        mix_signal, _ = sf.read(mix_file)
        # convert to tensor
        mix_signal = torch.from_numpy(mix_signal).type(torch.float32)
        ilen = torch.tensor(max(mix_signal.shape))
        # load the reference audio signal
        sim_signal, _ = sf.read(sim_file)
        sim_signal = torch.from_numpy(sim_signal).type(torch.float32)
        # load the location vector
        loc_data = scipy.io.loadmat(loc_file)['vector']
        return mix_signal, ilen, sim_signal, loc_data


# Example usage:
def get_dataloader(root_dir,split='train',batch_size=4,shuffle=True,num_workers=0,pin_memory=True):
    dataset = MyCustomDataset(root_dir=root_dir, split=split)
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            num_workers=num_workers,
                                            pin_memory=pin_memory)
    return dataloader