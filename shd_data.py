import os
import gzip
import shutil
import hashlib
import urllib.request
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, TensorDataset
from typing import Optional, Tuple
import random, time, argparse


# --------------------------------------------------------------------------------
# Download Utilities
# --------------------------------------------------------------------------------
def validate_file(fpath, file_hash, algorithm='auto', chunk_size=65535):
    """Validate file using either sha256 or md5 hash."""
    if (algorithm == 'sha256') or (algorithm == 'auto' and len(file_hash) == 64):
        hasher = 'sha256'
    else:
        hasher = 'md5'

    return str(_hash_file(fpath, hasher, chunk_size)) == str(file_hash)

def _hash_file(fpath, algorithm='sha256', chunk_size=65535):
    """Compute file hash."""
    if algorithm == 'sha256':
        hasher = hashlib.sha256()
    else:
        hasher = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_file(fname,
             origin,
             file_hash=None,
             cache_subdir='datasets',
             hash_algorithm='auto',
             cache_dir=None):
    """
    Downloads a file from a URL if not present, and makes sure it's the correct size/checksum.
    Returns the path to the downloaded file.
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.data-cache')

    datadir_base = os.path.expanduser(cache_dir)
    os.makedirs(datadir_base, exist_ok=True)

    datadir = os.path.join(datadir_base, cache_subdir)
    os.makedirs(datadir, exist_ok=True)

    fpath = os.path.join(datadir, fname)

    download = False
    if os.path.exists(fpath):
        # File found; verify integrity if a hash was provided.
        if file_hash is not None:
            if not validate_file(fpath, file_hash, algorithm=hash_algorithm):
                print('A local file was found, but it seems to be incomplete or outdated.')
                print('Re-downloading the data...')
                download = True
    else:
        download = True

    if download:
        print('Downloading data from', origin)
        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            urllib.request.urlretrieve(origin, fpath)
        except urllib.error.HTTPError as e:
            raise Exception(error_msg.format(origin, e.code, e.msg))
        except urllib.error.URLError as e:
            raise Exception(error_msg.format(origin, e.errno, e.reason))

    return fpath

def get_and_gunzip(origin, filename, file_hash=None, cache_dir=None, cache_subdir=None):
    """Downloads `filename` from `origin`, then gunzips it if needed."""
    gz_file_path = get_file(filename,
                            origin,
                            file_hash=file_hash,
                            cache_subdir=cache_subdir,
                            cache_dir=cache_dir)
    hdf5_file_path = gz_file_path[:-3]  # remove .gz
    # If the .h5 does not exist, or is older than the .gz, decompress again
    if (not os.path.isfile(hdf5_file_path) or
        os.path.getctime(gz_file_path) > os.path.getctime(hdf5_file_path)):
        print(f"Decompressing {gz_file_path}...")
        with gzip.open(gz_file_path, 'rb') as f_in, open(hdf5_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return hdf5_file_path

def download_shd(cache_dir, cache_subdir):
    """
    Download (if necessary) the SHD dataset files into `cache_dir/cache_subdir`.
    """
    base_url = "https://zenkelab.org/datasets"

    # Retrieve MD5 hashes from remote
    # (For simplicity, just hardcode them or read them if you've already read them.)
    # Example lines from the server:  [ "<md5>  <filename>" ]
    # Here we just specify them directly for demonstration. If you have them in a file, parse similarly.
    file_hashes = {
        "shd_train.h5.gz": "4c9bde4f3c354a47ea1a188d0ea2d902",
        "shd_test.h5.gz": "cdb1e533941cbd71e6143fd372cb9af4"
    }

    files = ["shd_train.h5.gz", "shd_test.h5.gz"]

    for fname in files:
        origin = f"{base_url}/{fname}"
        md5hash = file_hashes[fname]
        local_hdf5 = get_and_gunzip(origin,
                                   filename=fname,
                                   file_hash=md5hash,
                                   cache_dir=cache_dir,
                                   cache_subdir=cache_subdir)
        print(f"File {fname} available locally at: {local_hdf5}")

# --------------------------------------------------------------------------------
# Preprocessing Utilities
# --------------------------------------------------------------------------------
def preprocess_sparsetensor(times, units, time_bins, input_size):
    """
    Given arrays of firing times and units, produce a single sparse PyTorch tensor
    of shape (T, input_size), where T = len(time_bins) - 1 (i.e. # of intervals).
    """
    # Convert times to indices in time_bins
    firing_times = torch.tensor(np.digitize(times, time_bins) - 1).long()
    units_tensor = torch.tensor(units).long()

    stacked = torch.stack([firing_times, units_tensor], dim=0)
    values = torch.ones(stacked.shape[1], dtype=torch.float)

    time_steps = len(time_bins) - 1
    size = (time_steps, input_size)
    sparse_tensor = torch.sparse_coo_tensor(stacked, values, size)
    return sparse_tensor

def convert_h5_to_sparse_list(h5_file_path: str,
                              time_bins: np.ndarray,
                              input_size: int):
    """
    Opens the specified .h5 file and converts each sample (spikes/units) into
    a PyTorch sparse tensor. Returns a list of sparse tensors and a label array.
    """
    with h5py.File(h5_file_path, 'r') as f:
        x_spikes = f['spikes']
        y_labels = f['labels']

        # We'll store results here
        sparse_list = []
        labels = []

        for i in range(len(y_labels)):
            times_i = x_spikes['times'][i]
            units_i = x_spikes['units'][i]
            sparse_tensor = preprocess_sparsetensor(times_i, units_i,
                                                    time_bins, input_size)
            sparse_list.append(sparse_tensor)
            labels.append(y_labels[i])

        labels = np.array(labels, dtype=np.int64)  # for PyTorch
    return sparse_list, labels

def sparse_list_to_dense_tensor(sparse_list):
    """
    Convert a list of sparse tensors [sparse_coo_tensor, ...] to a single dense
    tensor of shape (N, T, input_size).
    """
    dense_data_list = []
    for sparse_tensor in sparse_list:
        dense_data_list.append(sparse_tensor.to_dense())
    return torch.stack(dense_data_list, dim=0)

# --------------------------------------------------------------------------------
# Main Class / Functions to retrieve the dataset
# --------------------------------------------------------------------------------
class SHDDataset(Dataset):
    """
    A Dataset class that can load preprocessed SHD data (dense or sparse).
    Optionally loads from .pt files if they are saved, or computes them on-the-fly.
    """
    def __init__(self,
                 data_pt_path: str,
                 label_pt_path: str):
        """
        data_pt_path: path to the .pt file containing data (shape: (N, T, input_size))
        label_pt_path: path to .pt file containing labels
        """
        # Load from disk
        self.data = torch.load(data_pt_path)
        self.labels = torch.load(label_pt_path)

        assert len(self.data) == len(self.labels), "Data and labels must have same length."

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Return a single sample and label
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

def build_and_save_shd_tensors(cache_dir,
                               cache_subdir,
                               time_steps=100,
                               max_time=1.4,
                               input_size=700):
    """
    1) Download raw SHD .h5 if needed
    2) Convert them to dense tensors
    3) Save them to disk as .pt files
    """
    # Step 1: Download (if needed)
    download_shd(cache_dir, cache_subdir)

    # Step 2: Preprocess
    time_bins = np.linspace(0, max_time, num=time_steps+1)
    train_h5_path = os.path.join(cache_dir, cache_subdir, "shd_train.h5")
    test_h5_path = os.path.join(cache_dir, cache_subdir, "shd_test.h5")

    # Convert train
    train_sparse_list, train_labels = convert_h5_to_sparse_list(
        train_h5_path, time_bins, input_size
    )
    train_dense = sparse_list_to_dense_tensor(train_sparse_list)

    # Convert test
    test_sparse_list, test_labels = convert_h5_to_sparse_list(
        test_h5_path, time_bins, input_size
    )
    test_dense = sparse_list_to_dense_tensor(test_sparse_list)

    # Step 3: Save to .pt files
    train_data_pt_path = os.path.join(cache_dir, cache_subdir, "shd_train_data.pt")
    train_label_pt_path = os.path.join(cache_dir, cache_subdir, "shd_train_labels.pt")
    test_data_pt_path = os.path.join(cache_dir, cache_subdir, "shd_test_data.pt")
    test_label_pt_path = os.path.join(cache_dir, cache_subdir, "shd_test_labels.pt")

    torch.save(train_dense, train_data_pt_path)
    torch.save(torch.tensor(train_labels, dtype=torch.long), train_label_pt_path)
    torch.save(test_dense, test_data_pt_path)
    torch.save(torch.tensor(test_labels, dtype=torch.long), test_label_pt_path)

def get_shd_datasets(cache_dir: str = None,
                     cache_subdir: str = "hdspikes",
                     time_steps: int = 100,
                     max_time: float = 1.4,
                     input_size: int = 700,
                     force_preprocess: bool = False
                     ) -> Tuple[Dataset, Dataset]:
    """
    Convenience function that returns train_dataset, test_dataset for SHD.

    If .pt files already exist, it will simply load them. Otherwise,
    it will download + preprocess + save them automatically.
    """

    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), "data")

    train_data_pt_path = os.path.join(cache_dir, cache_subdir, "shd_train_data.pt")
    train_label_pt_path = os.path.join(cache_dir, cache_subdir, "shd_train_labels.pt")
    test_data_pt_path = os.path.join(cache_dir, cache_subdir, "shd_test_data.pt")
    test_label_pt_path = os.path.join(cache_dir, cache_subdir, "shd_test_labels.pt")

    # Check if the .pt files exist
    pt_files_exist = all([
        os.path.exists(train_data_pt_path),
        os.path.exists(train_label_pt_path),
        os.path.exists(test_data_pt_path),
        os.path.exists(test_label_pt_path)
    ])

    # If we do not have .pt files or if user wants to force re-preprocessing, do it
    if (not pt_files_exist) or force_preprocess:
        build_and_save_shd_tensors(cache_dir,
                                   cache_subdir,
                                   time_steps,
                                   max_time,
                                   input_size)

    # Now, load them into PyTorch datasets
    train_dataset = SHDDataset(train_data_pt_path, train_label_pt_path)
    test_dataset = SHDDataset(test_data_pt_path, test_label_pt_path)

    return train_dataset, test_dataset

