# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import pathlib
import warnings

from logging import getLogger

import numpy as np
import pandas as pd

from decord import VideoReader, cpu

import torch

from src.datasets.utils.weighted_sampler import DistributedWeightedSampler

_GLOBAL_SEED = 0
logger = getLogger()


def make_videodataset(
    data_paths,
    batch_size,
    frames_per_clip=8,
    frame_step=4,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,
    filter_long_videos=int(10**9),
    transform=None,
    shared_transform=None,
    rank=0,
    world_size=1,
    datasets_weights=None,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    duration=None,
    log_dir=None,
):
    """
    Creates a VideoDataset and corresponding DataLoader with distributed sampling.

    Args:
        data_paths (list of str): Paths to video data files.
        batch_size (int): Number of samples per batch.
        frames_per_clip (int, optional): Number of frames in each video clip. Defaults to 8.
        frame_step (int, optional): Step size between frames in each clip. Defaults to 4.
        num_clips (int, optional): Number of clips to sample from each video. Defaults to 1.
        random_clip_sampling (bool, optional): Whether to randomly sample clips. Defaults to True.
        allow_clip_overlap (bool, optional): Whether to allow overlapping clips. Defaults to False.
        filter_short_videos (bool, optional): Whether to filter out short videos. Defaults to False.
        filter_long_videos (int, optional): Maximum allowed video duration in seconds. Defaults to large value.
        transform (callable, optional): Function/transform to apply to each clip.
        shared_transform (callable, optional): Transform to be applied to all clips.
        rank (int, optional): Rank of the current process for distributed training. Defaults to 0.
        world_size (int, optional): Number of processes for distributed training. Defaults to 1.
        datasets_weights (list of float, optional): Weights for each dataset for sampling.
        collator (callable, optional): Function to collate samples into batches.
        drop_last (bool, optional): Whether to drop the last incomplete batch. Defaults to True.
        num_workers (int, optional): Number of subprocesses for data loading. Defaults to 10.
        pin_mem (bool, optional): Whether to use pinned memory for data loading. Defaults to True.
        duration (int, optional): Duration in seconds to filter videos. Defaults to None.
        log_dir (str, optional): Directory for logging information. Defaults to None.

    Returns:
        tuple: A tuple containing the VideoDataset, DataLoader, and DistributedSampler.
    """

    dataset = VideoDataset(
        data_paths=data_paths,
        datasets_weights=datasets_weights,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        filter_short_videos=filter_short_videos,
        filter_long_videos=filter_long_videos,
        duration=duration,
        shared_transform=shared_transform,
        transform=transform)

    logger.info('VideoDataset dataset created')
    if datasets_weights is not None:
        dist_sampler = DistributedWeightedSampler(
            dataset.sample_weights,
            num_replicas=world_size,
            rank=rank,
            shuffle=True)
    else:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=num_workers > 0)
    logger.info('VideoDataset unsupervised data loader created')

    return dataset, data_loader, dist_sampler


class VideoDataset(torch.utils.data.Dataset):
    """ Video classification dataset. """

    def __init__(
        self,
        data_paths,
        datasets_weights=None,
        frames_per_clip=16,
        frame_step=4,
        num_clips=1,
        transform=None,
        shared_transform=None,
        random_clip_sampling=True,
        allow_clip_overlap=False,
        filter_short_videos=False,
        filter_long_videos=int(10**9),
        duration=None,  # duration in seconds
    ):
        """
        Initialize VideoDataset.

        Args:
            data_paths (list): List of paths to csv files containing video paths and labels.
            datasets_weights (list, optional): List of weights for each dataset, used by
                downstream weighted video sampler. Defaults to None.
            frames_per_clip (int, optional): Number of frames in each video clip. Defaults to 16.
            frame_step (int, optional): Step size between frames in each clip. Defaults to 4.
            num_clips (int, optional): Number of clips to sample from each video. Defaults to 1.
            transform (callable, optional): Function to apply to each video clip. Defaults to None.
            shared_transform (callable, optional): Function to apply to the video as a whole.
                Defaults to None.
            random_clip_sampling (bool, optional): Whether to sample random clips from each video.
                Defaults to True.
            allow_clip_overlap (bool, optional): Whether to allow overlapping clips. Defaults to False.
            filter_short_videos (bool, optional): Whether to filter out videos that are shorter than
                the specified duration. Defaults to False.
            filter_long_videos (int, optional): Maximum length of videos in seconds. Defaults to 10**9.
            duration (int, optional): Duration of the evaluation clips. Defaults to None.
        """
        self.data_paths = data_paths
        self.datasets_weights = datasets_weights
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.transform = transform
        self.shared_transform = shared_transform
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap
        self.filter_short_videos = filter_short_videos
        self.filter_long_videos = filter_long_videos
        self.duration = duration

        if VideoReader is None:
            raise ImportError('Unable to import "decord" which is required to read videos.')

        # Load video paths and labels
        samples, labels = [], []
        self.num_samples_per_dataset = []
        
        data = pd.read_csv(data_paths)
        samples += list(data.values[:, 0])
        labels += list(data.values[:, 1])
        num_samples = len(data)
        self.num_samples_per_dataset.append(num_samples)

        # [Optional] Weights for each sample to be used by downstream
        # weighted video sampler
        self.sample_weights = None
        if self.datasets_weights is not None:
            self.sample_weights = []
            for dw, ns in zip(self.datasets_weights, self.num_samples_per_dataset):
                self.sample_weights += [dw / ns] * ns

        self.samples = samples
        self.labels = labels

    def __getitem__(self, index):
        """
        Retrieve a single video sample by index, apply transformations, and return its clips.

        This method attempts to load a video sample, retrying with a new random index if 
        the initial attempt fails to load a valid video. After a successful load, it applies 
        shared and individual transformations to the video, splits it into clips, and returns 
        the processed clips along with the corresponding label and clip indices.

        Args:
            index (int): Index of the video sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - buffer (list of tensors): List of transformed video clips.
                - label (int): Label associated with the video.
                - clip_indices (list): Indices of the frames in the original video.
        """

        sample = self.samples[index]

        # Keep trying to load videos until you find a valid sample
        loaded_video = False
        while not loaded_video:
            buffer, clip_indices = self.loadvideo_decord(sample)  # [T H W 3]
            loaded_video = len(buffer) > 0
            if not loaded_video:
                index = np.random.randint(self.__len__())
                sample = self.samples[index]

        # Label/annotations for video
        label = self.labels[index]

        def split_into_clips(video):
            """ Split video into a list of clips """
            fpc = self.frames_per_clip
            nc = self.num_clips
            return [video[i*fpc:(i+1)*fpc] for i in range(nc)]

        # Parse video into frames & apply data augmentations
        if self.shared_transform is not None:
            buffer = self.shared_transform(buffer)
        buffer = split_into_clips(buffer)
        if self.transform is not None:
            buffer = [self.transform(clip) for clip in buffer]

        return buffer, label, clip_indices

    def loadvideo_decord(self, sample):
        """ Load video content using Decord """

        fname = sample
        if not os.path.exists(fname):
            warnings.warn(f'video path not found {fname=}')
            return [], None

        _fsize = os.path.getsize(fname)
        if _fsize < 1 * 1024:  # avoid hanging issue
            warnings.warn(f'video too short {fname=}')
            return [], None
        if _fsize > self.filter_long_videos:
            warnings.warn(f'skipping long video of size {_fsize=} (bytes)')
            return [], None

        try:
            vr = VideoReader(fname, num_threads=-1, ctx=cpu(0))
        except Exception:
            return [], None

        fpc = self.frames_per_clip
        fstp = self.frame_step
        if self.duration is not None:
            try:
                fps = vr.get_avg_fps()
                fstp = int(self.duration * fps / fpc)
            except Exception as e:
                warnings.warn(e)
        clip_len = int(fpc * fstp)

        if self.filter_short_videos and len(vr) < clip_len:
            warnings.warn(f'skipping video of length {len(vr)}')
            return [], None

        vr.seek(0)  # Go to start of video before sampling frames

        # Partition video into equal sized segments and sample each clip
        # from a different segment
        partition_len = len(vr) // self.num_clips

        all_indices, clip_indices = [], []
        for i in range(self.num_clips):

            if partition_len > clip_len:
                # If partition_len > clip len, then sample a random window of
                # clip_len frames within the segment
                end_indx = clip_len
                if self.random_clip_sampling:
                    end_indx = np.random.randint(clip_len, partition_len)
                start_indx = end_indx - clip_len
                indices = np.linspace(start_indx, end_indx, num=fpc)
                indices = np.clip(indices, start_indx, end_indx-1).astype(np.int64)
                # --
                indices = indices + i * partition_len
            else:
                # If partition overlap not allowed and partition_len < clip_len
                # then repeatedly append the last frame in the segment until
                # we reach the desired clip length
                if not self.allow_clip_overlap:
                    indices = np.linspace(0, partition_len, num=partition_len // fstp)
                    indices = np.concatenate((indices, np.ones(fpc - partition_len // fstp) * partition_len,))
                    indices = np.clip(indices, 0, partition_len-1).astype(np.int64)
                    # --
                    indices = indices + i * partition_len

                # If partition overlap is allowed and partition_len < clip_len
                # then start_indx of segment i+1 will lie within segment i
                else:
                    sample_len = min(clip_len, len(vr)) - 1
                    indices = np.linspace(0, sample_len, num=sample_len // fstp)
                    indices = np.concatenate((indices, np.ones(fpc - sample_len // fstp) * sample_len,))
                    indices = np.clip(indices, 0, sample_len-1).astype(np.int64)
                    # --
                    clip_step = 0
                    if len(vr) > clip_len:
                        clip_step = (len(vr) - clip_len) // (self.num_clips - 1)
                    indices = indices + i * clip_step

            clip_indices.append(indices)
            all_indices.extend(list(indices))

        buffer = vr.get_batch(all_indices).asnumpy()
        return buffer, clip_indices

    def __len__(self):
        return len(self.samples)
