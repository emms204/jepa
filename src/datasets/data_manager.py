# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger


_GLOBAL_SEED = 0
logger = getLogger()


def init_data(
    batch_size,
    transform=None,
    shared_transform=None,
    data='ImageNet',
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    copy_data=False,
    drop_last=True,
    tokenize_txt=True,
    subset_file=None,
    clip_len=8,
    frame_sample_rate=2,
    duration=None,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,
    filter_long_videos=int(1e9),
    decode_one_clip=True,
    datasets_weights=None,
    persistent_workers=False,
    repeat_wds=False,
    ipe=300,
    log_dir=None,
):

    """
    Initializes data loading for image or video datasets.

    Args:
        batch_size (int): Number of samples per batch.
        transform (callable, optional): Transformation to apply to each sample.
        shared_transform (callable, optional): Transformation to apply to shared data.
        data (str, optional): Type of dataset ('ImageNet', 'inat21', 'places205', or 'videodataset').
        collator (callable, optional): Function to merge a list of samples to form a mini-batch.
        pin_mem (bool, optional): Whether to pin memory during DataLoader.
        num_workers (int, optional): Number of subprocesses to use for data loading.
        world_size (int, optional): Number of processes involved in distributed data loading.
        rank (int, optional): Rank of the current process in distributed data loading.
        root_path (str, optional): Root directory path of the dataset.
        image_folder (str, optional): Directory containing images.
        training (bool, optional): Whether to load training data.
        copy_data (bool, optional): Whether to copy data to a new location.
        drop_last (bool, optional): Whether to drop the last incomplete batch.
        tokenize_txt (bool, optional): Whether to tokenize text data.
        subset_file (str, optional): Path to a file specifying a subset of the dataset.
        clip_len (int, optional): Number of frames per video clip.
        frame_sample_rate (int, optional): Frame sampling rate for video.
        duration (float, optional): Duration of video clips in seconds.
        num_clips (int, optional): Number of clips per video.
        random_clip_sampling (bool, optional): Whether to randomly sample clips.
        allow_clip_overlap (bool, optional): Whether to allow overlapping clips.
        filter_short_videos (bool, optional): Whether to filter out short videos.
        filter_long_videos (int, optional): Maximum duration of videos to include.
        decode_one_clip (bool, optional): Whether to decode only one clip per video.
        datasets_weights (list, optional): Weights for different datasets.
        persistent_workers (bool, optional): Whether workers are persistent across epochs.
        repeat_wds (bool, optional): Whether to repeat web datasets.
        ipe (int, optional): Iterations per epoch.
        log_dir (str, optional): Directory for logging information.

    Returns:
        tuple: A tuple containing the DataLoader, DistributedSampler, and the dataset.
    """

    if (data.lower() == 'imagenet') \
            or (data.lower() == 'inat21') \
            or (data.lower() == 'places205'):
        from src.datasets.image_dataset import make_imagedataset
        dataset, data_loader, dist_sampler = make_imagedataset(
            transform=transform,
            batch_size=batch_size,
            collator=collator,
            pin_mem=pin_mem,
            training=training,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            persistent_workers=persistent_workers,
            copy_data=copy_data,
            drop_last=drop_last,
            subset_file=subset_file)

    elif data.lower() == 'videodataset':
        from src.datasets.video_dataset import make_videodataset
        dataset, data_loader, dist_sampler = make_videodataset(
            data_paths=root_path,
            batch_size=batch_size,
            frames_per_clip=clip_len,
            frame_step=frame_sample_rate,
            duration=duration,
            num_clips=num_clips,
            random_clip_sampling=random_clip_sampling,
            allow_clip_overlap=allow_clip_overlap,
            filter_short_videos=filter_short_videos,
            filter_long_videos=filter_long_videos,
            shared_transform=shared_transform,
            transform=transform,
            datasets_weights=datasets_weights,
            collator=collator,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            drop_last=drop_last,
            log_dir=log_dir)

    return (data_loader, dist_sampler)
