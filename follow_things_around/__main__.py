#!/usr/bin/env python

## Copyright 2023 David Miguel Susano Pinto <pinto@robots.ox.ac.uk>
##
## Licensed under the Apache License, Version 2.0 (the "License"); you
## may not use this file except in compliance with the License.  You
## may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
## implied.  See the License for the specific language governing
## permissions and limitations under the License.

## Because "follow_things_around" is a package and not a module, a
## module named "__main__" is required so that we can call it like:
##
##     python -m follow_things_around ...

import requests
import argparse
import logging
import os
import sys
import tempfile
import os.path
import shutil
from typing import List

import follow_things_around


_logger = logging.getLogger(__name__)


## TODO: the list of available configs is duplicated in the notebook

THING_TO_TRACK = "Chimpanzee faces"  # @param ["Chimpanzee faces", "Chimpanzee bodies", "Birds", "Cats", "Dogs", "Horses", "Sheep", "Cows", "Elephants", "Bears", "Zebras", "Giraffes"]

THING_TO_MODEL_CONFIG = {
    "Chimpanzee faces": {
        "config-url": "https://thor.robots.ox.ac.uk/models/staging/chimp-tracking/faster_rcnn_R_50_FPN_1x-CFbootstrap.yaml",
        "class-idx": 0,
    },
    "Chimpanzee bodies": {
        "config-url": "https://thor.robots.ox.ac.uk/models/staging/chimp-tracking/faster_rcnn_R_50_FPN_1x-imdb_5k_sup.yaml",
        "class-idx": 0,
    },
    "Birds": {
        "config-url": "https://thor.robots.ox.ac.uk/models/staging/chimp-tracking/faster_rcnn_R_50_FPN_1x-coco2017.yaml",
        "class-idx": 14,
    },
    "Cats": {
        "config-url": "https://thor.robots.ox.ac.uk/models/staging/chimp-tracking/faster_rcnn_R_50_FPN_1x-coco2017.yaml",
        "class-idx": 15,
    },
    "Dogs": {
        "config-url": "https://thor.robots.ox.ac.uk/models/staging/chimp-tracking/faster_rcnn_R_50_FPN_1x-coco2017.yaml",
        "class-idx": 16,
    },
    "Horses": {
        "config-url": "https://thor.robots.ox.ac.uk/models/staging/chimp-tracking/faster_rcnn_R_50_FPN_1x-coco2017.yaml",
        "class-idx": 17,
    },
    "Sheep": {
        "config-url": "https://thor.robots.ox.ac.uk/models/staging/chimp-tracking/faster_rcnn_R_50_FPN_1x-coco2017.yaml",
        "class-idx": 18,
    },
    "Cows": {
        "config-url": "https://thor.robots.ox.ac.uk/models/staging/chimp-tracking/faster_rcnn_R_50_FPN_1x-coco2017.yaml",
        "class-idx": 19,
    },
    "Elephants": {
        "config-url": "https://thor.robots.ox.ac.uk/models/staging/chimp-tracking/faster_rcnn_R_50_FPN_1x-coco2017.yaml",
        "class-idx": 20,
    },
    "Bears": {
        "config-url": "https://thor.robots.ox.ac.uk/models/staging/chimp-tracking/faster_rcnn_R_50_FPN_1x-coco2017.yaml",
        "class-idx": 21,
    },
    "Zebras": {
        "config-url": "https://thor.robots.ox.ac.uk/models/staging/chimp-tracking/faster_rcnn_R_50_FPN_1x-coco2017.yaml",
        "class-idx": 22,
    },
    "Giraffes": {
        "config-url": "https://thor.robots.ox.ac.uk/models/staging/chimp-tracking/faster_rcnn_R_50_FPN_1x-coco2017.yaml",
        "class-idx": 23,
    },
}


TRACKING_MODEL = "https://thor.robots.ox.ac.uk/models/staging/chimp-tracking/tracking-model-20181031_e45.pth"


def main(argv: List[str]) -> int:
    logging.basicConfig()
    argv_parser = argparse.ArgumentParser()
    argv_parser.add_argument(
        "what",
        help="What to track",
    )
    argv_parser.add_argument(
        "video_fpath",
        help="Filepath for video",
    )
    argv_parser.add_argument(
        "results_dir",
        help="Path for results dir",
    )
    args = argv_parser.parse_args(argv[1:])

    frames_dir = os.path.join(args.results_dir, "frames")
    results_via_fpath = os.path.join(
        args.results_dir, "results-via-project.json"
    )
    results_csv_fpath = os.path.join(args.results_dir, "results.csv")
    tracks_video_fpath = os.path.join(args.results_dir, "tracks.mp4")

    detection_model_config = THING_TO_MODEL_CONFIG[args.what]["config-url"]
    detection_class_idx = THING_TO_MODEL_CONFIG[args.what]["class-idx"]
    detection_threshold = 0.6

    follow_things_around.DEFAULT_DEVICE = "cuda"
    follow_things_around.FRAMES_DIR = frames_dir
    follow_things_around.MATCH_OVERLAP_THRESHOLD = 0.2
    follow_things_around.NONMATCH_TRACKING_THRESHOLD = 0.9
    follow_things_around.UNKNOWN_TRACK_ID_MARKER = -1
    follow_things_around.USE_GPU = True
    follow_things_around.VIDEO_FILE = args.video_fpath
    follow_things_around.FFMPEG_LOG_LEVEL = "warning"

    ## TODO: this is duplicated in the notebook
    def local_path_for_model(path: str) -> str:
        if path.startswith("https://"):
            downloaded_fh = tempfile.NamedTemporaryFile(delete=False)
            r = requests.get(path)
            downloaded_fh.write(r.content)
            downloaded_fh.flush()
            return downloaded_fh.name
        else:
            return path

    detection_model_config_path = local_path_for_model(detection_model_config)
    tracking_model_path = local_path_for_model(TRACKING_MODEL)

    os.makedirs(frames_dir, exist_ok=True)
    follow_things_around.ffmpeg_video_to_frames(args.video_fpath, frames_dir)

    dataset = follow_things_around.FramesDirDataset(frames_dir)
    detections = follow_things_around.detect(
        dataset,
        detection_model_config_path,
        detection_class_idx,
        detection_threshold,
    )

    tracks = follow_things_around.track(
        dataset, detections, tracking_model_path
    )
    tracks.export_via_project(
        results_via_fpath,
        config={
            "frame_img_dir": dataset.frames_dir,
            "via_project_name": "",
        },
    )
    tracks.export_plain_csv(results_csv_fpath, {})

    ## TODO: this is duplicated in the notebook
    with tempfile.TemporaryDirectory() as out_frames_dir:
        tmp_tracks_fpath = os.path.join(out_frames_dir, "tracks.mp4")
        follow_things_around.make_frames_with_tracks(
            results_csv_fpath, frames_dir, out_frames_dir
        )
        follow_things_around.ffmpeg_video_from_frames_and_video(
            out_frames_dir, args.video_fpath, tmp_tracks_fpath
        )
        shutil.move(tmp_tracks_fpath, tracks_video_fpath)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
