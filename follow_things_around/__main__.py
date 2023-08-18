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

import argparse
import json
import logging
import os
import os.path
import shutil
import sys
from typing import List

import requests

import follow_things_around
import follow_things_around as fta
import follow_things_around.via


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


TRACKING_MODEL_URL = "https://thor.robots.ox.ac.uk/models/staging/chimp-tracking/tracking-model-20181031_e45.pth"


def score_value(arg: str) -> float:
    x = float(arg)
    if x < 0 or x > 1:
        raise ValueError("not a value between 0 and 1")
    return x


def main(argv: List[str]) -> int:
    logging.basicConfig()
    argv_parser = argparse.ArgumentParser()
    argv_parser.add_argument(
        "--no-reuse-frames-dir",
        action="store_true",
        help=(
            "By default, if an existing frames directory exists, the images in"
            " that directory.  This option makes it so that the directory is"
            " removed (and all of its contents), forcing the frame images to"
            " be recreated."
        ),
    )
    argv_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log info level messages (see also --debug option)",
    )
    argv_parser.add_argument(
        "--debug",
        action="store_true",
        help="Log debug level messages (see also --verbose option)",
    )
    argv_parser.add_argument(
        "--detection-threshold",
        type=score_value,
        default=0.6,
        help="Detections with scores below this value are ignored"
    )
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

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    frames_dir = os.path.join(args.results_dir, "frames")
    detections_via_fpath = os.path.join(
        args.results_dir, "detections-via.json"
    )
    results_via_fpath = os.path.join(
        args.results_dir, "results-via-project.json"
    )
    results_csv_fpath = os.path.join(args.results_dir, "results.csv")
    tracks_video_fpath = os.path.join(args.results_dir, "tracks.mp4")

    detection_model_config_url = THING_TO_MODEL_CONFIG[args.what]["config-url"]
    detection_class_idx = THING_TO_MODEL_CONFIG[args.what]["class-idx"]

    follow_things_around.DEFAULT_DEVICE = "cuda"
    follow_things_around.FRAMES_DIR = frames_dir
    follow_things_around.MATCH_OVERLAP_THRESHOLD = 0.2
    follow_things_around.NONMATCH_TRACKING_THRESHOLD = 0.9
    follow_things_around.UNKNOWN_TRACK_ID_MARKER = -1
    follow_things_around.USE_GPU = True
    follow_things_around.VIDEO_FILE = args.video_fpath
    follow_things_around.FFMPEG_LOG_LEVEL = "warning"

    if args.no_reuse_frames_dir:
        try:
            shutil.rmtree(frames_dir)
        except FileNotFoundError:
            pass  # dir does not exist, this is fine
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir, exist_ok=True)
        fta.ffmpeg_video_to_frames(args.video_fpath, frames_dir)

    dataset = follow_things_around.FramesDirDataset(frames_dir)
    detections = follow_things_around.detect(
        dataset,
        detection_model_config_url,
        detection_class_idx,
        args.detection_threshold,
    )
    with open(detections_via_fpath, "w") as fh:
        json.dump(
            follow_things_around.via.detections_to_via2(dataset, detections),
            fh,
        )

    tracks = follow_things_around.track(
        dataset, detections, TRACKING_MODEL_URL
    )
    tracks.export_via_project(
        results_via_fpath,
        config={
            "frame_img_dir": dataset.frames_dir,
            "via_project_name": "",
        },
    )
    tracks.export_plain_csv(results_csv_fpath, {})

    fta.make_video_with_tracks(
        args.video_fpath, tracks_video_fpath, frames_dir, results_csv_fpath
    )
    _logger.info("Video file with tracks created '%s'", tracks_video_fpath)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
