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

import os.path
from typing import Dict

from follow_things_around import DatasetDetections, Detection, FramesDirDataset


def empty_via2_project() -> Dict:
    return {
        "_via_attributes": {
            "file": {},
            "region": {},
        },
        "_via_data_format_version": "2.0.10",
        "_via_image_id_list": [],
        "_via_img_metadata": {},
        "_via_settings": {
            "core": {
                "buffer_size": 18,
                "default_filepath": "",
                "filepath": {},
            },
            "project": {
                "name": "",
            },
            "ui": {
                "annotation_editor_fontsize": 0.8,
                "annotation_editor_height": 25,
                "image": {
                    "on_image_annotation_editor_placement": "NEAR_REGION",
                    "region_color": "__via_default_region_color__",
                    "region_label": "__via_region_id__",
                    "region_label_font": "10px Sans",
                },
                "image_grid": {
                    "img_height": 80,
                    "rshape_fill": "none",
                    "rshape_fill_opacity": 0.3,
                    "rshape_stroke": "yellow",
                    "rshape_stroke_width": 2,
                    "show_image_policy": "all",
                    "show_region_shape": True,
                },
                "leftsidebar_width": 18,
            },
        },
    }


def empty_via2_project_detections() -> Dict:
    project = empty_via2_project()
    project["_via_attributes"]["file"]["frame_id"] = {
        "default_value": "",
        "description": "Unique Frame ID (effectively frame number)",
        "type": "text",
    }
    project["_via_attributes"]["region"]["box_id"] = {
        "default_value": "",
        "description": "Unique ID of a specific region/box within one frame",
        "type": "text",
    }
    project["_via_attributes"]["region"]["detection_score"] = {
        "default_value": "",
        "description": "Confidence score of the detection",
        "type": "text",
    }
    return project


def detection_to_via2_shape(detection: Detection):
    return {
        "name": "rect",
        "x": detection.x,
        "y": detection.y,
        "width": detection.w,
        "height": detection.h,
    }


def detections_to_via2(
    dataset: FramesDirDataset, detections: DatasetDetections
) -> Dict:
    project = empty_via2_project_detections()

    shot_id = "0"
    assert len(detections) == len(dataset)
    for frame_fname, frame_detections in zip(dataset.frames, detections):
        regions = []
        for box_idx, detection in enumerate(frame_detections):
            regions.append(
                {
                    "shape_attributes": detection_to_via2_shape(detection),
                    "region_attributes": {
                        "box_id": str(box_idx),
                        "detection_score": str(detection.score),
                    },
                }
            )

        frame_fpath = os.path.join(dataset.frames_dir, frame_fname)
        img_metadata = {
            "filename": frame_fname,
            "file_attributes": {
                "frame_id": frame_fname,
                "shot_id": shot_id,
            },
            "regions": regions,
            "size": os.path.getsize(frame_fpath),
        }
        project["_via_img_metadata"][frame_fname] = img_metadata
        project["_via_image_id_list"].append(frame_fname)

    return project
