# Follow Things Around v2 beta

<img src="tracks.gif" alt="Animal track predictions from Follow Things Around v2 on a panther video" />

**[Follow Things Around](https://www.robots.ox.ac.uk/~vgg/software/follow-things-around/) is a program for easily detecting and tracking animals in videos, no programming skills required.**

You can open this notebook in Google Colab using the button below:

<a target="_blank" href="https://colab.research.google.com/github/ox-vgg/follow-things-around/blob/v2_beta1/follow-things-around-v2-beta.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Models supported

In Follow Things Around v2, we support the following detection models and animal types:

| Detection model                                                      | What it can detect                                                             |
|----------------------------------------------------------------------|--------------------------------------------------------------------------------|
| [MegaDetectorV6](https://github.com/microsoft/CameraTraps/tree/main) | Various animals, people, and vehicles                                          |
| Our chimpanzee face detector (coming soon)                           | Chimpanzee faces                                                               |
| Our primate/chimp body detector (coming soon)                        | Primates/chimpanzees (more details coming soon)                                |
| COCO-pretrained model (coming soon)                                  | Birds, cats, dogs, horses, sheep, cows, elephants, bears, zebras, and giraffes |

## v2 updates

This notebook is an early beta version of Follow Things Around v2. If you would like to receive updates when the v2 version is officially released, please [sign up here](https://docs.google.com/forms/d/e/1FAIpQLSdT0sa4AsRwo1m0qGDhr7GI9t2Z-A8Vko7bgDERdbh-MHVnUA/viewform).

## How it works

Follow Things Around works by performing a two-step approach called
"tracking by detection".  First, it detects the "things" of interest
in each frame of a video, and then it tracks those objects by
associating detections across frames to form individual tracks, while
also filling in gaps from the detector.

## About

If you have any questions, feedback, or issues, please contact [Horace
Lee](mailto:horacelee@robots.ox.ac.uk) or David Pinto. For more details, please visit our [software webpage](https://www.robots.ox.ac.uk/~vgg/software/follow-things-around/).
