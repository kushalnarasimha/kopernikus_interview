# Image Processing Interview Project

## Introduction
This project aims to filter out redundant frames from a collection of images based on certain criteria. It is particularly useful for scenarios where continuous image data needs to be analyzed, such as in surveillance systems or video processing pipelines.

## Features
- Sorting images by camera ID and timestamp.
- Filtering out non-essential frames based on image differencing method.
- Support for storing non-essential frames if required.
- Detection of damaged images.
- Detailed logging of processing results.

## Installation

```bash
git clone https://github.com/username/repository.git
pip install -r requirements.txt
python imaging_interview.py --path /path/to/image/folder --store_non_ess_frames
# Replace /path/to/image/folder with the path to the folder containing the image files. Use the --store_non_ess_frames flag to store non-essential frames.
```
