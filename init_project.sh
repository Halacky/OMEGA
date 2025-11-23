#!/bin/bash

PROJECT_DIR="/home/kirill/projects_2/folium/NIR/OMEGA/30_10"

mkdir -p $PROJECT_DIR/config
mkdir -p $PROJECT_DIR/data
mkdir -p $PROJECT_DIR/models
mkdir -p $PROJECT_DIR/processing
mkdir -p $PROJECT_DIR/training
mkdir -p $PROJECT_DIR/evaluation
mkdir -p $PROJECT_DIR/visualization
mkdir -p $PROJECT_DIR/utils
mkdir -p $PROJECT_DIR/scripts

touch $PROJECT_DIR/__init__.py
touch $PROJECT_DIR/config/__init__.py
touch $PROJECT_DIR/data/__init__.py
touch $PROJECT_DIR/models/__init__.py
touch $PROJECT_DIR/processing/__init__.py
touch $PROJECT_DIR/training/__init__.py
touch $PROJECT_DIR/evaluation/__init__.py
touch $PROJECT_DIR/visualization/__init__.py
touch $PROJECT_DIR/utils/__init__.py
touch $PROJECT_DIR/scripts/__init__.py

touch $PROJECT_DIR/config/base.py
touch $PROJECT_DIR/config/experiments.py

touch $PROJECT_DIR/data/loaders.py

touch $PROJECT_DIR/models/cnn1d.py

touch $PROJECT_DIR/processing/segmentation.py
touch $PROJECT_DIR/processing/windowing.py
touch $PROJECT_DIR/processing/splitting.py

touch $PROJECT_DIR/training/trainer.py
touch $PROJECT_DIR/training/datasets.py

touch $PROJECT_DIR/evaluation/metrics.py
touch $PROJECT_DIR/evaluation/rotation.py

touch $PROJECT_DIR/visualization/base.py
touch $PROJECT_DIR/visualization/rotation.py

touch $PROJECT_DIR/utils/logging.py
touch $PROJECT_DIR/utils/artifacts.py

touch $PROJECT_DIR/scripts/train.py
touch $PROJECT_DIR/scripts/evaluate_rotation.py

touch $PROJECT_DIR/main.py
touch $PROJECT_DIR/requirements.txt
touch $PROJECT_DIR/README.md
