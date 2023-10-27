# Create conda environment
conda env create -f environment.yml
conda activate prisma
./download_models.sh

# Install 3rd party models

## 1. Install mmdet
cd bands/mmdet
pip install -r requirements/build.txt
pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"
pip install -v -e .  # or "python setup.py develop"
