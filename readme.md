
Run this command to clone this repository.
```
git clone https://github.com/Lukeasargen/targetClassifier-v2.git
cd targetClassifier-v2
```

Create a new conda environment with juypter kernel.
```
conda create --name uas_vision python=3.9 -y
conda activate uas_vision
conda install ipykernel jupyter -y
python -m ipykernel install --user --name uas_vision --display-name "uas_vision" -y
```

Get pytorch installed. Command generated here: https://pytorch.org/
```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y
```

Get the rest of the requirements.
```
pip install -r requirements.txt
```
