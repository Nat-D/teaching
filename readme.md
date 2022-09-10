# Install Library

1. install conda
	- create conda environment 
    `conda create -n learn python=3.9`
	- activate the conda environment `conda activate learn`

2. install pytorch (https://pytorch.org/get-started/locally/)

    without gpu:
    -  `conda install pytorch torchvision torchaudio cpuonly -c pytorch`
	
	or with gpu:
	- `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`
	
3. install pandas
	- `conda install pandas`

4. install tensorboard
	- `pip install tensorboard`

5. (segmentation) install albumentation
	- `conda install -c conda-forge imgaug`
	- `conda install -c conda-forge albumentations`

6. (seq2seq) install spacy
	- `pip install -U spacy`
	- `python -m spacy download en_core_web_sm`
	- `python -m spacy download de_core_news_sm`
7. (seq2seq) install torchtext and torch-data
	- `conda install -c pytorch torchtext torchdata`


# Download Dataset

`https://drive.google.com/drive/folders/1feL5X6epYQiGaT-dHpgV7yWU7_ZthVVI?usp=sharing`


# References

1. https://github.com/aladdinpersson/Machine-Learning-Collection
2. https://pytorch.org/tutorials/
3. https://www.kaggle.com/competitions/dogs-vs-cats
4. http://www.cvlibs.net/datasets/kitti/
