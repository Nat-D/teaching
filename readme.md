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

5. install albumentation
	- `pip install -U albumentations`


# Download Dataset

`https://drive.google.com/drive/folders/1feL5X6epYQiGaT-dHpgV7yWU7_ZthVVI?usp=sharing`


# References

1. https://github.com/aladdinpersson/Machine-Learning-Collection
2. https://pytorch.org/tutorials/
3. https://www.kaggle.com/competitions/dogs-vs-cats
4. http://www.cvlibs.net/datasets/kitti/
