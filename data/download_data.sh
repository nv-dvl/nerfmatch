# Download image retrieval pairs
# https://drive.google.com/file/d/1-gOqAA0-sdiUZmQLRYlGprQdk7ttldfE/view?usp=drive_link
gdown 1-gOqAA0-sdiUZmQLRYlGprQdk7ttldfE
tar -xvzf pairs.tar.gz
rm pairs.tar.gz


# Download data annotations
# https://drive.google.com/file/d/1iHwsGwKXZWHxF_o9OMZUnGYKCopO-cKS/view?usp=drive_link
gdown 1iHwsGwKXZWHxF_o9OMZUnGYKCopO-cKS
tar -xvzf  annotations.tar.gz
rm  annotations.tar.gz


# Download SAM masks for training NeRF on Cambridge Landmarks
# https://drive.google.com/file/d/1lGgLcA6kZPJcOOrtMFhUml2KpOYio2MO/view?usp=drive_link
gdown 1lGgLcA6kZPJcOOrtMFhUml2KpOYio2MO
tar -xvzf  mask_preprocessed.tar.gz
rm  mask_preprocessed.tar.gz

