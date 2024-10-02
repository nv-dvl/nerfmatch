# Download image retrieval pairs
# https://drive.google.com/file/d/1ZhBjKfP6jVOUvEffX5SUosUdBYpydaBS/view?usp=drive_link
gdown 1ZhBjKfP6jVOUvEffX5SUosUdBYpydaBS
tar -xvzf pairs.tar.gz
rm pairs.tar.gz


# Download data annotations
# https://drive.google.com/file/d/1iMuZU5GW0i4ySOuVDKWYdsehRNdTd-tE/view?usp=drive_link
gdown 1iMuZU5GW0i4ySOuVDKWYdsehRNdTd-tE
tar -xvzf  annotations.tar.gz
rm  annotations.tar.gz


# Download SAM masks for training NeRF on Cambridge Landmarks
# https://drive.google.com/file/d/1XXCJSRXkaw1m9jYQw9Y45BCjp7Ei2a99/view?usp=drive_link
gdown 1XXCJSRXkaw1m9jYQw9Y45BCjp7Ei2a99
tar -xvzf  mask_preprocessed.tar.gz
rm  mask_preprocessed.tar.gz

