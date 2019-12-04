# Multi-Label-Classification-with-CNN-and-RNN
This repo is our research for multi-label classification for artworks in iMet. Here we implemented and modifed 9 various neural networks, including attention, baseline, cbam, rethink, rethink1, seresnext1/2/3/4.

The data comes from research competition on Kaggle, which is a multi-label classifictaion problem for artwork in iMet. And finally we reached F2 score at 0.63.

The significant contribution of our research is the comparison between the different networks with CNN and RNN and innovative ideas of Bi-stage Rethinking network with semantic similarity of labels, which largely improves the accuracy for such problems.

Because the memory limitation, we do not upload the pertained weight(.pth). For more details, you can contact me. The codes are also deployed on GPU.

### attention: adapting attention mechanism in multi-label classification
### baseline: ResNet baseline
### cbam: using Cbam50 with and without similarity loss
### rethink; Bi-stage Rethink Net with similarity loss code and a embedding filter mask
### rethink1; Bi-stage Rethink Net with similarity loss code and without a embedding filter mask
### ser1: seResNext101 focal_similarity loss
### ser2: seResNext101 focal  loss
### ser3: seResNext101similarity loss
### ser4: seResNext101 BCE loss
### util create embedding, similarity matrix and do threshold modification

# Acknowledgement
This is a joint work with Lujia Bai and Xinyi Jiang.

# Citation
For the researcher in the deep learning field, our proposed neural networks may be helpful for you. You can try our networks and please kindly cite this repository and the authors if you think this is helpful.
