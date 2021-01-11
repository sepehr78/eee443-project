
# EEE 443 Neural Networks Image Captioning Project
**Brief summary of the project:**
In  this  work,  we  experimented  with  various  image  captioning  architectures  for  encoding  images and decoding their captions, including a simple CNN and RNN model, a model with attention,  and a fully convolutional  model. We train and test our models on the provided COCO2014 dataset, discussing in detail both the image and caption pre-processing needed. We then discuss our successes and shortcomings and compare our results with those in the literature. We were able to observe that the image captioning model with attention not only provided the best results out of all the other tested models, but it also outperformed the paper that it is based on.

**Final results**
Note that the BLEU scores below correspond to testing on the COCO2014-val dataset.

| Model                       | BLEU-1        | BLEU-2        | BLEU-3        | BLEU-4        |
|-----------------------------|---------------|---------------|---------------|---------------|
| Naive ResNet                | 68.5          | 49.6          | 35.1          | 24.1          |
| Attention w/o GloVe ($k=1$) | 71.0          | 53.8          | 40.0          | 28.8          |
| Attention w/ GloVe ($k=1$)  | 70.4          | 53.2          | 39.3          | 28.3          |
| Attention w/o GloVe ($k=3$) | 72.2          | 55.3          | **42.0** | **31.5** |
| Attention w/ GloVe ($k=3$)  | **72.4**          | **55.4**          | 42.0          | 31.4          |



**All trained models are available here**
https://drive.google.com/drive/folders/1EfcqQ0tsOArqxL-o3aStwtm56IaAy0CP

**Pre-processed COCO2014 dataset (85/15 train/val split using COCO2014-train and testing using COCO2014-val) is available here** 
https://1drv.ms/u/s!Av7-zYTFLro0q8kG8uAtUL25qEpQdA?e=qYsSgn

**To test on images using our best model (Attention w/o GloVe)**
1. Download the trained Attention model from https://drive.google.com/drive/folders/1EfcqQ0tsOArqxL-o3aStwtm56IaAy0CP.
2. Then, open show_tell/caption.py and change img_path to path to the image you want to test. Then, simply run the script. 
3. Top captions will be printed and also the attention will be visualized for each word.

**Model and code directory mapping**
* Naive CNN+RNN -> /naive
* Attention model w/ and w/o GloVe -> /show_tell
* Convolutional captioning model -> /convcap-REFINED
**To train any of our models**
4. Open the respective model's training script.
6. Ensure that the data path is correctly set to where you downloaded the pre-processed dataset (link above).
7. The script automatically runs on GPU if it detects any. 
