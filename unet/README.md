# U-Net: Convolutional Networks for Biomedical Image Segmentation

Paper link:

https://arxiv.org/pdf/1505.04597

Notes:

Implementation doesn't fully follow the paper due to the paper's age. For example original implementation didn't use paddings and used tiling of the image. From my experiments, padding=1 in convolutions worked much better
than trying to mirror dataset images at the borders. Overall architecture principle is unchanged.

Instructions:

1. Run `python unet/dummy_images.py`
2. Run `python unet/train.py`
3. Take a look at one of the segmented image(saved as `segmented_image.jpg`)