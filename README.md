# XRAY Detector

This project aims to detect weapon from Xray images. The whole pipeline is built on AWS sagemaker. For more information, please refer to https://aws.amazon.com/sagemaker/

## Annotation

For this project. I used LabelBox for annotating images. URL: https://labelbox.com/. There is some scripts inside this repo to help converting different annotation format to json format, which is used for training at sagemaker later.

## Training

All the relevant training process can be found at https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/object_detection_pascalvoc_coco/object_detection_image_json_format.ipynb
Note: A sample json annotation format file is uploaded in this repo. Use it as a reference.

## Deployment

The model can be deploy in 2 ways.

1. Deploy using sagemaker endpoint configuration. See how it does: https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-hosting.html
-Run <code> python streamer_with_cloud_model.py </code> to run main program.  

2. Deploy locally. The object detection model is trained using mxnet and some of the layers needs to be removed before entering deployment mode. See https://discuss.mxnet.io/t/deploy-sagemaker-trained-model-locally/1934 and https://github.com/zhreshold/mxnet-ssd#convert-model-to-deploy-mode for more details. 
-Run <code>python streamer.py</code> to run main program.

My model links (Processed and ready to deploy): https://drive.google.com/open?id=1zrKmQUDI7S0acSgavSw5BEs-62e3txt- 
