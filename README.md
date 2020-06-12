# Coco Detector

>Created my family's own dog detector for surviellance and habit recognition for future data analysis. Coco's presence is tracked around the house through cameras connected to a Raspberry Pi. Data will be analyzed from a local PostgreSQL database.

### Built With
- [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
- [OpenCV](https://github.com/opencv/opencv)
- [Twilio API](https://github.com/twilio/twilio-python)

### Features/Content
- Script run on Raspberry Pi 4
- Pi Camera used for constant surveillance
- Recognition written to PostgreSQL database
- Alerts via text messages
- Data analysis to determine habits
- All relevant files located under <b>models-master/research/object_detection</b>
- Trained model ~100 images was not accurate enough
    - Therefore, used pretrained model, "faster_rcnn_inception_v2_coco" from [COCO dataset](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
- <b>Note:</b> secret constants (credentials) denoted by CAPITAL LETTERS in separate file secrets.py



### To-do Checklist
- [x] Object Recognition
- [x] Trained model for Coco
- [x] Integrate Twilio API for customized alerts
- [x] Write to database
- [x] Migrate to Raspberry Pi
- [ ] Data Analysis 
- [ ] Live surveillance