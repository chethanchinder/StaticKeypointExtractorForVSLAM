import numpy as np
from pixellib.torchbackend.instance import instanceSegmentation

ins = instanceSegmentation()
ins.load_model("weights/pointrend_resnet50.pkl",detection_speed='rapid')

results, image=ins.segmentImage("images/sample.jpeg", show_bboxes=False)

masks=results["masks"]
masks = np.moveaxis(masks, -1,0)
print("result masks :", masks.shape)




# from pixellib.semantic import semantic_segmentation
#
# segment_image = semantic_segmentation()
# segment_image.load_ade20k_model("weights/deeplabv3_xception65_ade20k.h5")
# segment_image.segmentAsAde20k("images/sample.jpeg", output_image_name= "outputs/sample.jpeg")

