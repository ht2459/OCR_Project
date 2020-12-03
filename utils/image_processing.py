import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy import ndimage


class ImageGenerator(object):
    def __init__(self, image):
        """
        image: An input image.
        """
        self.image = cv2.imread(image)
        self.height, self.width, self.channels = self.image.shape
    
       
    def scale_image(self, image, ratio=0.25, position=(0, 0)):
        """
        param image: Image to be scaled
        param ratio: Optional Float specifying the scaling ratio of the input image.
                     Shrink: 0 - 1
                     Stretch: 1 - Inf
        param source: Optional tuple for the upper left corner.
        """
        
        image = Image.open(image)
        width, height = image.size
        
        width_scaled = int(min(width, height) * ratio)
        width_percent = (width_scaled / float(width))
        height_scaled = int((float(height) * float(width_percent)))

        image_scaled = image.resize((width_scaled, height_scaled))
        
        transparent_image = Image.new('RGB', (self.width, self.height), (255,255,255))
        transparent_image.paste(image_scaled, position)
        transparent_image.convert("RGBA")      
        transparent_data = transparent_image.getdata()
        output_image = []
                    
        for item in transparent_data:
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                output_image.append((255, 255, 255, 0))
            else:
                output_image.append(item)
                
        transparent_image.putdata(transparent_data)
          
        return transparent_image
    
    
    def overlay_image(self, image, style):
        """
        param image: Image to be processed
        param style: Optional String specifying what pattern need to be emphasized
        """
        if style == 'background':
            pass
        ##待解决1：处理印章和文字重合部分，尽可能模拟真实的重合
        ##待解决2：印章拖影
            
        
        
        
        

        