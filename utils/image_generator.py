import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import rotate


class ImageGenerator(object):
    def __init__(self, x, y):
        """
        :param x: A numpy array of input data. It has shape (num of samples, height, width, channels).
        :param y: A numpy vector of labels. I has shape (num of samples, ).
        """
        
        self.x = x.copy()
        self.y = y.copy() 
        self.N, self.height, self.width, self.channels = x.shape
        self.num_of_pixels_translated = None
        self.degree_of_rotation = None
        self.is_horizontal_flip = None
        self.is_vertical_flip = None
        self.is_add_noise=None
        
        self.translated = None
        self.rotated = None
        self.flipped = None
        self.added = None
        self.x_aug = self.x.copy()
        self.y_aug = self.y.copy()
        self.N_aug = self.N
        
        
    def create_aug_data(self):
        '''
        Combine all the data to form a augmented dataset 
        '''
        
        if self.translated:
            self.x_aug = np.vstack((self.x_aug,self.translated[0]))
            self.y_aug = np.hstack((self.y_aug,self.translated[1]))
        if self.rotated:
            self.x_aug = np.vstack((self.x_aug,self.rotated[0]))
            self.y_aug = np.hstack((self.y_aug,self.rotated[1]))
        if self.flipped:
            self.x_aug = np.vstack((self.x_aug,self.flipped[0]))
            self.y_aug = np.hstack((self.y_aug,self.flipped[1]))
        if self.added:
            self.x_aug = np.vstack((self.x_aug,self.added[0]))
            self.y_aug = np.hstack((self.y_aug,self.added[1]))
            
        print("Size of training data:{}".format(self.N_aug))
        
    def next_batch_gen(self, batch_size, shuffle=True):
        """
        A python generator function that yields a batch of data infinitely.
        :param batch_size: The number of samples to return for each batch.
        :param shuffle: If True, shuffle the entire dataset after every sample has been returned once.
                        If False, the order or data samples stays the same.
        :return: A batch of data with size (batch_size, width, height, channels).
        """
        num_of_samples=self.num_of_samples
        total_batch=num_of_samples//batch_size
        batch_count=0

        x=self.x
        y=self.y

        while True:
            if(batch_count<total_batch):
                batch_count+=1
                yield (x[batch_count*batch_size:(batch_count+1)*batch_size,:,:,:],y[batch_count*batch_size:(batch_count+1)*batch_size])
            else:

                #shuffle()
                index=np.random.choice(self.num_of_samples,self.num_of_samples,replace=False)
                #np.random.shuffle(x)
                self.x=x[index]
                self.y=y[index]
                #reset batch_count
                batch_count=0
        
        
    def show(self, images):
        """
        Plot the top 16 images (index 0~15) for visualization.
        :param images: images to be shown
        """
        fig = plt.figure(figsize=(10, 10))
        for i in range(16):
            ax = fig.add_subplot(4,4,i+1)
            plt.imshow(images[i,:].reshape(28,28), 'gray')
            ax.axis('off')
            
    def translate(self, shift_height, shift_width):
        """
        Translate self.x by the values given in shift.
        :param shift_height: the number of pixels to shift along height direction. Can be negative.
        :param shift_width: the number of pixels to shift along width direction. Can be negative.
        :return translated: translated dataset
        """

        self.trans_height += shift_height
        self.trans_width += shift_width
        translated = np.roll(self.x.copy(), (shift_width, shift_height), axis=(1, 2))
        print('Current translation: ', self.trans_height, self.trans_width)
        self.translated = (translated,self.y.copy())
        self.N_aug += self.N
        
        return translated
    
    
    
    def rotate(self, angle=0.0):
            """
            Rotate self.x by the angles (in degree) given.
            :param angle: Rotation angle in degrees.
            :return rotated: rotated dataset
            - https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.ndimage.interpolation.rotate.html
            """

            self.dor = angle
            rotated = rotate(self.x.copy(), angle,reshape=False,axes=(1, 2))
            print('Currrent rotation: ', self.dor)
            self.rotated = (rotated, self.y.copy())
            self.N_aug += self.N
            return rotated

    def flip(self, mode='h'):
        """
        Flip self.x according to the mode specified
        :param mode: 'h' or 'v' or 'hv'. 'h' means horizontal and 'v' means vertical.
        :return flipped: flipped dataset
        """
        
        x = self.x.copy()
        img = x.reshape(self.N, 28, 28)
        if mode == 'h':
            img_flipped = np.flip(img, axis=2)
            self.is_horizontal_flip = True
        if mode == 'v':
            img_flipped = np.flip(img, axis=2)
            self.is_vertical_flip = True
 
        return img_flipped.reshape(self.N, -1)

    
    def add_noise(self, portion, amplitude):
        """
        Add random integer noise to self.x.
        :param portion: The portion of self.x samples to inject noise. If x contains 10000 sample and portion = 0.1,
                        then 1000 samples will be noise-injected.
        :param amplitude: An integer scaling factor of the noise.
        :return added: dataset with noise added
        """
        x = self.x.copy()
        img = x.reshape(self.N, 28, 28)
        num_add_noise = int(self.N * portion)
        
        img_portion = img[0:num_add_noise,:,:]
        noise_img = img_portion.copy()
        
        for i in range(num_add_noise):
            noise_img[i,:,:] = img_portion[i,:,:] + np.random.randn(28,28) * amplitude
        self.is_add_noise = True

        return noise_img.reshape(num_add_noise, -1)