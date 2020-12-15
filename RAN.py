from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, AveragePooling2D, Input, ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization, Activation, Add, Multiply, Flatten, Dense, Dropout
from tensorflow.keras.models import Model


class ResidualAttentionNetwork():
    
    def __init__(self, input_shape, output_size, mask_activation, p=1, t=2, r=1):
         
        self.input_shape = input_shape
        self.output_size = output_size
        self.mask_activation = mask_activation
        self.p = p
        self.t = t
        self.r = r
        self.filter_dic = {'s1': [32,32,128],
                           's2': [64,64,256],
                           's3': [128,128,512],
                           'se': [256,256,1024]}
    
    def Attention_56(self):
        filter_dic = self.filter_dic
        
        input_data = Input(shape=self.input_shape)
        #  padded_data = ZeroPadding2D((4,4))(input_data)
        
        conv_1 = Conv2D(filters=64, kernel_size=(1,1), strides=(1,1), padding='same')(input_data)
        max_pool_1 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same')(conv_1)
        
        # Residual-Attention Module stage #1 
        filters_s1 = filter_dic['s1']
        res_unit_1 = self.ResidualUnit(max_pool_1, filters=filters_s1)
        am_unit_1 = self.AttentionModule(res_unit_1, filters=filters_s1, learning_mechanism='ARL', stage=1)
        
        max_pool_2 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same')(am_unit_1)        
        
        # Residual-Attention Module stage #2
        filters_s2 = filter_dic['s2']
        res_unit_2 = self.ResidualUnit(max_pool_2, filters=filters_s2)
        am_unit_2 = self.AttentionModule(res_unit_2, filters=filters_s2, learning_mechanism='ARL', stage=2)
        
        max_pool_3 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same')(am_unit_2)
        
        # Residual-Attention Module stage #3
        filters_s3 = filter_dic['s3']
        res_unit_3 = self.ResidualUnit(max_pool_3, filters=filters_s3)
        am_unit_3 = self.AttentionModule(res_unit_3, filters=filters_s3, learning_mechanism='ARL', stage=3)
        
        # ending Residual Units
        
        filters_ending = filter_dic['se']
        for i in range(3):
            am_unit_3 = self.ResidualUnit(am_unit_3, filters=filters_ending)
        
        # prediction 
        avg_pool = AveragePooling2D(pool_size=(4,4), strides=(1,1))(am_unit_3)
        
        flatten = Flatten()(avg_pool)
        output = Dense(self.output_size, activation='softmax')(flatten)
        
        # model construction
        model = Model(inputs=input_data, outputs=output)
        
        return model


     
    def AttentionModule(self, input_unit, filters, learning_mechanism, stage):
        attention_unit = input_unit
        skip_count = 3 - stage

        for _ in range(self.p):
            attention_unit = self.ResidualUnit(attention_unit, filters)

        trunk_unit = self.TrunkBranch(attention_unit, filters)
        soft_mask_unit = self.SoftMaskBranch(attention_unit, skip_count, filters)
        
        if learning_mechanism == 'NAL':
            attention_unit = self.NaiveAttentionLearning(trunk_unit, soft_mask_unit)
        else:
            attention_unit = self.AttentionResidualLearning(trunk_unit, soft_mask_unit)
        
        for _ in range(self.p):
            attention_unit = self.ResidualUnit(attention_unit, filters)
            
        return attention_unit

    
    def TrunkBranch(self, input_unit, filters):
        trunk_unit = input_unit
        for _ in range(self.t):
            trunk_unit = self.ResidualUnit(trunk_unit, filters)
    
        return trunk_unit
        
        
    def SoftMaskBranch(self, input_unit, skip_count, filters):
        soft_mask_unit = input_unit
        if skip_count == 2:
            skip_unit_1 = self.ResidualUnit(soft_mask_unit, filters)

        down_sample_unit_1 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same')(soft_mask_unit)
        for _ in range(self.r):
            down_sample_unit_1 = self.ResidualUnit(down_sample_unit_1, filters)

        if skip_count == 1:
            skip_unit_2 = self.ResidualUnit(down_sample_unit_1, filters)

        down_sample_unit_2 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same')(down_sample_unit_1)
        for _ in range(self.r*2):
            down_sample_unit_2 = self.ResidualUnit(down_sample_unit_2, filters)
        up_sample_unit_1 = UpSampling2D(size=(2,2))(down_sample_unit_2) 

        if skip_count == 1:
            up_sample_unit_1 = Add()([up_sample_unit_1, skip_unit_2])

        for _ in range(self.r):
            up_sample_unit_1 = self.ResidualUnit(up_sample_unit_1, filters)
        up_sample_unit_2 = UpSampling2D(size=(2,2))(up_sample_unit_1) 

        if skip_count == 2:
            up_sample_unit_2 = Add()([up_sample_unit_2, skip_unit_1])

        conv_filter = up_sample_unit_2.shape[-1]

        conv_1 = Conv2D(filters=conv_filter, kernel_size=(1,1), padding='same')(up_sample_unit_2)
        conv_2 = Conv2D(filters=conv_filter, kernel_size=(1,1), padding='same')(conv_1)
        output = Activation(self.mask_activation)(conv_2)

        return output        

        
    def AttentionResidualLearning(self, trunk_unit, soft_mask_unit):
        output = Multiply()([trunk_unit, soft_mask_unit])
        output = Add()([output, trunk_unit])

        return output   
    
    
    def NaiveAttentionLearning(self, trunk_unit, soft_mask_unit):
        output = Multiply()([trunk_unit, soft_mask_unit])
        
        return output

    

    # building block - ResUnit
    def ResidualUnit(self, residual_input, filters):
        """
        Implementation of Deeper Bottleneck Architectures
        :param residual_input: 4-D Tensor
        :param filters: number of filers
        """
        identity_x = residual_input
        
        filter1, filter2, filter3 = filters
        
        #the 1x1 layers are responsible for reducing and then increasing (restoring) dimensions
        batch_norm_1 = BatchNormalization()(residual_input)
        activation_1 = Activation('relu')(batch_norm_1)
        conv_1 = Conv2D(filters=filter1, kernel_size=(1,1), padding='same')(activation_1)
        
        #the 3x3 layer: a bottleneck with smallerinput/output dimensions.
        batch_norm_2 = BatchNormalization()(conv_1)
        activation_2 = Activation('relu')(batch_norm_2)
        conv_2 = Conv2D(filters=filter2, kernel_size=(3,3), padding='same')(activation_2)
        
        batch_norm_3 = BatchNormalization()(conv_2)
        activation_3 = Activation('relu')(batch_norm_2)
        conv_3 = Conv2D(filters=filter3, kernel_size=(1,1), padding='same')(activation_3)
        
        # skip connection
        if identity_x.shape[-1] != conv_3.shape[-1]:
            filter_c = conv_3.shape[-1]
            identity_x = Conv2D(filters=filter_c, kernel_size=(1,1), padding='same')(identity_x)
            
        # residual + identity
        output = Add()([identity_x, conv_3])
        
        return output

    