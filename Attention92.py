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
        
        
        
    def Attention_92(self):
        filter_dic = self.filter_dic
        
        input_data = Input(shape=self.input_shape)
        padded_data = ZeroPadding2D((4,4))(input_data)
        
        conv_1 = Conv2D(filters=64, kernel_size=(7,7), padding='same')(padded_data)
        max_pool_1 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same')(conv_1)
        
        # Residual-Attention Module stage #1 
        filters_s1 = filter_dic['s1']
        res_unit_1 = self.ResidualUnit(max_pool_1, filters=filters_s1)
        am_unit_1 = self.AttentionModule(res_unit_1, filters=filters_s1)
        am_unit_1 = self.AttentionModule(am_unit_1, filters=filters_s1)
        
        # Residual-Attention Module stage #2
        filters_s2 = filter_dic['s2']
        res_unit_2 = self.ResidualUnit(am_unit_1, filters=filters_s2)
        am_unit_2 = self.AttentionModule(res_unit_2, filters=filters_s2)
        am_unit_2 = self.AttentionModule(am_unit_2, filters=filters_s2)

        # Residual-Attention Module stage #3
        filters_s3 = filter_dic['s3']
        res_unit_3 = self.ResidualUnit(am_unit_2, filters=filters_s3)
        am_unit_3 = self.AttentionModule(res_unit_3, filters=filters_s3)
        am_unit_3 = self.AttentionModule(am_unit_3, filters=filters_s3)

        # ending Residual Units
        res_unit_ending = am_unit_3
        
        filters_ending = filter_dic['se']
        for i in range(3):
            res_unit_ending = self.ResidualUnit(res_unit_ending, filters=filters_ending)
        
        # prediction 
        avg_pool = AveragePooling2D(pool_size=(7,7), strides=(1,1))(res_unit_ending)
        
        flatten = Flatten()(avg_pool)
        output = Dense(self.output_size, activation='softmax')(flatten)
        
        # model construction
        model = Model(inputs=input_data, outputs=output)
        
        return model
    
    
    def AttentionModule(self, am_input, filters):
        
        am_unit = am_input
        # p iterations of ResUnit
        for _ in range(self.p):
            am_unit = self.ResidualUnit(am_unit, filters=filters)
        
        # spliting into t/m branches and combining using ARL
        trunk_unit = self.TrunkBranch(am_unit, filters=filters)
        mask_unit = self.MaskBranch(am_unit, filters=filters)    
            
        ##ARL
        arl_unit = self.AttentionResidualLearning(trunk_unit, mask_unit)        
        p iterations of ResUnit
        for _ in range(self.p):
            arl_unit = self.ResidualUnit(arl_unit, filters=filters)
        
        return arl_unit
    
    def TrunkBranch(self, trunk_input, filters):
        trunk_unit = trunk_input
        for _ in range(self.t):
            trunk_unit = self.ResidualUnit(trunk_unit, filters=filters)
            
        return trunk_unit
    
    
    def MaskBranch(self, mask_input, filters, iteration=2):
        
        downsample_unit = mask_input
        for _ in range(iteration):
            for _ in range(self.r):
                downsample_unit = self.ResidualUnit(downsample_unit, filters=filters)
            
            downsample_unit = MaxPool2D(pool_size=(2,2), strides=(2,2))(downsample_unit)     
        
        midstep_unit = downsample_unit
        for _ in range(2*self.r):
            midstep_unit = self.ResidualUnit(midstep_unit, filters=filters)  
        
        upsample_unit = midstep_unit
        for _ in range(iteration):
            for _ in range(self.r):
                upsample_unit = self.ResidualUnit(upsample_unit, filters=filters)
                
            upsample_unit = UpSampling2D(size=(2,2))(upsample_unit) 
        
        conv_filter = upsample_unit.shape[-1]
        
        conv_1 = Conv2D(filters=conv_filter, kernel_size=(1,1), padding='same')(upsample_unit)
        conv_2 = Conv2D(filters=conv_filter, kernel_size=(1,1), padding='same')(conv_1)
        output = Activation(self.mask_activation)(conv_2)
        
        return output
    
    
    # building block - ResUnit
    def ResidualUnit(self, residual_input, filters):
        
        identity_x = residual_input
        
        filter1, filter2, filter3 = filters
        
        # layer 1
        batch_norm_1 = BatchNormalization()(residual_input)
        activation_1 = Activation('relu')(batch_norm_1)
        conv_1 = Conv2D(filters=filter1, kernel_size=(1,1), padding='same')(activation_1)
        
        # layer 2
        batch_norm_2 = BatchNormalization()(conv_1)
        activation_2 = Activation('relu')(batch_norm_2)
        conv_2 = Conv2D(filters=filter2, kernel_size=(3,3), padding='same')(activation_2)
        
        # layer 3
        batch_norm_3 = BatchNormalization()(conv_2)
        activation_3 = Activation('relu')(batch_norm_2)
        conv_3 = Conv2D(filters=filter3, kernel_size=(1,1), padding='same')(activation_3)
        
        # shape alignment
        if identity_x.shape[-1] != conv_3.shape[-1]:
            filter_c = conv_3.shape[-1]
            identity_x = Conv2D(filters=filter_c, kernel_size=(1,1), padding='same')(identity_x)
            
        # residual + identity
        output = Add()([identity_x, conv_3])
        
        return output
    
    
    