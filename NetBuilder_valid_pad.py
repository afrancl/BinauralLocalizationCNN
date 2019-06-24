import pdb
class NetBuilder:

    def __init__(self):
        self.layer=0
        self.layer1=0
        self.layer2=0
        self.layer_fc1=0
        self.layer_fc2=0
        self.layer_out1=0
        self.layer_out2=0
    
    def build(self,config_array,subbands_batch,training_state,dropout_training_state,filter_dtype,padding,n_classes_localization,n_classes_recognition,branched):

#      config_array=[[["/gpu:0"],['conv',[2,50,32],[1,4]],['relu'],['pool',[1,4]]],[["/gpu:1"],['conv',[4,20,64],[1,1]],['bn'],['relu'],['pool',[1,4]],['conv',[8,8,128],[1,1]],['bn'],['relu'],['pool',[1,4]],['conv',[8,8,256],[1,1]],['bn'],['relu'],['pool',[1,8]],['conv',[8,8,512],[1,1]],['bn'],['relu'],['pool',[2,2]]],[["/gpu:2"],['fc',512],['fc_bn'],['fc_relu'],['dropout'],['out']]]

        import tensorflow as tf
        #net_input=tf.constant(1., shape=[16,72,30000,1],dtype=filter_dtype)
        #self.input=net_input
        self.input=subbands_batch
        self.input1=0
        self.input2=0
        self.layer=0
        self.layer1=0
        self.layer2=0
        self.layer_fc=0
        self.layer_fc1=0
        self.layer_fc2=0
        self.layer_out=0
        self.layer_out1=0
        self.layer_out2=0
        start_point=0
        branched_point = False
        second_branch = [["/gpu:2"],["/gpu:3"]]
        gpu2 = second_branch [0][0]
        for lst in config_array:
            gpu1 = lst[0][0]
            start_point=1
            if branched_point is False:
                with tf.device(gpu1):
                    for element in lst[1:]:
                        if element[0]=='conv':
                            size=self.input.get_shape()
                            kernel_size = [element[1][0],element[1][1],size[3],element[1][2]]
                            stride_size = [1,element[2][0],element[2][1],1]
                            filter_height = kernel_size[0]
                            in_height = int(size[1])
                            weight=tf.get_variable("wc_{}".format(self.layer),kernel_size,filter_dtype)
                            bias=tf.get_variable("wb_{}".format(self.layer),element[1][2],filter_dtype)
                            if (in_height % stride_size[1] == 0):
                                pad_along_height = max(filter_height -stride_size[1], 0)
                            else:
                                pad_along_height = max(filter_height - (in_height % stride_size[1]), 0)
                            pad_top = pad_along_height // 2
                            pad_bottom = pad_along_height - pad_top
                            if pad_along_height == 0 or padding == 'SAME':
                                weight=tf.nn.conv2d(self.input,weight,strides=stride_size,padding=padding)
                            else:
                                paddings = tf.constant([[0,0],[pad_top, pad_bottom], [0, 0],[0,0]])
                                input_padded = tf.pad(self.input,paddings)
                                weight=tf.nn.conv2d(input_padded,weight,strides=stride_size,padding=padding)

                            self.input=tf.nn.bias_add(weight,bias)
                            self.layer+=1
                            print(element)
                            print(self.input)

                        elif element[0]=='bn':
                            self.input=tf.layers.batch_normalization(self.input,training=training_state)
                            print(element)
                            print(self.input)
                        
                        elif element[0]=='relu':
                            self.input=tf.nn.relu(self.input)
                            print(element)
                            print(self.input)
                        
                        elif element[0]=='pool':
                            self.input=tf.nn.max_pool(self.input,ksize=[1,element[1][0],element[1][1],1],strides=[1,element[1][0],element[1][1],1],padding=padding)
                            tf.add_to_collection('checkpoints', self.input)
                            print(element)
                            print(self.input)
                        
                        elif element[0]=='fc':
                            dim=self.input.get_shape()
                            wd1=tf.get_variable("wc_fc_{}".format(self.layer_fc),[dim[3]*dim[1]*dim[2],element[1]],filter_dtype)
                            dense_bias1=tf.get_variable("wb_fc_{}".format(self.layer_fc1),element[1],filter_dtype)
                            pool_flat=tf.reshape(self.input,[-1,wd1.get_shape().as_list()[0]])
                            fc1=tf.add(tf.matmul(pool_flat,wd1),dense_bias1)
                            self.input=tf.cast(fc1,tf.float32)
                            
                            self.layer_fc+=1
                            print(element)
                            print(self.input)

                        elif element[0]=='fc_bn':
                            self.input=tf.cast(tf.layers.batch_normalization(self.input,training=training_state),filter_dtype)
                            print(element)
                            print(self.input)
                            
                        elif element[0]=='fc_relu':
                            self.input=tf.nn.relu(self.input)
                            print(element)
                            print(self.input)
                        
                        elif element[0]=='dropout':
                            self.input=tf.layers.dropout(self.input,training=dropout_training_state)
                            print(element)
                            print(self.input)
                   
                        elif element[0]=='out':
                            dim_1=self.input.get_shape()
                            w_out=tf.get_variable("wc_out_{}".format(self.layer_out),[dim_1[1],n_classes_localization],filter_dtype)
                            b_out=tf.get_variable("wb_out_{}".format(self.layer_out),[n_classes_localization],filter_dtype)
                            out=tf.add(tf.matmul(self.input,w_out),b_out)
                            self.input=tf.cast(out,tf.float32)
                            
                            self.layer_out+=1
                            print(element)
                            print(self.input)
                        
                        elif element[0]=='branch':
                            self.input1=self.input
                            self.input2=self.input
                            self.layer1=self.layer
                            self.layer2=self.layer
                            print(self.input1)
                            print(self.input2)
                            print(self.layer1)
                            print(self.layer2)
                            branched_point = True
                            start_point = lst.index(element)
                            print(start_point)
                            start_point +=1
                            break

            if branched_point is True:
                with tf.device(gpu1):
                    for element in lst [start_point:]:     ##############
                        if element[0]=='conv':
                            size1=self.input1.get_shape()
                            kernel_size = [element[1][0],element[1][1],size1[3],element[1][2]]
                            stride_size = [1,element[2][0],element[2][1],1]
                            filter_height = kernel_size[0]
                            in_height = size1[1]
                            weight1=tf.get_variable("wc1_{}".format(self.layer1),kernel_size,filter_dtype)
                            bias1=tf.get_variable("wb1_{}".format(self.layer1),element[1][2],filter_dtype)
                            if (in_height % stride_size[1] == 0):
                                pad_along_height = max(filter_height - stride_size[1], 0)
                            else:
                                pad_along_height = max(filter_height - (in_height % stride_size[1]), 0)
                            pad_top = pad_along_height // 2
                            pad_bottom = pad_along_height - pad_top
                            if pad_along_height == 0 or padding == 'SAME':
                                weight1=tf.nn.conv2d(self.input1,weight1,strides=stride_size,padding=padding)
                            else:
                                paddings = tf.constant([[0,0],[pad_top, pad_bottom], [0, 0],[0,0]])
                                input_padded = tf.pad(self.input1,paddings)
                                weight1=tf.nn.conv2d(input_padded,weight1,strides=stride_size,padding=padding)
                                
                            self.input1=tf.nn.bias_add(weight1,bias1)
                            self.layer1+=1
                            print(element)
                            print(self.input1)

                        elif element[0]=='bn':
                            self.input1=tf.layers.batch_normalization(self.input1,training=training_state)
                            print(element)
                            print(self.input1)
                        
                        elif element[0]=='relu':
                            self.input1=tf.nn.relu(self.input1)
                            print(element)
                            print(self.input1)
                        
                        elif element[0]=='pool':
                            self.input1=tf.nn.max_pool(self.input1,ksize=[1,element[1][0],element[1][1],1],strides=[1,element[1][0],element[1][1],1],padding=padding)
                            print(element)
                            print(self.input1)
                        
                        elif element[0]=='fc':
                            dim1=self.input1.get_shape()
                            wd1_1=tf.get_variable("wc1_fc_{}".format(self.layer_fc1),[dim1[3]*dim1[1]*dim1[2],element[1]],filter_dtype)
                            dense_bias1_1=tf.get_variable("wb1_fc_{}".format(self.layer_fc1),element[1],filter_dtype)
                            pool_flat1=tf.reshape(self.input1,[-1,wd1_1.get_shape().as_list()[0]])
                            fc1_1=tf.add(tf.matmul(pool_flat1,wd1_1),dense_bias1_1)
                            self.input1=tf.cast(fc1_1,tf.float32)
                            
                            self.layer_fc1+=1
                            print(element)
                            print(self.input1)

                        elif element[0]=='fc_bn':
                            self.input1=tf.cast(tf.layers.batch_normalization(self.input1,training=training_state),filter_dtype)
                            print(element)
                            print(self.input1)
                            
                        elif element[0]=='fc_relu':
                            self.input1=tf.nn.relu(self.input1)
                            print(element)
                            print(self.input1)
                        
                        elif element[0]=='dropout':
                            self.input1=tf.layers.dropout(self.input1,training=dropout_training_state)
                            print(element)
                            print(self.input1)
                   
                        elif element[0]=='out':
                            dim_1_1=self.input1.get_shape()
                            w_out1=tf.get_variable("wc1_out_{}".format(self.layer_out1),[dim_1_1[1],n_classes_localization],filter_dtype)
                            b_out1=tf.get_variable("wb1_out_{}".format(self.layer_out1),[n_classes_localization],filter_dtype)
                            out1=tf.add(tf.matmul(self.input1,w_out1),b_out1)
                            self.input1=tf.cast(out1,tf.float32)
                            
                            self.layer_out1+=1
                            print(element)
                            print(self.input1)

                with tf.device(gpu2):
                    for element in lst [start_point:]:
                        if element[0]=='conv':
                            size2=self.input2.get_shape()
                            kernel_size = [element[1][0],element[1][1],size2[3],element[1][2]]
                            stride_size = [1,element[2][0],element[2][1],1]
                            filter_height = kernel_size[0]
                            in_height = size2[1]
                            weight2=tf.get_variable("wc2_{}".format(self.layer2),kernel_size,filter_dtype)
                            bias2=tf.get_variable("wb2_{}".format(self.layer2),element[1][2],filter_dtype)
                            if (in_height % stride_size[1] == 0):
                                pad_along_height = max(filter_height - stride_size[1], 0)
                            else:
                                pad_along_height = max(filter_height - (in_height % stride_size[1]), 0)
                            pad_top = pad_along_height //2
                            pad_bottom = pad_along_height - pad_top
                            if pad_along_height == 0 or padding == 'SAME':
                                weight2=tf.nn.conv2d(self.input2,weight2,strides=stride_size,padding=padding)
                            else:
                                paddings = tf.constant([[0,0],[pad_top, pad_bottom], [0, 0],[0,0]])
                                input_padded = tf.pad(self.input2,paddings)
                                weight2=tf.nn.conv2d(self.input2,weight2,strides=stride_size,padding=padding)

                            self.input2=tf.nn.bias_add(weight2,bias2)
                            self.layer2+=1
                            print(element)
                            print(self.input2)

                        elif element[0]=='bn':
                            self.input2=tf.layers.batch_normalization(self.input2,training=training_state)
                            print(element)
                            print(self.input2)
                        
                        elif element[0]=='relu':
                            self.input2=tf.nn.relu(self.input2)
                            print(element)
                            print(self.input2)
                        
                        elif element[0]=='pool':
                            self.input2=tf.nn.max_pool(self.input2,ksize=[1,element[1][0],element[1][1],1],strides=[1,element[1][0],element[1][1],1],padding=padding)
                            print(element)
                            print(self.input2)
                        
                        elif element[0]=='fc':
                            dim2=self.input2.get_shape()
                            wd1_2=tf.get_variable("wc2_fc_{}".format(self.layer_fc2),[dim2[3]*dim2[1]*dim2[2],element[1]],filter_dtype)
                            dense_bias1_2=tf.get_variable("wb2_fc_{}".format(self.layer_fc2),element[1],filter_dtype)
                            pool_flat2=tf.reshape(self.input2,[-1,wd1_2.get_shape().as_list()[0]])
                            fc1_2=tf.add(tf.matmul(pool_flat2,wd1_2),dense_bias1_2)
                            self.input2=tf.cast(fc1_2,tf.float32)
                            
                            self.layer_fc2+=1
                            print(element)
                            print(self.input2)

                        elif element[0]=='fc_bn':
                            self.input2=tf.cast(tf.layers.batch_normalization(self.input2,training=training_state),filter_dtype)
                            print(element)
                            print(self.input2)

                        elif element[0]=='fc_relu':
                            self.input2=tf.nn.relu(self.input2)
                            print(element)
                            print(self.input2)
                        
                        elif element[0]=='dropout':
                            self.input2=tf.layers.dropout(self.input2,training=dropout_training_state)
                            print(element)
                            print(self.input2)
                   
                        elif element[0]=='out':
                            dim_1_2=self.input2.get_shape()
                            w_out2=tf.get_variable("wc2_out_{}".format(self.layer_out2),[dim_1_2[1],n_classes_recognition],filter_dtype)
                            b_out2=tf.get_variable("wb2_out_{}".format(self.layer_out2),[n_classes_recognition],filter_dtype)
                            out2=tf.add(tf.matmul(self.input2,w_out2),b_out2)
                            self.input2=tf.cast(out2,tf.float32)
                            
                            self.layer_out2+=1
                            print(element)
                            print(self.input2)
                                    
                gpu2 = second_branch[1][0]                         
        if branched:       
            return self.input1,self.input2
        else:
            return self.input
            
