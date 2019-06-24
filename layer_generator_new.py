def generate():
    import random
    import numpy as np
    import numpy.random
    min_layers=3
    max_layers=8
    num_layers=numpy.random.randint(min_layers,max_layers)  #number of total layers
    freq_stride=1
    time_stride=1
    freq_pool=1
    time_pool=4
    act=32
    gpu_num=0       #tracks gpu number
    dim=512
    count=0         #tracks total number of convolutions

    #create lists
    config_array=[]
    gpu_0=[]

    gpu=["/gpu:{}".format(gpu_num)]
    gpu_num+=1
    gpu_0.append(gpu)

    possible_pooling_kernels_length = [[2,4,8],[2,4],[2,4],[1,2],[1,2],[1,1,2],[1,1,1,2],[1,1,1,2]]
    possible_pooling_kernels_height = [[1,2],[1,2],[1,2],[1,2],[1,1,2],[1,1,2],[1,1,1,2],[1,1,1,2]]


    possible_conv_kernels_height = [[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]]
    possible_conv_kernels_length = [[4,8,16,32,64],[4,8,16,32],[2,4,8,16],[2,4,8],[2,4,8],[2,3,4],[2,3,4],[2,3,4]]



    def sample_conv_kernel(layer_idx):
        rand_choose_conv_kernel_length = np.random.randint(0,len(possible_conv_kernels_length[layer_idx]))
        rand_choose_conv_kernel_height = np.random.randint(0,len(possible_conv_kernels_height[layer_idx]))
        return [possible_conv_kernels_height[layer_idx][rand_choose_conv_kernel_height],
                possible_conv_kernels_length[layer_idx][rand_choose_conv_kernel_length]]

    def sample_conv_strides(layer_idx):
        rand_choose_conv_stride_length = np.random.randint(0,len(possible_conv_strides_length[layer_idx]))
        rand_choose_conv_stride_height = np.random.randint(0,len(possible_conv_strides_height[layer_idx]))
        return [possible_conv_strides_height[layer_idx][rand_choose_conv_stride_height],
                possible_conv_strides_length[layer_idx][rand_choose_conv_stride_length]]

    def sample_pool_kernel(layer_idx):
        # set the pooling for each layer
        rand_choose_poolkernel_length = np.random.randint(0,len(possible_pooling_kernels_length[layer_idx]))
        rand_choose_poolkernel_height = np.random.randint(0,len(possible_pooling_kernels_height[layer_idx]))
        return [possible_pooling_kernels_height[layer_idx][rand_choose_poolkernel_height],
                possible_pooling_kernels_length[layer_idx][rand_choose_poolkernel_length]]
    #Generate layers on second gpu. No more than 3 convolutions 
    #allowed before pooling. First layer has to be a convolutional.
    #Next layer has a 50% chance of being another convolution.
    #Convolution includes convolution, bias, batch norm, and relu. 
    conv_count=0     #tracks number of convolutions per layer
    layer_count= 0
    while  layer_count <= num_layers:
        if conv_count<3:
            a=random.random()
            if a >= .5 or conv_count==0:
                stride=[freq_stride,time_stride]
                kernal=sample_conv_kernel(layer_count)
                kernal.append(act)
                conv=['conv',kernal,stride]
                bn=['bn']
                relu=['relu']
                gpu_0.append(conv)
                gpu_0.append(relu)
                gpu_0.append(bn)
                conv_count+=1
            else:
                pool_kernel = sample_pool_kernel(layer_count)
                pool = ['pool',pool_kernel]
                gpu_0.insert(-2,pool)
                layer_count+=1
                if act < 512 and random.random() > .2:
                    act*=2
                conv_count=0
        else:
            pool_kernel = sample_pool_kernel(layer_count)
            pool = ['pool',pool_kernel]
            gpu_0.insert(-2,pool)
            num_layers-=1
            conv_count=0
            if act < 512 and random.random() > .2:
                act*=2


    #Make last layer before fully connected layers a pool    
    #if gpu_1[-1]==bn:
    #    gpu_1.append(pool)


    #Generate fully connected layers and add to second gpu
    num_fc=1
    while num_fc != 0 :
        fc=['fc',dim]
        fc_bn=['fc_bn']
        fc_relu=['fc_relu']
        gpu_0.append(fc)
        gpu_0.append(fc_relu)
        gpu_0.append(fc_bn)
        num_fc-=1

    #Add dropout and out to second gpu
    dropout=['dropout']
    out=['out']
    gpu_0.append(dropout)
    gpu_0.append(out)

    #Append lists gpu_0 and gpu_1 to config.array
    config_array.append(gpu_0)
    
    return config_array



def uniformly_select_conv_layers(range_num_conv_layers, possible_conv_layer_length, possible_conv_layer_height, possible_conv_layer_nums, possible_pooling_strides_length, possible_pooling_strides_height, possible_conv_strides_length, possible_conv_strides_height):
    """
    uniformly selects from the options within the provided inputs
    Args
    ----
    range_num_conv_layers (list) : possible depths of the network (number of conv layers)
    possible_conv_layer_length (list) : for each convolutional layer, the width of the kernels
    possible_conv_layer_height (list) : for each convolutional layer, the height of the kernels
    possible_conv_layer_nums (list) : for each convolutional layer, the number of kernels to include
    possible_pooling_strides_length (list) : for each pooling layer, the amount of pooling (width)
    possible_pooling_strides_height (list) : for each pooling layer, the amount of pooling (height)
    possible_conv_strides_length (list) : for each conv layer, the possible strides (width)
    possible_conv_strides_height (list) : for each conv layer, the possible stides (height)
    """
    num_conv_kernels = []
    convstrides = []
    poolstrides = []
    conv_kernels_sizes = []

    num_conv_layers = np.random.randint(range_num_conv_layers[0], range_num_conv_layers[1]+1)
    for layer_idx in np.arange(num_conv_layers):
        # choose number of conv kernels in each layer
        rand_choose_num_conv = np.random.randint(0,len(possible_conv_layer_nums[layer_idx]))
        num_conv_kernels.append(possible_conv_layer_nums[layer_idx][rand_choose_num_conv])

        # set the stride for each layer
        rand_choose_conv_stride_length = np.random.randint(0,len(possible_conv_strides_length[layer_idx]))
        rand_choose_conv_stride_height = np.random.randint(0,len(possible_conv_strides_height[layer_idx]))
        convstrides.append([possible_conv_strides_length[layer_idx][rand_choose_conv_stride_length], 
                            possible_conv_strides_height[layer_idx][rand_choose_conv_stride_height]])

        # set the pooling for each layer
        rand_choose_poolstrides_length = np.random.randint(0,len(possible_pooling_strides_length[layer_idx]))
        rand_choose_poolstrides_height = np.random.randint(0,len(possible_pooling_strides_height[layer_idx]))
        poolstrides.append([possible_pooling_strides_height[layer_idx][rand_choose_poolstrides_height],possible_pooling_strides_length[layer_idx][rand_choose_poolstrides_length]])

        # set the filter length and height for each layer
        rand_choose_conv_length = np.random.randint(0,len(possible_conv_layer_length[layer_idx]))
        rand_choose_conv_height = np.random.randint(0,len(possible_conv_layer_height[layer_idx]))

        conv_kernels_sizes.append([possible_conv_layer_height[layer_idx][rand_choose_conv_height],possible_conv_layer_length[layer_idx][rand_choose_conv_length]])

    return num_conv_kernels, convstrides, poolstrides, conv_kernels_sizes
