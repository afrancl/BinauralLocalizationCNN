def generate():
    import random
    import numpy as np
    import numpy.random
    min_layers=5
    max_layers=7
    num_layers=numpy.random.randint(min_layers,max_layers)  #number of total layers
    freq_stride=2
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
    gpu_1=[]


    #lists of variable values for first layer
    k_width=numpy.random.randint((2**(5-count))/2,((2**(5-count))+4*num_layers)/2)
    k_width*=2
    k_heighth=numpy.random.randint(1,2)
    k_heighth*=2
#    freq_stride=numpy.random.randint(1,2)
    stride=[freq_stride,time_stride]
    kernal=[k_heighth,k_width,act]
    pool_size=[freq_pool,time_pool]
    gpu=["/gpu:{}".format(gpu_num)]
    gpu_num+=1

    #generate layers and add to first gpu
    conv=['conv',kernal,stride]
    bn=['bn']
    relu=['relu']
    pool=['pool',pool_size]
    num_layers-=1
    count+=1

    gpu_0.append(gpu)
    gpu_0.append(conv)
    gpu_0.append(pool)
    gpu_0.append(relu)
    gpu_0.append(bn)


    #Make sure there is on 2x1 stride on the first gpu
#    if freq_stride == 1:
#        freq_stride = 2
#    else:
#        freq_stride = 1


   #Generate layer on first gpu. Before the first pool no more than 2
   #convolutions allowed. First layer has to be a convolutional layer.
   #Next layer has a 50% chance of being another convolution.
   #Convolution includes convolution, bias, batch norm, and relu. 
#     act*=2
#     i=0         #tracks number of convolutions per layer
#     while  num_layers != 0:
#         if i<2:
#             a=random.random()
#             if a >= .5:
#                 k_heighth=numpy.random.randint((2**count)/2,((2**count)+2*count)/2)
#                 k_heighth*=2
#                 k_width=numpy.random.randint((2**(5-count))/2,((2**(5-count))+2*num_layers)/2)
#                 k_width*=2
#                 stride=[freq_stride,time_stride]
#                 kernal=[k_heighth,k_width,act]
#                 conv=['conv',kernal,stride]
#                 bn=['bn']
#                 relu=['relu']
#                 gpu_0.append(conv)
#                 gpu_0.append(bn)
#                 gpu_0.append(relu)
#                 count+=1
#                 i+=1
#             else:
#                 if i==0:            
#                     k_heighth=numpy.random.randint((2**count)/2,((2**count)+2*count)/2)
#                     k_heighth*=2
#                     k_width=numpy.random.randint((2**(5-count))/2,((2**(5-count))+2*num_layers)/2)
#                     k_width*=2
#                     stride=[freq_stride,time_stride]
#                     kernal=[k_heighth,k_width,act]
#                     conv=['conv',kernal,stride]
#                     bn=['bn']
#                     relu=['relu']
#                     gpu_0.append(conv)
#                     gpu_0.append(bn)
#                     gpu_0.append(relu)
#                     count+=1
#                     i+=1
#                 else:
#                     gpu_0.append(pool)
#                     num_layers-=1
#                     act*=2
#                     break
#         else:
#             gpu_0.append(pool)
#             num_layers-=1
#             act*=2
#             break
   

    gpu=["/gpu:{}".format(gpu_num)]
    gpu_1.append(gpu)

    #Second layer has one convolution, batch norm, and relu before pooling.
    #Kernal sizes variable
    freq_stride=1
    act*=2
    k_heighth=numpy.random.randint((2**count)/2,((2**count)+2*count)/2)
    k_heighth*=2
    k_width=numpy.random.randint((2**(5-count))/2,((2**(5-count))+2*num_layers)/2)
    k_width*=2
    stride=[freq_stride,time_stride]
    kernal=[k_heighth,k_width,act]
    conv=['conv',kernal,stride]
    bn=['bn']
    relu=['relu']
    count+=1
    gpu_0.append(conv)
    gpu_0.append(pool)
    gpu_0.append(relu)
    gpu_0.append(bn)
    num_layers-=1
    act*=2
    
    
    
    #Generate layers on second gpu. No more than 3 convolutions 
    #allowed before pooling. First layer has to be a convolutional.
    #Next layer has a 50% chance of being another convolution.
    #Convolution includes convolution, bias, batch norm, and relu. 
    i=0     #tracks number of convolutions per layer
    while  num_layers != 0:
        if i<3:
            a=random.random()
            if a >= .5:
                k_width=numpy.random.randint(3,5)
                k_width*=2
                k_heighth=k_width
                stride=[freq_stride,time_stride]
                kernal=[k_heighth,k_width,act]
                conv=['conv',kernal,stride]
                bn=['bn']
                relu=['relu']
                gpu_1.append(conv)
                gpu_1.append(relu)
                gpu_1.append(bn)
                count+=1
                i+=1
            else:
                if i==0:            
                    k_width=numpy.random.randint(3,5)
                    k_width*=2
                    k_heighth=k_width
                    stride=[freq_stride,time_stride]
                    kernal=[k_heighth,k_width,act]
                    conv=['conv',kernal,stride]
                    bn=['bn']
                    relu=['relu']
                    gpu_1.append(conv)
                    gpu_1.append(relu)
                    gpu_1.append(bn)
                    count+=1
                    i+=1
                else:
                    gpu_1.insert(-2,pool)
                    num_layers-=1
                    if act < 512:
                        act*=2
                    i=0
        else:
            gpu_1.insert(-2,pool)
            num_layers-=1
            if act < 512:
                act*=2
            i=0


    #Make last layer before fully connected layers a pool    
    #if gpu_1[-1]==bn:
    #    gpu_1.append(pool)


    #Generate fully connected layers and add to second gpu
    num_fc=1
    while num_fc != 0 :
        fc=['fc',dim]
        fc_bn=['fc_bn']
        fc_relu=['fc_relu']
        gpu_1.append(fc)
        gpu_1.append(fc_relu)
        gpu_1.append(fc_bn)
        num_fc-=1

    #Add dropout and out to second gpu
    dropout=['dropout']
    out=['out']
    gpu_1.append(dropout)
    gpu_1.append(out)

    #Append lists gpu_0 and gpu_1 to config.array
    config_array.append(gpu_0)
    config_array.append(gpu_1)
    
    return config_array
