package list:
 1. numpy/1.14.0
 2. pytorch/0.3.1.post2
 3. sklearn/0.19.dev0

GPU: all gpu when set nn_config['cuda'] to True and the pytorch can detect available gpu. The default gpu device is 0, and
can change the device to set nn_config['gpu'] to the gpu device number available.

program instruction:
TODO: Normalize the data
Normlize data(seems not necessary in this work, because ..)
The performance of the module is not stable. To get the best performance, you may need to run it several times.
The best accuracy and f1 is 96.77%, when test written independent data. And this happens when loss is NaN
which might mean loss is a absolutely small value.

TODO:
need to ask if the method can be from another paper, but I implement it and use features from tom.

TODO:
add bidirectional weight-sharing networks. pre-train with few-shot data(cut into two parts), then use the pre-trained one
to prototypical network.
To see, 1. less time to train. 2. better performance.
in prototypical network, the linear layer before softmax can be pre-trained by this function.