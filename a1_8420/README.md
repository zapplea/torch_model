package list:
 1. numpy/1.14.0
 2. pytorch/0.3.1.post2
 3. sklearn/0.19.dev0

GPU: all gpu when set nn_config['cuda'] to True and the pytorch can detect available gpu. The default gpu device is 0, and
can change the device to set nn_config['gpu'] to the gpu device number available.

# constraings
The k_shot should not be less or equal to 90

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

limitation to number of instance is 300.

Whole process of research:
pre-train the linear layer with bi-directional model by using training data/validation data used to initialize the k-means.
Then train the whole model with training data. To show if the model can have better performance when training data is small.
This more like the cascading classifier which also trains on validation and use exception to knn.


# mathmatical
## proto
why proto is ok? because the mapping learns all the non-linear part and it makes transfer the non-linearly separable to a 
linearly separable in high dimension space.

Train the bidirection on validation can make it learn non-linear on validation better and make sure the generalization better
by using share weight. Assume: without share weight, the result will be worse. Previously, the non-linear is learned
in training data, now learn it on validation data.

## cascading
why the cascading not good?
the model is linear one.
knn is a non-linear model but the non-linear is not learned. the shortage is that if there is overlap between classes, it will not easy to
 separate the class. just use exceptions to construct the knn, this also explain whey the model is not stable, because each time 
 the knn will be chosen.
 
## bidirectional
just use the directional to learn a better non-linear.