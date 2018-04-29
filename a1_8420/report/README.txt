If you run the program for several times, please first delete the old report.

This folder contains training result of cascading network and prototypical network.

report_proto_with_share.txt:
This file contains the result of prototypical network with shared weights network to pre-train it.
The report contains the loss, difference between original image and the predicted image, of test after each epoch when shared weights network pre-trains the prototypical network.
And after the pre-training process, the prototypical network will start training and the report will show the accuracy and f1 score of test after each training epoch.

report_proto.txt:
This file contains the result of prototypical network without pre-training.
The report contains the accuracy and f1 score of test after each training epoch when the prototypical network starts training.

report_cascade.txt:
This file contains the result of cascading network.
It contains the accuracy and f1 score of test after each training epoch when the cascading network starts training.