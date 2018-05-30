If you run the program several times, please first delete the old report files in /code/report before run the program again.

This folder contains training result of cascading network and prototypical network.

report_proto_with_share.txt:
This file contains the result of prototypical network with shared weights network to pre-train it.
The report contains the accuracy and f1 score of test after each training epoch when the prototypical network starts training.

report_proto.txt:
This file contains the result of prototypical network without pre-training.
The report contains the accuracy and f1 score of test after each training epoch when the prototypical network starts training.

report_cascade.txt:
This file contains the result of cascading network.
It contains the accuracy and f1 score of test after each training epoch when the cascading network starts training.
