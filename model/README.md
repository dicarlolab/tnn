# UniCycle Architecture Builder
Universal Neural Interpretation and Cyclicity Engine (UNICYCLE)

Description coming soon, this stuff gets updated so often that I'll just write everything up when the MVP is rolling

-------

TO-DO

Things to do by next iteration of Unicycle:

- Look into Harbor generalization, merge Harbor_Dummy into Harbor

(1) Tests for small scale architectures like MNIST - make this a basic test.
This should test that standard MNIST works as promised and correctly trains to the end, and compares favorably to existing trained MNIST.

(2) Test MNIST + bypasses. This is primarily to ensure that the code works in a nontrivial circumstance, but should also achieve non-terrible numbers of possible.

(3) Test MNIST + feedbacks.

(4) Train on AlexNet and compare

(5) Train on DenseNet and compare

(6) Train on VGG and compare

For all things, make sure that:

-- code runs in a reasonable amount of time
-- using a reasonable amount of memory
-- appropriate training losses are achieved

NB: Items (1)-(3) above should be in the form of tests in the test suite