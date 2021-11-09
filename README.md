# Dataset
We conduct experiments on three different public dataset on the task of code summarization.
1. The first dataset can be accessible from the paper https://arxiv.org/abs/2005.00653
2. The second dataset can be accessible from the paper https://arxiv.org/abs/2004.02843
3. The third dataset can be accessible from the paper https://arxiv.org/abs/1909.09436v2
# Run the code
1. To different dataset, you can change its parameter setting in the file gscs/config.py.
2. To train the model, you can set the main function to run _train() in the file gscs/main.py.
3. To test the model, you can set the main function to run _test() and set the picked model file's path in the file gscs/main.py.
4. To evaluate the result, you can use three metrics in the floder evaluation.
