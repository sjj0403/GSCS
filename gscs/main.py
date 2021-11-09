import os

import config
import train
import eval
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
def _train(vocab_file_path=None, model_file_path=None):
    print('\nStarting the training process......\n')

    if vocab_file_path:
        code_vocab_path, nl_vocab_path = vocab_file_path
        print('Vocabulary will be built by given file path.')
        print('\tTokenized source code vocabulary path:\t', os.path.join(config.vocab_dir, code_vocab_path))
        print('\tCode comment vocabulary path:\t', os.path.join(config.vocab_dir, nl_vocab_path))
    else:
        print('Vocabulary will be built according to dataset.')

    if model_file_path:
        print('Model will be built by given state dict file path:', os.path.join(config.model_dir, model_file_path))
    else:
        print('Model will be created by program.')

    print('\nInitializing the training environments......\n')
    train_instance = train.Train(vocab_file_path=vocab_file_path, model_file_path=model_file_path)
    print('Environments built successfully.\n')
    print('Size of train dataset:', train_instance.train_dataset_size)
    print('\n type size:',train_instance.origin_type_vocab_size)
    print('\n code size:',train_instance.origin_code_vocab_size)
    print('\n nl size:',train_instance.origin_nl_vocab_size)


    if config.validate_during_train:
        print('\nValidate every', config.validate_every, 'batches and each epoch.')
        print('Size of validation dataset:', train_instance.eval_instance.dataset_size)
        config.logger.info('Size of validation dataset: {}'.format(train_instance.eval_instance.dataset_size))

    print('\nStart training......\n')
    config.logger.info('Start training.')
    best_model = train_instance.run_train()
    print('\nTraining is done.')
    config.logger.info('Training is done.')

    # writer = SummaryWriter('runs/CodePtr')
    # for _, batch in enumerate(train_instance.train_dataloader):
    #     batch_size = len(batch[0][0])
    #     writer.add_graph(train_instance.model, (batch, batch_size, train_instance.nl_vocab))
    #     break
    # writer.close()

    return best_model


def _test(model):
    print('\nInitializing the test environments......')
    test_instance = eval.Test(model)
    print('Environments built successfully.\n')
    print('Size of test dataset:', test_instance.dataset_size)
    config.logger.info('Size of test dataset: {}'.format(test_instance.dataset_size))

    config.logger.info('Start Testing.')
    print('\nStart Testing......')
    test_instance.run_test()
    print('Testing is done.')


if __name__ == '__main__':
    best_model_dict = _train()
    # best_model_dict = _train(model_file_path='/data2/java50/gat/model/20211002_151032/model_valid-bleu-0.1414_epoch-13_batch--1.pt')    
    # best_model_dict = '/data2/java50/gat/model/20211002_151032/model_valid-bleu-0.1503_epoch-4_batch-6000.pt'
    # best_model_d/ict = '/data2/java50/gat/model/20211002_151032/model_valid-bleu-0.1414_epoch-13_batch--1.pt'
    # best_model_dict = '/data2/java50/gat/model/20211005_012941/model_valid-bleu-0.1472_epoch-16_batch--1.pt'
    # best_model_dict = '/data2/java50/gat/model/20211005_012941/model_valid-bleu-0.1478_epoch-21_batch--1.pt'
    _test(best_model_dict)
    # _test(os.path.join('20200526_095918', 'best_epoch-2_batch-1000.pt'))
