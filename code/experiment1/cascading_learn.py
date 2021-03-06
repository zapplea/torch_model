from cascading import Cascading
from data_feeder import DataFeeder

def main(data_config,nn_config):
    df = DataFeeder(data_config)
    cl = Cascading(nn_config,df)
    cl.classifier()

if __name__=="__main__":
    seed = {'batch_size':30}
    data_config = {'train_data':'../dataset/optdigits.tra',
                   'test_data':'../dataset/optdigits.tes',
                   'data_filePath':'../dataset/data.pkl',
                   'train_data_len': 1934,
                   'truncated_len': 1434,
                   'validation_data_len': 946,
                   'test_data_len': 1797,
                   'batch_size':seed['batch_size'],
                   'k_shot': 50
                   }

    nn_config = {'feature_dim':64,
                 'label_dim':10,
                 'lr':0.03,
                 'weight_decay':0.00003,
                 'epoch':60,
                 'mod':20,
                 'batch_size':seed['batch_size'],
                 'theta':0.6,
                 'train_data_len': 1934,
                 'validation_data_len': 946,
                 'test_data_len': 1797,
                 'neigh_num':3,
                 'report_filePath':'../report/report1_cascade.txt',
                 'cuda':False,}
    print('runing ...')
    main(data_config,nn_config)
    print('finished')
