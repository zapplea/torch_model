from data_feeder import DataFeeder
from super_proto import SuperPrototypicalNet

def main(nn_config,data_config):
    df = DataFeeder(data_config)
    cl = SuperPrototypicalNet(nn_config,df)
    cl.classifier()

if __name__ =="__main__":
    seed = {'batch_size': 30}
    data_config = {'train_data': '../dataset/optdigits.tra',
                   'test_data': '../dataset/optdigits.tes',
                   'data_filePath': '../dataset/data.pkl',
                   'train_data_len': 1934,
                   'truncated_len': 1434,
                   'validation_data_len': 946,
                   'test_data_len': 1797,
                   'batch_size':seed['batch_size']
                   }

    nn_configs =[
        {'feature_dim': 64,
         'layer_dim': [100],
         'label_dim': 10,
         'lr': 0.00003,
         'weight_decay': 0.00003,
         'epoch': 60,
         'comp_epoch':30,
         'mod': 20,
         'batch_size': seed['batch_size'],
         'train_data_len': 1934,
         'validation_data_len': 946,
         'test_data_len': 1797,
         'report_filePath': '../report/report1',
         'cuda': False,
         'k_shot': 50,
         'is_share_weight': True},

        {'feature_dim': 64,
         'layer_dim': [100],
         'label_dim': 10,
         'lr': 0.03,
         'weight_decay': 0.00003,
         'epoch': 60,
         'mod': 20,
         'batch_size': seed['batch_size'],
         'train_data_len': 1934,
         'validation_data_len': 946,
         'test_data_len': 1797,
         'report_filePath': '../report/report1',
         'cuda': False,
         'k_shot': 50,
         'is_share_weight': False},
        ]
    print('running ...')
    for i in range(len(nn_configs)):
        nn_config = nn_configs[i]
        if nn_config['is_share_weight']:
            nn_config['report_filePath']=nn_config['report_filePath']+'_proto_with_share.txt'
        else:
            nn_config['report_filePath']=nn_config['report_filePath']+'_proto.txt'
        main(nn_config,data_config)
    print('finished')
