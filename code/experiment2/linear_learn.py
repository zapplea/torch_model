from data_feeder import DataFeeder
from super_proto_linear import SuperPrototypicalNet

def main(nn_config,data_config):
    df = DataFeeder(data_config)
    cl = SuperPrototypicalNet(nn_config,df)
    cl.classifier()

if __name__ =="__main__":
    seed = {'batch_size': 30}
    data_config = data_config = {'data_filePath': '../dataset/data_mnist.pkl',
                   'batch_size':seed['batch_size']
                   }

    nn_configs =[
        {'connect_feature_dim': 1024,
         'feature_dim':784, # the dim of input image
         'label_dim': 10,
         'lr': 0.00003,
         'weight_decay': 0.00003,
         'epoch': 60,
         'comp_epoch':30,
         'mod': 20,
         'batch_size': seed['batch_size'],
         'query_data_len': 30000,
         'support_data_len':30000,
         'test_data_len': 10000,
         'report_filePath': '../report/report2',
         'cuda': True,
         'k_shot': 50,
         'is_share_weight': True},

        {'connect_feature_dim': 1024,
         'feature_dim': 784, # the dim of input image
         'label_dim': 10,
         'lr': 0.03,
         'weight_decay': 0.00003,
         'epoch': 60,
         'mod': 20,
         'batch_size': seed['batch_size'],
         'query_data_len': 30000,
         'support_data_len':30000,
         'test_data_len': 10000,
         'report_filePath': '../report/report2',
         'cuda': True,
         'k_shot': 50,
         'is_share_weight': False},
        ]
    print('running ...')
    for i in range(len(nn_configs)):
        nn_config = nn_configs[i]
        if nn_config['is_share_weight']:
            nn_config['report_filePath']=nn_config['report_filePath']+'_share-proto_with_linear.txt'
        else:
            nn_config['report_filePath']=nn_config['report_filePath']+'_proto_with_linear.txt'
        main(nn_config,data_config)
    print('finished')
