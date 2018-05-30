from data_feeder import DataFeeder
from super_proto_cnn import SuperPrototypicalNet

def main(nn_config,data_config):
    df = DataFeeder(data_config)
    cl = SuperPrototypicalNet(nn_config,df)
    cl.classifier()

if __name__ =="__main__":
    seed = {'batch_size': 30}
    data_config = {'data_filePath': '../dataset/data_mnist.pkl',
                   'batch_size':seed['batch_size'],
                   'query_data_len': 30000,
                   'support_data_len': 30000,
                   'test_data_len': 10000,
                   }

    nn_configs =[
        {'cnn_feature_dim': 7*7*64,
         'connect_layer_dim': 1024,
         'feature_height_dim': 28,  # height of input images
         'feature_width_dim':28, # width of input images
         'label_dim': 10,
         'lr': 0.00003,
         'weight_decay': 0.00003,
         'epoch': 60,
         'comp_epoch':30,
         'mod': 20,
         'batch_size': seed['batch_size'],
         'report_filePath': '../report/report2',
         'cuda': True,
         'k_shot': 50,
         'is_share_weight': True},
        
        {'cnn_feature_dim': 7 * 7 * 64,
         'connect_layer_dim': 1024,
         'feature_height_dim': 28,  # height of input images
         'feature_width_dim': 28,  # width of input images
         'label_dim': 10,
         'lr': 0.03,
         'weight_decay': 0.00003,
         'epoch': 60,
         'mod': 20,
         'batch_size': seed['batch_size'],
         'report_filePath': '../report/report2',
         'cuda': True,
         'k_shot': 50,
         'is_share_weight': False},
        ]
    print('running ...')
    for i in range(len(nn_configs)):
        nn_config = nn_configs[i]
        if nn_config['is_share_weight']:
            nn_config['report_filePath']=nn_config['report_filePath']+'_share-proto_with_cnn.txt'
        else:
            nn_config['report_filePath']=nn_config['report_filePath']+'_proto_with_cnn.txt'
        main(nn_config,data_config)
    print('finished')
