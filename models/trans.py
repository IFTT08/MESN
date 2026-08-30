'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import model_dict
from models.util import FcLayer,ReLayer1,SepFcLayer,BottleLayer,MobileFcLayer,WrnFcLayer,ICLayer4,SepAtt,ShuFcLayer
from models.initializer import identity_init, ones_w_zeros_b_init, permutation_init, \
    random_permutation_mask_init, PsInvInit, SemiMatchMaskInit, AbsSemiMatchMaskInit
from .general import GeneralNet
from models.transform import Transform

class ShuffleBlock(nn.Module):
    def __init__(self, groups=2):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C//g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)
class Transform46(GeneralNet):
    def __init__(self, in_planes, planes):
        super(Transform46, self).__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, stride=1),
            nn.BatchNorm2d(planes, affine=True),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.transform(x)

class simfc(nn.Module):
    def __init__(self, in_planes, planes):
        super(simfc, self).__init__()
        self.op = nn.Sequential(
            nn.Linear(in_planes, planes)
        )
    def forward(self, x):

        return self.op(x)

class SimilarityNet(nn.Module):
    """ This class is responsible for inserting a linear transformation
        layer between two networks with identical architecture. The user has
        a choice to index at which layer should the transformation happen.
        If the indexed layer is a convolution, the first model is transferred
        to the second one by a conv1x1 layer. If the layer is not a convolution
        but a linear layer, the transformation will be a linear layer too.
    """
    # ==============================================================
    # CONSTRUCTORS
    # ==============================================================
    def __init__(self,
                 backbone_model_path,
                 fc_path,
                 transform46_path,
                 front_layer_name,
                 end_layer_name,
                 dataset_name: str,
                 n_cls,
                 channel,
                 init='random',
):
        super().__init__()
        # Init variables
        self.backbone_model_path = backbone_model_path
        self.fc_path=fc_path
        self.transform46_path=transform46_path
        self.front_layer_name = front_layer_name
        self.end_layer_name = end_layer_name
        self.dataset_name = dataset_name
        self.n_cls = n_cls
        self.channel = channel
        self.init_type = init
        self.init_front_layer_name="conv2.1.pointwise.2"
        self.init_end_layer_name = "conv2.0.op.7"


        # Derived variables
        self.front_model_backbone = self.load_backbone(self.backbone_model_path,self.n_cls)
        self.front_model_fc = self.load_fc(self.fc_path)

        self.end_model_backbone = self.load_backbone(self.backbone_model_path, self.n_cls)
        self.end_model_fc = self.load_fc(self.fc_path)

        # Define transformation layer
        self.intitfront = self._init_front()
        self.intitend = self._init_end()

        self.transform = self._get_transform_layer(self.channel, self.channel)
        #self.transform46 = self.load_S46(self.transform46_path)



        self.layer_names = [self.front_layer_name, self.end_layer_name]
        self.models = [self.front_model_backbone, self.front_model_fc, self.end_model_backbone, self.end_model_fc]
        # Prepare models and layers for transformation
        self.transform_input = None  # The tensor passed to transformation
        self.forced_output = None  # The tensor passed from front to end
        self.connection_enabled = False  # Either tensor should be passed or not
        self.last_m2_out = None
        self.prepare_models()



    # ==============================================================
    # PUBLIC FUNCTIONS
    # ==============================================================
    def load_backbone(self,model_path, n_cls):
        print('==> loading teacher model')
        model_t = self.get_teacher_name(model_path)
        model_t=model_t
        model = model_dict[model_t](num_classes=n_cls)
        model.load_state_dict(torch.load(model_path)['model'])
        print('==> done')
        return model
    def get_teacher_name(self,model_path):
        """parse teacher name"""
        segments = model_path.split('/')[-2].split('_')
        if segments[0] != 'wrn':
            return segments[0]
        else:
            return segments[0] + '_' + segments[1] + '_' + segments[2]

    def load_fc(self,model_path):
        print('==> loading fc model')
        fclayer = ReLayer1(num_classes=self.n_cls)
        fclayer.load_state_dict(torch.load(model_path)['model'])
        print('==> done')
        return fclayer

    def load_S6(self,model_path, n_cls):
        print('==> loading teacher model')
        model_t = self.get_teacher_name(model_path)
        model_t = model_t
        modelSpe = model_dict[model_t](num_classes=n_cls)
        # modelSpe.load_state_dict(torch.load(model_path)['model'],strict=False)
        print('==> done')
        return modelSpe

    def load_S4(self,model_path, n_cls):
        print('==> loading teacher model')
        model_t = self.get_teacher_name(model_path)
        model_t = model_t
        modelSpe = model_dict[model_t](num_classes=n_cls)
        #modelSpe.load_state_dict(torch.load(model_path)['model'],strict=False)
        print('==> done')
        return modelSpe

    def load_S46(self,model_path):
        print('==> loading teacher model')
        transform46 = Transform46(128,128)
        print("transform46",transform46)
        transform46.load_state_dict(torch.load(model_path)['model'])
        print('==> load_S46 done')
        return transform46

    def forward(self, orig_input):
        # Enabling overriding activations


        self.connection_enabled = True
        # Get middle activations & save middle activation
        feat_f, logit_f = self.front_model_backbone(orig_input, is_feat=True, preact=False)
        self.front_model_fc(feat_f)
        # Transform the activations and save it
        self.forced_output = self.transform(self.transform_input)


        #print("self.forced_output",self.forced_output)

        # Load forced output from hook end run the rest of the model
        feat_f, logit_f = self.end_model_backbone(orig_input, is_feat=True, preact=False)
        outputstage1, outputs_feature = self.end_model_fc(feat_f)

        #feat_f[2]=self.transform46(outputs_feature[2])
        #outputs, outputs_feature = self.end_model_fc(feat_f)

        # Disabling overriding activations
        self.connection_enabled = False
        return outputstage1



    def prepare_models(self):

        self._register_connection()
        for model in self.models:
            model.eval_mode()


    # ==============================================================
    # PRIVATE HELPERS
    # ==============================================================

    def _register_connection(self):
        ''' Register activation save on each neuron '''
        self._register_activation_save(self.front_layer_name)
        self._register_activation_load(self.end_layer_name)



    def _register_activation_save(self,front_layer_name):
        def save_activation(module, m_in, m_out):
            if self.connection_enabled:
                self.transform_input = m_out
                #print("m_out",m_out)

        connect_layer = self.front_model_backbone.get_layer(front_layer_name)
        connect_layer.register_forward_hook(save_activation)


    def _register_activation_load(self,end_layer_name):
        def override_activation(module, m_in, m_out):
            saved_output_exist = self.forced_output is not None
            should_override = saved_output_exist and self.connection_enabled
            activation = self.forced_output if should_override else m_out
            self.last_m2_out = m_out
            return activation

        connect_layer = self.end_model_fc.get_layer(self.end_layer_name)
        connect_layer.register_forward_hook(override_activation)

    def _get_transform_layer(self,input,output):
        # Calculate each layers shape
        #front_shape, end_shape = self._determine_trans_shapes()

        # transform_layer = Transform(front_shape,
        #                             end_shape,
        #                             )
        transform_layer = Transform(input,
                                    output,
                                    init_fn=self._get_transform_init(),
                                    )
        return transform_layer

    def _init_front(self):
        print('==> loading fc model')
        modelfront = self.load_S4(self.backbone_model_path, 10)
        modelfront.conv1 = self.front_model_backbone.conv1
        modelfront.layer1 = self.front_model_backbone.layer1
        modelfront.layer2 = self.front_model_backbone.layer2
        modelfront.layer3 = self.front_model_fc.scala4
        modelfront.layer4 = self.front_model_fc.scala5
        modelfront.linear = self.front_model_fc.fc2
        return modelfront

    def _init_end(self):
        print('==> loading fc model')
        modelend = self.load_S6(self.backbone_model_path, 10)
        modelend.conv1= self.front_model_backbone.conv1
        modelend.layer1 = self.front_model_backbone.layer1
        modelend.layer2 = self.front_model_fc.scala1
        modelend.layer3 = self.front_model_fc.scala2
        modelend.layer4 = self.front_model_fc.scala3
        modelend.linear = self.front_model_fc.fc1
        # transform46.load_state_dict(torch.load(model_path)['model'],strict=False)
        return modelend


    def _get_transform_init(self):
        name = self.init_type.lower()
        if name == 'random':
            return None  # this is the default behaviour of Transform
        elif name in ['identity', 'eye']:
            return identity_init
        elif name in ['ones-zeros']:
            return ones_w_zeros_b_init
        elif name in ['perm' or 'permutation']:
            return permutation_init
        elif name in ['ps_inv' or 'pseudo_inverse']:
            return PsInvInit(self.intitfront.cuda(),
                             self.intitend.cuda(),
                             self.init_front_layer_name,
                             self.init_end_layer_name,
                             self.dataset_name,
                             dataset_type="train",
                             )
        else:
            raise ValueError('Initializer {} is unknown.'.format(name))

    def _determine_trans_shapes(self):
        # Get input shape
        #inp_shape = get_datasets(self.dataset_name)['train'][0][0].shape
        inp_shape =  torch.Tensor(3,32,32).shape
        # Ask models to save activation shapes and make a forward pass
        for model in self.models:
            model.register_shape_fw_hooks()
            print(0)
            model.register_order_fw_hooks()
            print(1)
            model.simulate_forward_pass(inp_shape)
            print(2)
        # Save shapes
        front_shape = self.front_model.shapes[self.front_layer_name]['out']
        end_shape = self.end_model.shapes[self.end_layer_name]['out']
        # Remove forward hooks to speed up networks
        for model in self.models:
            model.remove_shape_fw_hooks()
            model.remove_order_fw_hooks()

        return front_shape, end_shape