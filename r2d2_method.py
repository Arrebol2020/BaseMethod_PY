import torch
import cv2
import numpy as np
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvf
from PIL import Image


class BaseNet (nn.Module):
    """ Takes a list of images as input, and returns for each image:
        - a pixelwise descriptor
        - a pixelwise confidence
    """
    def softmax(self, ux):
        if ux.shape[1] == 1:
            x = F.softplus(ux)
            return x / (1 + x)  # for sure in [0,1], much less plateaus than softmax
        elif ux.shape[1] == 2:
            return F.softmax(ux, dim=1)[:,1:2]

    def normalize(self, x, ureliability, urepeatability):
        return dict(descriptors = F.normalize(x, p=2, dim=1),
                    repeatability = self.softmax( urepeatability ),
                    reliability = self.softmax( ureliability ))

    def forward_one(self, x):
        raise NotImplementedError()

    def forward(self, imgs, **kw):
        res = [self.forward_one(img) for img in imgs]
        # merge all dictionaries into one
        res = {k:[r[k] for r in res if k in r] for k in {k for r in res for k in r}}
        return dict(res, imgs=imgs, **kw)



class PatchNet (BaseNet):
    """ Helper class to construct a fully-convolutional network that
        extract a l2-normalized patch descriptor.
    """
    def __init__(self, inchan=3, dilated=True, dilation=1, bn=True, bn_affine=False):
        BaseNet.__init__(self)
        self.inchan = inchan
        self.curchan = inchan
        self.dilated = dilated
        self.dilation = dilation
        self.bn = bn
        self.bn_affine = bn_affine
        self.ops = nn.ModuleList([])

    def _make_bn(self, outd):
        return nn.BatchNorm2d(outd, affine=self.bn_affine)

    def _add_conv(self, outd, k=3, stride=1, dilation=1, bn=True, relu=True, k_pool = 1, pool_type='max'):
        # as in the original implementation, dilation is applied at the end of layer, so it will have impact only from next layer
        d = self.dilation * dilation
        if self.dilated: 
            conv_params = dict(padding=((k-1)*d)//2, dilation=d, stride=1)
            self.dilation *= stride
        else:
            conv_params = dict(padding=((k-1)*d)//2, dilation=d, stride=stride)
        self.ops.append( nn.Conv2d(self.curchan, outd, kernel_size=k, **conv_params) )
        if bn and self.bn: self.ops.append( self._make_bn(outd) )
        if relu: self.ops.append( nn.ReLU(inplace=True) )
        self.curchan = outd
        
        if k_pool > 1:
            if pool_type == 'avg':
                self.ops.append(torch.nn.AvgPool2d(kernel_size=k_pool))
            elif pool_type == 'max':
                self.ops.append(torch.nn.MaxPool2d(kernel_size=k_pool))
            else:
                print(f"Error, unknown pooling type {pool_type}...")
    
    def forward_one(self, x):
        assert self.ops, "You need to add convolutions first"
        for n,op in enumerate(self.ops):
            x = op(x)
        return self.normalize(x)


class L2_Net (PatchNet):
    """ Compute a 128D descriptor for all overlapping 32x32 patches.
        From the L2Net paper (CVPR'17).
    """
    def __init__(self, dim=128, **kw ):
        PatchNet.__init__(self, **kw)
        add_conv = lambda n,**kw: self._add_conv((n*dim)//128,**kw)
        add_conv(32)
        add_conv(32)
        add_conv(64, stride=2)
        add_conv(64)
        add_conv(128, stride=2)
        add_conv(128)
        add_conv(128, k=7, stride=8, bn=False, relu=False)
        self.out_dim = dim


class Quad_L2Net (PatchNet):
    """ Same than L2_Net, but replace the final 8x8 conv by 3 successive 2x2 convs.
    """
    def __init__(self, dim=128, mchan=4, relu22=False, **kw ):
        PatchNet.__init__(self, **kw)
        self._add_conv(  8*mchan)
        self._add_conv(  8*mchan)
        self._add_conv( 16*mchan, stride=2)
        self._add_conv( 16*mchan)
        self._add_conv( 32*mchan, stride=2)
        self._add_conv( 32*mchan)
        # replace last 8x8 convolution with 3 2x2 convolutions
        self._add_conv( 32*mchan, k=2, stride=2, relu=relu22)
        self._add_conv( 32*mchan, k=2, stride=2, relu=relu22)
        self._add_conv(dim, k=2, stride=2, bn=False, relu=False)
        self.out_dim = dim



class Quad_L2Net_ConfCFS (Quad_L2Net):
    """ Same than Quad_L2Net, with 2 confidence maps for repeatability and reliability.
    """
    def __init__(self, **kw ):
        Quad_L2Net.__init__(self, **kw)
        # reliability classifier
        self.clf = nn.Conv2d(self.out_dim, 2, kernel_size=1)
        # repeatability classifier: for some reasons it's a softplus, not a softmax!
        # Why? I guess it's a mistake that was left unnoticed in the code for a long time...
        self.sal = nn.Conv2d(self.out_dim, 1, kernel_size=1) 

    def forward_one(self, x):
        assert self.ops, "You need to add convolutions first"
        for op in self.ops:
            x = op(x)
        # compute the confidence maps
        ureliability = self.clf(x**2)
        urepeatability = self.sal(x**2)
        return self.normalize(x, ureliability, urepeatability)


class Fast_Quad_L2Net (PatchNet):
    """ Faster version of Quad l2 net, replacing one dilated conv with one pooling to diminish image resolution thus increase inference time
    Dilation  factors and pooling:
        1,1,1, pool2, 1,1, 2,2, 4, 8, upsample2
    """
    def __init__(self, dim=128, mchan=4, relu22=False, downsample_factor=2, **kw ):

        PatchNet.__init__(self, **kw)
        self._add_conv(  8*mchan)
        self._add_conv(  8*mchan)
        self._add_conv( 16*mchan, k_pool = downsample_factor) # added avg pooling to decrease img resolution
        self._add_conv( 16*mchan)
        self._add_conv( 32*mchan, stride=2)
        self._add_conv( 32*mchan)
        
        # replace last 8x8 convolution with 3 2x2 convolutions
        self._add_conv( 32*mchan, k=2, stride=2, relu=relu22)
        self._add_conv( 32*mchan, k=2, stride=2, relu=relu22)
        self._add_conv(dim, k=2, stride=2, bn=False, relu=False)
        
        # Go back to initial image resolution with upsampling
        self.ops.append(torch.nn.Upsample(scale_factor=downsample_factor, mode='bilinear', align_corners=False))
        
        self.out_dim = dim
        
        
class Fast_Quad_L2Net_ConfCFS (Fast_Quad_L2Net):
    """ Fast r2d2 architecture
    """
    def __init__(self, **kw ):
        Fast_Quad_L2Net.__init__(self, **kw)
        # reliability classifier
        self.clf = nn.Conv2d(self.out_dim, 2, kernel_size=1)
        
        # repeatability classifier: for some reasons it's a softplus, not a softmax!
        # Why? I guess it's a mistake that was left unnoticed in the code for a long time...
        self.sal = nn.Conv2d(self.out_dim, 1, kernel_size=1) 
        
    def forward_one(self, x):
        assert self.ops, "You need to add convolutions first"
        for op in self.ops:
            x = op(x)
        # compute the confidence maps
        ureliability = self.clf(x**2)
        urepeatability = self.sal(x**2)
        return self.normalize(x, ureliability, urepeatability)


def model_size(model):
    ''' Computes the number of parameters of the model 
    '''
    size = 0
    for weights in model.state_dict().values():
        size += np.prod(weights.shape)
    return size


def load_network(model_fn): 
    checkpoint = torch.load(model_fn)
    print("\n>> Creating net = " + checkpoint['net']) 
    net = eval(checkpoint['net'])
    nb_of_weights = model_size(net)
    print(f" ( Model size: {nb_of_weights/1000:.0f}K parameters )")

    # initialization
    weights = checkpoint['state_dict']
    net.load_state_dict({k.replace('module.',''):v for k,v in weights.items()})
    return net.eval()


class NonMaxSuppression (torch.nn.Module):
    def __init__(self, rel_thr=0.7, rep_thr=0.7):
        torch.nn.Module.__init__(self)
        self.max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.rel_thr = rel_thr
        self.rep_thr = rep_thr
    
    def forward(self, reliability, repeatability, **kw):
        assert len(reliability) == len(repeatability) == 1
        reliability, repeatability = reliability[0], repeatability[0]

        # local maxima
        maxima = (repeatability == self.max_filter(repeatability))

        # remove low peaks
        maxima *= (repeatability >= self.rep_thr)
        maxima *= (reliability   >= self.rel_thr)

        return maxima.nonzero().t()[2:4]


RGB_mean = [0.485, 0.456, 0.406]
RGB_std  = [0.229, 0.224, 0.225]
norm_RGB = tvf.Compose([tvf.ToTensor(), tvf.Normalize(mean=RGB_mean, std=RGB_std)])


def extract_multiscale( net, img, detector, scale_f=2**0.25, 
                        min_scale=0.0, max_scale=1, 
                        min_size=256, max_size=1024, 
                        verbose=False):
    old_bm = torch.backends.cudnn.benchmark 
    torch.backends.cudnn.benchmark = False # speedup
    
    # extract keypoints at multiple scales
    B, three, H, W = img.shape
    assert B == 1 and three == 3, "should be a batch with a single RGB image"
    
    assert max_scale <= 1
    s = 1.0 # current scale factor
    
    X,Y,S,C,Q,D = [],[],[],[],[],[]
    while  s+0.001 >= max(min_scale, min_size / max(H,W)):
        if s-0.001 <= min(max_scale, max_size / max(H,W)):
            nh, nw = img.shape[2:]
            if verbose: print(f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}")
            # extract descriptors
            with torch.no_grad():
                res = net(imgs=[img])
                
            # get output and reliability map
            descriptors = res['descriptors'][0]
            reliability = res['reliability'][0]
            repeatability = res['repeatability'][0]

            # normalize the reliability for nms
            # extract maxima and descs
            y,x = detector(**res) # nms
            c = reliability[0,0,y,x]
            q = repeatability[0,0,y,x]
            d = descriptors[0,:,y,x].t()
            n = d.shape[0]

            # accumulate multiple scales
            X.append(x.float() * W/nw)
            Y.append(y.float() * H/nh)
            S.append((32/s) * torch.ones(n, dtype=torch.float32, device=d.device))
            C.append(c)
            Q.append(q)
            D.append(d)
        s /= scale_f

        # down-scale the image for next iteration
        nh, nw = round(H*s), round(W*s)
        img = F.interpolate(img, (nh,nw), mode='bilinear', align_corners=False)

    # restore value
    torch.backends.cudnn.benchmark = old_bm

    Y = torch.cat(Y)
    X = torch.cat(X)
    S = torch.cat(S) # scale
    scores = torch.cat(C) * torch.cat(Q) # scores = reliability * repeatability
    XYS = torch.stack([X,Y,S], dim=-1)
    D = torch.cat(D)
    return XYS, D, scores


# https://github.com/naver/r2d2#feature-extraction-with-kapture-datasets
def r2d2_m(img_path):
  #img_path = r"E:\datasets\bop_datasets\ycbv\crop_test\15\img\000050_000001.png"

  reliability_thr = 0.7
  repeatability_thr = 0.7
  scale_f = 2**0.25
  min_scale = 0
  max_scale = 1
  min_size = 256
  max_size = 1024


  # CUDA
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda:0" if use_cuda else "cpu")
  model = r'C:\Users\DELL\Desktop\projs\BaseMethod_PY\weights\r2d2\r2d2_WASF_N16.pt'
  net = load_network(model)

  net.to(device)

  # create the non-maxima detector
  detector = NonMaxSuppression(
      rel_thr = reliability_thr, 
      rep_thr = repeatability_thr)
  img = Image.open(img_path).convert('RGB')
  W, H = img.size
  img = norm_RGB(img)[None] 
  img = img.to(device)

  xys, desc, scores = extract_multiscale(net, img, detector,
            scale_f   = scale_f, 
            min_scale = min_scale, 
            max_scale = max_scale,
            min_size  = min_size, 
            max_size  = max_size, 
            verbose = True)
            
  xys = xys.cpu().numpy()

  img = cv2.imread(img_path)
  kps = []
  for x, y, scale in xys:
    #print(x, y, scale)
    kp = cv2.KeyPoint(x, y, 0)
    kps.append(kp)
  img_keypoints = np.empty((img.shape[0], img.shape[1], 3), dtype=np.uint8)
  cv2.drawKeypoints(img, kps, img_keypoints)
  #cv2.imwrite("keynet.png", img_keypoints)
  cv2.imshow("r2d2", img_keypoints)
  cv2.waitKey(0)
  print()